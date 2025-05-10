import numpy as np
import torch.nn as nn
from tqdm import tqdm

from src.gd.networks import get_network
from src.neugrasp.network.LoFTR import Attention
from src.neugrasp.network.aggregate_net import name2agg_net
from src.neugrasp.network.feature_volume import name2feature_volume
from src.neugrasp.network.img_encoder import name2img_encoder
from src.neugrasp.network.render_ops import *
from src.neugrasp.utils.field_utils import TSDF_SAMPLE_POINTS


class NeuGraspRenderer(nn.Module):
    base_cfg = {
        'agg_net_type': 'default',
        'agg_net_cfg': {},

        'fine_depth_use_all': False,

        'use_ray_mask': True,
        'ray_mask_view_num': 1,  # NOTE
        'ray_mask_point_num': 8,

        'disable_view_dir': False,

        'img_encoder_type': 'res_unet',
        'img_encoder_cfg': {},

        'feature_volume_type': '3d_unet',
        'feature_volume_cfg': {},
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.image_encoder = name2img_encoder[self.cfg['img_encoder_type']](self.cfg['img_encoder_cfg'])
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg)
        self.feature_volume = name2feature_volume[self.cfg['feature_volume_type']](self.cfg['feature_volume_cfg'])
        self.attention = Attention(32, 0.)

        self.use_sdf = self.cfg['agg_net_type'] in ['neus']

    def get_img_feats(self, ref_imgs_info, prj_dict):
        rfn, _, h, w = ref_imgs_info['imgs'].shape
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape

        img_feats = ref_imgs_info['img_feats']
        prj_img_feats = interpolate_feature_map(img_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                prj_dict['mask'].reshape(rfn, qn * rn * dn), h, w, )
        prj_dict['img_feats'] = prj_img_feats.reshape(rfn, qn, rn, dn, -1)

        resi_feats = ref_imgs_info['residual_feats']
        prj_resi_feats = interpolate_feature_map(resi_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                 prj_dict['mask'].reshape(rfn, qn * rn * dn), h, w, )
        prj_dict['residual_feats'] = prj_resi_feats.reshape(rfn, qn, rn, dn, -1)
        return prj_dict

    def network_rendering(self, prj_dict, que_dir, que_pts, que_depth, is_fine, is_train, is_sdf=False, sdf_only=False,
                          feat_volume=None, residual_volume=None):
        net = self.agg_net  # prj_dict, que_dir, que_pts, que_dists, feature_volume, residual_volume, is_train
        que_dists = depth2dists(que_depth) if que_depth is not None else None
        rendering_outputs = net(prj_dict, que_dir, que_pts, que_dists, feat_volume, residual_volume, is_train)
        outputs = {}
        if is_sdf:
            alpha_values, outputs['sdf_values'], colors, outputs['sdf_gradient_error'], outputs['s'] = rendering_outputs
            if sdf_only:
                return outputs
        else:
            density, colors = rendering_outputs
            alpha_values = 1.0 - torch.exp(-torch.relu(density))

        outputs['alpha_values'] = alpha_values
        outputs['colors_nr'] = colors
        outputs['hit_prob_nr'] = hit_prob = alpha_values2hit_prob(alpha_values)
        outputs['pixel_colors_nr'] = torch.sum(hit_prob.unsqueeze(-1) * colors, 2)  # synthetic novel-view pixels
        outputs['pixel_depths_nr'] = torch.sum(hit_prob * que_depth, -1).unsqueeze(-1)
        return outputs

    def render_by_depth(self, que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine):
        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        que_dists = depth2inv_dists(que_depth, que_imgs_info['depth_range'])  # [qn, rn, dn]
        # generate points with query depth
        que_pts, que_dir = depth2points(que_imgs_info, que_depth)
        if self.cfg['disable_view_dir']:
            que_dir = None
        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)

        outputs = self.network_rendering(prj_dict, que_dir, que_pts, que_depth, is_fine, is_train, is_sdf=self.use_sdf,
                                         feat_volume=ref_imgs_info['feature_volume'],
                                         residual_volume=ref_imgs_info['residual_volume'])

        if 'imgs' in que_imgs_info:
            outputs['pixel_colors_gt'] = interpolate_feats(  # pixel_colors_gt == pixel_colors_gt_fine
                que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)

        if 'depth' in que_imgs_info:
            outputs['pixel_depths_gt'] = interpolate_feats(
                que_imgs_info['depth'], que_imgs_info['coords'], align_corners=True)  # [qn, rn, 1]

        if self.cfg['use_ray_mask']:
            outputs['ray_mask'] = torch.sum(prj_dict['mask'].int(), 0) > self.cfg['ray_mask_view_num']  # qn,rn,dn,1
            outputs['ray_mask'] = torch.sum(outputs['ray_mask'], 2) > self.cfg['ray_mask_point_num']  # qn,rn,1
            outputs['ray_mask'] = outputs['ray_mask'][..., 0]  # qn,rn

        if self.cfg['render_depth']:
            # qn,rn,dn
            outputs['render_depth'] = torch.sum(outputs['hit_prob_nr'] * que_depth,
                                                -1)  # qn,rn     used to calculate depth_mae

        outputs['que_points'] = que_pts  # qn,rn,dn,3
        return outputs

    def fine_render_impl(self, coarse_render_info, que_imgs_info, ref_imgs_info, is_train):
        fine_depth = sample_fine_depth(coarse_render_info['depth'], coarse_render_info['hit_prob'].detach(),
                                       que_imgs_info['depth_range'], self.cfg['fine_depth_sample_num'], is_train)

        # qn, rn, fdn+dn
        if self.cfg['fine_depth_use_all']:  # NOTE: in original implementation, this is set to False
            que_depth = torch.sort(torch.cat([coarse_render_info['depth'], fine_depth], -1), -1)[0]
        else:
            que_depth = torch.sort(fine_depth, -1)[0]
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, True)
        return outputs

    def render_impl(self, que_imgs_info, ref_imgs_info, is_train):
        # [qn,rn,dn]
        # sample points along test ray at different depth
        que_depth, _ = sample_depth(que_imgs_info['depth_range'], que_imgs_info['coords'], self.cfg['depth_sample_num'],
                                    False)
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, False)
        if self.cfg['use_hierarchical_sampling']:
            coarse_render_info = {'depth': que_depth, 'hit_prob': outputs['hit_prob_nr']}
            fine_outputs = self.fine_render_impl(coarse_render_info, que_imgs_info, ref_imgs_info, is_train)  # --------
            for k, v in fine_outputs.items():
                outputs[k + "_fine"] = v
        return outputs

    def sample_volume(self, ref_imgs_info):
        ref_imgs_info = ref_imgs_info.copy()
        res = self.cfg['feature_volume_cfg']['volume_resolution']
        # NOTE: the second to last dimension represents z-axis
        que_pts = (torch.from_numpy(TSDF_SAMPLE_POINTS).to(ref_imgs_info['imgs'].device) +
                   torch.tensor(ref_imgs_info['bbox3d'][0], device=ref_imgs_info['imgs'].device)
                   ).reshape(1, res * res, res, 3)
        que_pts = torch.flip(que_pts, (2,))

        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)
        valid_ratio = torch.sum(prj_dict['mask'], dim=(1, 2, 3, 4)) / np.prod(list(prj_dict['mask'].shape)[1:])
        if torch.mean(valid_ratio) < 0.5:
            tqdm.write("!! too low ratio", valid_ratio)

        que_dir = torch.tensor([0, 0, 1], device=que_pts.device).reshape(1, 1, 1, 3).repeat(1, res * res, res, 1) if \
            not self.cfg['disable_view_dir'] else None

        feat_list = []
        mode = self.cfg['volume_type']
        if 'image' in mode:
            image_feat = torch.cat([prj_dict['rgb'], prj_dict['img_feats']], dim=-1)
            mean = torch.mean(image_feat, dim=-1)
            var = torch.var(image_feat, dim=-1)
            feat_list.append(torch.cat([image_feat, mean, var], dim=-1).reshape(1, res, res, res, -1).permute(1, -1))

        if 'alpha' in mode:
            outputs = self.network_rendering(prj_dict, que_dir, que_pts, None, False, False)
            feat_list.append(outputs['alpha_values'].reshape(1, 1, res, res, res))

        if 'sdf' in mode:
            outputs = self.network_rendering(prj_dict, que_dir, que_pts, None, False, False, is_sdf=self.use_sdf,
                                             sdf_only=True,
                                             feat_volume=ref_imgs_info['feature_volume'],
                                             residual_volume=ref_imgs_info['residual_volume'])
            feat_list.append(outputs['sdf_values'].reshape(1, 1, res, res, res))

        feat = torch.cat(feat_list, dim=1)
        feat = torch.flip(feat, (-1,))  # we sample from top to down, so we need to flip here
        return feat

    def render(self, que_imgs_info, ref_imgs_info, is_train):
        render_info_all = {}
        ray_batch_num = self.cfg["ray_batch_num"]  # 4096
        coords = que_imgs_info['coords']
        ray_num = coords.shape[1]  # train: 512  val: 288 * 512

        for ray_id in range(0, ray_num, ray_batch_num):  # train: all in  val: input a fixed number of rays per iter
            que_imgs_info['coords'] = coords[:, ray_id:ray_id + ray_batch_num]
            render_info = self.render_impl(que_imgs_info, ref_imgs_info, is_train)
            output_keys = [k for k in render_info.keys()]
            for k in output_keys:
                v = render_info[k]
                if k not in render_info_all:
                    render_info_all[k] = []
                render_info_all[k].append(v)

        for k, v in render_info_all.items():
            render_info_all[k] = torch.cat(v, 1)

        return render_info_all

    def gen_depth_loss_coords(self, h, w, device):
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).reshape(-1, 2).to(device)
        num = self.cfg['depth_loss_coords_num']
        idxs = torch.randperm(coords.shape[0])
        idxs = idxs[:num]
        coords = coords[idxs]
        return coords

    def forward(self, data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data
        src_imgs_info = data['src_imgs_info'].copy() if 'src_imgs_info' in data else None

        # extract image feature, input: [4, 3, 288, 512], output:[4, 32, 72, 128]
        ref_imgs_info['img_feats'] = self.image_encoder(ref_imgs_info['imgs'])
        ref_imgs_info['bg_feats'] = self.image_encoder(ref_imgs_info['bgs'])
        ref_imgs_info['feature_volume'] = self.feature_volume(ref_imgs_info['img_feats'], ref_imgs_info['poses'],
                                                              ref_imgs_info['Ks'], ref_imgs_info['bbox3d'],
                                                              ref_imgs_info['imgs'].shape)
        ref_imgs_info['residual_volume'] = self.feature_volume(ref_imgs_info['img_feats'] - ref_imgs_info['bg_feats'],
                                                               ref_imgs_info['poses'],
                                                               ref_imgs_info['Ks'], ref_imgs_info['bbox3d'],
                                                               ref_imgs_info['imgs'].shape)

        ref_imgs_info['residual_feats'] = self.attention(ref_imgs_info['bg_feats'],
                                                         ref_imgs_info['img_feats'],
                                                         pos=None)

        render_outputs = {}

        if self.cfg['render_rgb']:
            render_outputs = self.render(que_imgs_info, ref_imgs_info, is_train)

        if self.cfg['sample_volume']:
            render_outputs['volume'] = self.sample_volume(ref_imgs_info)

        render_outputs['img_feats'] = ref_imgs_info['img_feats']

        return render_outputs


class NeuGrasp(nn.Module):
    default_cfg_vgn = {
        'nr_initial_training_steps': 0,
        'freeze_nr_after_init': False
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg_vgn, **cfg}
        self.nr_net = NeuGraspRenderer(self.cfg)
        self.vgn_net = get_network("conv")

    def select(self, out, index):
        qual_out, rot_out, width_out = out
        batch_index = torch.arange(qual_out.shape[0])
        label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
        rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
        width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
        return label, rot, width

    def forward(self, data):
        if data['step'] < self.cfg['nr_initial_training_steps']:
            render_outputs = super().forward(data)
            with torch.no_grad():
                vgn_pred = self.vgn_net(render_outputs['volume'])
        elif self.cfg['freeze_nr_after_init']:
            with torch.no_grad():
                render_outputs = super().forward(data)
                vgn_pred = self.vgn_net(render_outputs['volume'])
        else:
            render_outputs = self.nr_net(data)
            vgn_pred = self.vgn_net(render_outputs['volume'])

        if 'full_vol' not in data:
            render_outputs['vgn_pred'] = self.select(vgn_pred, data['grasp_info'][0])
        else:
            render_outputs['vgn_pred'] = vgn_pred
        return render_outputs


name2network = {
    'neugrasp': NeuGrasp,
}
