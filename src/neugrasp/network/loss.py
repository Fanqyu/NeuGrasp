import torch
import sys

import numpy as np
import pyquaternion as pyq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

sys.path.append('/path/to/NeuGrasp')

from src.neugrasp.utils.base_utils import calc_rot_error_from_qxyzw


# from src.neugrasp.network.geocontra_ops import *


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        Args:
            keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class RenderLoss(Loss):
    default_cfg = {
        'use_ray_mask': True,
        'use_nr_fine_loss': False,
        'disable_at_eval': True,
        'render_loss_weight': 1e-1  # 0.001
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        if not is_train and self.cfg['disable_at_eval']:
            return {}
        rgb_gt = data_pr['pixel_colors_gt']  # 1,rn,3
        rgb_nr = data_pr['pixel_colors_nr']  # 1,rn,3

        def compute_loss(rgb_pr, rgb_gt):
            loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)  # b,n
            if self.cfg['use_ray_mask']:
                ray_mask = data_pr['ray_mask'].float()  # 1,rn
                loss = torch.sum(loss * ray_mask, 1) / (torch.sum(ray_mask, 1) + 1e-3)
            else:
                loss = torch.mean(loss, 1)
            return loss * self.cfg['render_loss_weight']

        results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine'] = compute_loss(data_pr['pixel_colors_nr_fine'], rgb_gt)
        return results


def compute_mae(pr, gt, mask):
    return np.mean(np.abs(pr[mask] - gt[mask]))  # the mask makes valid elements out

def log_transform(x, shift=1):
    # https://github.com/magicleap/Atlas
    """rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()


def log_transform_np(x, shift=1):
    # https://github.com/magicleap/Atlas
    """rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return np.sign(x) * np.log(1 + np.abs(x) / shift)


class SDFLoss(Loss):
    default_cfg = {
        'loss_sdf_weight': 1.,
        'loss_out_weight': 0.,
        'loss_eikonal_weight': 0.1,
        'show_sdf_mae': True,
        'record_s': True,
        'loss_s_weight': 0
    }

    def __init__(self, cfg):
        super().__init__(['loss_sdf'])
        self.cfg = {**self.default_cfg, **cfg}
        self.loss_fn = nn.SmoothL1Loss()

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        outputs = {}
        if self.cfg['show_sdf_mae']:
            sdf_pr = data_pr['volume'][0, 0].detach().cpu().numpy()
            sdf_gt = data_gt['ref_imgs_info']['sdf_gt'].detach().cpu().numpy()
            valid_mask = sdf_gt > -1.0
            invalid_mask = sdf_gt == -1.0
            outputs['l1'] = torch.tensor([compute_mae(sdf_pr, sdf_gt, valid_mask)], dtype=torch.float32)
            outputs['out_l1'] = torch.tensor([compute_mae(sdf_pr, -sdf_gt, invalid_mask)], dtype=torch.float32)

        if self.cfg['loss_sdf_weight'] > 0:
            valid_mask = data_gt['ref_imgs_info']['sdf_gt'] != -1.0
            invalid_mask = data_gt['ref_imgs_info']['sdf_gt'] == -1.0
            sdf_gt, sdf_pr = data_gt['ref_imgs_info']['sdf_gt'], data_pr['volume'][0, 0]  # NOTE
            outputs['loss_sdf'] = (self.loss_fn(sdf_gt * valid_mask, sdf_pr * valid_mask)[None] *
                                   self.cfg['loss_sdf_weight'])

        if self.cfg['loss_eikonal_weight'] > 0:
            outputs['loss_eikonal'] = (data_pr['sdf_gradient_error']).mean()[None] * self.cfg['loss_eikonal_weight']
        if self.cfg['record_s']:
            outputs['variance'] = data_pr['s'][None]
        if self.cfg['loss_s_weight'] > 0:  # loss_s_weight equals to 0, so there is no loss of s
            outputs['loss_s'] = torch.norm(data_pr['s']).mean()[None] * self.cfg['loss_s_weight']
        return outputs


class VGNLoss(Loss):
    default_cfg = {
        'loss_vgn_weight': 1e-2,
    }

    def __init__(self, cfg):
        super().__init__(['loss_vgn'])
        self.cfg = {**self.default_cfg, **cfg}

    def _loss_fn(self, y_pred, y, is_train):
        label_pred, rotation_pred, width_pred = y_pred
        _, label, rotations, width = y
        loss_qual = self._qual_loss_fn(label_pred, label)
        acc = self._acc_fn(label_pred, label)
        loss_rot_raw = self._rot_loss_fn(rotation_pred, rotations)
        loss_rot = label * loss_rot_raw
        loss_width_raw = 0.01 * self._width_loss_fn(width_pred, width)
        loss_width = label * loss_width_raw
        loss = loss_qual + loss_rot + loss_width
        loss_item = {'loss_vgn': loss.mean()[None] * self.cfg['loss_vgn_weight'],
                     'vgn_total_loss': loss.mean()[None], 'vgn_qual_loss': loss_qual.mean()[None],
                     'vgn_rot_loss': loss_rot.mean()[None], 'vgn_width_loss': loss_width.mean()[None],
                     'vgn_qual_acc': acc[None]}

        num = torch.count_nonzero(label)
        angle_torch = label * self._angle_error_fn(rotation_pred, rotations, 'torch')
        loss_item['vgn_rot_err'] = (angle_torch.sum() / num)[None] if num else torch.zeros((1,), device=label.device)
        return loss_item

    def _qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none")

    def _acc_fn(self, pred, target):
        return 100 * (torch.round(pred) == target).float().sum() / target.shape[0]

    def _pr_fn(self, pred, target):
        p, r = torchmetrics.functional.precision_recall(torch.round(pred).to(torch.int), target.to(torch.int), 'macro',
                                                        num_classes=2)
        return p[None] * 100, r[None] * 100

    def _rot_loss_fn(self, pred, target):
        loss0 = self._quat_loss_fn(pred, target[:, 0])
        loss1 = self._quat_loss_fn(pred, target[:, 1])
        return torch.min(loss0, loss1)

    def _angle_error_fn(self, pred, target, method='torch'):
        if method == 'np':
            def _angle_error(q1, q2, ):
                q1 = pyq.Quaternion(q1[[3, 0, 1, 2]])
                q1 /= q1.norm
                q2 = pyq.Quaternion(q2[[3, 0, 1, 2]])
                q2 /= q2.norm
                qd = q1.conjugate * q2
                qdv = pyq.Quaternion(0, qd.x, qd.y, qd.z)
                err = 2 * math.atan2(qdv.norm, qd.w) / math.pi * 180
                return min(err, 360 - err)

            q1s = pred.detach().cpu().numpy()
            q2s = target.detach().cpu().numpy()
            err = []
            for q1, q2 in zip(q1s, q2s):
                err.append(min(_angle_error(q1, q2[0]), _angle_error(q1, q2[1])))
            return torch.tensor(err, device=pred.device)
        elif method == 'torch':
            return calc_rot_error_from_qxyzw(pred, target)
        else:
            raise NotImplementedError

    def _quat_loss_fn(self, pred, target):
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

    def _width_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        return self._loss_fn(data_pr['vgn_pred'], data_gt['grasp_info'], is_train)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MVSDepthLoss(Loss):
    default_cfg = {
        'use_automask': True,
        'use_ssim': True,
        'use_feat': False,
        'mvs_depth_loss_weight': 5e-2,
        'disable_at_eval': True,
        'ray_img_size': [36, 64],
        'num_input_views': 4
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__([f'loss_mvsdepth'])
        self.ssim = SSIM()

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        if not is_train and self.cfg['disable_at_eval']:
            return {}

        outputs = {}

        que_id = data_gt['que_imgs_info']['que_id']
        ref_ids = data_gt['ref_imgs_info']['ref_ids']
        # que_pts = data_pr['que_points_fine']
        que_pts = torch.cat([data_pr['que_points'], data_pr['que_points_fine']], -2)  # [qn, rn, dn, 3]
        _, rn, dn, _ = que_pts.shape
        que_pts = torch.cat([que_pts, torch.ones([*que_pts.shape[:-1], 1]).type_as(que_pts)], dim=-1)
        # weight = data_pr['hit_prob_nr_fine']  # [qn, rn, dn]
        weight = torch.cat([data_pr['hit_prob_nr'], data_pr['hit_prob_nr_fine']], -1)

        curr_img = data_gt['que_imgs_info']['imgs']
        qn, dim, H, W = curr_img.shape
        curr_pose = torch.zeros([qn, 4, 4]).type_as(curr_img)
        curr_pose[:, :3, :4] = data_gt['que_imgs_info']['poses']  # w2c
        curr_pose[:, 3, 3] = 1.
        curr_Ks = torch.zeros([qn, 4, 4]).type_as(curr_img)
        curr_Ks[:, :3, :3] = data_gt['que_imgs_info']['Ks']
        curr_Ks[:, 3, 3] = 1.
        w2curr_img = curr_Ks @ curr_pose

        def cal_pixel(trans, coords):
            trans = trans[:, None, None, :, :].repeat(qn, rn, dn, 1, 1)  # [qn, rn, dn 4, 4]
            coords = coords.unsqueeze(-1)  # [qn, rn, dn, 4, 1]
            pixel = torch.matmul(trans, coords).squeeze(-1)  # [qn ,rn, dn, 4]
            mask = pixel[..., 2] > 0  # [qn, rn, dn]
            pixel = pixel[..., :2] / torch.clamp_min(pixel[..., 2:3], 1e-5)  # [qn ,rn, dn, 2]
            mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] < W) & \
                   (pixel[..., 1] > 0) & (pixel[..., 1] < H)
            return pixel, mask  # valid_mask

        que_id = (que_id % self.cfg['num_input_views']).int()
        if que_id.item() == 0:
            curr_featmap = data_pr['img_feats'][que_id:que_id + 1]

            left_img, left_featmap, left_Ks, left_pose = None, None, None, None
            right_img = data_gt['ref_imgs_info']['imgs'][que_id + 1:que_id + 2]
            right_featmap = data_pr['img_feats'][que_id + 1:que_id + 2]
            right_Ks = torch.zeros([qn, 4, 4]).type_as(curr_img)
            right_Ks[:, :3, :3] = data_gt['ref_imgs_info']['Ks'][que_id + 1:que_id + 2]
            right_Ks[:, 3, 3] = 1.
            right_pose = torch.zeros([qn, 4, 4]).type_as(curr_img)
            right_pose[:, :3, :4] = data_gt['ref_imgs_info']['poses'][que_id + 1:que_id + 2]
            right_pose[:, 3, 3] = 1.

            w2right_img = right_Ks @ right_pose
            right_pixel, right_mask = cal_pixel(w2right_img, que_pts)
        elif que_id.item() == self.cfg['num_input_views'] - 1:
            curr_featmap = data_pr['img_feats'][que_id:que_id + 1]

            right_img, right_featmap, right_Ks, right_pose = None, None, None, None
            left_img = data_gt['ref_imgs_info']['imgs'][que_id - 1:que_id]
            left_featmap = data_pr['img_feats'][que_id - 1:que_id]
            left_Ks = torch.zeros([qn, 4, 4]).type_as(curr_img)
            left_Ks[:, :3, :3] = data_gt['ref_imgs_info']['Ks'][que_id - 1:que_id]
            left_Ks[:, 3, 3] = 1.
            left_pose = torch.zeros([qn, 4, 4]).type_as(curr_img)
            left_pose[:, :3, :4] = data_gt['ref_imgs_info']['poses'][que_id - 1:que_id]
            left_pose[:, 3, 3] = 1.

            w2left_img = left_Ks @ left_pose
            left_pixel, left_mask = cal_pixel(w2left_img, que_pts)
        else:
            curr_featmap = data_pr['img_feats'][que_id:que_id + 1]

            left_img = data_gt['ref_imgs_info']['imgs'][que_id - 1:que_id]
            left_featmap = data_pr['img_feats'][que_id - 1:que_id]
            right_img = data_gt['ref_imgs_info']['imgs'][que_id + 1:que_id + 2]
            right_featmap = data_pr['img_feats'][que_id + 1:que_id + 2]
            left_Ks = torch.zeros([qn, 4, 4]).type_as(curr_img)
            left_Ks[:, :3, :3] = data_gt['ref_imgs_info']['Ks'][que_id - 1:que_id]
            left_Ks[:, 3, 3] = 1.
            right_Ks = torch.zeros([qn, 4, 4]).type_as(curr_img)
            right_Ks[:, :3, :3] = data_gt['ref_imgs_info']['Ks'][que_id + 1:que_id + 2]
            right_Ks[:, 3, 3] = 1.
            left_pose = torch.zeros([qn, 4, 4]).type_as(curr_img)
            left_pose[:, :3, :4] = data_gt['ref_imgs_info']['poses'][que_id - 1:que_id]
            left_pose[:, 3, 3] = 1.
            right_pose = torch.zeros([qn, 4, 4]).type_as(curr_img)
            right_pose[:, :3, :4] = data_gt['ref_imgs_info']['poses'][que_id + 1:que_id + 2]
            right_pose[:, 3, 3] = 1.

            w2right_img = right_Ks @ right_pose
            right_pixel, right_mask = cal_pixel(w2right_img, que_pts)
            w2left_img = left_Ks @ left_pose
            left_pixel, left_mask = cal_pixel(w2left_img, que_pts)

        def sample_pixel(pixel, imgs):
            # imgs: qn, 3, H, W
            # pixel: qn, rn, dn, 2
            _, _, H, W = imgs.shape
            pixel_ = pixel
            pixel = pixel_.clone()
            pixel[..., 0] /= W
            pixel[..., 1] /= H
            pixel = 2. * pixel - 1.
            pixel_rgb = F.grid_sample(imgs, pixel, align_corners=True)  # [qn, 3, rn, dn]
            return pixel_rgb.permute(0, 2, 3, 1)  # [qn, rn, dn, 3]

        def mask_invalid(mask, weight):
            new_weight = weight.clone()
            new_weight[~mask] = 0.
            return new_weight

        def compute_reprojection_loss_fn(pred, target, pred_new, weight, mask):
            abs_diff = torch.abs(target - pred)  # [qn, rn, dn, 3]
            l1_loss_ = abs_diff.mean(-1, True)  # [qn, rn, dn, 1]
            l1_loss = torch.sum(l1_loss_ * weight.unsqueeze(-1), dim=-2)  # [qn, rn, 1]

            if not self.cfg['use_automask']:
                reprojection_loss = l1_loss
            else:
                # NOTE: permute below first
                pred_reshape = pred_new.permute(0, 2, 1).reshape(qn, -1, *self.cfg['ray_img_size'])  # [qn, 3, 36, 64]
                target_reshape = target.permute(0, 3, 1, 2).squeeze(-1).reshape(qn, -1, *self.cfg['ray_img_size'])
                ssim_loss = self.ssim(pred_reshape, target_reshape).mean(1, True)  # [qn, 1, 36, 64]
                ssim_loss = ssim_loss.flatten(2).permute(0, 2, 1)  # [qn, rn, 1]
                reprojection_loss = 0.8 * ssim_loss + 0.2 * l1_loss  # TODO

            reprojection_loss[mask] = 1e3  # original: 1e3

            return reprojection_loss

        def compute_mask_reprojection_loss_fn(pred, target):
            abs_diff = torch.abs(target - pred)  # [qn, rn, 1, 3]
            l1_loss = abs_diff.mean(-1)  # [qn, rn, 1]

            if not self.cfg['use_automask']:
                mask_reprojection_loss = l1_loss
            else:
                pred_reshape = pred.permute(0, 3, 1, 2).squeeze(-1).reshape(qn, -1, *self.cfg['ray_img_size'])
                target_reshape = target.permute(0, 3, 1, 2).squeeze(-1).reshape(qn, -1, *self.cfg['ray_img_size'])
                ssim_loss = self.ssim(pred_reshape, target_reshape).mean(1, True)  # [qn, 1, 36, 64]
                ssim_loss = ssim_loss.flatten(2).permute(0, 2, 1)
                mask_reprojection_loss = 0.8 * ssim_loss + 0.2 * l1_loss  # TODO

            return mask_reprojection_loss

        curr_pixel = data_gt['que_imgs_info']['coords'].unsqueeze(-2)  # [qn, rn, dn, 2]
        curr_rgb_target = data_pr['pixel_colors_gt'].unsqueeze(-2)

        if left_img is not None and right_img is not None:
            left_rgb = sample_pixel(left_pixel, left_img)
            left_weight = mask_invalid(left_mask, weight)
            right_rgb = sample_pixel(right_pixel, right_img)
            right_weight = mask_invalid(right_mask, weight)

            left_weight = left_weight / torch.sum(left_weight, dim=-2, keepdim=True).clamp_min(1e-5)
            left_rgb_new = torch.sum(left_rgb * left_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]
            left_mask_new = torch.sum(left_mask, dim=-1) == 0  # [qn, rn] invalid_mask
            right_weight = right_weight / torch.sum(right_weight, dim=-2, keepdim=True).clamp_min(1e-5)
            right_rgb_new = torch.sum(right_rgb * right_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]
            right_mask_new = torch.sum(right_mask, dim=-1) == 0  # [qn, rn]

            left_rgb_target = sample_pixel(curr_pixel, left_img)  # [qn, rn, 1, 3]
            right_rgb_target = sample_pixel(curr_pixel, right_img)

            left_rgb_proj_loss = compute_reprojection_loss_fn(left_rgb, curr_rgb_target, left_rgb_new, left_weight,
                                                              left_mask_new)  # [qn, rn, 1]
            right_rgb_proj_loss = compute_reprojection_loss_fn(right_rgb, curr_rgb_target, right_rgb_new, right_weight,
                                                               right_mask_new)

            if not self.cfg['use_automask']:
                rgb_proj_loss = torch.cat([left_rgb_proj_loss, right_rgb_proj_loss], dim=-1)
            else:
                mask_left_rgb_proj_loss = compute_mask_reprojection_loss_fn(left_rgb_target, curr_rgb_target)  # [qn, rn, 1]
                mask_right_rgb_proj_loss = compute_mask_reprojection_loss_fn(right_rgb_target, curr_rgb_target)
                rgb_proj_loss = torch.cat([left_rgb_proj_loss,
                                       right_rgb_proj_loss,
                                       mask_left_rgb_proj_loss,
                                       mask_right_rgb_proj_loss], dim=-1)

            if not self.cfg['use_feat']:
                pass
            else:
                left_feat = sample_pixel(left_pixel, left_featmap)
                right_feat = sample_pixel(right_pixel, right_featmap)
                curr_feat_target = sample_pixel(curr_pixel, curr_featmap)
                left_feat_target = sample_pixel(curr_pixel, left_featmap)  # [qn, rn, 1, 3]
                left_feat_new = torch.sum(left_feat * left_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]
                right_feat_target = sample_pixel(curr_pixel, right_featmap)
                right_feat_new = torch.sum(right_feat * right_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]

                left_feat_proj_loss = compute_reprojection_loss_fn(left_feat, curr_feat_target, left_feat_new, left_weight,
                                                                  left_mask_new)  # [qn, rn, 1]
                right_feat_proj_loss = compute_reprojection_loss_fn(right_feat, curr_feat_target, right_feat_new, right_weight,
                                                                   right_mask_new)
                if not self.cfg['use_automask']:
                    feat_proj_loss = torch.cat([left_feat_proj_loss, right_feat_proj_loss], dim=-1)
                else:
                    mask_left_feat_proj_loss = compute_mask_reprojection_loss_fn(left_feat_target,
                                                                            curr_feat_target)  # [qn, rn, 1]
                    mask_right_feat_proj_loss = compute_mask_reprojection_loss_fn(right_feat_target, curr_feat_target)
                    feat_proj_loss = torch.cat([left_feat_proj_loss,
                                               right_feat_proj_loss,
                                               mask_left_feat_proj_loss,
                                               mask_right_feat_proj_loss], dim=-1)

        elif left_img is not None:
            left_rgb = sample_pixel(left_pixel, left_img)
            left_weight = mask_invalid(left_mask, weight)

            left_weight = left_weight / torch.sum(left_weight, dim=-2, keepdim=True).clamp_min(1e-5)
            left_rgb_new = torch.sum(left_rgb * left_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]
            left_mask_new = torch.sum(left_mask, dim=-1) == 0  # [qn, rn] invalid_mask

            left_rgb_target = sample_pixel(curr_pixel, left_img)  # [qn, rn, 1, 3]

            left_rgb_proj_loss = compute_reprojection_loss_fn(left_rgb, curr_rgb_target, left_rgb_new, left_weight,
                                                          left_mask_new)  # [qn, rn, 1]

            if not self.cfg['use_automask']:
                rgb_proj_loss = left_rgb_proj_loss
            else:
                mask_left_proj_loss = compute_mask_reprojection_loss_fn(left_rgb_target, curr_rgb_target)  # [qn, rn, 1]
                rgb_proj_loss = torch.cat([left_rgb_proj_loss,
                                       mask_left_proj_loss], dim=-1)

            if not self.cfg['use_feat']:
                pass
            else:
                left_feat = sample_pixel(left_pixel, left_featmap)
                curr_feat_target = sample_pixel(curr_pixel, curr_featmap)
                left_feat_target = sample_pixel(curr_pixel, left_featmap)  # [qn, rn, 1, 3]
                left_feat_new = torch.sum(left_feat * left_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]

                left_feat_proj_loss = compute_reprojection_loss_fn(left_feat, curr_feat_target, left_feat_new, left_weight,
                                                                  left_mask_new)  # [qn, rn, 1]
                if not self.cfg['use_automask']:
                    feat_proj_loss = left_feat_proj_loss
                else:
                    mask_left_feat_proj_loss = compute_mask_reprojection_loss_fn(left_feat_target,
                                                                            curr_feat_target)  # [qn, rn, 1]
                    feat_proj_loss = torch.cat([left_feat_proj_loss,
                                                        mask_left_feat_proj_loss], dim=-1)

        elif right_img is not None:
            right_rgb = sample_pixel(right_pixel, right_img)
            right_weight = mask_invalid(right_mask, weight)

            right_weight = right_weight / torch.sum(right_weight, dim=-2, keepdim=True).clamp_min(1e-5)
            right_rgb_new = torch.sum(right_rgb * right_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]
            right_mask_new = torch.sum(right_mask, dim=-1) == 0  # [qn, rn] invalid_mask

            right_rgb_target = sample_pixel(curr_pixel, right_img)  # [qn, rn, 1, 3]

            right_rgb_proj_loss = compute_reprojection_loss_fn(right_rgb, curr_rgb_target, right_rgb_new, right_weight,
                                                           right_mask_new)  # [qn, rn, 1]

            if not self.cfg['use_automask']:
                rgb_proj_loss = right_rgb_proj_loss
            else:
                mask_right_proj_loss = compute_mask_reprojection_loss_fn(right_rgb_target, curr_rgb_target)  # [qn, rn, 1]
                rgb_proj_loss = torch.cat([right_rgb_proj_loss,
                                       mask_right_proj_loss], dim=-1)

            if not self.cfg['use_feat']:
                pass
            else:
                right_feat = sample_pixel(right_pixel, right_featmap)
                curr_feat_target = sample_pixel(curr_pixel, curr_featmap)
                right_feat_target = sample_pixel(curr_pixel, right_featmap)  # [qn, rn, 1, 3]
                right_feat_new = torch.sum(right_feat * right_weight.unsqueeze(-1), dim=-2)  # [qn, rn, 3]

                right_feat_proj_loss = compute_reprojection_loss_fn(right_feat, curr_feat_target, right_feat_new, right_weight,
                                                                  right_mask_new)  # [qn, rn, 1]
                if not self.cfg['use_automask']:
                    feat_proj_loss = right_feat_proj_loss
                else:
                    mask_right_feat_proj_loss = compute_mask_reprojection_loss_fn(right_feat_target,
                                                                            curr_feat_target)  # [qn, rn, 1]
                    feat_proj_loss = torch.cat([right_feat_proj_loss,
                                                        mask_right_feat_proj_loss], dim=-1)

        else:
            raise NotImplementedError

        if not self.cfg['use_feat']:
            proj_loss, _ = torch.min(rgb_proj_loss, dim=-1)
        else:
            proj_loss = torch.min(rgb_proj_loss, dim=-1)[0] + torch.min(feat_proj_loss, dim=-1)[0]
        outputs['loss_mvs_reproj'] = proj_loss.mean()[None] * self.cfg['mvs_depth_loss_weight']

        return outputs


name2loss = {
    'render': RenderLoss,
    'vgn': VGNLoss,
    'sdf': SDFLoss,
    'mvsdepth': MVSDepthLoss
}

