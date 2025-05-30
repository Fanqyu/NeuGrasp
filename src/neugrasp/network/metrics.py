from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave
from skimage.metrics import structural_similarity

from src.neugrasp.network.loss import Loss
from src.neugrasp.utils.base_utils import color_map_backward
from src.neugrasp.utils.draw_utils import concat_images_list


def compute_psnr(img_gt, img_pr, use_vis_scores=False, vis_scores=None, vis_scores_thresh=1.5):
    if use_vis_scores:
        mask = vis_scores >= vis_scores_thresh
        mask = mask.flatten()
        img_gt = img_gt.reshape([-1, 3]).astype(np.float32)[mask]
        img_pr = img_pr.reshape([-1, 3]).astype(np.float32)[mask]
        mse = np.mean((img_gt - img_pr) ** 2, 0)

    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


def compute_mae(depth_pr, depth_gt):
    return np.mean(np.abs(depth_pr - depth_gt))


class PSNR_SSIM(Loss):
    default_cfg = {
        'eval_margin_ratio': 1.0,
    }

    def __init__(self, cfg):
        super().__init__([])
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgbs_gt = data_pr['pixel_colors_gt']  # 1,rn,3
        rgbs_pr = data_pr['pixel_colors_nr']  # 1,rn,3
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
        rgbs_pr = rgbs_pr.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_pr = color_map_backward(rgbs_pr)

        rgbs_gt = rgbs_gt.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_gt = color_map_backward(rgbs_gt)

        h, w, _ = rgbs_gt.shape
        h_margin = int(h * (1 - self.cfg['eval_margin_ratio'])) // 2
        w_margin = int(w * (1 - self.cfg['eval_margin_ratio'])) // 2
        rgbs_gt = rgbs_gt[h_margin:h - h_margin, w_margin:w - w_margin]
        rgbs_pr = rgbs_pr[h_margin:h - h_margin, w_margin:w - w_margin]

        psnr = compute_psnr(rgbs_gt, rgbs_pr)
        outputs = {
            'psnr_nr': torch.tensor([psnr], dtype=torch.float32),
        }

        def compute_psnr_prefix(suffix):
            if f'pixel_colors_{suffix}' in data_pr:
                rgbs_other = data_pr[f'pixel_colors_{suffix}']  # 1,rn,3
                # h, w = data_pr['shape']
                rgbs_other = rgbs_other.reshape([h, w, 3]).detach().cpu().numpy()
                rgbs_other = color_map_backward(rgbs_other)
                psnr = compute_psnr(rgbs_gt, rgbs_other)
                ssim = structural_similarity(rgbs_gt, rgbs_other, win_size=11, multichannel=True, data_range=255, channel_axis=2)
                outputs[f'psnr_{suffix}'] = torch.tensor([psnr], dtype=torch.float32)

        compute_psnr_prefix('nr_fine')

        depth_pr = data_pr['render_depth'].reshape([h, w]).detach().cpu().numpy()
        depth_gt = data_gt['que_imgs_info']['true_depth'][0, 0].cpu().numpy()
        outputs['depth_mae'] = torch.tensor([compute_mae(depth_pr, depth_gt)], dtype=torch.float32)  # higher is better ???
        return outputs


class VisualizeImage(Loss):
    def __init__(self, cfg):
        super().__init__([])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]

        def get_img(key):
            rgbs = data_pr[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, 3]).detach().cpu().numpy()
            rgbs = color_map_backward(rgbs)
            return rgbs

        def get_img_depth(key):
            depths = data_pr[key]  # 1,rn,1
            depths = depths.reshape([h, w]).detach().cpu().numpy()
            depths = color_map_backward(depths)
            return depths

        outputs = {}
        imgs = [get_img('pixel_colors_gt'), get_img('pixel_colors_nr')]
        depths = [get_img_depth('pixel_depths_gt'), get_img_depth('pixel_depths_nr')]

        if 'pixel_colors_dr' in data_pr: imgs.append(get_img('pixel_colors_dr'))
        if 'pixel_colors_nr_fine' in data_pr: 
            imgs.append(get_img('pixel_colors_nr_fine'))
            depths.append(get_img_depth('pixel_depths_nr_fine'))
        if 'pixel_colors_dr_fine' in data_pr: imgs.append(get_img('pixel_colors_dr_fine'))

        data_index = kwargs['data_index']
        model_name = kwargs['model_name']
        Path(f'data/vis_val/{model_name}').mkdir(exist_ok=True, parents=True)
        if h <= 64 and w <= 64:
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.png', concat_images_list(*imgs))
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}-depth.png', concat_images_list(*depths))
        else:
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.jpg', concat_images_list(*imgs))
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}-depth.jpg', concat_images_list(*depths))
        return outputs


name2metrics = {
    'psnr_ssim': PSNR_SSIM,
    'vis_img': VisualizeImage,
}


def psnr_nr(results):
    return np.mean(results['psnr_nr'])


def psnr_nr_fine(results):
    return np.mean(results['psnr_nr_fine'])


def depth_mae(results):
    return np.mean(results['depth_mae'])


def sdf_mae(results):
    return np.mean(results['sdf_mae'])


def loss_vgn(results):
    if 'loss_vgn' in results:
        return np.mean(results['loss_vgn'])
    else:
        return 1e6


name2key_metrics = {
    'psnr_nr': psnr_nr,
    'psnr_nr_fine': psnr_nr_fine,
    'depth_mae': depth_mae,
    'loss_vgn': loss_vgn,
    'sdf_mae': sdf_mae
}
