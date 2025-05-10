import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.ndimage import convolve

from src.gd.grasp import *
from src.gd.networks import load_network
from src.gd.perception import *
from src.gd.utils.transform import Rotation
from src.rd.stereo_matching import main_batch


# mpl.use('Qt5Agg')


class VGN(object):
    def __init__(self, model_path, rviz=False, args=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz
        self.size = 0.3
        self.resolution = 40
        self.voxel_size = self.size / self.resolution
        self.downSample = 0.8
        self.img_wh = (np.array([640, 360]) * self.downSample).astype(int)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.intrinsic = CameraIntrinsic(int(640 * self.downSample),
                                         int(360 * self.downSample),
                                         459.14 * self.downSample,
                                         459.14 * self.downSample,
                                         319.75 * self.downSample,
                                         179.75 * self.downSample)  # TODO

        self.origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0.0503])

    def get_depth(self, img_id):
        img_filename = os.path.join(self.args.log_root_dir, "rendered_results/" + str(self.args.logdir).split("/")[-1],
                                    "sim_depth/%04d.exr" % img_id)
        img = cv2.imread(img_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        img = cv2.resize(img, self.img_wh)
        return np.asarray(img, dtype=np.float32)

    def denoise_depth_map(self, depth_map, noise_value=0.):
        # 定义一个3x3的卷积核
        kernel = np.array([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)

        # 正规化卷积核
        kernel = kernel / np.sum(kernel)

        # 对整个深度图应用卷积，计算每个像素的周围平均值
        surrounding_sum = convolve(depth_map, kernel, mode='reflect')
        surrounding_count = convolve(np.ones_like(depth_map), kernel, mode='reflect')

        # 计算周围像素的平均值
        surrounding_avg = surrounding_sum / surrounding_count

        # 使用平均值替换噪声值
        denoised_map = np.where(depth_map < noise_value, surrounding_avg, depth_map)

        return denoised_map
    def get_pose(self, img_id):
        poses_ori = np.load(Path(self.args.renderer_root_dir) / 'camera_pose_my.npy')  # NOTE
        poses = [np.linalg.inv(p @ self.blender2opencv) for p in poses_ori]
        return poses[img_id].astype(np.float32).copy()

    def acquire_tsdf(self, test_view_id):
        tsdf = TSDFVolume(self.size, self.resolution)
        high_res_tsdf = TSDFVolume(self.size, self.resolution * 3)

        depths = [self.get_depth(i) for i in test_view_id]
        poses = [self.get_pose(i) for i in test_view_id]
        extrinsics = [Transform.from_matrix(p) * self.origin.inverse() for p in poses]

        timing = 0.0
        for i, extrinsic in enumerate(extrinsics):
            tic = time.time()
            tsdf.integrate(depths[i], self.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depths[i], self.intrinsic, extrinsic)

        # o3d.visualization.draw_geometries([high_res_tsdf.get_cloud()])

        return tsdf, high_res_tsdf, timing

    def vis(self, g):
        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)
        ax.voxels(np.logical_and(g < 0.5, g > 0.), edgecolor='k')
        ax.set_xlabel('X label')
        ax.set_xlabel('Y label')
        ax.set_xlabel('Z label')

        plt.savefig('/path/to/fig.png')

    def __call__(self, test_view_id, round_idx, n_grasp, choose_best=True, gt_tsdf=None):
        main_batch(os.path.join(self.args.log_root_dir, "rendered_results/" + str(self.args.logdir).split("/")[-1]))

        tsdf_vol, _, _ = self.acquire_tsdf(test_view_id)
        tsdf_vol = tsdf_vol.get_grid()

        np.save('/path/to/temp/vis.npy', tsdf_vol)

        tic = time.time()
        if gt_tsdf is not None:
            gt_tsdf = gt_tsdf.get_grid()
            mask = np.logical_and(gt_tsdf != 0., gt_tsdf != 0.)
            mask = ndimage.morphology.binary_dilation(mask, iterations=3)
            tsdf_vol[~mask] = 0.
            np.save('/path/to/temp/vis_gt.npy', gt_tsdf)
            np.save('/path/to/temp/vis_mask.npy', tsdf_vol)
            qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        else:
            qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        grasps, scores, indexs = select(qual_vol.copy(), rot_vol, width_vol)
        toc = time.time() - tic

        grasps, scores, indexs = np.asarray(grasps), np.asarray(scores), np.asarray(indexs)

        if len(grasps) > 0:
            if choose_best:  # TODO
                p = np.argsort(-scores)
                grasps = [from_voxel_coordinates(g, self.voxel_size) for g in grasps[p]]
                scores = scores[p]
                indexs = indexs[p]

            else:
                np.random.seed(self.args.seed + round_idx + n_grasp)
                p = np.random.permutation(len(grasps))
                grasps = [from_voxel_coordinates(g, self.voxel_size) for g in grasps[p]]
                scores = scores[p]
                indexs = indexs[p]

        # self.vis(qual_vol)

        if self.rviz:
            from src.gd import vis
            vis.draw_quality(qual_vol, self.voxel_size, threshold=0.01)

        return grasps, scores, toc, tsdf_vol


def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=1.0,
        min_width=1.33,
        max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)

    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores, indexs = [], [], []
    for index in np.argwhere(mask):
        indexs.append(index)
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores, indexs


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
