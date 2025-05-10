import numpy as np
import torch


def generate_grid_points_old(bound_min, bound_max, resolution):
    X = torch.linspace(bound_min[0], bound_max[0], resolution)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution)
    Z = torch.linspace(bound_max[2], bound_min[2], resolution)  # from top to down to be like with training rays
    XYZ = torch.stack(torch.meshgrid(X, Y, Z), dim=-1)

    return XYZ


RESOLUTION = 40
VOLUME_SIZE = 0.3
VOXEL_SIZE = VOLUME_SIZE / RESOLUTION
HALF_VOXEL_SIZE = VOXEL_SIZE / 2


def generate_grid_points():
    """
    [z-axis, y-axis, x-axis] in dim64000
    [x, y, z] in dim3
    """
    points = []
    voxels = []
    id = 0
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            for z in range(RESOLUTION):
                points.append([x * VOXEL_SIZE + HALF_VOXEL_SIZE,
                               y * VOXEL_SIZE + HALF_VOXEL_SIZE,
                               z * VOXEL_SIZE + HALF_VOXEL_SIZE])
                voxels.append([x, y, z, id])
                id += 1
    return np.array(points).astype(np.float32), np.array(voxels).astype(np.int_)


TSDF_SAMPLE_POINTS, TSDF_SAMPLE_VOXELS = generate_grid_points()

# GT_POINTS = np.load('points.npy')
# TSDF_VOLUME_MASK = np.zeros((1, 40, 40, 40), dtype=np.bool8)
# idxs = []
# for point in GT_POINTS:
#     i, j, k = np.floor(point / VOXEL_SIZE).astype(int)
#     TSDF_VOLUME_MASK[0, i, j, k] = True
#     idxs.append(i * (RESOLUTION * RESOLUTION) + j * RESOLUTION + k)
# print(TSDF_SAMPLE_POINTS.shape)
# assert np.allclose(TSDF_SAMPLE_POINTS[idxs], GT_POINTS)
# print(torch.tensor(TSDF_SAMPLE_VOXELS, dtype=torch.int))
########################################################################################################################
# NOTE: in order to verify the second to last dimension of que_pts represents z-axis
# print(TSDF_SAMPLE_POINTS / 0.3)
# res = 40
# que_pts = (torch.from_numpy(TSDF_SAMPLE_POINTS).to('cuda') +
#            torch.tensor([-0.15, -0.15, -0.05], device='cuda')
#            ).reshape(1, res * res, res, 3)  # 将坐标原点变换到场景中心
# print(que_pts[:, :, 0, :])
# que_pts = torch.flip(que_pts, (2,))
# print(que_pts[:, :, 0, :])
