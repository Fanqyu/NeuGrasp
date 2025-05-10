import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys

sys.path.append('/path/to/NeuGrasp')

from tqdm import tqdm
from src.gd.io import *

from addict import *
from pathlib import Path

import cv2
from colorama import Fore, init


init()
intrinsic = Dict()
intrinsic.width = 640
intrinsic.height = 360
intrinsic.fx = 459.14
intrinsic.fy = 459.14
intrinsic.cx = 319.75
intrinsic.cy = 179.75
blender2opencv = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
tsdf2blender_coord_T_shift = np.array([-0.15, -0.15, -0.0503])
tsdf2blender_world = np.diag([0., 0., 0., 0.])
tsdf2blender_world[:3, 3] = tsdf2blender_coord_T_shift

root_path = '/path/to/NeuGraspData/data/giga_hemisphere_train_0827/packed_full/6_M_rand'
output_path = '/path/to/NeuGraspData/data/giga_hemisphere_train_0827/scenes_tsdf_dep-nor'
if not os.path.exists(output_path):
    os.makedirs(output_path)
count = 0
for split in tqdm(os.listdir(root_path)):
    tqdm.write(Fore.RESET + f'\nBegin {count}-{count + 830}: ')
    split_path = os.path.join(root_path, split)
    for scene in tqdm(os.listdir(split_path)):
        type_list = os.listdir(os.path.join(split_path, scene))
        # tqdm.write(f'{type_list}')
        if 'depth' not in type_list:
            continue
        depth_files = os.path.join(split_path, scene, 'depth')
        depth_imgs = []
        for filename in sorted(os.listdir(depth_files)):  # NOTE: the order
            try:
                depth_h = cv2.imread(os.path.join(depth_files, filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
                depth_imgs.append(depth_h)
            except TypeError:
                print(Fore.RED + f'{scene}')
        depth_imgs = np.stack(depth_imgs, 0)
        extrinsics_ori = np.load(os.path.join(split_path, scene, 'camera_pose_neugrasp.npy'))
        #  NOTE: need to get inv
        extrinsics = np.asarray([np.linalg.inv((p - tsdf2blender_world) @ blender2opencv) for p in extrinsics_ori],
                                dtype=np.float32)
        tsdf = create_tsdf2(0.3, 40, depth_imgs, intrinsic, extrinsics)
        grid = tsdf.get_grid()
        # o3d.visualization.draw_geometries([tsdf.get_cloud()])
        write_voxel_grid(Path(output_path), scene, grid)
        # pc = tsdf.get_cloud()
        # o3d.io.write_point_cloud(os.path.join(output_path, scene + '.ply'), pc)  # real world size
        # pc_np = np.asarray(pc.points)
        # np.savez_compressed(os.path.join(output_path, scene + '.npz'), pc=pc_np)
    tqdm.write(Fore.GREEN + f'\n{count}-{count + 830}: done')
    count += 830