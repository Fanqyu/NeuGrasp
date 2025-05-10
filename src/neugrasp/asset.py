import os

DATA_ROOT_DIR = '/path/to/NeuGraspData/data/'  # NOTE
# DATA_ROOT_DIR = 'E:/NeRF/NeuGrasp/data/traindata_example/'
VGN_TRAIN_ROOT = DATA_ROOT_DIR + 'giga_hemisphere_train_0827'  # NOTE


def add_scenes(root, type, filter_list=None):
    scene_names = []
    splits = os.listdir(root)
    for split in splits:
        if filter_list is not None and split not in filter_list:
            continue
        scenes = os.listdir(os.path.join(root, split))
        scene_names += [f'vgn_syn/train/{type}/{split}/{fn}/w_0.8' for fn in scenes]  # type, split, scene_type,
        # scene_split, scene_id, background_size
    return scene_names


if os.path.exists(VGN_TRAIN_ROOT):
    vgn_pile_train_scene_names = sorted(add_scenes(os.path.join(VGN_TRAIN_ROOT, 'pile_full', '6_M_rand'), 'pile'),
                                        # NOTE
                                        key=lambda x: x.split('/')[4])
    vgn_pack_train_scene_names = sorted(add_scenes(os.path.join(VGN_TRAIN_ROOT, 'packed_full', '6_M_rand'), 'packed'),
                                        key=lambda x: x.split('/')[4])
    num_scenes_pile = len(vgn_pile_train_scene_names)
    num_scenes_pack = len(vgn_pack_train_scene_names)
    vgn_pack_train_scene_names = vgn_pack_train_scene_names[:num_scenes_pack]
    num_val_pile = 125  # NOTE
    num_val_pack = 125  # NOTE
    print(f"total: {num_scenes_pile + num_scenes_pack} pile: {num_scenes_pile} pack: {num_scenes_pack}")
    vgn_val_scene_names = vgn_pile_train_scene_names[-num_val_pile:] + vgn_pack_train_scene_names[-num_val_pack:]
    vgn_train_scene_names = vgn_pile_train_scene_names[:-num_val_pile] + vgn_pack_train_scene_names[:-num_val_pack]

# VGN_SDF_DIR = DATA_ROOT_DIR + "/giga_hemisphere_train_0606/scenes_tsdf_dep-nor"  # NOTE
# VGN_PC_DIR = DATA_ROOT_DIR + "/giga_hemisphere_train_0606/scenes_pc_dep-nor"  # NOTE
# VGN_SDF_GRIPPER = DATA_ROOT_DIR + "/giga_hemisphere_train_0606/gripper_tsdf"  # NOTE

VGN_SDF_DIR = DATA_ROOT_DIR + "/giga_hemisphere_train_0827/scenes_tsdf_dep-nor"  # NOTE
VGN_PC_DIR = DATA_ROOT_DIR + "/giga_hemisphere_train_0827/scenes_pc_dep-nor"  # NOTE
VGN_SDF_GRIPPER = DATA_ROOT_DIR + "/giga_hemisphere_train_0827/gripper_tsdf"  # NOTE

VGN_TEST_ROOT = ''
VGN_TEST_ROOT_PILE = os.path.join(VGN_TEST_ROOT, 'pile')
VGN_TEST_ROOT_PACK = os.path.join(VGN_TEST_ROOT, 'packed')
if os.path.exists(VGN_TEST_ROOT):
    fns = os.listdir(VGN_TEST_ROOT_PILE)
    vgn_pile_test_scene_names = [f'vgn_syn/test/pile//{fn}/w_0.8' for fn in fns]
    fns = os.listdir(VGN_TEST_ROOT_PACK)
    vgn_pack_test_scene_names = [f'vgn_syn/test/packed//{fn}/w_0.8' for fn in fns]

    vgn_test_scene_names = vgn_pile_test_scene_names + vgn_pack_test_scene_names

CSV_ROOT = DATA_ROOT_DIR + 'GIGA'  # NOTE
import pandas as pd
from pathlib import Path
import time

t0 = time.time()
VGN_PACK_TRAIN_CSV = pd.read_csv(Path(CSV_ROOT + '/data_packed_train_raw/grasps.csv'))
VGN_PILE_TRAIN_CSV = pd.read_csv(Path(CSV_ROOT + '/data_pile_train_raw/grasps.csv'))  # NOTE
print(f"finished loading csv in {time.time() - t0} s")
VGN_PACK_TEST_CSV = None
VGN_PILE_TEST_CSV = None
