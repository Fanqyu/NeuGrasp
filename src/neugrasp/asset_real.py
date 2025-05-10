import os

DATA_ROOT_DIR = '/path/to/NeuGrasp/data/NeuGraspData/data/'  # NOTE
# DATA_ROOT_DIR = 'E:/NeRF/NeuGrasp/data/traindata_example/'
VGN_TRAIN_ROOT = DATA_ROOT_DIR + 'train_raw_mix_real'  # 'giga_hemisphere_train_0606'  # NOTE

# VGN_TRAIN_ROOT = DATA_ROOT_DIR + 'giga_hemisphere_train_demo'


def add_scenes(root, type, filter_list=None):
    scene_names = []
    scenes = os.listdir(root)
    scene_names += [f'vgn_real/train/{type}/{0}/{fn}/w_0.8' for fn in scenes]  # type, split, scene_type,
    # scene_split, scene_id, background_size
    return scene_names


if os.path.exists(VGN_TRAIN_ROOT):
    vgn_real_train_scene_names = sorted(add_scenes(VGN_TRAIN_ROOT, 'mix'),
                                        # NOTE
                                        key=lambda x: x.split('/')[4])
    num_scenes = len(vgn_real_train_scene_names)
    num_val = 9  # NOTE
    print(f"total: {num_scenes}")
    vgn_val_scene_names = vgn_real_train_scene_names[-num_val:]
    vgn_train_scene_names = vgn_real_train_scene_names[:-num_val]

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
VGN_TRAIN_CSV = pd.read_csv(Path(CSV_ROOT + '/data_real_train/grasps.csv'))
print(f"finished loading csv in {time.time() - t0} s")
VGN_TEST_CSV = None