import numpy as np
from torch.utils.data import Dataset

from src.neugrasp.asset import *  # TODO
from src.neugrasp.dataset.database import parse_database_name, get_database_split
from src.neugrasp.utils.base_utils import get_coords_mask, color_map_forward, get_uniform_coords
from src.neugrasp.utils.dataset_utils import set_seed
from src.neugrasp.utils.imgs_info import build_imgs_info, random_crop, random_flip, pad_imgs_info, imgs_info_slice, \
    imgs_info_to_torch, grasp_info_to_torch
from src.neugrasp.utils.view_select import compute_nearest_camera_indices


# from src.neugrasp.asset_real import *  # TODO


def select_train_ids_for_real_estate(img_ids):
    num_frames = len(img_ids)
    window_size = 32
    shift = np.random.randint(low=-1, high=2)
    id_render = np.random.randint(low=4, high=num_frames - 4 - 1)

    right_bound = min(id_render + window_size + shift, num_frames - 1)
    left_bound = max(0, right_bound - 2 * window_size)
    candidate_ids = np.arange(left_bound, right_bound)
    # remove the query frame itself with high probability
    if np.random.choice([0, 1], p=[0.01, 0.99]):
        candidate_ids = candidate_ids[candidate_ids != id_render]

    id_feat = np.random.choice(candidate_ids, size=min(8, len(candidate_ids)), replace=False)
    img_ids = np.asarray(img_ids)
    return img_ids[id_render], img_ids[id_feat]


def add_depth_offset(depth, mask, region_min, region_max, offset_min, offset_max, noise_ratio, depth_length):
    coords = np.stack(np.nonzero(mask), -1)[:, (1, 0)]
    length = np.max(coords, 0) - np.min(coords, 0)
    center = coords[np.random.randint(0, coords.shape[0])]
    lx, ly = np.random.uniform(region_min, region_max, 2) * length
    diff = coords - center[None, :]
    mask0 = np.abs(diff[:, 0]) < lx
    mask1 = np.abs(diff[:, 1]) < ly
    masked_coords = coords[mask0 & mask1]
    global_offset = np.random.uniform(offset_min, offset_max) * depth_length
    if np.random.random() < 0.5:
        global_offset = -global_offset
    local_offset = np.random.uniform(-noise_ratio, noise_ratio, masked_coords.shape[0]) * depth_length + global_offset
    depth[masked_coords[:, 1], masked_coords[:, 0]] += local_offset


def build_src_imgs_info_select(database, ref_ids, ref_ids_all, cost_volume_nn_num, pad_interval=-1, self_ref=True):
    if not self_ref:
        # ref_ids - selected ref ids for rendering
        ref_idx_exp = compute_nearest_camera_indices(database, ref_ids, ref_ids_all)  # [6, 24]
        ref_idx_exp = ref_idx_exp[:, 1:1 + cost_volume_nn_num]  # [3, 3]
        ref_ids_all = np.asarray(ref_ids_all)
        ref_ids_exp = ref_ids_all[ref_idx_exp]  # rfn,nn
        ref_ids_exp_ = ref_ids_exp.flatten()
        ref_ids = np.asarray(ref_ids)
        ref_ids_in = np.unique(np.concatenate([ref_ids_exp_, ref_ids]))  # rfn'
        mask0 = ref_ids_in[None, :] == ref_ids[:, None]  # rfn,rfn'
        ref_idx_, ref_idx = np.nonzero(mask0)
        ref_real_idx = ref_idx[np.argsort(ref_idx_)]  # sort

        rfn, nn = ref_ids_exp.shape
        mask1 = ref_ids_in[None, :] == ref_ids_exp.flatten()[:, None]  # nn*rfn,rfn'
        ref_cv_idx_, ref_cv_idx = np.nonzero(mask1)
        ref_cv_idx = ref_cv_idx[np.argsort(ref_cv_idx_)]  # sort
        ref_cv_idx = ref_cv_idx.reshape([rfn, nn])

    else:  # not using extra view to construct cost volume
        ref_ids_in = ref_ids
        ref_real_idx = np.asarray(list(range(len(ref_ids))))  # [6,]
        ref_cv_idx = np.asarray([ref_real_idx for _ in range(len(ref_ids))])  # [6, 6]
    # print("ref_ids", ref_ids, "ref_ids_in", ref_ids_in, "ref_cv_idx", ref_cv_idx, "ref_real_idx", ref_real_idx)
    is_aligned = not database.database_name.startswith('space')
    ref_imgs_info = build_imgs_info(database, ref_ids_in, pad_interval, is_aligned)
    return ref_imgs_info, ref_cv_idx, ref_real_idx


class GeneralRendererDataset(Dataset):
    default_cfg = {
        'train_database_types': ['dtu_train', 'space', 'real_iconic', 'real_estate', 'gso'],
        'type2sample_weights': {'gso': 20, 'dtu_train': 20, 'real_iconic': 20, 'space': 10, 'real_estate': 10},
        'val_database_name': 'nerf_synthetic/lego/black_800',
        'val_database_split_type': 'val',

        "total_views": 24,
        "num_input_views": 3,
        'min_wn': 3,
        'max_wn': 4,
        'ref_pad_interval': 16,
        'train_ray_num': 2304,  # [64 * 36]; original:  512
        'foreground_ratio': 0.5,
        'resolution_type': 'hr',
        "use_consistent_depth_range": True,
        'use_depth_loss_for_all': False,
        "use_depth": True,
        "use_src_imgs": False,
        "cost_volume_nn_num": 3,

        "aug_gso_shrink_range_prob": 0.5,
        "aug_depth_range_prob": 0.05,  # NOTE
        'aug_depth_range_min': 0.95,
        'aug_depth_range_max': 1.05,
        "aug_use_depth_offset": True,
        "aug_depth_offset_prob": 0.25,
        "aug_depth_offset_region_min": 0.05,
        "aug_depth_offset_region_max": 0.1,
        'aug_depth_offset_min': 0.5,
        'aug_depth_offset_max': 1.0,
        'aug_depth_offset_local': 0.1,
        "aug_use_depth_small_offset": True,
        "aug_use_global_noise": True,
        "aug_global_noise_prob": 0.5,
        "aug_depth_small_offset_prob": 0.5,
        "aug_forward_crop_size": (400, 600),
        "aug_pixel_center_sample": False,
        "aug_view_select_type": "easy",

        "use_consistent_min_max": False,
        "revise_depth_range": False,
        'load_sdf': True,
        'load_pc': False,
        'exclude_ref_views': False,
    }

    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        self.is_train = is_train
        if is_train:
            self.num = len(vgn_train_scene_names)  # 999999
            self.type2scene_names, self.database_types, self.database_weights = {}, [], []
            if self.cfg['resolution_type'] == 'hr':
                type2scene_names = {'vgn_syn': vgn_train_scene_names}
            elif self.cfg['resolution_type'] == 'lr':
                type2scene_names = {'vgn_syn': vgn_train_scene_names}
            else:
                raise NotImplementedError

            for database_type in self.cfg['train_database_types']:
                self.type2scene_names[database_type] = type2scene_names[database_type]
                self.database_types.append(database_type)
                self.database_weights.append(self.cfg['type2sample_weights'][database_type])
                print(f"training scene num: {len(type2scene_names[database_type])}")
            assert (len(self.database_types) > 0)
            # normalize weights
            self.database_weights = np.asarray(self.database_weights)
            self.database_weights = self.database_weights / np.sum(self.database_weights)
        else:
            self.database = parse_database_name(self.cfg['val_database_name'])
            self.ref_ids, self.que_ids = get_database_split(self.database, self.cfg['val_database_split_type'])
            # ref_ids: [not in que_ids] shape: [21, or 20,]   que_ids: val: [2, 10, 18] shape: [3,] test: [0, 2, 10, 18]
            # shape: [4,]
            self.num = len(self.que_ids)

    def get_database_ref_que_ids(self, index):
        if self.is_train:
            database_type = np.random.choice(self.database_types, 1, False, p=self.database_weights)[0]
            database_scene_name = np.random.choice(self.type2scene_names[database_type])
            database = parse_database_name(database_scene_name)
            # if there is no depth for all views, we repeat random sample until find a scene with depth
            while True:
                ref_ids = database.get_img_ids(check_depth_exist=True)
                if len(ref_ids) == 0:
                    database_type = np.random.choice(self.database_types, 1, False, self.database_weights)[0]
                    database_scene_name = np.random.choice(self.type2scene_names[database_type])
                    database = parse_database_name(database_scene_name)
                else:
                    break
            que_id = np.random.choice(ref_ids)
            if database.database_name.startswith('real_estate'):
                que_id, ref_ids = select_train_ids_for_real_estate(ref_ids)
        else:
            database = self.database
            que_id, ref_ids = self.que_ids[index], self.ref_ids  # int, [21,]
        return database, que_id, np.asarray(ref_ids)

    def select_working_views_impl(self, database_name, dist_idx, ref_num):
        if self.cfg['aug_view_select_type'] == 'default':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError
        elif self.cfg['aug_view_select_type'] == 'easy':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = 3
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 20)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError
        elif self.cfg['aug_view_select_type'] == 'hard':
            if database_name.startswith('grasp'):
                dist_idx = dist_idx[80:]
            elif database_name.startswith('vgn'):
                dist_idx = dist_idx[8:]
            else:
                raise NotImplementedError
        return dist_idx

    def get_ref_que_ids(self, target_id):
        N = self.cfg['total_views']
        n = self.cfg['num_input_views']  # num_input_views = 4
        if self.is_train:
            interval = list(range(0, n))
            start = target_id - target_id % n
            res = [(start + i) for i in interval]
            new_que_id = np.random.choice([i for i in range(N) if i not in res])
        else:
            res = [16, 17, 18, 19]
            new_que_id = np.random.choice([i for i in range(N) if i not in res])
        return res, new_que_id

    def select_working_views(self, database, que_id, ref_ids):
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]
        if self.is_train:
            if np.random.random() > 0.02:  # 2% chance to include que image
                dist_idx = dist_idx[ref_ids[dist_idx] != que_id]
            ref_num = np.random.randint(self.cfg['min_wn'], self.cfg['max_wn'])  # [low, high)
            dist_idx = self.select_working_views_impl(database_name, dist_idx, ref_num)

            if database_name.startswith('grasp'):
                ref_id = np.random.randint(0, 256)
                ref_ids = [ref_id, (ref_id + 80) % 256, (ref_id + 160) % 256]
            elif database_name.startswith('vgn'):
                ref_ids, que_id = self.get_ref_que_ids(np.random.randint(0, self.cfg['total_views']))
            elif not database_name.startswith('real_estate'):
                # we already select working views for real estate dataset
                np.random.shuffle(dist_idx)
                dist_idx = dist_idx[:ref_num]
                ref_ids = ref_ids[dist_idx]
            else:
                ref_ids = ref_ids[:ref_num]
        else:
            if database_name.startswith('vgn'):
                ref_ids, que_id = self.get_ref_que_ids(que_id)  # [6,] int
            elif database_name.startswith('grasp'):
                ref_ids = [que_id, (que_id + 80) % 256, (que_id + 160) % 256]
            else:
                dist_idx = dist_idx[:self.cfg['min_wn']]
                ref_ids = ref_ids[dist_idx]
        return ref_ids, que_id

    def depth_range_aug_for_gso(self, depth_range, depth, mask):
        depth_range_new = depth_range.copy()
        if np.random.random() < self.cfg['aug_gso_shrink_range_prob']:
            rfn, _, h, w = depth.shape
            far_ratios, near_ratios = [], []
            for rfi in range(rfn):
                depth_val = depth[rfi][mask[rfi].astype(np.bool)]
                depth_val = depth_val[depth_val > 1e-3]
                depth_val = depth_val[depth_val < 1e4]
                depth_max = np.max(depth_val) * 1.1
                depth_min = np.min(depth_val) * 0.9
                near, far = depth_range[rfi]
                far_ratio = depth_max / far
                near_ratio = near / depth_min
                far_ratios.append(far_ratio)
                near_ratios.append(near_ratio)

            far_ratio = np.max(far_ratios)
            near_ratio = np.max(near_ratios)
            if far_ratio < 1.0: depth_range_new[:, 1] *= np.random.uniform(far_ratio, 1.0)
            if near_ratio < 1.0: depth_range_new[:, 0] /= np.random.uniform(near_ratio, 1.0)

        if np.random.random() < 0.8:
            ratio0, ratio1 = np.random.uniform(0.025, 0.1, 2)
            depth_range_new[:, 0] = depth_range_new[:, 0] * (1 - ratio0)
            depth_range_new[:, 1] = depth_range_new[:, 1] * (1 + ratio1)
        return depth_range_new

    def random_change_depth_range(self, depth_range, depth, mask, database_name):
        if database_name.startswith('gso'):
            depth_range_new = self.depth_range_aug_for_gso(depth_range, depth, mask)
        else:
            depth_range_new = depth_range.copy()
            if np.random.random() < self.cfg['aug_depth_range_prob']:
                depth_range_new[:, 0] *= np.random.uniform(self.cfg['aug_depth_range_min'], 1.0)
                depth_range_new[:, 1] *= np.random.uniform(1.0, self.cfg['aug_depth_range_max'])
        return depth_range_new

    def add_depth_noise(self, depths, masks, depth_ranges):
        rfn = depths.shape[0]
        depths_output = []
        for rfi in range(rfn):
            depth, mask, depth_range = depths[rfi, 0], masks[rfi, 0], depth_ranges[rfi]

            depth = depth.copy()
            near, far = depth_range
            depth_length = far - near
            if self.cfg['aug_use_depth_offset'] and np.random.random() < self.cfg['aug_depth_offset_prob']:
                add_depth_offset(depth, mask, self.cfg['aug_depth_offset_region_min'],
                                 self.cfg['aug_depth_offset_region_max'],
                                 self.cfg['aug_depth_offset_min'],
                                 self.cfg['aug_depth_offset_max'],
                                 self.cfg['aug_depth_offset_local'], depth_length)
            if self.cfg['aug_use_depth_small_offset'] and np.random.random() < self.cfg['aug_depth_small_offset_prob']:
                add_depth_offset(depth, mask, 0.1, 0.2, 0.01, 0.05, 0.005, depth_length)
            if self.cfg['aug_use_global_noise'] and np.random.random() < self.cfg['aug_global_noise_prob']:
                depth += np.random.uniform(-0.005, 0.005, depth.shape).astype(np.float32) * depth_length
            depths_output.append(depth)
        return np.asarray(depths_output)[:, None, :, :]

    def generate_coords_for_training(self, database, que_imgs_info):  # TODO
        if (database.database_name.startswith('real_estate') or database.database_name.startswith(
                'real_iconic') or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
            que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
            h, w = que_mask_cur.shape
            center_ratio = 0.8
            begin_ratio = (1 - center_ratio) / 2
            hb, he = int(h * begin_ratio), int(h * (center_ratio + begin_ratio))
            wb, we = int(w * begin_ratio), int(w * (center_ratio + begin_ratio))
            que_mask_cur[hb:he, wb:we] = True
            coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 0.9).reshape([1, -1, 2])
        else:
            que_mask_cur = que_imgs_info['masks'][0, 0] > 0
            coords = get_coords_mask(que_mask_cur,
                                     self.cfg['train_ray_num'],
                                     self.cfg['foreground_ratio']
                                     ).reshape([1, -1, 2])  # TODO
        return coords

    def consistent_depth_range(self, ref_imgs_info, que_imgs_info):
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if self.cfg['use_consistent_min_max']:
            depth_range_all[:, 0] = np.min(depth_range_all)
            depth_range_all[:, 1] = np.max(depth_range_all)
        else:
            range_len = depth_range_all[:, 1] - depth_range_all[:, 0]
            max_len = np.max(range_len)
            range_margin = (max_len - range_len) / 2
            ref_near = depth_range_all[:, 0] - range_margin
            ref_near = np.max(np.stack([ref_near, depth_range_all[:, 0] * 0.5], -1), 1)  # max(re_near, 原来最小值的一半)
            depth_range_all[:, 0] = ref_near
            depth_range_all[:, 1] = ref_near + max_len
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]

    def __getitem__(self, index):
        set_seed(index, self.is_train)  # NOTE: currently que_id and ref is all randomly sampled
        # print('\n', index, '\n')
        database, que_id, ref_ids_all = self.get_database_ref_que_ids(index)  # BaseDataBase, int, [24,]
        ref_ids, new_que_id = self.select_working_views(database, que_id, ref_ids_all)  # [6,], int
        if self.cfg['exclude_ref_views']:
            que_id = new_que_id
        # print(que_id, ref_ids)
        if self.cfg['use_src_imgs']:
            # src_imgs_info used in construction of cost volume
            ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(database, ref_ids, ref_ids_all,
                                                                                 self.cfg['cost_volume_nn_num'])
        else:
            ref_idx = compute_nearest_camera_indices(database, ref_ids)[:, 1:4]  # used in cost volume construction
            is_aligned = not database.database_name.startswith('space')
            ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned)
        que_imgs_info = build_imgs_info(database, [que_id], has_depth=True)
        ref_imgs_info['ref_ids'] = np.asarray(ref_ids)
        que_imgs_info['que_id'] = np.asarray([que_id])

        ref_imgs_info['bgs'] = color_map_forward(np.asarray([database.get_bg(ref_id) for ref_id in ref_ids])).transpose(
            [0, 3, 1, 2])  # TODO

        # data augmentation
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if database.database_name.startswith('gso'):  # only used in gso currently
            depth_all = np.concatenate([ref_imgs_info['depth'], que_imgs_info['depth']], 0)
            mask_all = np.concatenate([ref_imgs_info['masks'], que_imgs_info['masks']], 0)
        else:
            depth_all, mask_all = None, None
        depth_range_all = self.random_change_depth_range(depth_range_all, depth_all, mask_all, database.database_name)
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]

        if database.database_name.startswith('gso') and self.cfg['use_depth']:
            depth_aug = self.add_depth_noise(ref_imgs_info['depth'], ref_imgs_info['masks'],
                                             ref_imgs_info['depth_range'])
            ref_imgs_info['true_depth'] = ref_imgs_info['depth']
            ref_imgs_info['depth'] = depth_aug

        if database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space'):
            # crop all datasets
            ref_imgs_info, que_imgs_info = random_crop(ref_imgs_info, que_imgs_info, self.cfg['aug_forward_crop_size'])
            if np.random.random() < 0.5:
                ref_imgs_info, que_imgs_info = random_flip(ref_imgs_info, que_imgs_info)

        if self.cfg['use_depth_loss_for_all'] and self.cfg['use_depth']:
            if not database.database_name.startswith('gso'):
                ref_imgs_info['true_depth'] = ref_imgs_info['depth']

        if database.database_name.startswith('grasp') or database.database_name.startswith('vgn'):
            ref_imgs_info['true_depth'] = ref_imgs_info['depth']
            que_imgs_info['true_depth'] = que_imgs_info['depth']
        if self.cfg['use_consistent_depth_range']:
            self.consistent_depth_range(ref_imgs_info, que_imgs_info)

        # generate coords
        if self.is_train:
            coords = self.generate_coords_for_training(database, que_imgs_info)
        else:
            qn, _, hn, wn = que_imgs_info['imgs'].shape
            coords = np.stack(np.meshgrid(np.arange(wn), np.arange(hn)), -1)
            coords = coords.reshape([1, -1, 2]).astype(np.float32)
        que_imgs_info['coords'] = coords  # train: [1, 512, 2]  val: [1, 512 * 288, 2]
        ref_imgs_info = pad_imgs_info(ref_imgs_info, self.cfg['ref_pad_interval'])

        # don't feed depth to gpu
        if not self.cfg['use_depth']:
            if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
            if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
            if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')

        if self.cfg['use_src_imgs']:
            src_imgs_info = ref_imgs_info.copy()
            ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)  # ？？？
            ref_imgs_info['nn_ids'] = ref_cv_idx
        else:
            # 'nn_ids' used in constructing cost volume (specify source image ids)
            ref_imgs_info['nn_ids'] = ref_idx.astype(np.int64)

        if self.cfg['load_sdf']:
            ref_imgs_info['sdf_gt'] = database.get_sdf()  # [40, 40, 40]
            # ref_imgs_info['sdf_gripper'] = database.get_gripper()  # [1, 1, 36, 36, 36]

        if self.cfg['load_pc']:
            ref_imgs_info['pc_gt'] = database.get_pc()

        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)

        outputs = {'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info, 'scene_name': database.database_name}
        if self.cfg['use_src_imgs']:
            outputs['src_imgs_info'] = imgs_info_to_torch(src_imgs_info)

        if database.database_name.startswith('vgn'):
            outputs['grasp_info'] = grasp_info_to_torch(database.get_grasp_info())
        return outputs

    def __len__(self):
        return self.num


class FinetuningRendererDataset(Dataset):
    default_cfg = {
        "database_name": "nerf_synthetic/lego/black_800",
        "database_split_type": "val_all"
    }

    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        self.is_train = is_train
        self.train_ids, self.val_ids = get_database_split(parse_database_name(self.cfg['database_name']),
                                                          self.cfg['database_split_type'])

    def __getitem__(self, index):
        output = {'index': index}
        return output

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return len(self.val_ids)


class FinetuningRealDataset(Dataset):
    default_cfg = {
        'train_database_types': ['dtu_train', 'space', 'real_iconic', 'real_estate', 'gso'],
        'type2sample_weights': {'gso': 20, 'dtu_train': 20, 'real_iconic': 20, 'space': 10, 'real_estate': 10},
        'val_database_name': 'nerf_synthetic/lego/black_800',
        'val_database_split_type': 'val',

        "total_views": 24,
        "num_input_views": 3,
        'min_wn': 3,
        'max_wn': 4,
        'ref_pad_interval': 16,
        'train_ray_num': 2304,  # [64 * 36]; original:  512
        'foreground_ratio': 0.5,
        'resolution_type': 'hr',
        "use_consistent_depth_range": True,
        'use_depth_loss_for_all': False,
        "use_depth": False,
        "use_src_imgs": False,
        "cost_volume_nn_num": 3,

        "aug_gso_shrink_range_prob": 0.5,
        "aug_depth_range_prob": 0.05,  # NOTE
        'aug_depth_range_min': 0.95,
        'aug_depth_range_max': 1.05,
        "aug_use_depth_offset": True,
        "aug_depth_offset_prob": 0.25,
        "aug_depth_offset_region_min": 0.05,
        "aug_depth_offset_region_max": 0.1,
        'aug_depth_offset_min': 0.5,
        'aug_depth_offset_max': 1.0,
        'aug_depth_offset_local': 0.1,
        "aug_use_depth_small_offset": True,
        "aug_use_global_noise": True,
        "aug_global_noise_prob": 0.5,
        "aug_depth_small_offset_prob": 0.5,
        "aug_forward_crop_size": (400, 600),
        "aug_pixel_center_sample": False,
        "aug_view_select_type": "easy",

        "use_consistent_min_max": False,
        "revise_depth_range": False,
        'load_sdf': False,
        'load_pc': False,
        'exclude_ref_views': False,
    }

    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        self.is_train = is_train
        if is_train:
            self.num = len(vgn_train_scene_names)  # 999999
            self.type2scene_names, self.database_types, self.database_weights = {}, [], []
            if self.cfg['resolution_type'] == 'hr':
                type2scene_names = {'vgn_real': vgn_train_scene_names}
            elif self.cfg['resolution_type'] == 'lr':
                type2scene_names = {'vgn_real': vgn_train_scene_names}
            else:
                raise NotImplementedError

            for database_type in self.cfg['train_database_types']:
                self.type2scene_names[database_type] = type2scene_names[database_type]
                self.database_types.append(database_type)
                self.database_weights.append(self.cfg['type2sample_weights'][database_type])
                print(f"training scene num: {len(type2scene_names[database_type])}")
            assert (len(self.database_types) > 0)
            # normalize weights
            self.database_weights = np.asarray(self.database_weights)
            self.database_weights = self.database_weights / np.sum(self.database_weights)
        else:
            self.database = parse_database_name(self.cfg['val_database_name'])
            self.ref_ids, self.que_ids = get_database_split(self.database, self.cfg['val_database_split_type'])
            # ref_ids: [not in que_ids] shape: [21, or 20,]   que_ids: val: [2, 10, 18] shape: [3,] test: [0, 2, 10, 18]
            # shape: [4,]
            self.num = len(self.que_ids)

    def build_src_imgs_info_select(self, database, ref_ids, ref_ids_all, cost_volume_nn_num, pad_interval=-1, self_ref=True):
        if not self_ref:
            # ref_ids - selected ref ids for rendering
            ref_idx_exp = compute_nearest_camera_indices(database, ref_ids, ref_ids_all)  # [6, 24]
            ref_idx_exp = ref_idx_exp[:, 1:1 + cost_volume_nn_num]  # [3, 3]
            ref_ids_all = np.asarray(ref_ids_all)
            ref_ids_exp = ref_ids_all[ref_idx_exp]  # rfn,nn
            ref_ids_exp_ = ref_ids_exp.flatten()
            ref_ids = np.asarray(ref_ids)
            ref_ids_in = np.unique(np.concatenate([ref_ids_exp_, ref_ids]))  # rfn'
            mask0 = ref_ids_in[None, :] == ref_ids[:, None]  # rfn,rfn'
            ref_idx_, ref_idx = np.nonzero(mask0)
            ref_real_idx = ref_idx[np.argsort(ref_idx_)]  # sort

            rfn, nn = ref_ids_exp.shape
            mask1 = ref_ids_in[None, :] == ref_ids_exp.flatten()[:, None]  # nn*rfn,rfn'
            ref_cv_idx_, ref_cv_idx = np.nonzero(mask1)
            ref_cv_idx = ref_cv_idx[np.argsort(ref_cv_idx_)]  # sort
            ref_cv_idx = ref_cv_idx.reshape([rfn, nn])

        else:  # not using extra view to construct cost volume
            ref_ids_in = ref_ids
            ref_real_idx = np.asarray(list(range(len(ref_ids))))  # [6,]
            ref_cv_idx = np.asarray([ref_real_idx for _ in range(len(ref_ids))])  # [6, 6]
        # print("ref_ids", ref_ids, "ref_ids_in", ref_ids_in, "ref_cv_idx", ref_cv_idx, "ref_real_idx", ref_real_idx)
        is_aligned = not database.database_name.startswith('space')
        ref_imgs_info = build_imgs_info(database, ref_ids_in, pad_interval, is_aligned, has_depth=False, has_mask=False)
        return ref_imgs_info, ref_cv_idx, ref_real_idx

    def get_database_ref_que_ids(self, index):
        if self.is_train:
            database_type = np.random.choice(self.database_types, 1, False, p=self.database_weights)[0]
            database_scene_name = np.random.choice(self.type2scene_names[database_type])
            database = parse_database_name(database_scene_name)
            # if there is no depth for all views, we repeat random sample until find a scene with depth
            while True:
                ref_ids = database.get_img_ids(check_depth_exist=True)
                if len(ref_ids) == 0:
                    database_type = np.random.choice(self.database_types, 1, False, self.database_weights)[0]
                    database_scene_name = np.random.choice(self.type2scene_names[database_type])
                    database = parse_database_name(database_scene_name)
                else:
                    break
            que_id = np.random.choice(ref_ids)
            if database.database_name.startswith('real_estate'):
                que_id, ref_ids = select_train_ids_for_real_estate(ref_ids)
        else:
            database = self.database
            que_id, ref_ids = self.que_ids[index], self.ref_ids  # int, [21,]
        return database, que_id, np.asarray(ref_ids)

    def select_working_views_impl(self, database_name, dist_idx, ref_num):
        if self.cfg['aug_view_select_type'] == 'default':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError
        elif self.cfg['aug_view_select_type'] == 'easy':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = 3
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 20)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError
        elif self.cfg['aug_view_select_type'] == 'hard':
            if database_name.startswith('grasp'):
                dist_idx = dist_idx[80:]
            elif database_name.startswith('vgn'):
                dist_idx = dist_idx[8:]
            else:
                raise NotImplementedError
        return dist_idx

    def get_ref_que_ids(self, target_id):
        N = self.cfg['total_views']
        n = self.cfg['num_input_views']  # num_input_views = 4
        id_list0 = [0, 1, 2, 3]
        id_list1 = [16, 17, 18, 19]
        if self.is_train:
            foo = np.random.choice([0, 1])
            if foo == 0:
                res = id_list0
            else:
                res = id_list1
            new_que_id = np.random.choice([i for i in id_list0 + id_list1 if i not in res])
        else:
            res = [16, 17, 18, 19]  # [15, 16, 17]
            new_que_id = np.random.choice([i for i in id_list0 + id_list1 if i not in res])
        return res, new_que_id

    def select_working_views(self, database, que_id, ref_ids):
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]
        if self.is_train:
            if np.random.random() > 0.02:  # 2% chance to include que image
                dist_idx = dist_idx[ref_ids[dist_idx] != que_id]
            ref_num = np.random.randint(self.cfg['min_wn'], self.cfg['max_wn'])  # [low, high)
            dist_idx = self.select_working_views_impl(database_name, dist_idx, ref_num)

            if database_name.startswith('grasp'):
                ref_id = np.random.randint(0, 256)
                ref_ids = [ref_id, (ref_id + 80) % 256, (ref_id + 160) % 256]
            elif database_name.startswith('vgn'):
                ref_ids, que_id = self.get_ref_que_ids(np.random.randint(0, self.cfg['total_views']))
            elif not database_name.startswith('real_estate'):
                # we already select working views for real estate dataset
                np.random.shuffle(dist_idx)
                dist_idx = dist_idx[:ref_num]
                ref_ids = ref_ids[dist_idx]
            else:
                ref_ids = ref_ids[:ref_num]
        else:
            if database_name.startswith('vgn'):
                ref_ids, que_id = self.get_ref_que_ids(que_id)  # [6,] int
            elif database_name.startswith('grasp'):
                ref_ids = [que_id, (que_id + 80) % 256, (que_id + 160) % 256]
            else:
                dist_idx = dist_idx[:self.cfg['min_wn']]
                ref_ids = ref_ids[dist_idx]
        return ref_ids, que_id

    def depth_range_aug_for_gso(self, depth_range, depth, mask):
        depth_range_new = depth_range.copy()
        if np.random.random() < self.cfg['aug_gso_shrink_range_prob']:
            rfn, _, h, w = depth.shape
            far_ratios, near_ratios = [], []
            for rfi in range(rfn):
                depth_val = depth[rfi][mask[rfi].astype(np.bool)]
                depth_val = depth_val[depth_val > 1e-3]
                depth_val = depth_val[depth_val < 1e4]
                depth_max = np.max(depth_val) * 1.1
                depth_min = np.min(depth_val) * 0.9
                near, far = depth_range[rfi]
                far_ratio = depth_max / far
                near_ratio = near / depth_min
                far_ratios.append(far_ratio)
                near_ratios.append(near_ratio)

            far_ratio = np.max(far_ratios)
            near_ratio = np.max(near_ratios)
            if far_ratio < 1.0: depth_range_new[:, 1] *= np.random.uniform(far_ratio, 1.0)
            if near_ratio < 1.0: depth_range_new[:, 0] /= np.random.uniform(near_ratio, 1.0)

        if np.random.random() < 0.8:
            ratio0, ratio1 = np.random.uniform(0.025, 0.1, 2)
            depth_range_new[:, 0] = depth_range_new[:, 0] * (1 - ratio0)
            depth_range_new[:, 1] = depth_range_new[:, 1] * (1 + ratio1)
        return depth_range_new

    def random_change_depth_range(self, depth_range, depth, mask, database_name):
        if database_name.startswith('gso'):
            depth_range_new = self.depth_range_aug_for_gso(depth_range, depth, mask)
        else:
            depth_range_new = depth_range.copy()
            if np.random.random() < self.cfg['aug_depth_range_prob']:
                depth_range_new[:, 0] *= np.random.uniform(self.cfg['aug_depth_range_min'], 1.0)
                depth_range_new[:, 1] *= np.random.uniform(1.0, self.cfg['aug_depth_range_max'])
        return depth_range_new

    def add_depth_noise(self, depths, masks, depth_ranges):
        rfn = depths.shape[0]
        depths_output = []
        for rfi in range(rfn):
            depth, mask, depth_range = depths[rfi, 0], masks[rfi, 0], depth_ranges[rfi]

            depth = depth.copy()
            near, far = depth_range
            depth_length = far - near
            if self.cfg['aug_use_depth_offset'] and np.random.random() < self.cfg['aug_depth_offset_prob']:
                add_depth_offset(depth, mask, self.cfg['aug_depth_offset_region_min'],
                                 self.cfg['aug_depth_offset_region_max'],
                                 self.cfg['aug_depth_offset_min'],
                                 self.cfg['aug_depth_offset_max'],
                                 self.cfg['aug_depth_offset_local'], depth_length)
            if self.cfg['aug_use_depth_small_offset'] and np.random.random() < self.cfg['aug_depth_small_offset_prob']:
                add_depth_offset(depth, mask, 0.1, 0.2, 0.01, 0.05, 0.005, depth_length)
            if self.cfg['aug_use_global_noise'] and np.random.random() < self.cfg['aug_global_noise_prob']:
                depth += np.random.uniform(-0.005, 0.005, depth.shape).astype(np.float32) * depth_length
            depths_output.append(depth)
        return np.asarray(depths_output)[:, None, :, :]

    def generate_coords_for_training(self, database, que_imgs_info):  # TODO
        if (database.database_name.startswith('real_estate') or database.database_name.startswith(
                'real_iconic') or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
            que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
            h, w = que_mask_cur.shape
            center_ratio = 0.8
            begin_ratio = (1 - center_ratio) / 2
            hb, he = int(h * begin_ratio), int(h * (center_ratio + begin_ratio))
            wb, we = int(w * begin_ratio), int(w * (center_ratio + begin_ratio))
            que_mask_cur[hb:he, wb:we] = True
            coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 0.9).reshape([1, -1, 2])
        else:
            coords = get_uniform_coords((288, 512), self.cfg['train_ray_num']).reshape([1, -1, 2])

        return coords

    def consistent_depth_range(self, ref_imgs_info, que_imgs_info):
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if self.cfg['use_consistent_min_max']:
            depth_range_all[:, 0] = np.min(depth_range_all)
            depth_range_all[:, 1] = np.max(depth_range_all)
        else:
            range_len = depth_range_all[:, 1] - depth_range_all[:, 0]
            max_len = np.max(range_len)
            range_margin = (max_len - range_len) / 2
            ref_near = depth_range_all[:, 0] - range_margin
            ref_near = np.max(np.stack([ref_near, depth_range_all[:, 0] * 0.5], -1), 1)  # max(re_near, 原来最小值的一半)
            depth_range_all[:, 0] = ref_near
            depth_range_all[:, 1] = ref_near + max_len
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]

    def __getitem__(self, index):
        set_seed(index, self.is_train)  # NOTE: currently que_id and ref is all randomly sampled
        # print('\n', index, '\n')
        database, que_id, ref_ids_all = self.get_database_ref_que_ids(index)  # BaseDataBase, int, [24,]
        ref_ids, new_que_id = self.select_working_views(database, que_id, ref_ids_all)  # [6,], int
        if self.cfg['exclude_ref_views']:
            que_id = new_que_id
        # print(que_id, ref_ids)
        if self.cfg['use_src_imgs']:
            # src_imgs_info used in construction of cost volume
            ref_imgs_info, ref_cv_idx, ref_real_idx = self.build_src_imgs_info_select(database, ref_ids, ref_ids_all,
                                                                                 self.cfg['cost_volume_nn_num'])
        else:
            ref_idx = compute_nearest_camera_indices(database, ref_ids)[:, 1:4]  # used in cost volume construction
            is_aligned = not database.database_name.startswith('space')
            ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned, has_depth=False, has_mask=False)
        que_imgs_info = build_imgs_info(database, [que_id], has_depth=False, has_mask=False)
        ref_imgs_info['ref_ids'] = np.asarray(ref_ids)
        que_imgs_info['que_id'] = np.asarray([que_id])

        ref_imgs_info['bgs'] = color_map_forward(np.asarray([database.get_bg(ref_id) for ref_id in ref_ids])).transpose(
            [0, 3, 1, 2])  # TODO

        # data augmentation
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if database.database_name.startswith('gso'):  # only used in gso currently
            depth_all = np.concatenate([ref_imgs_info['depth'], que_imgs_info['depth']], 0)
            mask_all = np.concatenate([ref_imgs_info['masks'], que_imgs_info['masks']], 0)
        else:
            depth_all, mask_all = None, None
        depth_range_all = self.random_change_depth_range(depth_range_all, depth_all, mask_all, database.database_name)
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]

        if database.database_name.startswith('gso') and self.cfg['use_depth']:
            depth_aug = self.add_depth_noise(ref_imgs_info['depth'], ref_imgs_info['masks'],
                                             ref_imgs_info['depth_range'])
            ref_imgs_info['true_depth'] = ref_imgs_info['depth']
            ref_imgs_info['depth'] = depth_aug

        if database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space'):
            # crop all datasets
            ref_imgs_info, que_imgs_info = random_crop(ref_imgs_info, que_imgs_info, self.cfg['aug_forward_crop_size'])
            if np.random.random() < 0.5:
                ref_imgs_info, que_imgs_info = random_flip(ref_imgs_info, que_imgs_info)

        if self.cfg['use_depth_loss_for_all'] and self.cfg['use_depth']:
            if not database.database_name.startswith('gso'):
                ref_imgs_info['true_depth'] = ref_imgs_info['depth']

        if self.cfg['use_consistent_depth_range']:
            self.consistent_depth_range(ref_imgs_info, que_imgs_info)

        # generate coords
        if self.is_train:
            coords = self.generate_coords_for_training(database, que_imgs_info)
        else:
            qn, _, hn, wn = que_imgs_info['imgs'].shape
            coords = np.stack(np.meshgrid(np.arange(wn), np.arange(hn)), -1)
            coords = coords.reshape([1, -1, 2]).astype(np.float32)
        que_imgs_info['coords'] = coords  # train: [1, 512, 2]  val: [1, 512 * 288, 2]

        # don't feed depth to gpu
        if not self.cfg['use_depth']:
            if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
            if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
            if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')

        if self.cfg['use_src_imgs']:
            src_imgs_info = ref_imgs_info.copy()
            ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)  # ？？？
            ref_imgs_info['nn_ids'] = ref_cv_idx
        else:
            # 'nn_ids' used in constructing cost volume (specify source image ids)
            ref_imgs_info['nn_ids'] = ref_idx.astype(np.int64)

        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)

        outputs = {'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info, 'scene_name': database.database_name}
        if self.cfg['use_src_imgs']:
            outputs['src_imgs_info'] = imgs_info_to_torch(src_imgs_info)

        if database.database_name.startswith('vgn'):
            outputs['grasp_info'] = grasp_info_to_torch(database.get_grasp_info())
        return outputs

    def __len__(self):
        return self.num



if __name__ == '__main__':
    # exp = GeneralRendererDataset()
    N = 24
    interval = list(range(0, N, N // 6))  # num_input_views = 6
    res = [(2 + i) % N for i in interval]
    que_id = (np.random.choice(res) + np.random.randint(1, N // 6)) % N
    print(res, que_id)
