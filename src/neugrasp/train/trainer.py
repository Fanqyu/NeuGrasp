import os
import random
import sys

import numpy as np
import torch
from colorama import Fore
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/path/to/NeuGrasp')

from src.neugrasp.dataset.name2dataset import name2dataset
from src.neugrasp.network.loss import name2loss
from src.neugrasp.network.renderer import name2network
from src.neugrasp.train.lr_common_manager import name2lr_manager
from src.neugrasp.network.metrics import name2metrics
from src.neugrasp.train.train_tools import to_cuda, Logger
from src.neugrasp.train.train_valid import ValidationEvaluator
from src.neugrasp.utils.dataset_utils import dummy_collate_fn
# from src.neugrasp.asset_real import vgn_val_scene_names  # TODO
from src.neugrasp.asset import vgn_val_scene_names  # TODO

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 1.0e-4,
            "decay_step": 100000,
            "decay_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 2500,  # 1000,
        "worker_num": 8,
        "fix_seed": False
    }

    def _init_dataset(self):
        self.train_data = name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
        self.train_set = DataLoader(self.train_data, 1, True, num_workers=self.cfg['worker_num'],
                                    collate_fn=dummy_collate_fn)
        print(Fore.GREEN + f'train set len: {len(self.train_set)}' + Fore.RESET)
        self.val_set_list, self.val_set_names = [], []
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            if 'val_scene_num' in val_set_cfg:
                num = val_set_cfg['val_scene_num']
                num = len(vgn_val_scene_names) if num == -1 else num
                names, val_types = [name] * num, [val_type] * num
                val_cfgs = []
                for i in range(num):
                    val_cfgs.append({**val_cfg, **{'val_database_name': vgn_val_scene_names[i]}})
            else:
                names, val_types, val_cfgs = [name], [val_type], [val_cfg]
        for name, val_type, val_cfg in zip(names, val_types, val_cfgs):
            val_set = name2dataset[val_type](val_cfg, False)
            val_set = DataLoader(val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
        print('[I] ' + Fore.GREEN + f'Val set len: {len(self.val_set_list)}' + Fore.RESET)

    def _init_network(self):
        self.network = name2network[self.cfg['network']](self.cfg).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))

        # metrics
        self.val_metrics = []
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(self.optimizer, self.network)

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['fix_seed']:
            seed = 0
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            print("fix seed")
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['group_name'], cfg['name'])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        assert self.cfg["key_metric_prefer"] in ['higher', 'lower']
        self.better = lambda x, y: x > y if self.cfg["key_metric_prefer"] == 'higher' else x < y

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para, start_step = self._load_model()
        if self.cfg["key_metric_prefer"] == 'lower' and start_step == 0:
            best_para = 1e6
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], ncols=80)
        pbar.set_description('   itr')
        pbar.update(start_step)
        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                del self.train_set
                self.train_set = DataLoader(self.train_data, 1, True, num_workers=self.cfg['worker_num'],
                                            collate_fn=dummy_collate_fn)
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info = {}
            outputs = self.train_network(train_data)

            for loss in self.train_losses:
                loss_results = loss(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v

            loss = 0
            for k, v in log_info.items():
                if k.startswith('loss'):
                    loss = loss + torch.mean(v)

            loss.backward()
            self.optimizer.step()
            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')

            if step == 0 or (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                _tqdm1 = tqdm(total=len(self.val_set_list), ncols=80)
                for vi, val_set in enumerate(self.val_set_list):
                    _tqdm1.set_description('   val_epoch')
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        key = f'{self.val_set_names[vi]}-{k}'
                        if key not in val_results:
                            val_results[key] = v
                        else:
                            val_results[key] += v
                    val_para += val_para_cur
                    _tqdm1.update(1)
                _tqdm1.close()

                # average all items 
                for k, v in val_results.items():
                    val_results[k] /= len(self.val_set_list)
                val_para /= len(self.val_set_list)

                if step and self.better(val_para, best_para):  # do not save the first step
                    tqdm.write(
                        Fore.GREEN + f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}'
                        + Fore.RESET)
                    best_para = val_para
                    self.best_pth_fn = os.path.join(self.model_dir, f'best_model_{step + 1}.pth')
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                self._save_model(step + 1, best_para)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr)
            pbar.update(1)
            del loss, log_info
        pbar.close()

    def _load_model(self):
        def _step(fname):
            return int(os.path.splitext(fname)[0].split('_')[-1])

        best_ckpts = sorted([f for f in os.listdir(self.model_dir) if f.startswith('best_model_')], key=_step)
        latest_ckpts = sorted([f for f in os.listdir(self.model_dir) if f.startswith('model_')], key=_step)

        if latest_ckpts:
            ckpt_path = os.path.join(self.model_dir, latest_ckpts[-1])
        elif best_ckpts:
            ckpt_path = os.path.join(self.model_dir, best_ckpts[-1])
        else:
            return 0, 0  # nothing to resume

        checkpoint = torch.load(ckpt_path, map_location='cuda')
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_para = checkpoint['best_para']
        start_step = checkpoint['step']
        print(f'==> Resumed from {ckpt_path}, step {start_step}, best {best_para:.5f}')
        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = os.path.join(self.model_dir, f'model_{step}.pth') if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)

