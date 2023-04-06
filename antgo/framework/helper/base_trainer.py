from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.framework.helper.parallel.utils import is_module_wrapper
from antgo.framework.helper.models.proxy_module import ProxyModule
from antgo.framework.helper.runner.test import single_gpu_test

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader, dataloader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import (build_dataset, build_dataloader, build_kv_dataloader, build_iter_dataloader)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import *
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.utils import *
from antgo.framework.helper.models.dummy_module import *
from antgo.framework.helper.models.proxy_module import *
from antgo.framework.helper.models.distillation import *
from antgo.framework.helper.runner.hooks.hook import *
from antgo.framework.helper.utils import Config, build_from_cfg
from antgo.framework.helper.task_flag import *
import traceback
from thop import profile
import copy

import torch.distributed as dist
from contextlib import contextmanager
import json


class BaseTrainer(object):
    def __init__(self, cfg_dict, work_dir="./", device='cuda', distributed=False, diff_seed=True, deterministic=True):
        self.cfg = Config.fromstring(json.dumps(cfg_dict), '.json')

        self.data_loaders = None
        self.runner = None
        self.work_dir = work_dir
        self.train_generator = None
        self.val_dataloader = None
        self.distributed = distributed
        self.meta = {}

        # set multi-process settings
        # 如 opencv_num_threads, OMP_NUM_THREADS, MKL_NUM_THREADS
        setup_multi_processes(self.cfg)

        if self.distributed:
            init_dist(**self.cfg.get('dist_params', {}))
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)

        # set random seeds
        seed = init_random_seed(self.cfg.get('seed', 0), device=device)
        seed = seed + dist.get_rank() if diff_seed else seed
        set_random_seed(seed, deterministic=deterministic)
        self.cfg.seed = seed
        self.meta['seed'] = seed
        self.device = device

        self.submodule_optimizer = {}
        self.submodule_optimizer_config = {}

    def config_dataloader(self, with_validate=False):
        # 创建数据集
        dataset = build_dataset(self.cfg.data.train)

        # 创建数据集加载器
        train_dataloader_default_args = dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            # `num_gpus` will be ignored if distributed
            num_gpus=1,
            dist=self.distributed,
            seed=self.cfg.seed,
            runner_type='EpochBasedRunner',
            persistent_workers=True)

        train_loader_cfg = {
            **train_dataloader_default_args,
            **self.cfg.data.get('train_dataloader', {})
        }
        if getattr(dataset, 'is_kv', False):
            self.train_generator = build_kv_dataloader(dataset, **train_loader_cfg)
        elif isinstance(dataset, torch.utils.data.IterableDataset):
            self.train_generator = build_iter_dataloader(dataset, **train_loader_cfg)
        else:
            self.train_generator = build_dataloader(dataset, **train_loader_cfg)

        if with_validate:
            val_dataloader_default_args = dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                dist=self.distributed,
                shuffle=False,
                persistent_workers=True)
            
            val_dataloader_args = {
                    **val_dataloader_default_args,
                    **self.cfg.data.get('val_dataloader', {})
            }
            val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
            if getattr(dataset, 'is_kv', False):
                self.val_dataloader = build_kv_dataloader(val_dataset, **val_dataloader_args)
            elif isinstance(dataset, torch.utils.data.IterableDataset):
                val_dataset.is_infinite = False
                self.val_dataloader = build_iter_dataloader(val_dataset, **val_dataloader_args)
            else:
                self.val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

    def _config_runner(self, runner_config, model, logger):
        self.runner = build_runner(
            runner_config,        # 忽略max_epochs，开发者控制最大epoch
            default_args=dict(
                model=model,
                optimizer=self.submodule_optimizer,
                work_dir=self.work_dir,
                logger=logger,
                meta=self.meta))

    def _config_training_hooks(self):
        # 仅对通用hook进行配置
        custom_hooks = self.cfg.get('custom_hooks', None)        

        if 'policy' not in self.cfg.lr_config:
            for submodule_name, lr_config in self.cfg.lr_config.items():
                lr_config.update({'name': submodule_name})
                self.runner.register_lr_hook(lr_config)
        else:
            self.runner.register_lr_hook(self.cfg.lr_config)

        self.runner.register_checkpoint_hook(self.cfg.checkpoint_config)
        self.runner.register_logger_hooks(self.cfg.log_config)
        self.runner.register_custom_hooks(custom_hooks)

    def _config_optimizer(self, model):
        if len(self.submodule_optimizer) > 0:
            return

        if is_module_wrapper(model):
            model = model.module

        for submodule_name, optimizer_dict in self.cfg.optimizer.items():
            self.submodule_optimizer[submodule_name] = build_optimizer(getattr(model, submodule_name), optimizer_dict)
            self.submodule_optimizer_config[submodule_name] = self.cfg.optimizer_config.get(submodule_name, dict())

    def config_model(self, model_builder=None, resume_from=None, load_from=None, revise_keys=[(r'^module\.', '')]):
        # prepare network
        logger = get_logger('model', log_level=self.cfg.log_level)
        logger.info("Creating model and optimizer...")

        # 构建网络
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)

        # 模型初始化
        model.init_weights()

        if self.distributed:
            if self.cfg.get('syncBN', True):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = build_ddp(
                model,
                self.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = build_dp(model, self.device, device_ids=self.cfg.gpu_ids)

        # config optimizer
        self._config_optimizer(model)
        
        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        # build training strategy
        custom_runner_config = self.cfg.get('runner', dict(type='EpochBasedRunner', max_epochs=1))
        self._config_runner(custom_runner_config, model, logger)

        # config default training hooks
        self._config_training_hooks()

        # an ugly workaround to make .log and .log.json filenames the same
        self.runner.timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())

        if self.distributed:
            if isinstance(self.runner, EpochBasedRunner):
                self.runner.register_hook(DistSamplerSeedHook())

        if resume_from is not None:
            self.runner.resume(resume_from)
        elif load_from is not None:
            self.runner.load_checkpoint(load_from, revise_keys=revise_keys)

    def start_train(self, max_epochs, **kwargs):
        try:
            running_flag(self.cfg.get('root', None))
            self.runner.run([self.train_generator], [('train', max_epochs)], max_epochs)
            finish_flag(self.cfg.get('root', None))
        except Exception:
            stop_flag(self.cfg.get('root', None))
            traceback.print_exc()

    def start_eval(self, **kwargs):
        try:
            running_flag(self.cfg.get('root', None))
            self.runner.run([self.train_generator], [('val', 1)], 1)
            finish_flag(self.cfg.get('root', None))
        except:
            stop_flag(self.cfg.get('root', None))
            traceback.print_exc()
