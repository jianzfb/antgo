from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import (build_dataloader, build_dataset)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.utils.setup_env import *
import torch.distributed as dist
from contextlib import contextmanager
import json

'''
cfg_dict = dict(
    optimizer = dict(type='Adam', lr=1e-4),
    optimizer_config = dict(grad_clip=None),
    lr_config = dict(
        policy='step',
        by_epoch=True,
        warmup=None,
        gamma=0.1,
        step=[15, 17]),
    log_config = dict(
        interval=5,    
        hooks=[
            dict(type='TextLoggerHook'),
        ]),
    checkpoint_config = dict(interval=1,out_dir='./'),        
    seed=0,
    data = dict(
        train=dict(
                    type='InterHand26MReader',
                    dir='/root/paddlejob/workspace/env_run/portrait/InterHand2.6M/data/InterHand2.6M',
                    trans_test='rootnet',
                    output_hm_shape=(64, 64, 64),
                    input_img_shape=(256, 256),
                    bbox_3d_size=400,      
                    output_root_hm_shape=64, 
                    bbox_3d_size_root=400,
                    pipeline=[dict(type='Compose', processes=[['ToTensor', {}]])],
                    inputs_def = {
                        'fields': [
                            ['image'],
                            ['joint_coord','rel_root_depth','hand_type'],
                            ['joint_valid', 'root_valid', 'hand_type_valid', 'inv_trans', 'capture', 'cam', 'frame']
                        ]
                    }
                ),
        train_dataloader=dict(samples_per_gpu=32,workers_per_gpu=2),
        val=dict(
                type='InterHand26MReader',
                dir='/root/paddlejob/workspace/env_run/portrait/InterHand2.6M/data/InterHand2.6M',
                train_or_test='val',
                trans_test='rootnet',
                output_hm_shape=(64, 64, 64),
                input_img_shape=(256, 256),
                bbox_3d_size=400,  
                output_root_hm_shape=64, 
                bbox_3d_size_root=400,
                pipeline=[dict(type='Compose', processes=[['ToTensor', {}]])],
                inputs_def = {
                    'fields': [
                        ['image'],
                        ['joint_coord','rel_root_depth','hand_type'],
                        ['joint_valid', 'root_valid', 'hand_type_valid', 'inv_trans', 'capture', 'cam', 'frame', 'TID']
                    ]
                }
            ),
        val_dataloader=dict(samples_per_gpu=16, workers_per_gpu=1),
        ),
    gpu_ids=[0],
    log_level=logging.INFO,
    evaluation=dict(out_dir='./out')
)
'''
class Trainer(object):
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
            init_dist("pytorch", **self.cfg.get('dist_params', {}))
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

    def make_dataloader(self, with_validate=False):
        # 创建数据集
        dataset = build_dataset(self.cfg.data.train)

        # 创建数据集加载器
        train_dataloader_default_args = dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(self.cfg.gpu_ids),
            dist=self.distributed,
            seed=self.cfg.seed,
            runner_type='EpochBasedRunner',
            persistent_workers=True)

        train_loader_cfg = {
            **train_dataloader_default_args,
            **self.cfg.data.get('train_dataloader', {})
        }
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
            self.val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

    def make_model(self, model_builder=None, resume_from=None, load_from=None):
        # prepare network
        logger = get_logger('model', log_level=self.cfg.log_level)
        logger.info("Creating graph and optimizer...")
        model = model_builder()

        if self.distributed:
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

        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        # build optimizer
        optimizer = build_optimizer(model, self.cfg.optimizer)

        # build training strategy
        self.runner = build_runner(
            dict(type='EpochBasedRunner', max_epochs=1),        # 忽略max_epochs，开发者控制最大epoch
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=self.work_dir,
                logger=logger,
                meta=self.meta))

        # an ugly workaround to make .log and .log.json filenames the same
        self.runner.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        optimizer_config = self.cfg.optimizer_config
        if self.distributed and 'type' not in self.cfg.optimizer_config:
            optimizer_config = OptimizerHook(**self.cfg.optimizer_config)

        self.runner.register_training_hooks(
            self.cfg.lr_config,                                     # 学习率调整策略，比如step,warmup等
            optimizer_config,                                       # 优化器的相关后处理，比如限制梯度操作等
            self.cfg.checkpoint_config,                             # checkpoint相关处理
            self.cfg.log_config,                                         
            self.cfg.get('momentum_config', None),                       
            custom_hooks_config=self.cfg.get('custom_hooks', None))

        if self.distributed:
            if isinstance(self.runner, EpochBasedRunner):
                self.runner.register_hook(DistSamplerSeedHook())

        if self.val_dataloader is not None:
            eval_cfg = self.cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = True
            if 'interval' not in eval_cfg:
                eval_cfg['interval'] = 1

            eval_hook = DistEvalHook if self.distributed else EvalHook

            if 'metric' in eval_cfg:
                metric = build_measure(eval_cfg['metric'])
                eval_cfg['metric'] = metric
            self.runner.register_hook(
                eval_hook(self.val_dataloader, **eval_cfg), priority='LOW')

        if resume_from is not None:
            self.runner.resume(resume_from)
        elif load_from is not None:
            self.runner.load_checkpoint(load_from)
    
    @contextmanager
    def train_context(self, max_epochs):
        self.runner._max_epochs = max_epochs
        self.runner.call_hook('before_run')
        yield self.runner

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.runner.call_hook('after_run')

    def run_on_train(self, **kwargs):
        self.runner.train(self.train_generator, **kwargs)

    def run_on_val(self, **kwargs):
        self.runner.val(self.val_dataloader, **kwargs)