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
from timer import Timer
from torch.nn.parallel.data_parallel import DataParallel
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import (build_dataloader, build_dataset)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
import torch.distributed as dist
from contextlib import contextmanager
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.runner.test import multi_gpu_test, single_gpu_test
import json


class Tester(object):
    def __init__(self, cfg_dict, work_dir="./", device='cuda', distributed=False):
        self.cfg = Config.fromstring(json.dumps(cfg_dict), '.json')
        self.data_loaders = None
        self.device = device
        self.distributed = distributed

        # set multi-process settings
        # 如 opencv_num_threads, OMP_NUM_THREADS, MKL_NUM_THREADS
        setup_multi_processes(self.cfg)

        if self.cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if self.cfg.work_dir is None
            self.cfg.work_dir = './work_dirs'

        # 非分布式下，仅支持单卡
        self.cfg.gpu_ids = [self.cfg.get('gpu_id',0)]

        if self.distributed:
            init_dist("pytorch", **self.cfg.get('dist_params', {}))
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)

        # build the dataloader
        self.dataset = build_dataset(self.cfg.data.test)
        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **self.cfg.data.get('test_dataloader', {})
        }        
        self.data_loader = build_dataloader(self.dataset, **test_loader_cfg)

    def make_model(self, model_builder=None, checkpoint='', fuse_conv_bn=False):
        # build the model and load checkpoint
        self.model = model_builder()
        if checkpoint == '':
            checkpoint = self.cfg.get('checkpoint', checkpoint)
        checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
        
        if fuse_conv_bn:
            print('use fuse conv_bn')
            self.model = fuse_conv_bn(self.model)
        
        if not self.distributed:
            self.model = build_dp(self.model, self.device, device_ids=self.cfg.gpu_ids)
        else:
            print('build model in multi')
            self.model = build_ddp(
                self.model,
                self.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)

    def evaluate(self):
        rank, _ = get_dist_info()        
        # allows not to create
        if self.cfg.work_dir is not None and rank == 0:
            if not os.path.exists(osp.abspath(self.cfg.work_dir)):
                os.makedirs(osp.abspath(self.cfg.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(self.cfg.work_dir, f'eval_{timestamp}.json')

        if not self.distributed:
            outputs = single_gpu_test(self.model, self.data_loader)
        else:
            outputs = multi_gpu_test(self.model, self.data_loader)

        if rank == 0:
            metric_func = None
            if 'metric' in self.cfg:
                metric_func = build_measure(self.cfg['metric'])         

            eval_kwargs = self.cfg.get('evaluation', {}).copy()  
            if metric_func is None:
                metric = self.dataset.evaluate(
                    outputs, **eval_kwargs)
            else:
                gts = self.dataset.get_ann_info(None)
                if gts is None:
                    gts = []
                    for gt_i in len(self.dataset):
                        gts.append(self.dataset.sample(gt_i))
                metric = metric_func(outputs, gts)

            print(metric)
            metric_dict = dict(metric=metric)
            if self.cfg.work_dir is not None and rank == 0:
                json.dump(metric_dict, json_file)
