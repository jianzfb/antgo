from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from asyncio.log import logger

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
from antgo.framework.helper.dataset import (build_dataloader,build_kv_dataloader, build_dataset, build_iter_dataloader)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.models.builder import *
import torch.distributed as dist
from contextlib import contextmanager
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.runner.test import multi_gpu_test, single_gpu_test
from antgo.framework.helper.cnn.utils import fuse_conv_bn
from antgo.framework.helper.task_flag import *
from thop import profile
import json


class Tester(object):
    def __init__(self, cfg, work_dir='./', gpu_id=-1, distributed=False):
        if isinstance(cfg, dict):
            self.cfg = Config.fromstring(json.dumps(cfg), '.json')
        else:
            self.cfg = cfg

        device = 'cpu' if gpu_id < 0 else 'cuda'
        self.device = device
        self.distributed = distributed

        # set multi-process settings
        # 如 opencv_num_threads, OMP_NUM_THREADS, MKL_NUM_THREADS
        setup_multi_processes(self.cfg)

        self.work_dir = work_dir
        # 非分布式下，仅支持单卡
        self.cfg.gpu_ids = [gpu_id] if gpu_id >=0 else []

        if self.distributed:
            init_dist(**self.cfg.get('dist_params', {}))
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)


        assert(self.cfg.data.get('test', None) is not None)
        assert(self.cfg.get('evaluation', None) is not None)

        if not isinstance(self.cfg.data.test, list):
            self.cfg.data.test = [self.cfg.data.test]
        if not isinstance(self.cfg.evaluation, list):
            self.cfg.evaluation = [self.cfg.evaluation]

        assert(len(self.cfg.data.test) == len(self.cfg.evaluation))

        self.dataset = []
        self.data_loader = []
        for data_cfg, eval_cfg in zip(self.cfg.data.test, self.cfg.evaluation):
            dataset = build_dataset(data_cfg)
            test_dataloader_default_args = dict(
                samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

            test_loader_cfg = {
                **test_dataloader_default_args,
                **self.cfg.data.get('test_dataloader', {})
            }        

            assert(not test_loader_cfg['shuffle'])
            if getattr(dataset, 'is_kv', False):
                data_loader = build_kv_dataloader(dataset, **test_loader_cfg)
            elif isinstance(dataset, torch.utils.data.IterableDataset):
                data_loader = build_iter_dataloader(dataset, **test_loader_cfg)
            else:
                data_loader = build_dataloader(dataset, **test_loader_cfg)

            self.dataset.append(dataset)
            self.data_loader.append(data_loader)

    def config_model(self, model_builder=None, checkpoint='', revise_keys=[(r'^module\.', '')], is_fuse_conv_bn=False, strict=True):
        # build the model and load checkpoint
        if model_builder is not None:
            self.model = model_builder()
        else:
            self.model = build_model(self.cfg.model)

        if checkpoint == '':
            checkpoint = self.cfg.get('checkpoint', checkpoint)

        if checkpoint is None or checkpoint == '':
            logger.error('Missing checkpoint file')
        else:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu', revise_keys=revise_keys, strict=strict)

        if is_fuse_conv_bn:
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

    def evaluate(self, json_file=None):
        rank, _ = get_dist_info()        
        if json_file is None and self.work_dir is not None and rank == 0:
            if not os.path.exists(osp.abspath(self.work_dir)):
                os.makedirs(osp.abspath(self.work_dir))
            timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())
            json_file = osp.join(self.work_dir, f'eval_{timestamp}.json')

        all_metric = []
        for dataset, data_loader, eval_cfg in zip(self.dataset, self.data_loader, self.cfg.evaluation):
            eval_cfg = eval_cfg.copy()
            metric_func = None
            if 'metric' in eval_cfg:
                metric_func = build_measure(eval_cfg['metric'])         

            needed_info = []
            if metric_func is not None:
                needed_info = metric_func.keys()['gt']

            if not self.distributed:
                outputs = single_gpu_test(self.model, data_loader, needed_info=needed_info)
            else:
                outputs = multi_gpu_test(self.model, data_loader, needed_info=needed_info)

            if rank == 0:
                if metric_func is None:
                    metric = dataset.evaluate(outputs, **eval_cfg)
                else:
                    gts = []
                    for sample in outputs:
                        gt = {}
                        for k in needed_info:
                            v = sample[k]
                            gt[k] = v
                            sample.pop(k)
                        gts.append(gt)

                    metric = metric_func(outputs, gts)

                metric_dict = dict(metric=metric)
                all_metric.append(metric_dict)

        if self.work_dir is not None and rank == 0:
            with open(json_file, 'w') as fp:
                json.dump(all_metric[0] if len(all_metric) == 1 else all_metric, fp)
        print(all_metric)