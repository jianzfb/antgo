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
from antgo.framework.helper.dataset import (build_dataloader,build_kv_dataloader, build_dataset)
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
from thop import profile
import json


class Tester(object):
    def __init__(self, cfg_dict, work_dir="./", device='cuda', distributed=False):
        self.cfg = Config.fromstring(json.dumps(cfg_dict), '.json')
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
            init_dist(**self.cfg.get('dist_params', {}))
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
        
        if not getattr(self.dataset, 'is_kv', False):
            self.data_loader = build_dataloader(self.dataset, **test_loader_cfg)
        else:
            self.data_loader = build_kv_dataloader(self.dataset, **test_loader_cfg)

    def make_model(self, model_builder=None, checkpoint='', revise_keys=[(r'^module\.', '')], is_fuse_conv_bn=False):
        # build the model and load checkpoint
        if model_builder is not None:
            self.model = model_builder()
        else:
            self.model = build_model(self.cfg.model)

        if checkpoint == '':
            checkpoint = self.cfg.get('checkpoint', checkpoint)
        checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu', revise_keys=revise_keys)
        
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

    def export(self, dummy_input, checkpoint=None, model_builder=None, path='./', prefix='model'):
        model = None
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)

        # 加载checkpoint
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=True)

        # 获得浮点模型的 FLOPS、PARAMS
        # model = DummyModelWarp(f32_model)        
        # model = model.eval()
        model.eval()
        model.forward = model.onnx_export
        model = model.to('cpu')
        dummy_input = dummy_input.to('cpu')
        flops, params = profile(model, inputs=(dummy_input,))
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        if not os.path.exists(path):
            os.makedirs(path)

        # Export the model
        torch.onnx.export(
                model,                                      # model being run
                dummy_input,                                # model input (or a tuple for multiple inputs)
                os.path.join(path, f'{prefix}.onnx'),       # where to save the model (can be a file or file-like object)
                export_params=True,                         # store the trained parameter weights inside the model file
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = ['input'],                    # the model's input names
        )

    def evaluate(self):
        rank, _ = get_dist_info()        
        # allows not to create
        json_file = './result.json'
        if self.cfg.work_dir is not None and rank == 0:
            if not os.path.exists(osp.abspath(self.cfg.work_dir)):
                os.makedirs(osp.abspath(self.cfg.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(self.cfg.work_dir, f'eval_{timestamp}.json')

        eval_cfg = self.cfg.get('evaluation', {}).copy()
        metric_func = None
        if 'metric' in eval_cfg:
            metric_func = build_measure(eval_cfg['metric'])         

        needed_info = []
        if metric_func is not None:
            needed_info = metric_func.keys()['gt']

        if not self.distributed:
            outputs = single_gpu_test(self.model, self.data_loader, needed_info=needed_info)
        else:
            outputs = multi_gpu_test(self.model, self.data_loader, needed_info=needed_info)

        if rank == 0:
            if metric_func is None:
                metric = self.dataset.evaluate(
                    outputs, **eval_cfg)
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

            print(metric)
            metric_dict = dict(metric=metric)
            if self.cfg.work_dir is not None and rank == 0:
                with open(json_file, 'w') as fp:
                    json.dump(metric_dict, fp)
