# Copyright (c) MLTALKER. All rights reserved.
import sys
sys.path.append('/root/paddlejob/workspace/env_run/portrait/antgo')
sys.path.append('/root/paddlejob/workspace/env_run/portrait/InterHand26M/common')
sys.path.append('/root/paddlejob/workspace/env_run/portrait/InterHand26M/main')
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np

import torch
import torch.distributed as dist
from antgo.framework.helper.utils import Config, DictAction
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.utils import get_git_hash
from antgo.framework.helper.utils import (collect_env,update_data_root)
from antgo.framework.helper.utils import (collect_env, get_logger, update_data_root)
from antgo.framework.helper.utils.util_distribution import get_device
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.runner.utils import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import *
from antgo.framework.helper.tools.util import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.cnn import fuse_conv_bn
from antgo.framework.helper.runner.test import multi_gpu_test, single_gpu_test
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.utils.util_distribution import get_device
import json


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print(f'checkpoint {args.checkpoint}')
    # set multi-process settings
    # 如 opencv_num_threads, OMP_NUM_THREADS, MKL_NUM_THREADS
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None and args.work_dir != '':
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 非分布式下，仅支持单卡
    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(**cfg.get('dist_params', {}))

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if cfg.work_dir is not None and rank == 0:
        if not os.path.exists(osp.abspath(cfg.work_dir)):
            os.makedirs(osp.abspath(cfg.work_dir))
        timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())
        json_file = osp.join(cfg.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if args.ext_module != '':
        # 导入ext module
        print(f'load ext model from {args.ext_module}')
        load_extmodule(args.ext_module)
    
    model = \
        build_model(cfg.model, test_cfg=cfg.get('test_cfg', None))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    cfg.device = get_device()
    if not distributed:
        model = build_dp(model, cfg.get('device', 'cuda'), device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = build_ddp(
            model,
            cfg.get('device', 'cuda'),
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        metric_func = None
        if 'metric' in cfg:
            metric_func = build_measure(cfg['metric'])         

        eval_kwargs = cfg.get('evaluation', {}).copy()  
        if metric_func is None:
            metric = dataset.evaluate(
                outputs, **eval_kwargs)
        else:
            gts = dataset.get_ann_info(None)
            if gts is None:
                gts = []
                for gt_i in len(dataset):
                    gts.append(dataset.sample(gt_i))
            metric = metric_func(outputs, gts)

        print(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if cfg.work_dir is not None and rank == 0:
            json.dump(metric_dict, json_file)

if __name__ == '__main__':
    main()
