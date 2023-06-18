# Copyright (c) MLTALKER. All rights reserved.
import sys
# sys.path.append('/root/paddlejob/workspace/env_run/portrait/InterHand26M/common')
# sys.path.append('/root/paddlejob/workspace/env_run/portrait/InterHand26M/main')

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
from antgo.framework.helper.runner.utils import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import *
from antgo.framework.helper.tools.util import *


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

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

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    cfg.gpu_ids = [args.gpu_id]
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(**cfg.get('dist_params', {}))
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger('model',log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # 1.step 创建模型并初始化
    if args.ext_module != '':
        # 导入ext module
        load_extmodule(args.ext_module)

    model = \
        build_model(cfg.model, train_cfg=cfg.get('train_cfg', {}), test_cfg=cfg.get('test_cfg', {}))
    model.init_weights()

    # 2.step 创建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 3.step 训练模型过程
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
