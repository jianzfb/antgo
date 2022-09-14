# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 13:18
# @File    : args.py
# @Author  : jian<jian@mltalker.com>

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import importlib
import os
import sys


def parse_args(model_name='defalut'):
    parser = argparse.ArgumentParser(description=f'Train {model_name}')
    group_gpus = parser.add_mutually_exclusive_group()
    parser.add_argument('--config', type=str, default="/root/paddlejob/workspace/env_run/portrait/InterHand2.6M/main/test_config.py", help='train config file path')
    parser.add_argument(
        '--resume-from', default=None, help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')        
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='is distributed.')        
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='fuse conv and bn'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='',
        help='work dir'
    )    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='checkpoint'
    ) 
    parser.add_argument(
        '--exp_name',
        type=str,
        default='',
        help='experiment name'
    )
    parser.add_argument('--ext-module', type=str, default='ext_module.py', help='introduce ext module py file')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def load_extmodule(ext_module_file):
    key_model = ext_module_file
    dot_pos = key_model.rfind(".")
    if dot_pos != -1:
        key_model = key_model[0:dot_pos]

    key_model = os.path.normpath(key_model)
    key_model = key_model.replace('/', '.')
    importlib.import_module(key_model)