from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import importlib
import os
import sys
parser = argparse.ArgumentParser(description=f'ANTGO')

def DEFINE_int(name, default, var_help):
    global parser
    parser.add_argument(
    f'--{name}',
    type=int,
    default=default,
    help=var_help)

def DEFINE_float(name, default, var_help):
    global parser
    parser.add_argument(
    f'--{name}',
    type=float,
    default=default,
    help=var_help)

def DEFINE_indicator(name, default, var_help):
    global parser
    parser.add_argument(
    f'--{name}',
    action='store_true' if default else 'store_false',
    help=var_help)   

def DEFINE_string(name, default, var_help):
    global parser
    parser.add_argument(
    f'--{name}',
    action=default,
    help=var_help)   


def nn_args():
    global parser
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
        '--max-epochs',
        type=int,
        default=30,
        help='max epochs'
    )    
    parser.add_argument(
        '--exp',
        type=str,
        default='',
        help='experiment name'
    )
    parser.add_argument('--ext-module', type=str, default='ext_module.py', help='introduce ext module py file')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args