from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import importlib
import os
import sys
from pprint import pprint
parser = argparse.ArgumentParser(description=f'ANTGO')

def DEFINE_int(name, default, var_help):
    global parser
    parser.add_argument(
        f'--{name}',
        type=int,
        default=default,
        help=var_help
    )

def DEFINE_float(name, default, var_help):
    global parser
    parser.add_argument(
        f'--{name}',
        type=float,
        default=default,
        help=var_help
    )

def DEFINE_indicator(name, default, var_help):
    global parser
    parser.add_argument(
        f'--{name}',
        action='store_true' if default else 'store_false',
        help=var_help
    )   

def DEFINE_string(name, default, var_help):
    global parser
    parser.add_argument(
        f'--{name}',
        type=str,
        default=default,
        help=var_help
    )   

def DEFINE_choices(name, default, choices, var_help):
    global parser
    parser.add_argument(
        f'--{name}',
        choices=choices,
        default=default,
        help=var_help
    )    


def DEFINE_nn_args():
    global parser
    group_gpus = parser.add_mutually_exclusive_group()
    parser.add_argument('--config', type=str, default="config.py", help='train config file path')
    parser.add_argument('--extra-config', type=str, default="", help='train extra config file path')
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
        type=str,
        default='0',
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
    parser.add_argument('--local-rank', type=int, default=0)
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
        default=None,
        help='checkpoint'
    ) 
    parser.add_argument(
        '--process',
        choices=['train', 'test', 'export', 'activelearning'],
        default='train',
        help='train or test process'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=-1,
        help='train max epochs'
    )    
    parser.add_argument(
        '--root',
        type=str,
        default='',
        help='root path'        
    )
    parser.add_argument(
        '--exp',
        type=str,
        default='',
        help='experiment name'
    )
    parser.add_argument(
        '--find-unused-parameters',
        action='store_true',
        help='only work on multi gpu training process'
    )
    parser.add_argument('--ext-module', type=str, default='', help='introduce ext module py file')


def parse_args():
    global parser
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        if getattr(args, 'local_rank', None):
           os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def print_args(s):
    s_dict = {}
    for k,v in s._get_kwargs():
        s_dict[k] = v
    pprint(f'exp {s.exp} config')
    pprint(s_dict)
