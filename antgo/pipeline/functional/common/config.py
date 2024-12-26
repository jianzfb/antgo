from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np

__graph_info = {}

def add_op_info(op_name, op_index, op_args, op_kwargs, op_config=None):
    global __graph_info
    if 'op' not in __graph_info:
        __graph_info['op'] = []

    # for k in op_kwargs.keys():
    #     if isinstance(op_kwargs[k], np.ndarray):
    #         op_kwargs[k] = op_kwargs[k].tolist()

    __graph_info['op'].append({
        'op_name': op_name,     # 算子名称
        'op_index': op_index,   # 上下游数据流转
        'op_args': op_args,     # 算子参数
        'op_kwargs': op_kwargs,  # 算子参数
        'op_config': op_config
    })


def add_input_config(input_config):
    global __graph_info
    __graph_info['input'] = input_config


def add_output_config(output_config):
    global __graph_info
    __graph_info['output'] = output_config


def get_graph_info():
    global __graph_info
    return __graph_info


def clear_grap_info():
    global __graph_info
    __graph_info.clear()