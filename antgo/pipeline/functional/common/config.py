from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

__graph_info = []

def add_op_info(op_name, op_index, op_args, op_kwargs):
    global __graph_info
    __graph_info.append({
        'op_name': op_name,     # 算子名称
        'op_index': op_index,   # 上下游数据流转
        'op_args': op_args,     # 算子参数
        'op_kwargs': op_kwargs  # 算子参数
    })

def get_graph_info():
    global __graph_info
    return __graph_info