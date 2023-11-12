# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine import *
from antgo.pipeline.engine.execution.base_data import *
import os
import base64
import numpy as np


@register 
class stack(object):
    def __init__(self, axis=0, transpose_dims=None) -> None:
        self.axis = axis
        self.transpose_dims = transpose_dims
    
    def __call__(self, x):
        x = np.stack(x, self.axis)
        if self.transpose_dims is not None:
            x = np.transpose(x, self.transpose_dims)
        return x


@register
class global_op(object):
    def __init__(self, *args, mode='resident', **kwargs) -> None:
        assert(len(args) == 0 or len(kwargs) == 0)
        if len(args) > 0:
            self.memory_obj = args if len(args) > 1 else args[0]
        else:
            self.memory_obj = {}
            self.memory_obj.update(kwargs)
        
        self.mode = mode
        self.first_call = True

    def __call__(self, *args, **kwargs):
        if self.mode == 'resident':
            if self.first_call:
                self.first_call = False
                return self.memory_obj
            
            return NoUpdate()
        else:
            return self.memory_obj


@register
class condition_op(object):
    def __init__(self, init_val=True, mode='resident'):
        # mode: 
        # resident: 常驻模式, 多次运行始终维持一个变量，适合运行连续帧计算管线
        # once: 每次运行重新初始化
        if not isinstance(init_val, bool):
            raise TypeError(f'Op bool as init_val, but {type(init_val)} are given.')
        assert(mode in ['resident', 'once'])
        self.val = np.array([init_val], dtype=bool)
        self.mode = mode
        self.first_call = True

    def __call__(self):
        if self.mode == 'resident':
            if self.first_call:
                self.first_call = False
                return self.val

            return NoUpdate()
        else:
            return self.val


@register
class const_op(object):
    def __init__(self, init_val):
        self.val = init_val

    def __call__(self):
        return self.val


@register
class filepath_parse(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs):
        file_path = args[0]
        file_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        return file_dir, base_name


@register
class ifelseend_op(object):
    def __init__(self, true_func, false_func):
        from antgo.pipeline.engine.factory import op
        func_obj, func_args = true_func
        self.true_func = op(func_obj.replace('_','-'), kwargs=func_args)
        func_obj, func_args = false_func
        self.false_func = op(func_obj.replace('_','-'), kwargs=func_args)
    
    def __call__(self, *args):
        condition_val = args[0]
        if condition_val:
            return self.true_func(*args[1:])
        else:
            return self.false_func(*args[1:])