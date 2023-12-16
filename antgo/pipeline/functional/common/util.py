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
class state_op(object):
    def __init__(self, init_val=0, mode='resident'):
        assert(mode in ['resident', 'once'])
        self.val = np.array([init_val], dtype=np.int32)
        self.first_call = True
        self.mode = mode

    def __call__(self):
        if self.mode == 'resident':
            if self.first_call:
                self.first_call = False
                return self.val

            return NoUpdate()
        else:
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


@register
class switch_op(object):
    def __init__(self, stride=0):
        self.stride = stride

    def __call__(self, *args):
        state_i = int(args[0])
        if self.stride == 0:
            return args[state_i+1]

        return args[state_i+1::self.stride]


@register
class scalar_float64_op(object):
    def __init__(self, init_val=0, is_placeholder=False):
        self.init_val = np.array([init_val]).astype(np.float64)
        self.is_placeholder = is_placeholder

    def __call__(self, *args):
        return self.init_val


@register
class scalar_float32_op(object):
    def __init__(self, init_val=0, is_placeholder=False):
        self.init_val = np.array([init_val]).astype(np.float64)
        self.is_placeholder = is_placeholder

    def __call__(self, *args):
        return self.init_val


@register
class scalar_int32_op(object):
    def __init__(self, init_val=0, is_placeholder=False):
        self.init_val = np.array([init_val]).astype(np.int32)
        self.is_placeholder = is_placeholder

    def __call__(self, *args):
        return self.init_val


@register
class bool_greater_compare_op(object):
    def __init__(self, thres=0):
        # bool value
        self.thres = thres

    def __call__(self, *args):
        if len(args) == 1:
            value = args[0]
            return np.array([value[0] > self.thres], dtype=np.bool)
        else:
            a, b = args[:2]
            return np.array([a[0] > b[0]], dtype=np.bool)


@register
class bool_greater_equal_compare_op(object):
    def __init__(self, thres=0):
        # bool value
        self.thres = thres

    def __call__(self, *args):
        if len(args) == 1:
            value = args[0]
            return np.array([value[0] >= self.thres], dtype=np.bool)
        else:
            a, b = args[:2]
            return np.array([a[0] >= b[0]], dtype=np.bool)


@register
class state_greater_equal_compare_op(object):
    def __init__(self, thres_list=[]):
        # thres_list: 有序值
        self.thres_list = thres_list

    def __call__(self, *args):
        if len(args) == 1:
            value = args[0]
            for index in range(len(self.thres_list)-1):
                if value >= self.thres_list[index] and value < self.thres_list[index+1]:
                    return np.array([index], dtype=np.int32)

            return np.array([0], dtype=np.int32)
        else:
            a, b = args[:2]
            if float(a) >= float(b):
                return np.array([1], dtype=np.int32)
            else:
                return np.array([0], dtype=np.int32)
