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
class expand_dim_op(object):
    def __init__(self, axis=0):
        self.axis = axis
    
    def __call__(self, x):
        x = np.expand_dims(x, self.axis)
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
    def __init__(self, init_val=0, is_mutable=True):
        assert(mode in ['resident', 'once'])
        self.val = np.array([init_val], dtype=np.int32)
        self.first_call = True
        self.is_mutable = is_mutable

    def __call__(self):
        if self.is_mutable:
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
    def __init__(self, init_val=0, is_placeholder=False, is_mutable=True):
        self.init_val = np.array([init_val]).astype(np.float64)
        self.is_placeholder = is_placeholder
        self.first_call = True
        self.is_mutable = is_mutable

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
            return self.init_val


@register
class scalar_float32_op(object):
    def __init__(self, init_val=0, is_placeholder=False, is_mutable=True):
        self.init_val = np.array([init_val]).astype(np.float64)
        self.is_placeholder = is_placeholder
        self.is_mutable = is_mutable
        self.first_call = True

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
            return self.init_val


@register
class scalar_int32_op(object):
    def __init__(self, init_val=0, is_placeholder=False, is_mutable=True):
        self.init_val = np.array([init_val]).astype(np.int32)
        self.is_placeholder = is_placeholder
        self.is_mutable = is_mutable
        self.first_call = True

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
            return self.init_val


@register
class tensor_int32_op(object):
    def __init__(self, init_val, is_placeholder=False, is_mutable=True):
        assert(init_val.dtype == np.int32)
        self.init_val = init_val
        self.is_placeholder = is_placeholder
        self.is_mutable = is_mutable
        self.first_call = True

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
            return self.init_val

@register
class tensor_float32_op(object):
    def __init__(self, init_val, is_placeholder=False, is_mutable=True):
        assert(init_val.dtype == np.float32)
        self.init_val = init_val
        self.is_placeholder = is_placeholder
        self.is_mutable = is_mutable
        self.first_call = True

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
            return self.init_val


@register
class tensor_float64_op(object):
    def __init__(self, init_val, is_placeholder=False, is_mutable=True):
        assert(init_val.dtype == np.float64)
        self.init_val = init_val
        self.is_placeholder = is_placeholder
        self.is_mutable = is_mutable
        self.first_call = True

    def __call__(self, *args):
        if self.is_mutable:
            if self.first_call:
                self.first_call = False
                return self.init_val

            return NoUpdate()
        else:
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
class bool_equal_compare_op(object):
    def __init__(self, init_val=0):
        # bool value
        self.init_val = init_val

    def __call__(self, *args):
        if len(args) == 1:
            value = args[0]
            return np.array([value[0] == self.init_val], dtype=np.bool)
        else:
            a, b = args[:2]
            return np.array([a[0] == b[0]], dtype=np.bool)


@register
class empty_op(object):
    def __call__(self, *args):
        return np.empty((0), dtype=np.float32)


@register
class state_static_select_op(object):
    def __init__(self, thres_list=[]):
        # thres_list: 有序值(从小到打)
        self.thres_list = thres_list

    def __call__(self, *args):
        value = args[0]
        if value < self.thres_list[index]:
            return np.array([0], dtype=np.int32)

        for index in range(len(self.thres_list)-1):
            if value >= self.thres_list[index] and value < self.thres_list[index+1]:
                return np.array([index], dtype=np.int32)

        return np.array([len(self.thres_list)-1], dtype=np.int32)


@register
class state_dynamic_select_op(object):
    def __call__(self, *args):
        a, b = args[:2]
        if float(a) >= float(b):
            return np.array([1], dtype=np.int32)
        else:
            return np.array([0], dtype=np.int32)


@register
class init_op(object):
    def __init__(self):
        self.value = None

    def __call__(self, *args):
        if self.value is None:
            self.value = args
        return self.value if len(self.value) > 1 else self.value[0]


@register
class command_op(object):
    def __init__(self, func=None):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)