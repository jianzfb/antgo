# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine import *
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
class global_var(object):
    def __init__(self, *args, **kwargs) -> None:
        assert(len(args) == 0 or len(kwargs) == 0)
        if len(args) > 0:
            self.memory_obj = args if len(args) > 1 else args[0]
        else:
            self.memory_obj = {}
            self.memory_obj.update(kwargs)

    def __call__(self, *args, **kwargs):
        return self.memory_obj

@register
class filepath_parse(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs):
        file_path = args[0]
        file_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        return file_dir, base_name
