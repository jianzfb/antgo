# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine import *
import cv2
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