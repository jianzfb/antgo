# -*- coding: UTF-8 -*-
# @Time    : 2022/9/14 23:25
# @File    : plot.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from antgo.pipeline.engine import *
import cv2
import numpy as np


@register
class resize_op(object):
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }
    def __init__(self, size, interp='LINEAR') -> None:
        self.resize_w = size[0]
        self.resize_h = size[1]
        self.interp = interp

    def __call__(self, image):
        image = cv2.resize(
            image, 
            (self.resize_w, self.resize_h), 
            interpolation=self.interp_dict[self.interp])
        return image


@register
class keep_ratio_op(object):
    def __init__(self, aspect_ratio) -> None:
        self.aspect_ratio = aspect_ratio

    def __call__(self, image):
        rhi, rwi = image.shape[:2]
        if abs(rwi / rhi - self.aspect_ratio) > 0.0001:
            if rwi / rhi > self.aspect_ratio:
                nwi = rwi
                nhi = int(rwi / self.aspect_ratio)
            else:
                nhi = rhi
                nwi = int(rhi * self.aspect_ratio)

            # 随机填充
            assert(nhi >= rhi)
            top_padding = (nhi - rhi) // 2
            bottom_padding = (nhi - rhi) - top_padding

            assert(nwi >= rwi)
            left_padding = (nwi - rwi)//2
            right_padding = (nwi - rwi) - (nwi - rwi)//2
  
            # 调整image
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                            cv2.BORDER_CONSTANT, value=(128, 128, 128))
            
        return image


@register
class preprocess_op(object):
    def __init__(self, mean, std, permute=None) -> None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        
        self.mean = np.float64(mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(std.reshape(1, -1))
        self.permute = permute
    
    def __call__(self, image):
        image = image.astype(np.float32)
        cv2.subtract(image, self.mean, image)  # inplace
        cv2.multiply(image, self.stdinv, image)  # inplace
        
        if self.permute is not None:
            image = np.transpose(image, self.permute)
        return image

@register
def convert_rgb2bgr_op(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)    
    return x