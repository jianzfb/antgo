# -*- coding: UTF-8 -*-
# @Time    : 2022/9/14 23:25
# @File    : plot.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from typing import Any
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
    def __init__(self, out_size, interp='LINEAR') -> None:
        self.resize_w = out_size[0]
        self.resize_h = out_size[1]
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
    def __init__(self, mean, std, permute=None, expand_dim=False) -> None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        
        self.mean = np.float64(mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(std.reshape(1, -1))
        self.permute = permute
        self.expand_dim = expand_dim
    
    def __call__(self, image):
        image = image.astype(np.float32)
        cv2.subtract(image, self.mean, image)  # inplace
        cv2.multiply(image, self.stdinv, image)  # inplace
        
        if self.permute is not None:
            image = np.transpose(image, self.permute)
            
        if self.expand_dim:
            image = np.expand_dims(image, 0)
        return image

@register
class center_crop_op(object):
    def __init__(self, size) -> None:
        self.crop_height = size[0]
        self.crop_width = size[1]
    
    def __call__(self, image):
        image_height, image_width = image.shape[:2]
        if self.crop_width > image_width or self.crop_height > image_height:
            padding_ltrb = [
                (self.crop_width - image_width) // 2 if self.crop_width > image_width else 0,
                (self.crop_height - image_height) // 2 if self.crop_height > image_height else 0,
                (self.crop_width - image_width + 1) // 2 if self.crop_width > image_width else 0,
                (self.crop_height - image_height + 1) // 2 if self.crop_height > image_height else 0,
            ]
            pad_left, pad_top, pad_right, pad_bottom = padding_ltrb
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=(0, 0))

            image_height, image_width = image.shape[:2]
            if self.crop_width == image_width and self.crop_height == image_height:
                return image
        
        crop_top = int(round((image_height - self.crop_height) / 2.))
        crop_left = int(round((image_width - self.crop_width) / 2.))
        crop_image = image[crop_top:crop_top+self.crop_height, crop_left:crop_left+self.crop_width].copy()
        return crop_image


@register
class nms(object):
    def __init__(self, iou_thres=0.2) -> None:
        self.iou_thres = iou_thres
    
    def __call__(self, boxes, labels) -> Any:
        """ 非极大值抑制 """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2-x1) * (y2-y1)
        keep = []

        # 按置信度进行排序
        index = np.argsort(scores)[::-1]

        while(index.size):
            # 置信度最高的框
            i = index[0]
            keep.append(index[0])

            if(index.size == 1): # 如果只剩一个框，直接返回
                break

            # 计算交集左下角与右上角坐标
            inter_x1 = np.maximum(x1[i], x1[index[1:]])
            inter_y1 = np.maximum(y1[i], y1[index[1:]])
            inter_x2 = np.minimum(x2[i], x2[index[1:]])
            inter_y2 = np.minimum(y2[i], y2[index[1:]])
            # 计算交集的面积
            inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
            # 计算当前框与其余框的iou
            iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
            ids = np.where(iou < self.iou_thres)[0]
            index = index[ids+1]

        return boxes[keep], labels[keep]


@register
def convert_rgb2bgr_op(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)    
    return x