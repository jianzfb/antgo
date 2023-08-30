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

def perspective_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    new_pt[:2] = new_pt[:2] / new_pt[-1]
    return new_pt[:2]


@register
class sync_layout_op(object):
    def __init__(self, layout_gen, layout_id=1):
        self.layout_gen = layout_gen if isinstance(layout_gen, list) else [layout_gen]
        if not isinstance(layout_id, list):
            layout_id = [layout_id]
        if len(layout_id) != len(self.layout_gen):
            layout_id = [layout_id[0]] * len(self.layout_gen)
        self.layout_id = layout_id

    def extract(self, layout_image, layout_info, layout_id):
        layout_h, layout_w = layout_image.shape[:2]
        layout_mask = np.zeros((layout_h, layout_w), dtype=np.uint8)

        layout_alpha = layout_image[:,:,3]
        pos = np.where(layout_alpha > 128)
        y0 = np.min(pos[0])
        y1 = np.max(pos[0])
        x0 = np.min(pos[1])
        x1 = np.max(pos[1])

        layout_image = layout_image[y0:y1, x0:x1]
        layout_alpha = layout_image[:,:,3] / 255
        layout_mask = layout_mask[y0:y1, x0:x1]
        layout_mask = layout_mask * (1-layout_alpha) + layout_alpha * layout_id

        layout_h, layout_w = layout_image.shape[:2]
        layout_points = None
        if 'points' in layout_info:
            layout_points = layout_info['points']
            layout_points = layout_points - np.array([[x0, y0]])
        return layout_image, layout_mask, layout_points

    def __call__(self, image):
        # layout_image: HxWx4
        # layout_info: {'points': [[],[]]}
        layout_image, layout_info = self.layout_gen[0](image)
        layout_image, layout_mask, layout_points = self.extract(layout_image, layout_info, self.layout_id[0])
        layout_h, layout_w = layout_image.shape[:2]

        for layout_i in range(1, len(self.layout_gen)):
            overlap_layout_image, overlap_layout_info = self.layout_gen[layout_i](image)
            overlap_layout_image, overlap_layout_mask, overlap_layout_points = self.extract(overlap_layout_image, overlap_layout_info, self.layout_id[layout_i])
            overlap_layout_h, overlap_layout_w = overlap_layout_image.shape[:2]

            # 随机叠加
            random_min_scale, random_max_scale = self.layout_gen[layout_i].scale()
            scale_value = (layout_w * (random_min_scale + (random_max_scale-random_min_scale) * np.random.random()))/overlap_layout_w
            scaled_w = int(overlap_layout_w * scale_value)
            scaled_h = int(overlap_layout_w * scale_value)
            overlap_layout_image = cv2.resize(overlap_layout_image, (scaled_w, scaled_h))
            overlap_layout_mask = cv2.resize(overlap_layout_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

            overlap_layout_mask_flag = (overlap_layout_mask > 0).astype(np.uint8)

            paste_cx = np.random.random()*(layout_w - scaled_w)
            paste_cy = np.random.random()*(layout_h - scaled_h)
            paste_cx = int(paste_cx)
            paste_cy = int(paste_cy)
            layout_image[paste_cy:paste_cy+scaled_h, paste_cx:paste_cx+scaled_w] = \
                layout_image[paste_cy:paste_cy+scaled_h, paste_cx:paste_cx+scaled_w] * (1-np.expand_dims(overlap_layout_mask_flag,-1)) + overlap_layout_image * np.expand_dims(overlap_layout_mask_flag, -1)
            layout_mask[paste_cy:paste_cy+scaled_h, paste_cx:paste_cx+scaled_w] = \
                layout_mask[paste_cy:paste_cy+scaled_h, paste_cx:paste_cx+scaled_w] * (1-overlap_layout_mask_flag) + self.layout_id[layout_i] * overlap_layout_mask_flag

            if overlap_layout_points is not None:
                overlap_layout_points = overlap_layout_points + np.array([[paste_cx, paste_cy]])
                if layout_points is None:
                    layout_points = overlap_layout_points
                else:
                    layout_points = np.concatenate([layout_points, overlap_layout_points], 0)

        return {
            'layout_image': layout_image,
            'layout_mask': layout_mask,
            'layout_points': layout_points
        }


@register
class sync_op(object):
    def __init__(self, min_scale=0.5, max_scale=0.8, border_fill=0, hard_paste=False):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.border_fill = border_fill
        self.hard = hard_paste

    def __call__(self, image, *args):
        image = image.copy()
        image_h, image_w, _ = image.shape        
        mask = np.ones((image_h, image_w), dtype=np.uint8) * self.border_fill
        sync_points = None
        # step 1. 图像调整到适合尺寸
        random_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale

        for layout_info in args:
            layout_image = layout_info['layout_image']
            layout_mask = layout_info['layout_mask']
            layout_points = layout_info['layout_points']

            object_image = layout_image
            obj_h, obj_w, _ = object_image.shape

            min_size = max(image_w, image_h) * random_scale
            obj_scale = min_size / max(obj_h, obj_w)

            if obj_w * obj_scale > image_w * random_scale or obj_h * obj_scale > image_h * random_scale:
                if obj_w * obj_scale > image_w * random_scale:
                    obj_scale = image_w * random_scale / obj_w

                if obj_h * obj_scale > image_h * random_scale:
                    obj_scale = image_h * random_scale / obj_h

            object_image = cv2.resize(object_image, dsize=(int(obj_w * obj_scale), int(obj_h * obj_scale)))
            obj_h, obj_w = object_image.shape[:2]

            if layout_points is not None:
               layout_points = layout_points * obj_scale

            object_paste_mask = np.ones((obj_h, obj_w)).astype(np.uint8)
            if object_image.shape[-1] == 4:
                object_paste_mask = object_image[:,:, 3] / 255
                object_image = object_image[:,:,:3]

            layout_mask = cv2.resize(layout_mask, dsize=(obj_w, obj_h), interpolation=cv2.INTER_NEAREST)

            # step 2. 随机透射变换
            # 生成随机变换矩阵 3x3
            p1 = np.array([[0,0], [obj_w-1, 0], [0, obj_h-1], [obj_w-1, obj_h-1]]).astype(np.float32)
            tgt_p1 = [np.random.randint(0, obj_w//8), np.random.randint(0, obj_h//8)]
            tgt_p2 = [np.random.randint(obj_w-obj_w//8, obj_w), np.random.randint(0, obj_h//8)]
            tgt_p3 = [np.random.randint(0, obj_w//8), np.random.randint(obj_h-obj_h//8, obj_h)]
            tgt_p4 = [np.random.randint(obj_w-obj_w//8, obj_w), np.random.randint(obj_h-obj_h//8, obj_h)]
            p2 = np.array([tgt_p1, tgt_p2, tgt_p3, tgt_p4]).astype(np.float32)
            M = cv2.getPerspectiveTransform(p1, p2)
            object_image = cv2.warpPerspective(object_image, M, (obj_w, obj_h))
            object_paste_mask = cv2. warpPerspective(object_paste_mask, M, (obj_w, obj_h))
            if layout_points is not None:
                for point_i in range(layout_points.shape[0]):
                    layout_points[point_i, :2] = perspective_transform(layout_points[point_i, :2], M)

            layout_mask = \
                    cv2.warpPerspective(
                        layout_mask,
                        M, 
                        (obj_w, obj_h), 
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=self.border_fill,
                        flags=cv2.INTER_NEAREST)

            # 合成
            paste_x = 0
            if image_w > obj_w:
                paste_x = np.random.randint(0, image_w - obj_w)
            paste_y = 0
            if image_h > obj_h:
                paste_y = np.random.randint(0, image_h - obj_h)

            print(f'image {image_w} {image_h}')
            print(f'obj {obj_w} {obj_h}')
            object_paste_mask_expand = np.expand_dims(object_paste_mask, -1)
            image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask_expand) + object_image * object_paste_mask_expand

            if layout_points is not None:
                layout_points = layout_points + np.float32([[paste_x, paste_y]])
                if sync_points is None:
                    sync_points = []
                sync_points.append(layout_points)

            if self.hard:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = layout_mask
            else:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask) + layout_mask * object_paste_mask

        sync_image = image
        sync_mask = mask
        if sync_points is not None:
            sync_points = np.concatenate(sync_points, 0)

        return {
            'layout_image': sync_image,
            'layout_mask': sync_mask,
            'layout_points': sync_points
        }