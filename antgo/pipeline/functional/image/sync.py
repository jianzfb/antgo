# -*- coding: UTF-8 -*-
# @Time    : 2022/9/14 23:25
# @File    : sync.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from typing import Any
from antgo.pipeline.engine import *
from antgo.pipeline.functional.mixins.yolo_format_func import YOLOFormatGen
from antgo.pipeline.functional.mixins.coco_format_func import COCOFormatGen
from antgo.pipeline.functional.mixins.inner_format_func import InnerFormatGen
from antgo.utils.sample_gt import *
from antgo.pipeline.functional.entity import *
import cv2
import json
import numpy as np
import json


def perspective_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    new_pt[:2] = new_pt[:2] / new_pt[-1]
    return new_pt[:2]


@register
class sync_op(object):
    def __init__(self, min_scale=0.2, max_scale=0.5, border_fill=0, hard_paste=False, layout_label=None):
        # 目标在背景图上所占比例约束
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.border_fill = border_fill
        self.hard = hard_paste
        self.layout_label = layout_label

    def __call__(self, image, *args):
        image = image.copy()
        image_h, image_w, _ = image.shape        
        mask = np.ones((image_h, image_w), dtype=np.uint8) * self.border_fill
        sync_points = None

        for layout_info in args:
            layout_image = layout_info['layout_image']
            layout_id = layout_info['layout_id']
            if layout_id.dtype != np.uint8:
                layout_id = layout_id.astype(np.uint8)
            layout_points = layout_info.get('layout_points', None)

            object_image = layout_image
            obj_h, obj_w, _ = object_image.shape

            # step 1. 图像调整到适合尺寸
            random_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale

            obj_scale = (min(image_w, image_h) * random_scale) / min(obj_h, obj_w)
            object_image = cv2.resize(object_image, dsize=(int(obj_w * obj_scale), int(obj_h * obj_scale)))
            obj_h, obj_w = object_image.shape[:2]

            if layout_points is not None:
               layout_points = layout_points * obj_scale

            object_paste_mask = np.ones((obj_h, obj_w)).astype(np.uint8)
            if object_image.shape[-1] == 4:
                object_paste_mask = object_image[:,:, 3] / 255
                object_image = object_image[:,:,:3]

            layout_id = cv2.resize(layout_id, dsize=(obj_w, obj_h), interpolation=cv2.INTER_NEAREST)

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
            object_paste_mask = \
                cv2. warpPerspective(
                    object_paste_mask, 
                    M,
                    (obj_w, obj_h), 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=self.border_fill, 
                    flags=cv2.INTER_NEAREST)

            if layout_points is not None:
                for point_i in range(layout_points.shape[0]):
                    layout_points[point_i, :2] = perspective_transform(layout_points[point_i, :2], M)

            layout_id = \
                    cv2.warpPerspective(
                        layout_id,
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

            object_paste_mask_expand = np.expand_dims(object_paste_mask, -1)
            image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask_expand) + object_image * object_paste_mask_expand

            if layout_points is not None:
                layout_points = layout_points + np.float32([[paste_x, paste_y]])
                if sync_points is None:
                    sync_points = []
                sync_points.append(layout_points)

            if self.hard:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = layout_id
            else:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask) + layout_id * object_paste_mask

        sync_image = image
        sync_mask = mask
        if sync_points is not None:
            sync_points = np.concatenate(sync_points, 0)

        bboxes = []
        labels = []
        if self.layout_label is None:
            self.layout_label = {1: 'object'}
        for layout_id, layout_label in self.layout_label.items():
            pos = np.where(sync_mask == layout_id)
            min_y = int(np.min(pos[0]))
            min_x = int(np.min(pos[1]))

            max_y = int(np.max(pos[0]))
            max_x = int(np.max(pos[1]))

            bboxes.append([min_x, min_y, max_x, max_y])
            # 标签=图层编号-1
            labels.append(layout_id - 1)

        # if self.keep_layout is not None:
        #     keep_sync_mask = np.zeros(sync_mask.shape, dtype=np.uint8)
        #     for layout_index, layout_id in enumerate(self.keep_layout):
        #         keep_sync_mask[np.where(sync_mask == layout_id)] = layout_index+1
        #     sync_mask = keep_sync_mask

        # image, segments
        sync_info = {
            'image': sync_image,
            'segments': sync_mask
        }

        if len(bboxes) > 0 and len(labels) > 0:
            sync_info.update({
                'bboxes': bboxes, 
                'labels': labels
            })

        # joints2d
        if sync_points is not None:
            sync_info.update({
                'joints2d': sync_points
            })

        return sync_info


@register
class save_sync_info_op(object):
    def __init__(self, folder, category_map={}, dataset_format='yolo', mode='detect', prefix="data", stage='train', sample_num=None):
        self.folder = folder
        self.index = 0
        self.gen_op = None
        self.mode = mode
        self.prefix = prefix
        self.stage = stage
        self.dataset_format = dataset_format
        self.sample_num = None

    def __call__(self, sync_info):
        if self.gen_op is None:
            if self.dataset_format == 'yolo':
                self.gen_op = YOLOFormatGen(self.folder, self.category_map, self.mode, self.prefix)
            elif self.dataset_format == 'coco':
                self.gen_op = COCOFormatGen(self.folder, self.category_map, self.mode, self.prefix)
            else:
                self.gen_op = InnerFormatGen(self.folder, self.category_map, self.mode, self.prefix)

        assert(self.gen_op is not None)
        info = {
            'image': sync_info['image'],
            'labels': np.array(sync_info['labels']) if isinstance(sync_info['labels'], list) else sync_info['labels'],
            'bboxes': np.array(sync_info['bboxes']) if isinstance(sync_info['bboxes'], list) else sync_info['bboxes']
        }

        self.gen_op.add(Entity(**info), self.stage)
        self.index += 1

        if self.sample_num is not None:
            if self.index >= self.sample_num:
                self.gen_op.save()
