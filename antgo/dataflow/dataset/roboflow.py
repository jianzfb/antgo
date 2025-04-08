# -*- coding: UTF-8 -*-
# @Time : 2018/8/24
# @File : roboflow.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import copy
import itertools
import json
import time
import numpy as np
from collections import defaultdict
import sys
from urllib.request import urlretrieve
from antgo.dataflow.dataset.dataset import *
from pycocotools.coco import COCO
from itertools import filterfalse, groupby


__all__ = ['Roboflow']

class Roboflow(Dataset):
    def __init__(self, train_or_test="train", dir=None):
        super(Roboflow, self).__init__(train_or_test, dir)
        self.dir = dir
        self.all_data_list = []
        part_list = []
        if train_or_test == 'trainval':
            part_list = ['train', 'valid']
        elif train_or_test == 'all':
            part_list = ['train', 'valid', 'test']
        else:
            part_list = [train_or_test]

        for dataset_name in os.listdir(self.dir):
            if dataset_name[0] == '.':
                continue
            for part_name in part_list:
                folder = os.path.join(self.dir, dataset_name, part_name)
                if not os.path.exists(folder):
                    continue

                anno_file = os.path.join(folder, '_annotations.coco.json')
                instance_list = []
                image_list = []
                coco = COCO(anno_file)
                for img_id in coco.getImgIds():
                    img = coco.loadImgs(img_id)[0]
                    img.update({
                        'img_id':
                        img_id,
                        'img_path':
                        os.path.join(folder, img['file_name']),
                    })
                    image_list.append(img)

                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    for ann in coco.loadAnns(ann_ids):
                        instance_info = self.parse_data_info(
                            dict(raw_ann_info=ann, raw_img_info=img, dataset=dataset_name))

                        # skip invalid instance annotation.
                        if not instance_info:
                            continue
                        instance_list.append(instance_info)

                del coco
                data_list = self._get_bottomup_data_infos(instance_list, image_list)
                self.all_data_list.extend(data_list)

        self.sample_num = len(self.all_data_list)

    def parse_data_info(self, raw_data_info: dict):
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']
        dataset_name = raw_data_info['dataset']

        # width, height not accuracy
        # couldnt use width, height to clip
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1, y1 ,x2, y2 = x, y, x+w, y+h

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': None,
            'id': ann['id'],
            'category_id': np.array([ann.get('category_id',1)]),
        }
        return data_info

    def _get_bottomup_data_infos(self, instance_list, image_list):
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_id, data_infos in groupby(instance_list,
                                          lambda x: x['img_id']):
            used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_path = data_infos[0]['img_path']
            data_info_bu = {
                'img_id': img_id,
                'img_path': img_path,
            }

            # group all instance in one image
            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq

            # rename key
            if 'bbox' in data_info_bu:
                data_info_bu['bboxes'] = data_info_bu['bbox']
                data_info_bu.pop('bbox')
            if 'bbox_score' in data_info_bu:
                data_info_bu['bboxes_score'] = data_info_bu['bbox_score']
                data_info_bu.pop('bbox_score')

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)
        return data_list_bu

    @staticmethod
    def _is_valid_instance(data_info):
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid bbox
        if 'bbox' in data_info:
            bbox = data_info['bbox'][0]
            w, h = bbox[2:4] - bbox[:2]
            if w <= 0 or h <= 0:
                return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    @property
    def size(self):
        return self.sample_num

    def sample(self, id):
        anno_info = self.all_data_list[id]
        image = cv2.imread(anno_info['img_path'])
        if image is None:
            return {
                'image': np.zeros((256,256,3), dtype=np.uint8),
                'bboxes': np.empty((0,4), dtype=np.float32),
                'labels': np.zeros((0), dtype=np.int32)
            }

        bboxes = np.array(anno_info['bboxes'], dtype=np.float32)
        labels = np.array(anno_info['category_id'], dtype=np.int32)

        return {
            'image': image,
            'bboxes': bboxes,
            'labels': np.zeros((len(bboxes)), dtype=np.int32)
        }


# rf = Roboflow(train_or_test='train', dir='/workspace/dataset/helmet-dataset/roboflow')
# print(rf.size)