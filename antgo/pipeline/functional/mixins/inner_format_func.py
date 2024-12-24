import json
import cv2
import os
import yaml
from antgo.utils.sample_gt import *


class InnerFormatGen(object):
    def __init__(self, save_path, category_map, mode='detect', prefix="data"):
        self.sgt = SampleGTTemplate()
        self.save_path = save_path
        self.data_folder = os.path.join(save_path, 'data')
        os.makedirs(self.data_folder, exist_ok=True)
        self.image_folder = os.path.join(self.data_folder, 'image')
        os.makedirs(self.image_folder, exist_ok=True)
        self.mask_folder = os.path.join(self.data_folder, 'mask')
        os.makedirs(self.mask_folder, exist_ok=True)

        self.anno_info_list = []
        self.category_map = category_map
        self.mode = mode
        self.prefix = prefix
        self.index = 0
        self.stage = 'train'

    def add(self, sample_info, stage='train'):
        gt_info = self.sgt.get()
        image = sample_info.__dict__['image']
        image_h, image_w = image.shape[:2]
        self.stage = stage
        image_path = os.path.join(self.data_folder, 'image', f'{self.index}.webp')
        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

        gt_info['image_file'] = f'data/image/{self.index}.webp'
        gt_info['height'] = image_h
        gt_info['width'] = image_w

        mask_path = ''
        if 'segments' in sample_info.__dict__:
            segments = sample_info.__dict__['segments']
            mask_path = f'data/mask/{self.index}.png'
            cv2.imwrite(os.path.join(self.data_folder, 'mask', f'{self.index}.png'), segments)
        gt_info['semantic_file'] = mask_path

        joints2d = []
        if 'joints2d' in sample_info.__dict__:
            joints2d = sample_info.__dict__['joints2d']
        joints3d = []
        if 'joints3d' in sample_info.__dict__:
            joints3d = sample_info.__dict__['joints3d']
        joints_vis = []
        if 'joints_vis' in sample_info.__dict__:
            joints_vis = sample_info.__dict__['joints_vis']

        gt_info['joints2d'] = joints2d
        gt_info['joints3d'] = joints3d
        gt_info['joints_vis'] = joints_vis

        bboxes = []
        labels = []
        if 'bboxes' in sample_info.__dict__:
            bboxes = sample_info.__dict__['bboxes']
            if bboxes.shape[-1] != 4:
                labels = bboxes[:, -1].tolist()
                bboxes = bboxes[:, :4].tolist()
            else:
                bboxes = bboxes.tolist()

        labels = []
        if 'labels' in sample_info.__dict__:
            labels = sample_info.__dict__['labels']
            labels = labels.tolist()

        gt_info['bboxes'] = bboxes
        gt_info['labels'] = labels

        image_label = -1
        if 'image_label' in sample_info.__dict__:
            image_label = sample_info.__dict__['image_label']
        gt_info['image_label'] = image_label

        self.anno_info_list.append(gt_info)
        self.index += 1

    def save(self):
        with open(os.path.join(self.save_path, f'{self.prefix}-{self.stage}.json'), 'w') as fp:
            json.dump(self.anno_info_list, fp)
