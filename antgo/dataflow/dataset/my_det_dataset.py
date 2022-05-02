# -*- coding: UTF-8 -*-
# @Time    : 2022/4/27 20:38
# @File    : my_det_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import gzip
import random
import numpy as np
from ...utils import logger
from .dataset import Dataset
import time
import copy
import json


class CustomDetDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(CustomDetDataset, self).__init__(train_or_test, dir, params)

    self.roidbs = None
    self.load_roidb()

  def load_roidb(self):
    anno_json = getattr(self, 'anno_json', f"{self.train_or_test}.json")
    anno_path = os.path.join(self.dir, anno_json)
    with open(anno_path) as anno_file:
      anno = json.load(anno_file)

    logger.info(f"Image num = {len(anno)}")
    records = []
    for im_id, a in enumerate(anno):
      image_name = a['image_id']
      dataset_name = a['dataset']
      if 'train2017' in dataset_name:
        dataset_name = 'coco_train'
      elif 'val2017' in dataset_name:
        dataset_name = 'coco_val'

      image_path = os.path.join(self.dir, "images", dataset_name, image_name)
      if not os.path.exists(image_path):
        print(f"Missing {image_path}")
        continue

      coco_rec = {
        'im_file': image_path,
        'im_id': np.array([im_id]),
        'h': 0,
        'w': 0,
      }
      gt_bbox = []
      gt_class = []
      gt_score = []
      is_crowd = []
      difficult = []
      gt_poly = []
      keypoints = []
      keypoints_from_bbox_i = []

      for human_name, human_bbox in a['human_annotations'].items():
        xmin, ymin, xmax, ymax = human_bbox
        if xmin > xmax and ymin > ymax:
          human_bbox = [xmax, ymax, xmin, ymin]
          xmin, ymin, xmax, ymax = human_bbox

        joints_3d = np.zeros((21, 3), dtype=np.float)
        joints_3d_vis = np.zeros((21, 3), dtype=np.float)

        current_keypoint = np.array(a['keypoint_annotations'][human_name]).reshape(-1, 3)
        joints = current_keypoint[:, 0:2]
        joints_vis = current_keypoint[:, -1]

        assert joints.shape[0] == 21, \
          'joint num diff: {} vs {}'.format(joints.shape[0], 21)

        joints_3d[:, 0:2] = joints[:, 0:2]
        joints_3d_vis[:, 0] = joints_vis[:]
        joints_3d_vis[:, 1] = joints_vis[:]
        joints_3d_vis[joints_3d_vis == 2] = 1
        joints_3d_vis[joints_3d_vis == 3] = 0
        joints_3d_vis[joints_3d_vis > 1] = 0

        # 设置可见性
        joints_3d[:, 2] = joints_3d_vis[:, 0]

        # 人脸keypoint
        try:
          face_keypoint = np.array(a['face_ketpoint'][human_name]).reshape(-1, 2)
          joints_3d[18, 0] = face_keypoint[16, 0]
          joints_3d[18, 1] = face_keypoint[16, 1]
          joints_3d_vis[18, 0:2] = 1
          if face_keypoint[16, 0] == 0 and face_keypoint[16, 0] == 0:
            joints_3d_vis[18, 0:2] = 0
        except:
          if dataset_name != 'zhuohua':
            joints_3d[18, :] = 0
            joints_3d_vis[18, 0:2] = 0

        # 左手keypoint
        try:
          left_hand_keypoint = np.array(a['left_hand'][human_name])
          joints_3d[19, 0] = (left_hand_keypoint[0] + left_hand_keypoint[2]) / 2
          joints_3d[19, 1] = (left_hand_keypoint[1] + left_hand_keypoint[3]) / 2
          joints_3d_vis[19, 0:2] = 1
          if left_hand_keypoint[0] == 0 and left_hand_keypoint[2] == 0:
            joints_3d_vis[19, 0:2] = 0
        except:
          if dataset_name != 'zhuohua':
            joints_3d[19, :] = 0
            joints_3d_vis[19, 0:2] = 0

        # 右手keypoint
        try:
          right_hand_keypoint = np.array(a['right_hand'][human_name])
          joints_3d[20, 0] = (right_hand_keypoint[0] + right_hand_keypoint[2]) / 2
          joints_3d[20, 1] = (right_hand_keypoint[1] + right_hand_keypoint[3]) / 2
          joints_3d_vis[20, 0:2] = 1
          if right_hand_keypoint[0] == 0 and right_hand_keypoint[2] == 0:
            joints_3d_vis[20, 0:2] = 0
        except:
          if dataset_name != 'zhuohua':
            joints_3d[20, :] = 0
            joints_3d_vis[20, 0:2] = 0

        # 加入人体框和对应类别 (类别0)
        gt_bbox.append(human_bbox)
        gt_class.append(0)
        gt_score.append(1.0)
        is_crowd.append(0)
        difficult.append(0)
        gt_poly.append(None)

        # 人体keypoint
        joints_3d = np.maximum(joints_3d, 0)
        keypoints.append(joints_3d)
        keypoints_from_bbox_i.append(len(gt_bbox) - 1)

        # # 加入人脸框和对应类别
        # if human_name in a['face_box']:
        #     face_w = a['face_box'][human_name][2] - a['face_box'][human_name][0]
        #     face_h = a['face_box'][human_name][3] - a['face_box'][human_name][1]
        #     if face_w > 0.0 and face_h > 0.0:
        #         gt_bbox.append(a['face_box'][human_name])
        #         gt_class.append(0)
        #         gt_score.append(1.0)
        #         is_crowd.append(0)
        #         difficult.append(0)
        #         gt_poly.append(None)

        # image = cv2.imread(image_path)
        # image = cv2.rectangle(image,
        #             ((int)(xmin), (int)(ymin)),
        #             ((int)(xmax), (int)(ymax)),
        #             (0, 255, 0), 1)
        # cv2.imwrite("./ss.png", image)

      if len(gt_bbox) == 0:
        print('skip')
        continue

      # 添加人体类别
      is_crowd = np.stack(is_crowd, 0)
      is_crowd = np.expand_dims(is_crowd, -1)
      gt_class = np.stack(gt_class, 0)
      gt_class = np.expand_dims(gt_class, -1)
      gt_score = np.stack(gt_score, 0)
      gt_score = np.expand_dims(gt_score, -1)
      gt_bbox = np.stack(gt_bbox, 0)
      difficult = np.stack(difficult, 0)
      difficult = np.expand_dims(difficult, -1)
      keypoints = np.stack(keypoints, 0)
      keypoints_from_bbox_i = np.stack(keypoints_from_bbox_i, 0)

      coco_rec.update({
        'is_crowd': is_crowd,
        'gt_class': gt_class,
        'gt_bbox': gt_bbox,
        'gt_score': gt_score,
        'gt_poly': gt_poly,
        'gt_keypoint': keypoints,
        'gt_keypoint_from_bbox_i': keypoints_from_bbox_i,
      })

      records.append(coco_rec)

    self.roidbs = records

  @property
  def size(self):
    return len(self.roidbs)

  def at(self, id):
    return None, self.roidbs[id]

  def data_pool(self):
    for index in range(len(self.roidbs)):
      yield self.at(index)