# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : vggface.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import numpy as np
import time
from antgo.dataflow.dataset import *
from antgo.framework.helper.fileio.file_client import *
import cv2
from filelock import FileLock


class VGGFace(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    if train_or_test != 'train':
      train_or_test = 'test'
    super(VGGFace, self).__init__(train_or_test, dir, ext_params=ext_params)
    assert(train_or_test in ['train', 'test', 'val'])
    lock = FileLock('DATASET.lock')
    with lock:
      if not os.path.exists(os.path.join(self.dir, 'data')) or not os.path.exists(os.path.join(self.dir, 'meta')):
        # 数据集不存在，需要重新下载，并创建标记
        ali = AliBackend()
        ali.download('ali:///dataset/vgg-face2/data', self.dir)
        ali.download('ali:///dataset/vgg-face2/meta', self.dir)

        os.system(f'cd {os.path.join(self.dir, "data")} && tar -xf vggface2_train.tar.gz && tar -xf vggface2_test.tar.gz')
        os.system(f'cd {os.path.join(self.dir, "meta")} && tar -xf bb_landmark.tar.gz')

    meta_file_name = 'loose_bb_train.csv' if self.train_or_test == 'train' else 'loose_bb_test.csv'
    
    self.data_list = []
    person_id_map = {}
    with open(os.path.join(self.dir, 'meta', 'bb_landmark', meta_file_name), 'r') as fp:
      # skip first line
      content = fp.readline()
      # file_name, x,y,w,h
      content = fp.readline()
      content = content.strip()
      while content:
        file_name, x,y,w,h = content.split(',')
        file_name = file_name[1:-1]
        person_id, _ = file_name.split('/')
        if person_id not in person_id_map:
          person_id_map[person_id] = len(person_id_map)
        
        self.data_list.append({
          'filepath': os.path.join(self.dir, 'data', self.train_or_test, f'{file_name}.jpg'),
          'person_id': person_id_map[person_id],
          'bbox': [int(x), int(y), int(x)+int(w), int(y)+int(h)]
        })
              
        content = fp.readline()
        content = content.strip()


    if self.train_or_test != 'train':
      # 重新设置person_id_map
      person_id_map = {}
      with open(os.path.join(self.dir, 'meta', 'bb_landmark', 'loose_bb_train.csv'), 'r') as fp:
        # skip first line
        content = fp.readline()
        # file_name, x,y,w,h
        content = fp.readline()
        content = content.strip()
        while content:
          file_name, x,y,w,h = content.split(',')
          file_name = file_name[1:-1]
          person_id, _ = file_name.split('/')
          if person_id not in person_id_map:
            person_id_map[person_id] = len(person_id_map)
                
          content = fp.readline()
          content = content.strip()

    print(f'person id num {len(person_id_map)}')
    print(f'sample num {len(self.data_list)}')
    
  @property
  def size(self):
    return len(self.data_list)

  def at(self, id):
    info = self.data_list[id]
    image = cv2.imread(info['filepath'])
    h,w = image.shape[:2]
    x0,y0,x1,y1 = info['bbox']
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    id = info['person_id']
    crop_image = image[y0:y1,x0:x1].copy()
    return crop_image, {'label':id, 'image_meta': {'image_shape': crop_image.shape[:2]}}

# vgg = VGGFace('test', '/opt/tiger/handdetJ/dataset/vggface2')
# size = vgg.size
# print(f'vgg size {size}')
# for i in range(size):
#   data = vgg.sample(i)
#   cv2.imwrite('./aabb.png', data['image'])
#   print(i)