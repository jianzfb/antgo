# -*- coding: UTF-8 -*-
# @Time : 29/03/2018
# @File : lip.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
from antgo.dataflow.dataset import *
import os
import numpy as np
import cv2
import time
from antgo.framework.helper.fileio.file_client import *
from filelock import FileLock


__all__ = ['LIP']
class LIP(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(LIP, self).__init__(train_or_test, dir, ext_params)
    assert (train_or_test in ['train', 'val'])
    lock = FileLock('DATASET.lock')
    with lock:
      if not os.path.exists(os.path.join(self.dir, 'train_id.txt')):
        # 数据集不存在，需要重新下载，并创建标记
        if not os.path.exists(self.dir):
          os.makedirs(self.dir)

        ali = AliBackend()
        ali.download('ali:///dataset/lip/TrainVal_images.zip', self.dir)
        ali.download('ali:///dataset/lip/Testing_images.zip', self.dir)
        ali.download('ali:///dataset/lip/TrainVal_parsing_annotations.zip', self.dir)
        ali.download('ali:///dataset/lip/train_id.txt', self.dir)
        ali.download('ali:///dataset/lip/val_id.txt', self.dir)

        os.system(f'cd {self.dir} && unzip TrainVal_images.zip && unzip Testing_images.zip && unzip TrainVal_parsing_annotations.zip')

    self.data_list = []    
    parse_file = os.path.join(self.dir, '%s' % ('train_id.txt' if self.train_or_test == 'train' else 'val_id.txt'))
    image_folder = os.path.join(self.dir, '%s_images' % ('train' if self.train_or_test == 'train' else 'val'))
    label_folder = os.path.join(self.dir, '%s_segmentations' % ('train' if self.train_or_test == 'train' else 'val'))
    with open(parse_file, 'r') as fp:
      content = fp.readline()
      while content:
        content = content.replace('\n', '')
        self.data_list.append((os.path.join(image_folder, f'{content}.jpg'),
                               os.path.join(label_folder, f'{content}.png')))

        content = fp.readline()

    self.ids = list(range(len(self.data_list)))

    self.inv_category_map = {
      0: 'background',
      1: 'Hat',
      2: 'Hair',
      3: 'Sunglasses',
      4: 'Upper-clothes',
      5: 'Dress',
      6: 'Coat',
      7: 'Socks',
      8: 'Pants',
      9: 'Glove',
      10: 'Scarf',
      11: 'Skirt',
      12: 'Jumpsuits',
      13: 'Face',
      14: 'Right-arm',
      15: 'Left-arm',
      16: 'Right-leg',
      17: 'Left-leg',
      18: 'Right-shoe',
      19: 'Left-shoe'
    }
    self.category_map = {v: k for k, v in self.inv_category_map.items()}

  @property
  def size(self):
    return len(self.ids)

  def at(self, id):
    image_file, label_file = self.data_list[id]
    image = cv2.imread(image_file)
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    return (
      image,
      {
        'segments': label,
        'image_meta': {
            'image_shape': (image.shape[0], image.shape[1]),
            'image_file': image_file
          }   
      }
    )

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    val_dataset = LIP('val', self.dir)

    return self, val_dataset

# p2012 = LIP('train', '/root/workspace/dataset/B')
# print(f'p2012 size {p2012.size}')
# for i in range(p2012.size):
#   result = p2012.sample(i)
#   ss = result['segments']
#   # cv2.imwrite('./1234.png', (ss/20*255).astype(np.uint8))
#   print(i)
# value = p2012.sample(0)
# print(value.keys())
# value = p2012.sample(1)
# print(value)