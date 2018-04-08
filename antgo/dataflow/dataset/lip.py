# -*- coding: UTF-8 -*-
# @Time : 29/03/2018
# @File : lip.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np

__all__ = ['LIP']
class LIP(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(LIP, self).__init__(train_or_test, dir, params)
    assert (train_or_test in ['train', 'sample', 'val'])

    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    self.data_list = []
    parse_file = os.path.join(self.dir, '%s' % ('train_id.txt' if self.train_or_test == 'train' else 'val_id.txt'))
    image_folder = os.path.join(self.dir, '%s_images' % ('train' if self.train_or_test == 'train' else 'val'))
    label_folder = os.path.join(self.dir, '%s_segmentations' % ('train' if self.train_or_test == 'train' else 'val'))
    with open(parse_file, 'r') as fp:
      content = fp.readline()
      while content:
        content = content.replace('\n', '')
        self.data_list.append((os.path.join(image_folder, content),
                               os.path.join(label_folder, content)))

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

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      # idxs = np.arange(len(self.ids))
      idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(idxs)

      for k in idxs:
        image_file, label_file = self.data_list[k]
        image = imread('%s.jpg' % image_file)
        label = imread('%s.png' % label_file)
        if len(label.shape) > 2:
          label = label[:, :, 0]

        included_cls = getattr(self, 'included', None)
        if included_cls is not None:
          included_cls = [int(a) for a in included_cls.split(',')]

        excluded_cls = getattr(self, 'excluded', None)
        if excluded_cls is not None:
          excluded_cls = [int(a) for a in excluded_cls.split(',')]

        existed_cls = None
        if included_cls is not None or excluded_cls is not None:
          existed_cls = list(set(label.flatten().tolist()))
          existed_cls.sort()

        valid_objs_num = -1
        valid_objs_index = []
        if included_cls is not None:
          valid_objs_num = len(existed_cls)

          for cls_label in existed_cls:
            if cls_label not in included_cls:
              cls_position = np.where(label == cls_label)
              label[cls_position] = 0
              valid_objs_num -= 1
            else:
              valid_objs_index.append(cls_label)

        if excluded_cls is not None:
          for cls_label in existed_cls:
            if cls_label in excluded_cls:
              cls_position = np.where(label == cls_label)
              label[cls_position] = 0
              valid_objs_num -= 1
            else:
              valid_objs_index.append(cls_label)

        if valid_objs_num == 0:
          continue

        # split to channels for every object
        segmentation_objs = []
        height, width = image.shape[0:2]
        for obj_index in valid_objs_index:
          seg = np.zeros((height, width), dtype=np.uint8)
          seg[np.where(label == obj_index)] = 1

          segmentation_objs.append(seg)

        yield [image, {'segmentation': segmentation_objs, 'segmentation_map': label, 'id': k}]

  def at(self, id):
    image_file, label_file = self.data_list[id]
    image = imread(image_file)
    label = imread(label_file)
    return [image, {'segmentation': label, 'segmentation_map': label, 'id': id}]

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    val_dataset = LIP('val', self.dir)

    return self, val_dataset
