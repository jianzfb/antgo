# -*- coding: UTF-8 -*-
# File: pascal_voc.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import os, sys
import numpy as np
import random
import six
from six.moves import urllib, range
import copy
import logging
import tarfile
import xml.etree.ElementTree as ET
import scipy.sparse
from antgo.utils.fs import download
from antgo.utils.fs import maybe_here_match_format
from antgo.utils import logger, get_rng
from antgo.dataflow.dataset.dataset import *

__all__ = ['Pascal2007', 'Pascal2012']

PASCAL2007_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007"
PASCAL2012_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012"


class PascalBase(Dataset):
  def __init__(self, year, image_set, dir=None, ext_params=None):
    super(PascalBase, self).__init__(image_set, dir, ext_params)
    self._year = year
    self._image_set = image_set
    self._devkit_path = dir

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self._image_index = self.load_samples()
      return

    if self._year == '2007':
      if not os.path.exists(os.path.join(self._devkit_path, 'VOCdevkit')):
        self.download(self.dir, ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar'], default_url=PASCAL2007_URL)
      maybe_data_path = maybe_here_match_format(self._devkit_path, 'VOC' + self._year)
      if maybe_data_path is None:
        # auto untar
        tar = tarfile.open(os.path.join(self.dir,'VOCtrainval_06-Nov-2007.tar'), 'r')
        tar.extractall(self.dir)
        tar.close()

        tar = tarfile.open(os.path.join(self.dir, 'VOCtest_06-Nov-2007.tar'), 'r')
        tar.extractall(self.dir)
        tar.close()
    else:
      if not os.path.exists(os.path.join(self._devkit_path, 'VOCdevkit')):
        self.download(self.dir, ['VOCtrainval_11-May-2012.tar'], default_url=PASCAL2012_URL)
        self.download(self.dir, ['VOC2012test.tar'], default_url='http://host.robots.ox.ac.uk:8080/eval/downloads/')

      maybe_data_path = maybe_here_match_format(self._devkit_path, 'VOC' + self._year)
      if maybe_data_path is None:
        # auto untar
        tar = tarfile.open(os.path.join(self.dir, 'VOCtrainval_11-May-2012.tar'), 'r')
        tar.extractall(self.dir)
        tar.close()

        tar = tarfile.open(os.path.join(self.dir, 'VOC2012test.tar'), 'r')
        tar.extractall(self.dir)
        tar.close()

    self._data_path = os.path.join(self.dir,'VOCdevkit', 'VOC' + self._year)
    self._classes = ('background',
                     'aeroplane',
                     'bicycle',
                     'bird',
                     'boat',
                     'bottle',
                     'bus',
                     'car',
                     'cat',
                     'chair',
                     'cow',
                     'diningtable',
                     'dog',
                     'horse',
                     'motorbike',
                     'person',
                     'pottedplant',
                     'sheep',
                     'sofa',
                     'train',
                     'tvmonitor')

    self._c_2_rgb = {0: (0, 0, 0),
                     1: (128, 0, 0),
                     2: (0, 128, 0),
                     3: (128, 128, 0),
                     4: (0, 0, 128),
                     5: (128, 0, 128),
                     6: (0, 128, 128),
                     7: (128, 128, 128),
                     8: (64, 0, 0),
                     9: (192, 0, 0),
                     10: (64, 128, 0),
                     11: (192, 128, 0),
                     12: (64, 0, 128),
                     13: (192, 0, 128),
                     14: (64, 128, 128),
                     15: (192, 128, 128),
                     16: (0, 64, 0),
                     17: (128, 64, 0),
                     18: (0, 192, 0),
                     19: (128, 192, 0),
                     20: (0, 64, 128)}
    self._c_2_rgb_index = {}
    for k, v in self._c_2_rgb.items():
      self._c_2_rgb_index[k] = v[0]*255*255+v[1]*255+v[2]

    self._action_classes = {'phoning': 1,
                            'playinginstrument': 2,
                            'reading': 3,
                            'ridingbike': 4,
                            'ridinghorse': 5,
                            'running': 6,
                            'takingphoto': 7,
                            'usingcomputer': 8,
                            'walking': 9,
                            'jumping': 10,
                            'other': 0}

    self._num_classes = len(self._classes)
    self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 2}

    assert os.path.exists(self._data_path), 'path does not exist: {}'.format(self._data_path)

    self._aug_dataset = []
    self._aug_seg = []
    if self._year == '2012':
      train_aug = getattr(self, 'aug', None)
      task_type = getattr(self, 'task_type', None)
      if train_aug is not None and task_type == 'SEGMENTATION':
        assert(os.path.exists(os.path.join(self.dir, 'SegmentationClassAug')))
        assert(os.path.exists(os.path.join(self.dir, 'trainaug.txt')))
        with open(os.path.join(self.dir, 'trainaug.txt'), 'r') as fp:
          content = fp.readline()

          while(content):
            content = content.strip()
            if content == "":
              break

            self._aug_dataset.append(os.path.join(self.dir, 'VOCdevkit/VOC2012/JPEGImages', '%s.jpg'%content))
            self._aug_seg.append(os.path.join(self.dir, 'SegmentationClassAug', '%s.png'%content))
            content = fp.readline()

  @property
  def size(self):
    return len(self._image_index) + len(self._aug_dataset)

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = list(range(self.size))
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

      # data index shuffle
      idxs = np.arange(len(self._image_index))
      if self.rng:
        self.rng.shuffle(idxs)

      for k in idxs:
        # real index
        index = self._image_index[k]
        if self.train_or_test == 'test' and self._year == '2012':
          image = imread(self.image_path_from_index(index))
          yield [image, {'file_id': index + self._image_ext}]

          continue

        # annotation
        gt_roidb = self._load_roidb(index)

        task_type = getattr(self, 'task_type', None)
        if task_type is not None:
          if task_type == 'ACTION_CLASSIFICATION':
            keep_index = []
            for obj_index, obj_name in enumerate(gt_roidb['category']):
              if obj_name == 'person':
                keep_index.append(obj_index)

            if len(keep_index) == 0:
              continue

            keep_index = [ki for ki in keep_index if gt_roidb['person_action'][ki] is not None]
            if len(keep_index) == 0:
              continue

            gt_roidb['category_id'] = np.concatenate([gt_roidb['person_action'][ki].reshape((1,-1)) for ki in keep_index])
            gt_roidb['bbox'] = gt_roidb['bbox'][keep_index,:]
            gt_roidb['area'] = gt_roidb['area'][keep_index]
            gt_roidb['difficult'] = [gt_roidb['difficult'][ki] for ki in keep_index]

            gt_roidb.pop('person_action')

        # image
        image = imread(self.image_path_from_index(index))
        # image original size
        gt_roidb['info'] = (image.shape[0], image.shape[1], image.shape[2])
        gt_roidb['id'] = k
        
        # [img, groundtruth]
        gt_roidb.update({'file_id': index})
        yield [image, gt_roidb]

      if len(self._aug_dataset) > 0:
        aug_indxs = np.arange(len(self._aug_dataset))
        if self.rng:
          self.rng.shuffle(aug_indxs)

        for k in aug_indxs:
          image_path = self._aug_dataset[k]
          image_seg_path = self._aug_seg[k]

          image = imread(image_path)
          seg_img = imread(image_seg_path)
          if len(seg_img.shape) == 3:
            seg_img = seg_img[:,:,0]

          seg_img[np.where(seg_img == 255)] = 0

          yield image, {'segmentation_map': seg_img,
                        'id': k+len(self._image_index),
                        'info': (image.shape[0], image.shape[1], image.shape[2])}

  
  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    index = self._image_index[id]
    gt_roidb = self._load_roidb(index)

    task_type = getattr(self, 'task_type', None)
    if task_type is not None:
      if task_type == 'ACTION_CLASSIFICATION':
        keep_index = []
        for obj_index, obj_name in enumerate(gt_roidb['category']):
          if obj_name == 'person':
            keep_index.append(obj_index)

        if len(keep_index) == 0:
          return [None, None]

        keep_index = [ki for ki in keep_index if gt_roidb['person_action'][ki] is not None]
        if len(keep_index) == 0:
          return [None, None]

        gt_roidb['category_id'] = np.concatenate([gt_roidb['person_action'][ki].reshape((1, -1)) for ki in keep_index])
        gt_roidb['bbox'] = gt_roidb['bbox'][keep_index, :]
        gt_roidb['area'] = gt_roidb['area'][keep_index]
        gt_roidb['difficult'] = [gt_roidb['difficult'][ki] for ki in keep_index]

        gt_roidb.pop('person_action')

    image = imread(self.image_path_from_index(index))
    gt_roidb['info'] = (image.shape[0], image.shape[1], image.shape[2])
    gt_roidb['id'] = id

    # [img,groundtruth]
    return [image, gt_roidb]
  
  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
    return image_path
    
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def _load_roidb(self, index):
    return self._load_pascal_annotation(index)
    
  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')

    task_type = getattr(self, 'task_type', None)
    if task_type is not None:
      if task_type == 'ACTION_CLASSIFICATION':
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Action',
                                  self._image_set + '.txt')
      elif task_type == 'SEGMENTATION':
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Segmentation',
                                  self._image_set + '.txt')

    assert os.path.exists(image_set_file), \
            'Path Does not Exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index
    
  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')

    # Exclude the samples labeled as difficult
    # non_diff_objs = [
    #     obj for obj in objs if int(obj.find('difficult').text) == 0]
    # if len(non_diff_objs) != len(objs):
    #     print 'Removed {} difficult objects'.format(
    #         len(objs) - len(non_diff_objs))
    # objs = non_diff_objs

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    category_id = np.zeros((num_objs), dtype=np.int32)
    # "Seg" area for pascal is just the box area
    area = np.zeros((num_objs), dtype=np.float32)
    category = []
    difficult = []

    segmented = tree.find('segmented')
    has_seg = 0
    if segmented is not None:
      has_seg = int(segmented.text)

    seg_img = None
    if has_seg:
      seg_file = os.path.join(self._data_path, 'SegmentationClass', index + '.png')
      seg_img = imread(seg_file)

    # Load object bb and segmentation into a data frame.
    segmentation = []
    segmentation_map = None
    person_action = []
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      category_id[ix] = cls
      area[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
      category.append(obj.find('name').text.lower().strip())
      difficult.append(int(obj.find('difficult').text))

      if has_seg and seg_img is not None:
        obj_seg = np.zeros((seg_img.shape[0], seg_img.shape[1]), np.uint8)
        seg_img_int = seg_img.astype(np.int32)
        seg_img_index = seg_img_int[:,:,0]*255*255 + seg_img_int[:,:,1]*255 + seg_img_int[:,:,2]

        obj_seg[np.where(seg_img_index == self._c_2_rgb_index[cls])] = cls
        segmentation.append(obj_seg)
        if segmentation_map is None:
          segmentation_map = np.zeros((seg_img.shape[0], seg_img.shape[1]), np.uint8)

        segmentation_map[np.where(seg_img_index == self._c_2_rgb_index[cls])] = cls

      if category[-1] == 'person':
        action_cls = obj.find('actions')
        if action_cls is not None:
          action_label = np.zeros((11), np.int32)
          for action_cc in action_cls.getchildren():
            action_label[self._action_classes[action_cc.tag]] = int(action_cc.text)

          person_action.append(action_label)
        else:
          person_action.append(None)
      else:
        person_action.append(None)

    annotation = {'bbox': boxes,
                  'category_id': category_id,
                  'category': category,
                  'flipped': False,
                  'difficult': difficult,
                  'area': area,
                  'person_action': person_action}

    if has_seg:
      annotation.update({'segmentation': segmentation, 'segmentation_map': segmentation_map})

    return annotation


class Pascal2007(PascalBase):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(Pascal2007,self).__init__('2007', train_or_test, dir, ext_params)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    validation_pascal2007 = Pascal2007('val', self.dir)

    return self, validation_pascal2007


class Pascal2012(PascalBase):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(Pascal2012,self).__init__('2012', train_or_test, dir, ext_params)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    validation_pascal2012 = Pascal2012('val', self.dir)

    return self, validation_pascal2012