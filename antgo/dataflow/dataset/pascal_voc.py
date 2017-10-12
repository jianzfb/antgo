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
from ...utils.fs import download
from ...utils.fs import maybe_here_match_format
from ...utils import logger, get_rng
from .dataset import *

__all__ = ['Pascal2007', 'Pascal2012']

PASCAL2007_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007"
PASCAL2012_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012"

class PascalBase(Dataset):
  def __init__(self, year, image_set, dir=None, ext_params=None):
    super(PascalBase, self).__init__(image_set, dir, ext_params)
    self._year = year
    self._image_set = image_set
    self._devkit_path = dir

    if self._year == '2007':
      self.download(self.dir, ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar'],default_url=PASCAL2007_URL)
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
      self.download(self.dir, ['VOCtrainval_11-May-2012.tar'], default_url=PASCAL2012_URL)
      maybe_data_path = maybe_here_match_format(self._devkit_path, 'VOC' + self._year)
      if maybe_data_path is None:
        # auto untar
        tar = tarfile.open(os.path.join(self.dir, 'VOCtrainval_11-May-2012.tar'), 'r')
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
    self.dataset_size = len(self._image_index)

  @property
  def size(self):
    return self.dataset_size

  def data_pool(self):
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
        # annotation
        gt_roidb = self._load_roidb(index)

        # label info
        gt_roidb = self.filter_by_condition(gt_roidb, ['segmentation'])
        if gt_roidb is None:
          continue

        # image
        image = imread(self.image_path_from_index(index))
        # image original size
        gt_roidb['info'] = (image.shape[0], image.shape[1], image.shape[2])
        gt_roidb['id'] = k

        # [img, groundtruth]
        yield [image, gt_roidb]
  
  def at(self, id):
    index = self._image_index[id]
    gt_roidb = self._load_roidb(index)
  
    # label info
    gt_roidb = self.filter_by_condition(gt_roidb, ['segmentation'])
    if gt_roidb is None:
      return [None, None]
  
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
    non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
    # if len(non_diff_objs) != len(objs):
    #     print 'Removed {} difficult objects'.format(
    #         len(objs) - len(non_diff_objs))
    objs = non_diff_objs

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
        obj_seg[np.where(seg_img[:, :, 0] == cls)] = 255
        segmentation.append(obj_seg)

    annotation = {'bbox': boxes,
                  'category_id': category_id,
                  'category': category,
                  'flipped': False,
                  'difficult': difficult,
                  'area': area}

    if has_seg:
      annotation.update({'segmentation': segmentation})

    return annotation


class Pascal2007(PascalBase):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(Pascal2007,self).__init__('2007', train_or_test, dir,ext_params)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    validation_pascal2007 = Pascal2007('val', self.dir)

    return self, validation_pascal2007


class Pascal2012(PascalBase):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(Pascal2012,self).__init__('2012', train_or_test, dir,ext_params)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    validation_pascal2012 = Pascal2012('val', self.dir)

    return self, validation_pascal2012