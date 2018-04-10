# -*- coding: UTF-8 -*-
# @Time    : 18-3-14
# @File    : ali_fashionai_landmark.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np

__all__ = ['FashionAILandmark']
class FashionAILandmark(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(FashionAILandmark, self).__init__(train_or_test, dir)
    
    assert(train_or_test in ['train', 'test'])
    if train_or_test == 'train':
      if not (os.path.exists(os.path.join(self.dir,'train','Images')) and
                os.path.exists(os.path.join(self.dir, 'train','Annotations'))):
        logger.error('FashionAILandmark train dataset must download from \n https://tianchi.aliyun.com/competition/information.htm?raceId=231648')
        sys.exit(0)
    else:
      if not (os.path.exists(os.path.join(self.dir, 'test','Images')) and
                os.path.exists(os.path.join(self.dir, 'test','test.csv'))):
        logger.error('FashionAILandmark test dataset must download from \n https://tianchi.aliyun.com/competition/information.htm?raceId=231648')
        sys.exit(0)
    
    self.key_points = ['neckline_left',     #左领部
                      'neckline_right',     #右领部
                      'center_front',       #中线
                      'shoulder_left',      #左肩部
                      'shoulder_right',     #右肩部
                      'armpit_left',        #左腋窝
                      'armpit_right',       #右腋窝
                      'waistline_left',     #左腰部
                      'waistline_right',    #右腰部
                      'cuff_left_in',       #左袖口内
                      'cuff_left_out',      #左袖口外
                      'cuff_right_in',      #右袖口内
                      'cuff_right_out',     #右袖口外
                      'top_hem_left',       #左衣摆
                      'top_hem_right',      #右衣摆
                      'waistband_left',     #左腰部
                      'waistband_right',    #右腰部
                      'hemline_left',       #左裙摆
                      'hemline_right',      #右裙摆
                      'crotch',             #裆部
                      'bottom_left_in',     #左裤脚内
                      'bottom_left_out',    #左裤脚外
                      'bottom_right_in',    #右裤脚内
                      'bottom_right_out',   #右裤脚外
                      ]

    self.category_landmark = {'blouse': {
                                0:0,
                                1:1,
                                2:2,
                                3:3,
                                4:4,
                                5:5,
                                6:6,
                                9:7,
                                10:8,
                                11:9,
                                12:10,
                                13:11,
                                14:12},
                              'dress': {
                                0:0,
                                1:1,
                                2:2,
                                3:3,
                                4:4,
                                5:5,
                                6:6,
                                7:7,
                                8:8,
                                9:9,
                                10:10,
                                11:11,
                                12:12,
                                17:13,
                                18:14
                              },
                              'outwear': {
                                0:0,
                                1:1,
                                3:2,
                                4:3,
                                5:4,
                                6:5,
                                7:6,
                                8:7,
                                9:8,
                                10:9,
                                11:10,
                                12:11,
                                13:12,
                                14:13
                              },
                              'skirt': {
                                15:0,
                                16:1,
                                17:2,
                                18:3,
                              },
                              'trousers': {
                                15:0,
                                16:1,
                                19:2,
                                20:3,
                                21:4,
                                22:5,
                                23:6
                              }}

    self.category_map = {
      'blouse': 0,
      'dress': 1,
      'outwear': 2,
      'skirt': 3,
      'trousers': 4,
    }
    
    self.annotation = []
    self.images = []
    if train_or_test == 'train':
      with open(os.path.join(self.dir, 'train', 'Annotations', 'train.csv')) as fp:
        # skip first row
        content = fp.readline()
        content = fp.readline()
        while content:
          key_terms = content.split(',')
          image_id = key_terms[0]
          category = key_terms[1]
          # record image id
          self.images.append((image_id, category))

          # record annotation
          sample_annotation = {}
          sample_annotation['category'] = category
          sample_annotation['category_id'] = self.category_map[category]
          sample_annotation['landmark'] = []
          sample_annotation['id'] = len(self.images)
          key_point_annotation = key_terms[2:]
          for kp_index, kp in enumerate(key_point_annotation):
            x, y, visible = kp.split('_')
            x = int(x)
            y = int(y)
            visible = int(visible)

            if kp_index in self.category_landmark[category]:
              sample_annotation['landmark'].append((self.category_landmark[category][kp_index], kp_index, x, y, visible))

          self.annotation.append(sample_annotation)

          # read next line
          content = fp.readline()

      with open(os.path.join(self.dir, 'train', 'Annotations', 'annotations.csv')) as fp:
        # skip first row
        content = fp.readline()
        content = fp.readline()
        while content:
          key_terms = content.split(',')
          image_id = key_terms[0]
          category = key_terms[1]
          # record image id
          self.images.append((image_id, category))
    
          # record annotation
          sample_annotation = {}
          sample_annotation['category'] = category
          sample_annotation['category_id'] = self.category_map[category]
          sample_annotation['landmark'] = []
          sample_annotation['id'] = len(self.images)
          key_point_annotation = key_terms[2:]
          for kp_index, kp in enumerate(key_point_annotation):
            x, y, visible = kp.split('_')
            x = int(x)
            y = int(y)
            visible = int(visible)
      
            if kp_index in self.category_landmark[category]:
              sample_annotation['landmark'].append(
                (self.category_landmark[category][kp_index], kp_index, x, y, visible))
    
          self.annotation.append(sample_annotation)
    
          # read next line
          content = fp.readline()
    else:
      with open(os.path.join(self.dir, 'test', 'test.csv')) as fp:
        # skip first row
        content = fp.readline()
        content = fp.readline()
        content = content.replace('\n','')
        while content:
          key_terms = content.split(',')
          image_id = key_terms[0]
          category = key_terms[1]
          # record image id
          self.images.append((image_id, category))

          # read next line
          content = fp.readline()
          content = content.replace('\n','')
    
    # data index list
    self.ids = list(range(len(self.images)))
    # fixed seed
    self.seed = 0
    
  @property
  def size(self):
    return len(self.ids)
  
  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1
    
      idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(idxs)
    
      for k in idxs:
        image_file, category = self.images[k]
        image_path = os.path.join(self.dir, self.train_or_test, image_file)
        image = imread(image_path)
        
        if self.train_or_test == 'train':
          data_annotation = copy.deepcopy(self.annotation[k])
          data_annotation['info'] = [image.shape[0], image.shape[1], image.shape[2]]
          yield [(image, category, image_file), data_annotation]
        else:
          yield [(image, category, image_file), None]

  def at(self, id):
    image_file, category_id = self.images[id]
    image_path = os.path.join(self.dir, self.train_or_test, image_file)
    image = imread(image_path)
    return (image, category_id, image_file)

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert (split_method in ['repeated-holdout', 'bootstrap', 'kfold', 'holdout'])

    # set fixed random seed
    np.random.seed(np.int64(self.seed))
    category_ids = None
    if split_method == 'kfold':
      category_ids = [i for i in range(len(self.ids))]
      np.random.shuffle(category_ids)
    else:
      category_ids = [0 for _ in range(len(self.ids))]

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = FashionAILandmark(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = FashionAILandmark(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids

    return train_dataset, val_dataset  # split data by their label
