# -*- coding: UTF-8 -*-
# @Time    : 18-3-26
# @File    : fashionai_attribute.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import sys
import numpy as np
from antgo.dataflow.dataset import *

__all__ = ['FashionAIAttribute']
class FashionAIAttribute(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(FashionAIAttribute, self).__init__(train_or_test, dir, params)
    assert(train_or_test in ['train', 'test'])
    self.train_or_test = train_or_test
    
    if self.train_or_test == 'train':
      if not os.path.exists(os.path.join(self.dir, 'base')) or not os.path.exists(os.path.join(self.dir, 'web')):
        logger.error('FashionAIAttribute must be download from \n https://tianchi.aliyun.com/competition/information.htm?spm=5176.100071.5678.2.778c2b9eBime0R&raceId=231649')
        sys.exit(0)
    else:
      if not os.path.exists(os.path.join(self.dir , 'rank')):
        logger.error('FashionAIAttribute must be download from \n https://tianchi.aliyun.com/competition/information.htm?spm=5176.100071.5678.2.778c2b9eBime0R&raceId=231649')
        sys.exit(0)
    
    self.images = []
    self.cloth_attribs = {
      'skirt_length_labels': ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length'],
      'coat_length_labels': ['Invisible', 'High Waist Length', 'Regular Length', 'Long Length', 'Micro Length', 'Knee Length', 'Midi Length', 'Ankle&Floor Length'],
      'collar_design_labels': ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar'],
      'lapel_design_labels': ['Invisible', 'Notched', 'Collarless', 'Shawl Collar', 'Plus Size Shawl'],
      'neck_design_labels': ['Invisible', 'Turtle Neck', 'Ruffle Semi-High Collar', 'Low Turtle Neck', 'Draped Collar'],
      'neckline_design_labels': ['Invisible', 'Strapless Neck', 'Deep V Neckline', 'Straight Neck', 'V Neckline', 'Square Neckline', 'Off Shoulder', 'Round Neckline', 'Sweat Heart Neck', 'OneShoulder Neckline'],
      'pant_length_labels': ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length', 'Cropped Pant', 'Full Length'],
      'sleeve_length_labels': ['Invisible', 'Sleeveless', 'Cup Sleeves', 'Short Sleeves', 'Elbow Sleeves', '3/4 Sleeves','Wrist Length','Long Sleeves','Extra Long Sleeves']
    }
    
    self.cloth_attribs_index = {
      'skirt_length_labels': 0,
      'coat_length_labels': 1,
      'collar_design_labels': 2,
      'lapel_design_labels': 3,
      'neck_design_labels': 4,
      'neckline_design_labels': 5,
      'pant_length_labels': 6,
      'sleeve_length_labels': 7,
    }
    
    if self.train_or_test == 'train':
      # 1.step parse from base folder
      with open(os.path.join(self.dir, 'base',  'Annotations', 'label.csv')) as fp:
        content = fp.readline()
        while content:
          image_file, cloth_attrib, cloth_attrib_value = content.split(',')
          cloth_attrib_value = cloth_attrib_value.replace('\n','')
          assert(len(self.cloth_attribs[cloth_attrib]) == len(cloth_attrib_value))
          cloth_attrib_label = np.zeros((len(cloth_attrib_value)), dtype=np.int32)
          for i in range(len(cloth_attrib_value)):
            if cloth_attrib_value[i] == 'y':
              cloth_attrib_label[i] = 1
            elif cloth_attrib_value[i] == 'n':
              cloth_attrib_label[i] = 0
            else:
              cloth_attrib_label[i] = -1
              
          self.images.append((image_file, cloth_attrib, cloth_attrib_label, 'base'))
          content = fp.readline()
      # 2.step parse from web folder
      with open(os.path.join(self.dir, 'web', 'Annotations', 'skirt_length_labels.csv')) as fp:
        content = fp.readline()
        while content:
          image_file, cloth_attrib, cloth_attrib_value = content.split(',')
          cloth_attrib_value = cloth_attrib_value.replace('\n', '')
          assert (len(self.cloth_attribs[cloth_attrib]) == len(cloth_attrib_value))
          cloth_attrib_label = np.zeros((len(cloth_attrib_value)), dtype=np.int32)
          for i in range(len(cloth_attrib_value)):
            if cloth_attrib_value[i] == 'y':
              cloth_attrib_label[i] = 1
            elif cloth_attrib_value[i] == 'n':
              cloth_attrib_label[i] = 0
            else:
              cloth_attrib_label[i] = -1
    
          self.images.append((image_file, cloth_attrib, cloth_attrib_label, 'web'))
          content = fp.readline()
    else:
      with open(os.path.join(self.dir, 'rank', 'Tests', 'question.csv')) as fp:
        content = fp.readline()
        while content:
          image_file, cloth_attrib, _ = content.split(',')
          self.images.append((image_file, cloth_attrib, None, 'rank'))
          content = fp.readline()
    
    # data index list
    self.ids = list(range(len(self.images)))
    # fixed seed
    self.seed = getattr(self, 'seed', 0)
  
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
        image_file, cloth_attrib, cloth_attrib_label, sub_folder = self.images[k]
        image_path = os.path.join(self.dir, sub_folder, image_file)
        image = imread(image_path)
      
        if self.train_or_test == 'train':
          yield [(image, self.cloth_attribs_index[cloth_attrib], image_file),
                 {'category_id': cloth_attrib_label, 'category': cloth_attrib, 'id': k}]
        else:
          yield [(image, self.cloth_attribs_index[cloth_attrib], image_file), None]
    
  def at(self, id):
    image_file, cloth_attrib, cloth_attrib_label, sub_folder = self.images[id]
    image_path = os.path.join(self.dir, sub_folder, image_file)
    image = imread(image_path)
    
    return (image, cloth_attrib, image_file)
  
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
    train_dataset = FashionAIAttribute(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = FashionAIAttribute(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids

    return train_dataset, val_dataset  # split data by their label
