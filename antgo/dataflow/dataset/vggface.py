# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : vggface.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np

VGGFACE_URL = 'http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz'
class VGGFace(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(VGGFace, self).__init__(train_or_test, dir)
    assert(train_or_test in ['train', 'sample'])
    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    # 0.step maybe download
    if not os.path.exists(os.path.join(self.dir, 'vgg_face_dataset.tar.gz')):
      self.download(self.dir, file_names=[], default_url=VGGFACE_URL, auto_untar=True, is_gz=True)
    
    if not os.path.exists(os.path.join(self.dir, 'data')) or len(os.listdir(os.path.join(self.dir, 'data'))) < 500:
      if not os.path.exists(os.path.join(self.dir, 'data')):
        os.makedirs(os.path.join(self.dir, 'data'))
      # download all images from web
      for file in os.listdir(os.path.join(self.dir, 'vgg_face_dataset', 'files')):
        if file[0] == '.':
          continue
        
        with open(os.path.join(self.dir, 'vgg_face_dataset', 'files', file), 'r') as fp:
          content = fp.readline()
          while content:
            image_id, image_url, x_1, y_1, x_2, y_2, pose, _1, _2 =\
              [i for i in content.replace('\n', '').split(' ') if i != '']
            
            # download every image as image_id.jpg
            name = os.path.normpath(image_url).split('/')[-1]
            if not os.path.exists(os.path.join(self.dir, 'data', name)):
              try:
                download(image_url, os.path.join(self.dir, 'data'))
              except:
                logger.error('%s couldnt be downloaded'%image_url)
            else:
              logger.info('%s has been existed'%name)
            
            content = fp.readline()
    
    # 1.step data files
    self._persons_file = []
    self._persons_annotation = []
    self._persons_id_str = []
    self._persons_id = []
    for file in os.listdir(os.path.join(self.dir, 'vgg_face_dataset', 'files')):
      if file[0] == '.':
        continue
  
      with open(os.path.join(self.dir, 'vgg_face_dataset', 'files', file), 'r') as fp:
        content = fp.readline()
        while content:
          image_id, _url, x_1, y_1, x_2, y_2, pose, _1, _2 =\
            [i for i in content.replace('\n', '').split(' ') if i != '']
          image_name = os.path.normpath(_url).split('/')[-1]
          if os.path.exists(os.path.join(self.dir, 'data', image_name)):
            self._persons_file.append(os.path.join(self.dir, 'data', image_name))
            bbox = np.zeros((1, 4))
            bbox[0, 0] = float(x_1)
            bbox[0, 1] = float(y_1)
            bbox[0, 2] = float(x_2)
            bbox[0, 3] = float(y_2)
            self._persons_annotation.append({'bbox': bbox, 'pose': float(pose), 'category': file.split('.')[0]})
            self._persons_id_str.append(file)
          
          content = fp.readline()

    id_set = set(self._persons_id_str)
    person_id_map = {}
    for s_i, s in enumerate(id_set):
      person_id_map[s] = s_i

    for person_id_str in self._persons_id_str:
      person_id = person_id_map[person_id_str]
      self._persons_id.append(person_id)

    self.ids = list(range(len(self._persons_file)))

    # fixed seed
    self.seed = time.time()
  
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
        person_file = self._persons_file[k]
        person_image = imread(person_file)
        person_annotation = self._persons_annotation[k]
        person_id = self._persons_id[k]
        person_annotation.update({'category_id': person_id,
                                  'id': k,
                                  'info': [person_image.shape[0], person_image.shape[1], person_image.shape[2]]})
        yield person_image, person_annotation
  
  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    person_file = self._persons_file[id]
    person_image = imread(person_file)
    person_id = self._persons_id[id]
    person_annotation = self._persons_annotation[id]
    person_annotation.update({'category_id': person_id,
                              'id': id,
                              'info': [person_image.shape[0], person_image.shape[1], person_image.shape[2]]})
    return person_image, person_annotation

  def split(self, split_params={}, split_method=''):
    assert (self.train_or_test == 'train')
    assert (split_method in ['repeated-holdout', 'bootstrap', 'kfold'])
  
    category_ids = None
    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      category_ids = [i for i in range(len(self.ids))]
      np.random.shuffle(category_ids)
    else:
      category_ids = [i for i in range(len(self.ids))]
  
    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = VGGFace(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids
  
    val_dataset = VGGFace(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids
  
    return train_dataset, val_dataset