# -*- coding: UTF-8 -*-
# @Time    : 17-8-2
# @File: dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import os
import random
import numpy as np
from antgo.utils import get_rng
from antgo.dataflow.core import *
from antgo.utils.fs import *
from functools import reduce
import re
import multiprocessing
import tarfile
import zipfile
from antgo import config
import subprocess
from antgo.utils.serialize import *
import copy
import cv2
try:
    import queue
except:
    import Queue as queue
import functools

Config = config.AntConfig


def imread(file):
  return cv2.imread(file)


def imwrite(file, img):
  cv2.imwrite(file, img)


def imresize(image, size):
  return cv2.resize(image, (size[1],size[0]))


class Dataset(BaseNode):
  _BASE_DATASET = True

  def __init__(self, train_or_test="train", dir=None, ext_params=None, name=None):
    super(Dataset, self).__init__(name)
    assert(train_or_test in ["train", "val", "test", "sample", 'unlabeled'])

    # for random episodes
    self.nb_samples = 0
    self.nb_samples_per_class = 0

    # basic info
    self.train_or_test = train_or_test
    self.dir = dir
    
    self._dataset_type = 'IMAGE'
    # config extent params
    # included, excluded, transform(cls->id)
    self.ext_params = ext_params
    if ext_params is not None:
      for k, v in ext_params.items():
        if k != 'self':
          setattr(self, k, v)

    self.data_rng = None

    self.epochs = None
    self._epoch = 0
    self.data_generator = None
    self._lock = multiprocessing.Lock()
    self._ids = []
    # data rng flag
    self._is_data_rng = False

    self._candidate_file = None
    self._unlabeled_list_file = None
    if self.dir is not None:
      self._candidate_file = os.path.join(self.dir, 'candidates.txt')
      self._unlabeled_list_file = os.path.join(self.dir, 'unlabeled_list.txt')

  def init(self, *args, **kwargs):
    pass
  
  def close(self):
    pass

  @property
  def unlabeled_tag(self):
    return 'unlabeled'

  @property
  def unlabeled_folder(self):
    return os.path.join(self.dir, self.unlabeled_tag)

  @property
  def candidate_folder(self):
    return os.path.join(self.dir, 'candidates')

  @property
  def candidate_file(self):
    return self._candidate_file
  @candidate_file.setter
  def candidate_file(self, val):
    self._candidate_file = val

  @property
  def unlabeled_list_file(self):
    return self._unlabeled_list_file
  @unlabeled_list_file.setter
  def unlabeled_list_file(self, val):
    self._unlabeled_list_file = val

  def unlabeled_size(self):
    unlabeled_folder = os.path.join(self.dir, self.unlabeled_tag)
    if not os.path.exists(unlabeled_folder):
      return 0

    if not os.path.exists(self.unlabeled_list_file):
      # 构建无标注数据文件列表
      unorder_list = []
      for file in os.listdir(unlabeled_folder):
        if file[0] == '.':
          continue
        unorder_list.append(file)
      
      order_list = sorted(unorder_list)
      with open(self.unlabeled_list_file, 'w') as fp:
        for file in order_list:
          fp.write('%s/%s,%d\n' % (self.unlabeled_tag, file, 0))

    has_labeled_list = []
    if os.path.exists(self.candidate_file):
      with open(self.candidate_file, 'r') as fp:
        line_content = fp.readline()
        while line_content:
          _, data_file, _ = line_content.split(',')
          has_labeled_list.append(data_file.split('/')[-1])
          line_content = fp.readline()
    else:
      # build empty candidates file
      with open(self.candidate_file, 'w') as fp:
        pass
    
    unlabeled_num = 0
    with open(self.unlabeled_list_file, 'r') as fp:
      line_content = fp.readline()
      while line_content:
        file_name, _ = line_content.split(',')
        if file_name.split('/')[-1] not in has_labeled_list:
          unlabeled_num += 1

        line_content = fp.readline()

    return unlabeled_num

  def unlabeled(self, tag=''):
    unlabeled_folder = os.path.join(self.dir, self.unlabeled_tag)
    if not os.path.exists(unlabeled_folder):
      return None

    if not os.path.exists(self.unlabeled_list_file):
      # 构建无标注数据文件列表
      unorder_list = []
      for file in os.listdir(unlabeled_folder):
        if file[0] == '.':
          continue
        unorder_list.append(file)
      
      order_list = sorted(unorder_list)
      with open(self.unlabeled_list_file, 'w') as fp:
        for file in order_list:
          fp.write('%s/%s,%d\n' % (self.unlabeled_tag, file, 0))

    has_labeled_list = []
    if os.path.exists(self.candidate_file):
      with open(self.candidate_file, 'r') as fp:
        line_content = fp.readline()
        while line_content:
          _, data_file, _ = line_content.split(',')
          has_labeled_list.append(data_file.split('/')[-1])
          line_content = fp.readline()
    else:
      # build empty candidates file
      with open(self.candidate_file, 'w') as fp:
        pass

    unlabeled_list = []
    unlabeled_index = 0
    with open(self.unlabeled_list_file, 'r') as fp:
      line_content = fp.readline()
      while line_content:
        file_name, _ = line_content.split(',')
        unlabeled_list.append((unlabeled_index,file_name))

        line_content = fp.readline()
        unlabeled_index += 1
    
    unlabeled_data = []
    for index, file in unlabeled_list:
      if file.split('/')[-1] not in has_labeled_list:
        unlabeled_data.append({'id': index, 'file_id': file})

    return unlabeled_data

  def make_candidate(self, unlabeled_id, unlabeled_file, label_file, status=''):
    # status: 'SKIP/OK'
    if not os.path.exists(os.path.join(self.dir, 'candidates')):
      os.makedirs(os.path.join(self.dir, 'candidates'))
      os.makedirs(os.path.join(self.dir, 'candidates', 'data'))
      os.makedirs(os.path.join(self.dir, 'candidates', 'label'))

    with open(self.candidate_file, 'a') as fp:
      if status == 'OK':
        # copy data file to candidates/data
        shutil.copy(os.path.join(self.dir, unlabeled_file), os.path.join(self.dir, 'candidates', 'data'))

        # copy label file to candidates/label
        shutil.copy(label_file, os.path.join(self.dir, 'candidates', 'label'))

        # write to canndidates.txt
        fp.write('%d,%s,%s\n'%(unlabeled_id, 'candidates/data/%s'%unlabeled_file.split('/')[-1], 'candidates/label/%s'%label_file.split('/')[-1]))

  def check_candidate(self, unlabeled_files, finished_label_folder):
    # 检查准备进入候选列表的标注数据，与等待的未标注数据一致
    consistent_sample_num = 0
    logger.info("Finshed label folder %s"%finished_label_folder)

    for sample_file, _ in unlabeled_files:
      if os.path.exists(os.path.join(finished_label_folder, sample_file)):
        consistent_sample_num += 1

    logger.info('Consistent unlabeled file and labeled file %d'%consistent_sample_num)
    if consistent_sample_num != len(unlabeled_files):
      return False

    return True

  def candidates(self, candidate_type='IMAGE'):
    '''

    :return:
    '''
    if not os.path.exists(self.candidate_file):
      raise StopIteration

    with open(self.candidate_file, 'r') as fp:
      line_content = fp.readline()
      while line_content:
        _, data_file, label_file = line_content.split(',')
        data_file = data_file.strip()
        label_file = label_file.strip()
        if candidate_type == 'IMAGE':
          yield imread(os.path.join(self.dir, data_file)), \
                imread(os.path.join(self.dir, label_file))
        else:
          yield os.path.join(self.dir, data_file), \
                os.path.join(self.dir, label_file)

        line_content = fp.readline()

    raise StopIteration

  def candidates_size(self):
    if not os.path.exists(self.candidate_file):
      return 0

    count = 0
    with open(self.candidate_file, 'r') as fp:
      line_content = fp.readline()
      while line_content:
        count += 1
        line_content = fp.readline()

    return count

  def waiting_process_num(self):
    return 0

  def finish_process_num(self):
    return 0

  @property
  def ids(self):
    return self._ids
  @ids.setter
  def ids(self, val):
    self._ids = val

  @property
  def epochs(self):
    return self._epochs
  @epochs.setter
  def epochs(self, val):
    self._epochs = val
    
  @property
  def epoch(self):
    return self._epoch
  @epoch.setter
  def epoch(self, val):
    self._epoch = val
  
  @property
  def dataset_type(self):
    return self._dataset_type
  @dataset_type.setter
  def dataset_type(self, val):
    self._dataset_type = val
  
  @property
  def data_generator(self):
    return self._data_generator
  @data_generator.setter
  def data_generator(self, val):
    self._data_generator = val

  def _reset_iteration_state(self):
    self.data_generator = None
  
  def set_value(self, new_value):
    pass
  
  def get_value(self):
    try:
      if DIRTY == self._value:
        if self.data_generator is None:
          self.data_generator = self.data_pool()
        self._value = next(self.data_generator)
        #self._set_dependents_dirty()

      return self._value
    except:
      raise StopIteration
  
  def _force_inputs_dirty(self):
    self._value = DIRTY
  
  value = property(get_value, set_value)

  def iterator_value(self):
    while True:
      self._value = self.get_value()
      yield self._value
      self._force_inputs_dirty()

  def __iter__(self):
    return self.iterator_value()

  def __len__(self):
    return self.size

  def __getitem__(self, index):
    return self.at(index)

  def at(self, id):
    raise NotImplementedError

  def get_ann_info(self, idx=None):
    raise NotImplementedError

  def get_cat_ids(self, idx):
    raise NotImplementedError
  
  def sample(self, id):
    image, annotation = self.at(id)
    if image is None:
      # None 跳过
      return None
    
    if not isinstance(annotation, dict):
      annotation = {'label': annotation}

    if image is not None:
      annotation['image'] = image
    return annotation

  def evaluate(self, preds, **kwargs):
    raise NotImplementedError

  @property
  def size(self):
    raise NotImplementedError

  @property
  def rng(self):
    if self._is_data_rng:
      self.data_rng = get_rng(self)
    return self.data_rng

  def reset_state(self):
    """
    Reset state of the dataflow. Will always be called before consuming data points.
    for example, RNG **HAS** to be reset here if used in the DataFlow.
    Otherwise it may not work well with prefetching, because different
    processes will have the same RNG state.
    """
    self._is_data_rng = True

  def split(self, split_params, split_method):
    pass

  def _split(self, idx=[], split_params={}, split_method='holdout'):
    '''
    
    :param idx:
    :param split_params:
    :param split_method:
    :return:
    '''
    if split_method == 'holdout':
      if 'ratio' not in split_params:
        split_params['ratio'] = None
      if 'is_stratified' not in split_params:
        split_params['is_stratified'] = True

      return self._split_holdout(split_params['ratio'], split_params['is_stratified'], idx)
    elif split_method == 'repeated-holdout':
      return self._split_repeated_holdout(split_params['ratio'], split_params['is_stratified'], idx)
    elif split_method == 'bootstrap':
      return self._split_bootstrap(idx)
    elif split_method == 'kfold':
      return self._split_kfold(split_params['kfold'], split_params['k'], idx)

    return None

  def _split_custom_holdout(self, split_ratio, is_stratified_sampling,idx):
    # implemented at child class
    # train/val
    raise NotImplementedError()

  def _split_holdout(self, split_ratio, is_stratified_sampling=True, idx=[]):
    assert(self.train_or_test in ['train','sample'])
    try:
      train_idx, val_idx = self._split_custom_holdout(split_ratio, is_stratified_sampling, idx)
      return train_idx, val_idx
    except:
      return self._split_repeated_holdout(split_ratio, is_stratified_sampling, idx)

  def _split_repeated_holdout(self, split_ratio, is_stratified_sampling=True, idx=[]):
    assert(self.train_or_test in ['train','sample'])

    # split by t/v
    if is_stratified_sampling:
      labels_num = len(set(idx))
      labels_index = [[] for _ in range(labels_num)]
      for index, label in enumerate(idx):
        labels_index[label].append(index)

      train_idx = [[] for _ in range(labels_num)]
      validation_idx = [[] for _ in range(labels_num)]
      for label in range(labels_num):
        label_samples = labels_index[label]
        get_rng().shuffle(label_samples)

        top_k = int(split_ratio * len(label_samples))
        train_idx[label].extend(label_samples[0:top_k])
        validation_idx[label].extend((label_samples[top_k:]))

      train_idx = reduce(lambda x, y: x+y, train_idx)
      validation_idx = reduce(lambda x, y: x+y, validation_idx)

      return train_idx, validation_idx
    else:
      index = range(len(idx))
      get_rng().shuffle(index)
      top_k = int(split_ratio * len(idx))
      train_idx = index[0:top_k]
      validation_idx = (index[top_k:])
      return train_idx, validation_idx

  def _split_bootstrap(self, idx):
    assert(self.train_or_test in ['train','sample'])

    is_ok = False
    train_idx = []
    validation_idx = []
    while not is_ok:
      selected_idx = get_rng().randint(low=0, high=len(idx)-1, size=len(idx)).tolist()
      train_idx = selected_idx
      validation_idx = [i for i in range(len(idx)) if i not in selected_idx]

      if len(validation_idx) > 0:
        is_ok = True

    return train_idx, validation_idx

  def _split_kfold(self, k_fold, k, idx):
    assert(self.train_or_test in ['train','sample'])
    assert(k < k_fold)

    size = len(idx)
    fold_size = int(float(size) / float(k_fold))
    k_start = k * fold_size
    k_end = (k+1) * fold_size if k < k_fold - 1 else len(idx)

    validation_idx = [idx[i] for i in np.arange(k_start, k_end, dtype=np.uint64)]
    train_idx = [i for i in range(len(idx)) if i not in validation_idx]

    return train_idx, validation_idx

  def data_pool(self):
    '''
    
    :return:
    '''
    raise NotImplementedError

  def make_data(self, data, label=None, support=None,transform_func=None):
    # data: [[],[],[],...]
    # label:[0,1,2,...]
    # support: [(0,[]),[1,[]],...]
    # return: [[label_index,label,data]]
    if label == None:
      label = range(len(data))

    if support == None:
      samples = [(i_index, i_val, d) for i_index, i_val in enumerate(label) for d in data[i_index]]
      if self.rng != None:
        self.rng.shuffle(samples)
      return samples
    else:
      # building samples by support
      support_labels, support_datas = zip(*support)
      table = {}
      for l_i, l_v in enumerate(support_labels):
        table[l_v] = l_i

      samples = None
      if transform_func is None:
        samples = [(table[i_val], i_val, d) for i_index, i_val in enumerate(label) \
                   if i_val in support_labels \
                   for d_index, d in enumerate(data[i_val]) \
                   if d_index not in support_datas[table[i_val]]]
      else:
        samples = [(table[i_val], i_val, d) for i_index, i_val in enumerate(label) \
                   if i_val in support_labels \
                   for d in data[i_val] \
                   if transform_func(d) not in support_datas[table[i_val]]]

      if self.rng != None:
        self.rng.shuffle(samples)
      return samples

  def make_sequence_data(self, data, label=None, nb_samples=0, nb_samples_per_class=0):
    '''
    only for image sequence
    :param data: [[],[],[],...]
    :param label: [0,1,2,...]
    :param nb_samples: sampling classes number
    :param nb_samples_per_class: sampling examples from per class
    :return:
    '''
    if label == None:
      label = range(len(data))

    label_num = len(label)

    # sample classes
    data_and_labels = zip(data, label)
    data_and_labels = [term for term in data_and_labels if len(term[0]) > 0]

    classes_samples = random.sample(data_and_labels, nb_samples)

    # sample example from every class
    sampler = lambda x: random.sample(x, nb_samples_per_class)
    samples = [(i, classes_samples[i][1], choice, classes_samples[i][0][choice]) \
               for i in range(nb_samples) \
               for choice in sampler(range(len(classes_samples[i][0])))]

    # shuffle
    if self.rng != None:
      self.rng.shuffle(samples)

    return samples

  def set_sequence(self, nb_samples, nb_samples_per_class):
    self.nb_samples = nb_samples
    self.nb_samples_per_class = nb_samples_per_class
    if nb_samples * nb_samples_per_class > 0:
      self.is_sequence = True

  def load_image(self,image_path):
    return imread(image_path)

  def reorganize_data(self,data,label,class_num=0):
    '''

    Parameters
    ----------
    data list (image list)
    label list (label list)

    Returns
    -------
        [[image,index],[],...]
    '''
    if class_num == 0:
      class_num = len(set(label))

    data_group = [[] for i in range(class_num)]
    for index in range(len(data)):
      data_group[label[index]].append((data[index],index))

    return data_group

  def regorganize_object_data(self,data,data_label,info):
    '''

    Parameters
    ----------
    data list (image list)
    data_label list (image label list)
    info class (dict)

    Returns
    -------
        [[data,data_index,(x1,y1,x2,y2)],[],...]
    '''

    data_group = [[] for i in range(len(info))]
    for index in range(len(data)):
      object_boxes = data_label[index]['bbox']
      object_classes = data_label[index]['category_id']

      for ob_index in range(len(object_boxes)):
        data_group[int(object_classes[ob_index])].append((data[index],index,object_boxes[ob_index]))

    return data_group

  def filter_by_id(self, id):
    id_filter = getattr(self, 'filter', None)
    if id_filter is not None:
      if id in id_filter:
        return True
      else:
        return False

    return True

  def filter_by_condition(self, label, ext_annotation=None, ext_filter=None):
    #
    included_cls = getattr(self, 'included', None)
    excluded_cls = getattr(self, 'excluded', None)
    transform_cls = getattr(self, 'transform', None)
    # object must be larger than min_size
    min_size = getattr(self, 'min_size', None)
    # object aspect ratio must be suffer to predefined value
    # (aspect ratio = width / height)
    min_aspect_ratio = getattr(self, 'min_aspect_ratio', None)
    max_aspect_ratio = getattr(self, 'max_aspect_ratio', None)

    if included_cls is not None:
      keep_index = []
      # cls filter
      for obj_index, obj in enumerate(label['category']):
        if obj in included_cls:
          if 'bbox' in label:
            x0, y0, x1, y1 = label['bbox'][obj_index, :]
            # size filter
            if min_size is not None:
              if (x1 - x0) < min_size or (y1 - y0) < min_size:
                continue

            if min_aspect_ratio is not None:
              if float(x1 - x0) / float(y1 - y0) < min_aspect_ratio:
                continue

            if max_aspect_ratio is not None:
              if float(x1 - x0) / float(y1 - y0) > max_aspect_ratio:
                continue

          # ext filter
          if ext_filter is not None:
            is_skip = False
            for f in ext_filter:
              remained_tag = getattr(self, f, None)
              if remained_tag is not None:
                if label[f][obj_index] != remained_tag:
                  is_skip = True
                  break
            if is_skip:
              continue

          # keep obj index
          keep_index.append(obj_index)

        if len(keep_index) == 0:
          return None

        if 'bbox' in label:
          label['bbox'] = label['bbox'][keep_index, :]
        label['category_id'] = label['category_id'][keep_index]
        label['category'] = [label['category'][i] for i in keep_index]

        if ext_annotation is not None:
          for annotation_name in ext_annotation:
            if annotation_name in label:
              if type(label[annotation_name]) == list:
                label[annotation_name] = [label[annotation_name][i] for i in keep_index]
              else:
                label[annotation_name] = label[annotation_name][keep_index]

    elif excluded_cls is not None:
      keep_index = []
      for obj_index, obj in enumerate(label['category']):
        if obj not in excluded_cls:
          # size filter
          if 'bbox' in label:
            x0, y0, x1, y1 = label['bbox'][obj_index, :]
            if min_size is not None:
              if (x1 - x0) < min_size or (y1 - y0) < min_size:
                continue

            if min_aspect_ratio is not None:
              if float(x1 - x0) / float(y1 - y0) < min_aspect_ratio:
                continue

            if max_aspect_ratio is not None:
              if float(x1 - x0) / float(y1 - y0) > max_aspect_ratio:
                continue

          # ext filter
          if ext_filter is not None:
            is_skip = False
            for f in ext_filter:
              remained_tag = getattr(self, f, None)
              if remained_tag is not None:
                if label[f][obj_index] != remained_tag:
                  is_skip = True
                  break
            if is_skip:
              continue

          # keep obj index
          keep_index.append(obj_index)

      if len(keep_index) == 0:
        return None

      if 'bbox' in label:
        label['bbox'] = label['bbox'][keep_index, :]
      label['category_id'] = label['category_id'][keep_index]
      label['category'] = [label['category'][i] for i in keep_index]

      if ext_annotation is not None:
        for annotation_name in ext_annotation:
          if annotation_name in label:
            if type(label[annotation_name]) == list:
              label[annotation_name] = [label[annotation_name][i] for i in keep_index]
            else:
              label[annotation_name] = label[annotation_name][keep_index]

    if transform_cls is not None:
      for obj_index, obj in enumerate(label['category']):
        if obj in transform_cls:
          label['category_id'][obj_index] = transform_cls[obj]

    return label

  def load_samples(self, sample_folder=None):
    assert(self.train_or_test == 'sample')
    sample_file = None
    if sample_folder is None:
      sample_folder = os.path.join(self.dir, self.train_or_test)

    for file in os.listdir(sample_folder):
      if file[0] == '.':
        continue
      if file.split('.')[-1].lower() == 'sample':
        sample_file = os.path.join(sample_folder, file)
        break

    sample_data = []
    if sample_file is not None:
      with open(sample_file, 'rb') as fp:
        sample_data = loads(fp.read())
    return sample_data, list(range(len(sample_data)))

  def download(self, target_path,
               file_names=[],
               default_url=None,
               auto_untar=False,
               is_gz=False,
               auto_unzip=False,
               shell=None):
    dataset_url = getattr(self, 'dataset_url', None)
    if dataset_url is None or len(dataset_url) == 0:
      dataset_url = default_url

    if dataset_url is not None:
      # dataset dont locate local
      mltalker_root = getattr(Config, 'server_ip', None)
      is_mltalker = False

      if mltalker_root is not None and mltalker_root in dataset_url:
        terms = dataset_url.split('/')
        dataset_url = '/'.join(terms[0:-2])
        dataset_url = os.path.join(dataset_url, self.train_or_test)
        is_mltalker = True

      # validate crowdsource address
      is_http = re.match('^((https|http|ftp|rtsp|mms)?://)', dataset_url)
      if is_http is not None:
        # 3rdpart dataset
        if not os.path.exists(target_path):
          os.makedirs(target_path)

        if len(file_names) > 0:
          for file_name in file_names:
            if maybe_here(target_path, file_name) is None:
              download(os.path.join(dataset_url, file_name), target_path)

              if auto_unzip:
                with zipfile.ZipFile(os.path.join(target_path, file_name), mode='r') as zpfd:
                  zpfd.extractall(target_path)
        else:
          if not is_mltalker:
            download_path = os.path.join(target_path, dataset_url.split('/')[-1])
            if not os.path.exists(download_path):
              download(dataset_url, target_path)

              if auto_untar:
                with tarfile.open(download_path, 'r:gz' if is_gz else 'r') as tar:
                  tar.extractall(target_path)

              if auto_unzip:
                with zipfile.ZipFile(download_path, mode='r') as zpfd:
                  zpfd.extractall(target_path)
                  
              if shell is not None:
                logger.info('execute: %s' % shell)
                subprocess.call(shell, shell=True, cwd=target_path)
          else:
            download_path = os.path.join(target_path, 'data.tar')
            if not os.path.exists(download_path):
              download(dataset_url, target_path, 'data.tar')

              if auto_untar:
                with tarfile.open(download_path, 'r:gz' if is_gz else 'r') as tar:
                  tar.extractall(target_path)
              
              if shell is not None:
                logger.info('execute: %s' % shell)
                subprocess.call(shell, shell=True, cwd=target_path)
                
      # # validate ipfs address
      # is_ipfs = re.match('^(ipfs://)', dataset_url)
      # if is_ipfs:
      #   dataset_name = os.path.normpath(target_path).split('/')[-1]
      #   process = multiprocessing.Process(target=data_download_local, args=(dataset_name, dataset_url))
      #   process.start()
      #   process.join()
      #   return

      # validate shell (download and reorganize data manully)
      is_shell = re.match('^(shell:)', dataset_url)
      if is_shell is not None:
        for f in file_names:
          if os.path.exists(os.path.join(target_path, f)):
            cmd_str = dataset_url[6:]
            cmds = cmd_str.split('\n')
            for cmd in cmds:
              if '{file_placeholder}' in cmd:
                cmd = cmd.format(file_placeholder=f)
                
              logger.info('execute: %s'%cmd)
              subprocess.call(cmd, shell=True, cwd=target_path)
          else:
            logger.error('perhaps %s must be prepared manully and '
                         'put it at %s'%(f, target_path))
            os._exit(-1)


class UnlabeledDataset(Dataset):
  def __init__(self, dataset):
    super(UnlabeledDataset, self).__init__()
    self.dataset = dataset
    self.unlabeled_data = dataset.unlabeled()
    self.dir = dataset.dir

  def data_pool(self):
    for id in range(self.size):
      yield self.at(id)

  def at(self, unlabeled_id):
    id = self.unlabeled_data[unlabeled_id]['id']
    file_id = self.unlabeled_data[unlabeled_id]['file_id']
    data = self.dataset.at(id, file_id)
    data[1].update({'file_id': file_id, 'id': id})
    return data

  @property
  def size(self):
    return len(self.unlabeled_data)


class CandidateDataset(Dataset):
  def __init__(self, dataset):
    super(CandidateDataset, self).__init__()
    self.dataset = dataset
    self.dir = dataset.dir
    self.candidates = []

    # 使用candidate文件夹中的数据(已经完成标注的数据)
    if os.path.exists(self.dataset.candidate_file):
      with open(self.dataset.candidate_file, 'r') as fp:
        content = fp.readline()
        content = content.strip()

        while content != '':
          id, file_id, annotation_file_id = content.split(',')

          self.candidates.append({
            'id': id,
            'file_id': file_id,
            'annotation_file_id': annotation_file_id
          })
          content = fp.readline()
          content = content.strip()

  def data_pool(self):
    for id in range(self.size):
      yield self.at(id)

  def at(self, local_id):
    id = self.candidates[local_id]['id']
    file_id = self.candidates[local_id]['file_id']
    annotation_file_id = self.candidates[local_id]['annotation_file_id']
 
    data = self.dataset.at((int)(id), file_id)
    data[1].update({'file_id': file_id, 'id': id, 'annotation_file_id': os.path.join(self.dataset.dir, annotation_file_id)})

    return data

  def split(self, split_params, split_method):
    _, b = self.dataset.split(split_params, split_method)
    return self, b

  @property
  def size(self):
    return len(self.candidates)


class DataAnnotationSplitDataset(object):
  def __init__(self, dataset):
      super().__init__()
      self.dataset = dataset
    
  def data_pool(self):
    pass

  def at(self, id):
    self.dataset.at(id)
    pass

  def annotation_at(self, id):
    
    return

  @property
  def size(self):
    return len(self.dataset)

  def waiting_process_num(self):
    return self.dataset.waiting_process_num()

  def finish_process_num(self):
    return self.dataset.finish_process_num()
