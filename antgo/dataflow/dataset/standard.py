# encoding=utf-8
# @Time    : 17-8-2
# @File    : standard.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import *
from antgo.dataflow.core import *
from antgo.utils.fs import maybe_here_match_format
from antgo.dataflow.basic import *
import numpy as np
import copy
import time
from antgo.utils.dht import *
import multiprocessing

class Standard(Dataset):
  is_complete = {}

  def __init__(self, train_or_test, dataset_dir=None, ext_params=None):
    dataset_name = dataset_dir.split('/')[-1]
    super(Standard, self).__init__(train_or_test, dataset_dir, ext_params, dataset_name)

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    # maybe dataset has not prepared now
    if not os.path.exists(os.path.join(dataset_dir,train_or_test)):
      os.makedirs(os.path.join(dataset_dir,train_or_test))

    # data queue (provide data from another independent process)
    self.data_queue = None
    dataset_url = getattr(self, 'dataset_url', None)
    if dataset_url is not None and dataset_url != 'http://xxx.xxx.xxx.com':
      if '%s_%s'%(dataset_name, train_or_test) not in Standard.is_complete:
        logger.info('download %s (%s dataset) from dht asynchronously'%(dataset_name, train_or_test))
        # set complete flag
        Standard.is_complete['%s_%s' % (dataset_name, train_or_test)] = False
        # db reader
        self._record_reader = RecordReader(os.path.join(dataset_dir, train_or_test),
                                           read_only=False)

        # launch independent process
        dataset_url = dataset_url.replace('ipfs://','')
        self.data_queue = multiprocessing.Queue()
        dht_process = multiprocessing.Process(target=dataset_download_dht,
                                              args=(dataset_dir,
                                                    train_or_test,
                                                    dataset_url,
                                                    self.data_queue,
                                                    self._record_reader,
                                                    2))
        dht_process.start()
    else:
      Standard.is_complete['%s_%s' % (dataset_name, train_or_test)] = True
      # db reader
      self._record_reader = RecordReader(os.path.join(dataset_dir, train_or_test),
                                         read_only=True)

    # dataset basic property
    dataset_attrs = {}
    if self.data_queue is not None:
      dataset_attrs = self.data_queue.get()
    else:
      dataset_attrs = self._record_reader.record_attrs()

    for k, v in dataset_attrs.items():
      setattr(self, k, v)

    # dataset index
    self.ids = np.arange(0, int(self.count)).tolist()

    if Standard.is_complete['%s_%s'%(dataset_name, train_or_test)]:
      self.ok_ids = np.arange(0, int(self.size)).tolist()
    else:
      self.ok_ids = []

    # fixed seed
    self.seed = time.time()

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    self.epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if self.epoch >= max_epoches:
        break

      if not Standard.is_complete['%s_%s'%(self.name, self.train_or_test)]:
        # for !train dataset, we have to wait until complete
        while True:
          # receive data list
          data_list = self.data_queue.get()
          for data in data_list:
            if type(data) == tuple:
              data_index, data_info, data_label, block_id = data
              # record sample
              self._record_reader.write(Sample(data=data_info, label=data_label), data_index)
              # record block
              self._record_reader.put(block_id, 'true')

              # update ids (sample index)
              self.ok_ids.append(data_index)
            else:
              # update ids (sample index)
              self.ok_ids.append(int(data))

          # check whether dataset is complete
          if self.size == len(self.ok_ids):
            Standard.is_complete['%s_%s' % (self.name, self.train_or_test)] = True
            break

          if self.train_or_test == 'train':
            break

      ids = copy.copy(self.ids)
      if self.rng:
        self.rng.shuffle(ids)

      # filter by ids
      filter_ids = getattr(self, 'filter', None)
      if filter_ids is not None:
        ids = [i for i in ids if i in filter_ids]

      for id in ids:
        if not Standard.is_complete['%s_%s'%(self.name, self.train_or_test)]:
          if id not in self.ok_ids:
            continue

        data, label = self._record_reader.read(id, 'data', 'label')
        # print((id, data,label))
        # filter by condition
        if type(label) == dict:
          if 'category' in label and 'category_id' in label:
            label = self.filter_by_condition(label)
            if label is None:
              continue
          
          label['id'] = id
        yield [data, label]

      # increment epoch
      self.epoch += 1

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')

    val_dataset = Standard('val', self.dir, self.ext_params)
    return self, val_dataset

  @property
  def size(self):
    if self.train_or_test == 'sample':
      return len(self.ids)

    return int(self.count)
  
  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    data, label = self._record_reader.read(id, 'data', 'label')
    return data, label