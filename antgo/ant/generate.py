# -*- coding: UTF-8 -*-
# @Time    : 17-9-5
# @File    : generate.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.dataflow.basic import *
from antgo.utils.dht import *
from antgo.utils.serialize import *

if sys.version > '3':
    PY3 = True
else:
    PY3 = False

Config = config.AntConfig


class AntGenerate(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               dump_dir,
               ant_token,
               dataset_name):
    super(AntGenerate, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_folder = ant_data_folder
    self.ant_context.ant = self
    self.dataset_name = dataset_name
    self.dump_dir = dump_dir

  def start(self):
    # 1.step check basic parameters
    if self.context.data_generator is None:
      logger.error('Must set data generator.')
      return

    if self.dataset_name is None:
      logger.error('Must set dsataset name.')
      return

    # 2.step build experiment folder
    now_time_stamp = datetime.fromtimestamp(self.time_stamp).strftime('%Y%m%d.%H%M%S.%f')
    if not os.path.exists(os.path.join(self.dump_dir, now_time_stamp, self.dataset_name)):
      os.makedirs(os.path.join(self.dump_dir, now_time_stamp, self.dataset_name))
    experiment_folder = os.path.join(self.dump_dir, now_time_stamp, self.dataset_name)
    logger.info('Dataset would be saved in %s.'%now_time_stamp)

    # 3.step build dataset (rocksdb)
    # dataset "train", 'val', 'test'
    for dataset_flag in ['train', 'val', 'test']:
      dataset_generateor = self.context.data_generator(dataset_flag)
      if dataset_generateor is not None:
        logger.info('Start build %s set for %s.'%(dataset_flag, self.dataset_name))
        if not os.path.exists(os.path.join(experiment_folder, dataset_flag)):
          os.makedirs(os.path.join(experiment_folder, dataset_flag))

        record_writer = RecordWriter(os.path.join(experiment_folder, dataset_flag))
        try:
          for data, label in dataset_generateor:
            record_writer.write(Sample(data=data, label=label))
        except:
          pass

        logger.info('Finish build %s set for %s.'%(dataset_flag, self.dataset_name))
      else:
        logger.warn('Couldn build %s set for %s.'%(dataset_flag, self.dataset_name))
