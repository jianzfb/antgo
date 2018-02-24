# -*- coding: UTF-8 -*-
# @Time    : 17-9-5
# @File    : generate.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.dataflow.basic import *
from antgo.utils.p2p_data import *
import multiprocessing
if sys.version > '3':
    PY3 = True
else:
    PY3 = False


class AntGenerate(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_token):
    super(AntGenerate, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_folder = ant_data_folder
    self.ant_context.ant = self
    
  def start(self):
    self.stage = 'GENERATE-DATA'
    if self.context.data_generator is None:
      logger.error('must set data generator')
      return
    
    dataset_name = None
    dataset_is_local = None
    dataset_is_public = None
    # 1.step check self.ant_token, sychronize dataset
    is_ok = False
    while not is_ok:
      # 1.1 step input 'dataset_name'
      if PY3:
        dataset_name = input('dataset name: ')
      else:
        dataset_name = raw_input('dataset name: ')
        
      # 1.2 step input 'is_local'
      if PY3:
        dataset_is_local = input('dataset is local: ')
      else:
        dataset_is_local = raw_input('dataset is local: ')
      
      try:
        if not (dataset_is_local.lower() == 'yes' or dataset_is_local.lower() == 'no'):
          continue
      except:
        logger.error('yes or no')
        continue

      if dataset_is_local.lower() == 'yes':
        dataset_is_local = 1
      else:
        dataset_is_local = 0

      # 1.3 step input 'is_public'
      if PY3:
        dataset_is_public = input('dataset is public: ')
      else:
        dataset_is_public = raw_input('dataset is public: ')

      try:
        if not (dataset_is_public.lower() == 'yes' or dataset_is_public.lower() == 'no'):
          continue
      except:
        logger.error('yes or no')
        continue

      if dataset_is_public.lower() == 'yes':
        dataset_is_public = 1
      else:
        dataset_is_public = 0

      if dataset_name is None or \
              dataset_is_local is None or \
              dataset_is_public is None:
        logger.error('must set dataset basic information')
        continue
      
      is_ok = True
      
    # 2.step generate training data
    is_train_data_ok = False
    if not os.path.exists(os.path.join(self.ant_data_folder, dataset_name, 'train')):
      os.makedirs(os.path.join(self.ant_data_folder, dataset_name, 'train'))
      
    dataset_record = RecordWriter(os.path.join(self.ant_data_folder, dataset_name, 'train'))
    try:
      logger.info('generate train data')
      for data, label in self.context.data_generator('train'):
        dataset_record.write(Sample(data=data, label=label))
      
      is_train_data_ok = True
    except:
      logger.error('error in generating train data')
      logger.error(sys.exc_info())
      os._exit(-1)
    
    # 3.step generate val data
    is_val_data_ok = False
    if not os.path.exists(os.path.join(self.ant_data_folder, dataset_name, 'val')):
      os.makedirs(os.path.join(self.ant_data_folder, dataset_name, 'val'))

    dataset_record = RecordWriter(os.path.join(self.ant_data_folder, dataset_name, 'val'))
    try:
      logger.info('generate val data')
      for data, label in self.context.data_generator('val'):
        dataset_record.write(Sample(data=data, label=label))
  
      is_val_data_ok = True
    except:
      logger.error('error in generating val data')
      logger.error(sys.exc_info())
      os._exit(-1)

    # 4.step generate test data
    is_test_data_ok = False
    if not os.path.exists(os.path.join(self.ant_data_folder, dataset_name, 'test')):
      os.makedirs(os.path.join(self.ant_data_folder, dataset_name, 'test'))

    dataset_record = RecordWriter(os.path.join(self.ant_data_folder, dataset_name, 'test'))
    try:
      logger.info('generate test data')
      for data, label in self.context.data_generator('test'):
        dataset_record.write(Sample(data=data, label=label))
      
      is_test_data_ok = True
    except:
      logger.error('error in generating test data')
      logger.error(sys.exc_info())
      os._exit(-1)
      
    if not is_train_data_ok and not is_test_data_ok:
      logger.error('fail to generate train/test dataset')
      return

    if self.token is not None:
      # 5.step create dataset in mltalker
      create_dataset_remote_api = 'hub/api/terminal/create/dataset'
      response = self.remote_api_request(create_dataset_remote_api,
                                         action='post',
                                         data={'dataset-name': dataset_name,
                                               'dataset-is-local': int(dataset_is_local),
                                               'dataset-is-public': int(dataset_is_public)})

      if response['status'] != 'OK':
        logger.error('fail to synchronize (maybe dataset name not unique)')
        return

      # 6.step publish to DHT
      if not dataset_is_local:
        logger.info('publish to DHT')
        process = multiprocessing.Process(target=data_publish_dht, args=(dataset_name, self.token))
        process.start()
        process.join()
