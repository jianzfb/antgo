# -*- coding: UTF-8 -*-
# @Time    : 17-9-5
# @File    : generate.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.dataflow.basic import *
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
        dataset_is_local = input('is local(0 or 1): ')
      else:
        dataset_is_local = raw_input('is local(0 or 1): ')
      
      try:
        dataset_is_local = int(dataset_is_local)
        if dataset_is_local not in [0, 1]:
          continue
      except:
        logger.error('is_local must is 0 or 1')
        continue
      
      # 1.3 step input 'is_public'
      if PY3:
        dataset_is_public = input('is public(0 or 1): ')
      else:
        dataset_is_public = raw_input('is public(0 or 1): ')

      try:
        dataset_is_public = int(dataset_is_public)
        if dataset_is_public not in [0, 1]:
          continue
      except:
        logger.error('is_public must is 0 or 1')
        continue

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
      
    if not is_train_data_ok and not is_test_data_ok:
      logger.error('fail to generate train/test dataset')
      return
      
    # 4.step synchronize with cloud
    if self.token is not None:
      create_dataset_remote_api = 'hub/api/terminal/create/dataset'
      response = self.remote_api_request(create_dataset_remote_api,
                                         action='post',
                                         data={'dataset-name': dataset_name,
                                               'dataset-is-local': int(dataset_is_local),
                                               'dataset-is-public': int(dataset_is_public)})
      if response['status'] != 'OK':
        logger.error('fail to synchronize (maybe dataset name not unique)')
        return
      
      if not dataset_is_local:
        # TODO: synchronize dataset
        pass