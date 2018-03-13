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
               dataset_name,
               dataset_public=False,
               dataset_local=False):
    super(AntGenerate, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_folder = ant_data_folder
    self.ant_context.ant = self
    self.dataset_name = dataset_name
    self.dataset_is_public = dataset_public
    self.dataset_is_local = dataset_local
    self.dump_dir = dump_dir

  def start(self):
    self.stage = 'GENERATE-DATA'
    if self.context.data_generator is None:
      logger.error('must set data generator')
      return

    if self.token is None:
      logger.error('must set your token')
      return

    if self.dataset_name is None:
      logger.error('must set dsataset name')
      return

    dataset_url = ''
    if not self.dataset_is_local:
      now_time_stamp = datetime.fromtimestamp(self.time_stamp).strftime('%Y%m%d.%H%M%S.%f')
      if not os.path.exists(os.path.join(self.dump_dir, now_time_stamp)):
        os.makedirs(os.path.join(self.dump_dir, now_time_stamp))

      # preprocess data generator
      categories = ['train', 'val', 'test', 'sample']
      data_generators = {}
      for category in categories:
        if category not in self.context.params.content:
          continue

        data_generators[category] = {}

        # data sample generator
        data_generators[category]['generator'] = self.context.data_generator(category)
        # data sample number
        data_generators[category]['num'] = self.context.params.content[category]['num']
        data_generators[category]['block'] = self.context.params.content[category]['block']

      # publish at dht
      dataset_hash_code = dataset_upload_dht(self.dataset_name,
                                             data_generators,
                                             os.path.join(self.dump_dir,now_time_stamp))
      dataset_url = 'ipfs://%s'%dataset_hash_code
      # print(dataset_hash_code)
    else:
      # build dataset in local
      categories = ['train', 'val', 'test', 'sample']
      for category in categories:
        if category not in self.context.params.content:
          continue

        dataset_record_path = os.path.join(Config.data_factory, self.dataset_name, category)

        if not os.path.exists(dataset_record_path):
          os.makedirs(dataset_record_path)

        data_writer = RecordWriter(dataset_record_path)
        for data,label in self.context.data_generator(category):
          data_writer.write(Sample(data=data, label=label))

    # rectify is_public and is_local
    if self.dataset_is_local:
      self.dataset_is_public = False

    # create dataset on mltalker
    create_dataset_remote_api = 'hub/api/terminal/create/dataset'
    response = self.remote_api_request(create_dataset_remote_api,
                                       action='post',
                                       data={'dataset-name': self.dataset_name,
                                             'dataset-is-public': int(self.dataset_is_public),
                                             'dataset-is-local': int(self.dataset_is_local),
                                             'dataset-url': dataset_url})

    if response['status'] != 'OK':
      logger.error('fail to synchronize (maybe dataset name not unique)')
      return

    logger.info('success to register to mltalker')
