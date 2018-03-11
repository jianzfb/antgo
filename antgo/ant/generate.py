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


class AntGenerate(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               dump_dir,
               ant_token,
               dataset_name,
               dataset_public=False):
    super(AntGenerate, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_folder = ant_data_folder
    self.ant_context.ant = self
    self.dataset_name = dataset_name
    self.dataset_is_public = dataset_public
    self.dump_dir = dump_dir

  def start(self):
    self.stage = 'GENERATE-DATA'
    if self.context.data_generator is None:
      logger.error('must set data generator')
      return

    if self.token is None:
      logger.error('must give your token')
      return

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
    dataset_hash_url = 'ipfs://%s'%dataset_hash_code
    # print(dataset_hash_code)

    # create dataset on mltalker
    create_dataset_remote_api = 'hub/api/terminal/create/dataset'
    response = self.remote_api_request(create_dataset_remote_api,
                                       action='post',
                                       data={'dataset-name': self.dataset_name,
                                             'dataset-is-public': int(self.dataset_is_public),
                                             'dataset-url': dataset_hash_url})

    if response['status'] != 'OK':
      logger.error('fail to synchronize (maybe dataset name not unique)')
      return

    logger.info('success to register to mltalker')
