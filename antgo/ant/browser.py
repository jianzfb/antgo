# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:29
# @File    : browser.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.crowdsource.browser_server import *
from multiprocessing import Process, Queue
from antgo.dataflow.dataset.queue_dataset import *
from antgo.dataflow.recorder import *
import requests
from jinja2 import Environment, FileSystemLoader


class BrowserDataRecorder(object):
  def __init__(self, maxsize=10):
    self.queue = Queue(maxsize=maxsize)
    self.dump_dir = ''
    self.dataset_flag = 'TRAIN'
    self.dataset_size = 0

  def _transfer_image(self, data):
    try:
      if len(data.shape) == 2:
        if data.dtype == np.uint8:
          transfer_result = data
        else:
          data_min = np.min(data)
          data_max = np.max(data)
          transfer_result = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
      else:
        assert (data.shape[2] == 3)
        transfer_result = data.astype(np.uint8)

      # save path
      if not os.path.exists(self.dump_dir):
        os.makedirs(self.dump_dir)

      file_name = '%s.png' % str(uuid.uuid4())
      image_path = os.path.join(self.dump_dir, file_name)
      scipy.misc.imsave(image_path, transfer_result)
      return file_name
    except:
      logger.error('couldnt transfer image data')
      return None

  def record(self, val):
    # 加入到队列中
    assert(type(val) == dict)

    # 重新组织数据格式
    data = {}
    for key, value in val.items():
      xxyy = key.split('_')
      data_name = xxyy[0]
      if data_name not in data:
        data[data_name] = {}

      # TYPE, TAG
      if(len(xxyy) == 1):
        data[data_name]['data'] = value
        data[data_name]['title'] = data_name
      else:
        if xxyy[-1] == "TYPE":
          data[data_name]['type'] = value
        elif xxyy[-1] == "TAG":
          data[data_name]['tag'] = value
        elif xxyy[-1] == 'ID':
          data[data_name]['id'] = value

    # 转换数据到适合web
    web_data = []
    for name, body in data.items():
      if 'type' in body:
        if body['type'] == 'IMAGE':
          body['width'] = body['data'].shape[1]
          body['height'] = body['data'].shape[0]
          body['data'] = '/static/data/%s'%self._transfer_image(body['data'])

      if 'type' not in body:
        body['type'] = 'STRING'

      if 'tag' not in body:
        body['tag'] = []

      body['dataset_flag'] = self.dataset_flag
      body['dataset_size'] = self.dataset_size
      web_data.append(body)

    # 加入队列，如果队列满，将阻塞
    self.queue.put(web_data)


class AntBrowser(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_host_ip,
               ant_host_port,
               ant_data_folder,
               ant_dataset,
               ant_dump_dir,
               **kwargs):
    super(AntBrowser, self).__init__(ant_name, ant_context)
    self.ant_data_source = ant_data_folder
    self.dataset_name = ant_dataset
    self.dump_dir = ant_dump_dir
    self.host_ip = ant_host_ip
    self.host_port = ant_host_port
    self.from_experiment = kwargs.get('from_experiment', None)
    self.rpc = None

  def start(self):
    # 1.step 获得数据集解析
    parse_flag = ''
    dataset_cls = AntDatasetFactory.dataset(self.dataset_name, parse_flag)

    if dataset_cls is None:
      logger.error('couldnt find dataset parse class')
      return

    # 2.step 配置记录器
    self.context.recorder = BrowserDataRecorder()

    # 3.step 启动浏览web服务
    browser_params = getattr(self.context.params, 'browser', None)
    tags = []
    if browser_params is not None:
      tags = browser_params.get('tags', []) \

    # 3.1.step 准备web服务资源
    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    browser_static_dir = os.path.join(self.dump_dir, 'browser')
    if os.path.exists(browser_static_dir):
      shutil.rmtree(browser_static_dir)

    shutil.copytree(os.path.join(static_folder, 'resource', 'browser'), browser_static_dir)

    # 3.2.step 准备有效端口
    self.host_port = _pick_idle_port(self.host_port)
    self.rpc = HttpRpc("v1", "browser-api", "127.0.0.1", self.host_port)

    # base_url = '{}:{}'.format(self.host_ip, self.host_port)
    #
    # # 3.3.step 准备web配置文件
    # template_file_folder = os.path.join(static_folder, 'resource', 'browser', 'static')
    # file_loader = FileSystemLoader(template_file_folder)
    # env = Environment(loader=file_loader)
    # template = env.get_template('config.json')
    # output = template.render(BASE_URL=base_url)
    #
    # with open(os.path.join(browser_static_dir, 'static', 'config.json'), 'w') as fp:
    #   fp.write(output)

    # 3.4.step 启动
    process = multiprocessing.Process(target=browser_server_start,
                                      args=(os.path.join(self.ant_data_source, self.dataset_name),
                                            self.dump_dir,
                                            self.context.recorder.queue,
                                            tags,
                                            self.host_port))

    process.daemon = True
    process.start()

    logger.info('wating 60s to launch browser server')
    time.sleep(60)

    train_offset, val_offset, test_offset = 0, 0, 0
    offset_config = {
      'dataset_flag': 'TRAIN',
      'dataset_offset': train_offset
    }
    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'TRAIN' in self.context.params.browser['offset']:
        train_offset = self.context.params.browser['offset']['TRAIN']
        offset_config = {
          'dataset_flag': 'TRAIN',
          'dataset_offset': train_offset
        }
    self.rpc.config.post(offset_config=json.dumps(offset_config))

    offset_config = {
      'dataset_flag': 'VAL',
      'dataset_offset': val_offset
    }
    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'VAL' in self.context.params.browser['offset']:
        val_offset = self.context.params.browser['offset']['VAL']
        offset_config = {
          'dataset_flag': 'VAL',
          'dataset_offset': val_offset
        }
    self.rpc.config.post(offset_config=json.dumps(offset_config))

    offset_config = {
      'dataset_flag': 'TEST',
      'dataset_offset': test_offset
    }
    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'TEST' in self.context.params.browser['offset']:
        test_offset = self.context.params.browser['offset']['TEST']
        offset_config = {
          'dataset_flag': 'TEST',
          'dataset_offset': test_offset
        }
    self.rpc.config.post(offset_config=json.dumps(offset_config))

    # 4.step 启动数据生成
    # 4.1.step 训练集
    train_dataset = dataset_cls('train', os.path.join(self.ant_data_source, self.dataset_name))
    self.context.recorder.dataset_flag = 'TRAIN'
    self.context.recorder.dataset_size = train_dataset.size
    self.context.recorder.dump_dir = os.path.join(self.dump_dir, 'browser', 'static', 'data')

    if train_dataset.size > 0:
      logger.info('train dataset browser')
      try:
        # 设置数据基本信息
        profile_config={
          'dataset_flag': 'TRAIN',
          'samples_num': train_dataset.size,
          'samples_num_checked': train_offset
        }
        self.rpc.config.post(profile_config=json.dumps(profile_config))

        # 设置当前状态
        self.rpc.config.post(state='TRAIN')

        count = 0
        for data in self.context.data_generator(train_dataset):
          if count < train_offset:
            count += 1
            continue

          logger.info('push train data to wait check')
          self.context.recorder.record(data)
      except:
        pass

    while not self.context.recorder.queue.empty():
      logger.info('waiting browser train dastaset')
      time.sleep(30)

    # 4.2.step 验证集
    val_dataset = dataset_cls('val', os.path.join(self.ant_data_source, self.dataset_name))
    self.context.recorder.dataset_flag = "VAL"
    self.context.recorder.dataset_size = val_dataset.size
    self.context.recorder.dump_dir = os.path.join(self.dump_dir, 'browser', 'static', 'data')
    if val_dataset.size > 0:
      logger.info('val dataset browser')
      # 设置数据基本信息
      profile_config = {
        'dataset_flag': 'VAL',
        'samples_num': val_dataset.size,
        'samples_num_checked': val_offset
      }
      self.rpc.config.post(profile_config=json.dumps(profile_config))

      # 设置当前状态
      self.rpc.config.post(state='VAL')

      count = 0
      for data in self.context.data_generator(val_dataset):
        if count < val_offset:
          count += 1
          continue

        logger.info('push val data to wait check')
        self.context.recorder.record(data)

    while not self.context.recorder.queue.empty():
      logger.info('waiting browser val dastaset')
      time.sleep(30)

    # 4.3.step 测试集
    test_dataset = dataset_cls('test', os.path.join(self.ant_data_source, self.dataset_name))
    self.context.recorder.dataset_flag = "TEST"
    self.context.recorder.dataset_size = test_dataset.size
    self.context.recorder.dump_dir = os.path.join(self.dump_dir, 'browser', 'static', 'data')
    if test_dataset.size > 0:
      logger.info('test dataset browser')

      # 设置数据基本信息
      profile_config = {
        'dataset_flag': 'TEST',
        'samples_num': test_dataset.size,
        'samples_num_checked': test_offset
      }
      self.rpc.config.post(profile_config=json.dumps(profile_config))

      # 设置当前状态
      self.rpc.config.post(state='TEST')

      count = 0
      for data in self.context.data_generator(test_dataset):
        if count < test_offset:
          count += 1
          continue

        logger.info('push test data to wait check')
        self.context.recorder.record(data)

    while not self.context.recorder.queue.empty():
      time.sleep(30)

    logger.info('finish data generate process')