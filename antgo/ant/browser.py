# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:29
# @File    : browser.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
# from antgo.dataflow import dataset
from antgo.dataflow.dataset.queue_dataset import *
from antgo.dataflow.recorder import *
from antvis.client.httprpc import *
from antgo.crowdsource.browser_server import *
import requests
import json
import traceback
import threading
import cv2


class BrowserDataRecorder(object):
  def __init__(self, rpc, maxsize=30):
    # self.queue = multiprocessing.Queue()              # 不设置队列最大缓冲
    self.prepare_queue = queue.Queue(maxsize=maxsize)
    self.dump_dir = ''
    self.dataset_flag = 'TRAIN'
    self.dataset_size = 0
    self.tag_dir = ''
    self.rpc = rpc

    # 5个线程，等待处理
    for _ in range(1):
      t = threading.Thread(target=self.asyn_record)
      t.daemon = True
      t.start()

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
      cv2.imwrite(image_path, transfer_result)
      return file_name
    except:
      logger.error('Couldnt transfer image data.')
      return None

  def asyn_record(self):
    while True:
      val = self.prepare_queue.get()
      if val is None:
        break

      # 加入到队列中
      assert(type(val) == dict)

      # 重新组织数据格式
      data = {}
      id = None
      for key, value in val.items():
        if type(value) != dict:
          if key.lower() == 'id' or key.lower() == 'image_file':
            id = value
          continue

        data_name = key
        if data_name not in data:
          data[data_name] = {}

        if 'data' in value or 'DATA' in value:
          if 'data' in value:
            data[data_name]['data'] = value['data']
          else:
            data[data_name]['data'] = value['DATA']
        if 'type' in value or 'TYPE' in value:
          if 'type' in value:
            data[data_name]['type'] = value['type']
          else:
            data[data_name]['type'] = value['TYPE']
        
          if 'bboxes' in value and data[data_name]['type'] == 'IMAGE':
            data[data_name]['bboxes'] = value['bboxes']

        data[data_name]['title'] = data_name

      if id is None:
        logger.error('Missing id flag, please return {"id": ...}')
        continue

      for k, v in data.items():
        v['id'] = id

      # 转换数据到适合web
      web_data = []
      for name, body in data.items():
        if 'type' in body:
          if body['type'].upper() == 'IMAGE':
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

      self.rpc.browser.sample.fresh.post(samples=json.dumps([web_data]))

  def record(self, val):
    self.prepare_queue.put(val)

  def close(self):
    pass
  
class AntBrowser(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config,
               ant_dataset,
               **kwargs):
    super(AntBrowser, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.ant_data_source = ant_data_folder
    self.ant_task_config = ant_task_config
    self.dataset_name = ant_dataset
    self.dump_dir = ant_dump_dir
    self.host_ip = self.context.params.system.get('ip', '127.0.0.1')
    self.host_port = self.context.params.system.get('port', 8901)
    self.rpc = None
    self._running_dataset = None
    self._running_task = None
    self.p = None

  @property
  def running_dataset(self):
    return self._running_dataset

  @property
  def running_task(self):
    return self._running_task

  def ping_until_ok(self):
    while True:
      content = self.rpc.ping.get()
      if content['status'] != 'ERROR':
          break
      # 暂停5秒钟，再进行尝试
      time.sleep(5)

  def wait_until_stop(self):
    self.p.join()

  def download(self):
    if not os.path.exists(os.path.join(self.dump_dir, 'check')):
      os.makedirs(os.path.join(self.dump_dir, 'check'))

    folder = os.path.join(self.dump_dir, 'check')
    file_name = f'check.json'
    self.rpc.browser.download(
      file_folder=folder,
      file_name=file_name
    )    
    with open(os.path.join(folder, file_name), 'r') as fp:
      content = json.load(fp)

      return content
  
  def waiting(self, until_exit=False):
    # 需要等待本轮标准完成
    while True:
      response = self.rpc.info.get()
      if response['status'] == 'ERROR':
        print('rpc error...')
        time.sleep(5)
        continue

      if response['content']['project_state']['stage'] == 'finish' and not until_exit:
        break
      # 等待10分钟后检查
      time.sleep(10)
  
  def start(self, *args, **kwargs):
    # 1.step 获得数据集解析
    running_ant_task = None
    if self.token is not None:
      # 1.1.step load challenge task
      response = mlogger.info.challenge.get(command=type(self).__name__)
      if response['status'] == 'ERROR':
        # invalid token
        logger.error('couldnt load challenge task')
        self.token = None
      elif response['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('prohibit submit challenge task frequently')
        exit(-1)
      elif response['status'] == 'OK':
        content = response['content']
        # maybe user token or task token
        if 'task' in content:
          challenge_task = create_task_from_json(content)
          if challenge_task is None:
            logger.error('couldnt load challenge task')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('unknow error')
        exit(-1)

    if running_ant_task is None:
        # 1.2.step load custom task
        if self.ant_task_config is not None:
            custom_task = create_task_from_xml(self.ant_task_config, self.context)
            if custom_task is None:
                logger.error('Couldnt load custom task.')
                exit(-1)
            running_ant_task = custom_task

    self._running_task = running_ant_task
    if running_ant_task is not None and running_ant_task.dataset is not None:
      self.dataset_name = running_ant_task.dataset_name
    dataset_cls = running_ant_task.dataset

    # 2.step 配置记录器
    self.rpc = HttpRpc("v1", "antgo/api", self.host_ip, self.host_port)
    self.context.recorder = \
      BrowserDataRecorder(rpc=self.rpc)

    # 3.step 启动浏览web服务
    browser_params = getattr(self.context.params, 'browser', None)
    tags = []
    if browser_params is not None:
      tags = browser_params.get('tags', [])

    # 3.1.step 准备web服务资源
    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    if os.path.exists(self.dump_dir):
      shutil.rmtree(self.dump_dir)
    browser_static_dir = os.path.join(self.dump_dir, 'browser', 'static')
    shutil.copytree(os.path.join(static_folder, 'resource', 'app'), browser_static_dir)

    # 3.2.step 准备有效端口
    self.host_port = _pick_idle_port(self.host_port)

    # 3.3.step 状态
    train_offset, val_offset, test_offset = 0, 0, 0
    offset_configs = [{
      'dataset_flag': 'TRAIN',
      'dataset_offset': train_offset
    }, {
      'dataset_flag': 'VAL',
      'dataset_offset': val_offset
    }, {
      'dataset_flag': 'TEST',
      'dataset_offset': test_offset
    }]

    if self.context.params.browser is not None and \
      'offset' in self.context.params.browser.keys():
      if 'TRAIN' in self.context.params.browser.offset.keys():
        train_offset = self.context.params.browser.offset.TRAIN
        offset_config = {
          'dataset_flag': 'TRAIN',
          'dataset_offset': train_offset
        }
        offset_configs[0] = offset_config

    if self.context.params.browser is not None and \
      'offset' in self.context.params.browser.keys():
      if 'VAL' in self.context.params.browser.offset:
        val_offset = self.context.params.browser.offset.VAL
        offset_config = {
          'dataset_flag': 'VAL',
          'dataset_offset': val_offset
        }
        offset_configs[1] = offset_config

    if self.context.params.browser is not None and \
      'offset' in self.context.params.browser.keys():
      if 'TEST' in self.context.params.browser.offset:
        test_offset = self.context.params.browser.offset.TEST
        offset_config = {
          'dataset_flag': 'TEST',
          'dataset_offset': test_offset
        }
        offset_configs[2] = offset_config

    dataset_flag = 'train'
    if self.context.params.browser is not None and \
        'dataset_flag' in self.context.params.browser.keys():
      if self.context.params.browser.dataset_flag.lower() in ['train', 'val', 'test']:
        dataset_flag = \
          self.context.params.browser.dataset_flag.lower()

    data_json_file = kwargs.get('json_file', None)
    if dataset_cls is not None and data_json_file is None:
      train_dataset = \
        dataset_cls(dataset_flag, os.path.join(self.ant_data_source, self.dataset_name))
      self._running_dataset = train_dataset

    self.context.recorder.dataset_flag = dataset_flag.upper()
    self.context.recorder.dataset_size = \
      self.running_dataset.size if self.running_dataset is not None else self.context.params.browser.size
    self.context.recorder.dump_dir = os.path.join(self.dump_dir, 'browser', 'static', 'data')
    if not os.path.exists(self.context.recorder.dump_dir):
      os.makedirs(self.context.recorder.dump_dir)
    self.context.recorder.tag_dir = os.path.join(self.dump_dir, 'record')
    if not os.path.exists(self.context.recorder.tag_dir):
      os.makedirs(self.context.recorder.tag_dir)

    sample_meta = {}
    sample_list = []
    sample_folder = None
    if data_json_file is not None:
      # step1: 加载样本信息
      # 直接使用来自于data_json_file中的样本
      # 兼容两种常用的存储样本信息方式（1.纯json格式，所有样本以list形式存储；2.样本按行存储，每个样本的信息是json格式）
      try:
        # try 1 (纯json格式，所有样本以list形式存储)
        with open(data_json_file, 'r', encoding="utf-8") as fp:
          sample_list = json.load(fp)
          self.context.recorder.dataset_size = len(sample_list)
      except:
        # try 2 (样本信息按行存储，每个样本信息是json格式)
        with open(data_json_file, 'r', encoding="utf-8") as fp:
          sample_info_str = fp.readline()
          sample_info_str = sample_info_str.strip()
          
          while sample_info_str:
            sample_info = json.loads(sample_info_str)
            sample_list.append(sample_info)
            sample_info_str = fp.readline()
            sample_info_str = sample_info_str.strip()
            if sample_info_str == '':
              break        
            
      # step2: 尝试加载样本集meta信息
      sample_folder = os.path.dirname(data_json_file)
      data_meta_file = os.path.join(sample_folder, 'meta.json')
      if os.path.exists(data_meta_file):
        with open(data_meta_file, 'r', encoding="utf-8") as fp:
          sample_meta = json.load(fp)
              
    sample_offset = train_offset
    if dataset_flag == 'test':
      sample_offset = test_offset
    elif dataset_flag == 'val':
      sample_offset = val_offset
    logger.info('Browser %s dataset.' % dataset_flag)

    # 设置数据基本信息
    profile_config = {
      'dataset_flag': dataset_flag.upper(),
      'samples_num': self.context.recorder.dataset_size,
      'samples_num_checked': sample_offset
    }

    # 设置记录器偏移
    self.context.recorder.sample_index = sample_offset

    white_users = \
      self.context.params.browser.white_users.get() if self.context.params.browser.white_users is not None else None
    if len(white_users) == 0:
      white_users = None
    user_input = self.context.params.browser.user_input
    # 在独立进程中启动webserver
    self.p = \
      multiprocessing.Process(
        target=browser_server_start,
        args=(self.dump_dir,
              tags,
              self.host_port,
              offset_configs,
              profile_config,
              sample_folder, sample_list, sample_meta, user_input, white_users)
      )
    self.p.daemon = True
    self.p.start()

    # 等待直到http服务开启
    self.ping_until_ok()

    if self.context.is_interact_mode:
      return

    # 3.3.step 启动数据处理
    try:
      count = 0
      for data in self.context.data_processor.iterator(self.running_dataset):
        logger.info('Record data %d for browser.' % count)
        self.context.recorder.record(data)
        count += 1
    except Exception as e:
      traceback.print_exc()
      logger.info('Finish all records in browser %s dataset.' % dataset_flag)

    # 不结束webserver
    self.p.join()
    logger.info('Stop Browser.')