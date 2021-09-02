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
from antgo.dataflow.dataset.queue_dataset import *
from antgo.dataflow.recorder import *
from antgo.dataflow.dataset.parallel_read import MultiprocessReader
from antvis.client.httprpc import *
from antgo.dataflow.dataset.parallel_read import *
import requests
import json
from jinja2 import Environment, FileSystemLoader
import traceback

class BrowserDataRecorder(object):
  def __init__(self, maxsize=30):
    self.queue = queue.Queue()  # 不设置队列最大缓冲
    self.prepare_queue = queue.Queue(maxsize=maxsize)
    self.dump_dir = ''
    self.dataset_flag = 'TRAIN'
    self.dataset_size = 0
    self.tag_dir = ''

    # 5个线程，等待处理
    for _ in range(5):
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
      imwrite(image_path, transfer_result)
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
        data_name = key
        if data_name not in data:
          data[data_name] = {}

        if 'data' in value or 'DATA' in value:
          if 'data' in value:
            data[data_name]['data'] = value['data']
            if data_name == 'id' or data_name == 'ID':
              id = value['data']
          else:
            data[data_name]['data'] = value['DATA']
            if data_name == 'id' or data_name == 'ID':
              id = value['data']

        if 'type' in value or 'TYPE' in value:
          if 'type' in value:
            data[data_name]['type'] = value['type']
          else:
            data[data_name]['type'] = value['TYPE']

        # if 'tag' in value or 'TAG' in value:
        #   if 'tag' in value:
        #     data[data_name]['tag'] = value['tag']
        #   else:
        #     data[data_name]['tag'] = value['TAG']

        # if 'id' in value or 'ID' in value:
        #   if 'id' in value:
        #     data[data_name]['id'] = value['id']
        #   else:
        #     data[data_name]['id'] = value['ID']

        data[data_name]['title'] = data_name

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

      # 如果从指定实验加载，则找寻是否存在以筛选标记
      if id is None:
        continue

      if os.path.exists(os.path.join(self.tag_dir, self.dataset_flag, '%s.json' % str(id))):
        # 加载json文件
        logger.info('Load record json from %s/%s'%(self.dataset_flag, '%s.json.' % str(id)))
        with open(os.path.join(self.tag_dir, self.dataset_flag, '%s.json' % str(id)), 'r') as fp:
          annotation = json.load(fp)
          # 确保标注数量和数据数量是一致的
          assert (len(web_data) == len(annotation))

          for a in web_data:
            is_update = False
            for b in annotation:
              # 检测数据和标注是否对应
              is_consistent = b['title'] == a['title']
              if 'id' in b:
                is_consistent = is_consistent and (a['id'] == b['id'])

                if not (a['id'] == b['id']):
                  logger.warn('Please whether data and record are consistent.')

              if is_consistent:
                a['tag'] = b['tag']
                is_update = True
                break

            if not is_update:
              if 'id' in a:
                logger.info('Annotation and data not consistence for %s.'%a['id'])
              else:
                logger.info('Annotation and data not consistence for %s.' % a['data'])

      # 加入队列，如果队列满，将阻塞
      self.queue.put(web_data)

  def record(self, val):
    self.prepare_queue.put(val)
    

class AntBrowser(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_token,
               ant_host_ip,
               ant_host_port,
               ant_data_folder,
               ant_dataset,
               ant_dump_dir,
               **kwargs):
    super(AntBrowser, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.ant_data_source = ant_data_folder
    self.dataset_name = ant_dataset
    self.dump_dir = ant_dump_dir
    self.host_ip = ant_host_ip
    self.host_port = ant_host_port
    self.rpc = None

  def start(self):
    # 1.step 获得数据集解析
    running_ant_task = None
    if self.token is not None:
      # 1.1.step load challenge task
      response = mlogger.getEnv().dashboard.challenge.get(command=type(self).__name__)
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

    if running_ant_task is not None:
      self.dataset_name = running_ant_task.dataset_name

    parse_flag = ''
    dataset_cls = AntDatasetFactory.dataset(self.dataset_name, parse_flag)

    if dataset_cls is None:
      logger.error('Couldnt find dataset parse class.')
      return

    # 2.step 配置记录器
    self.context.recorder = BrowserDataRecorder()

    # 3.step 启动浏览web服务
    browser_params = getattr(self.context.params, 'browser', None)
    tags = []
    if browser_params is not None:
      tags = browser_params.get('tags', [])

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

    # 3.3.step 状态
    train_offset, val_offset, test_offset = 0, 0, 0
    offset_configs = [{
      'dataset_flag': 'TRAIN',
      'dataset_offset': train_offset
    },{
      'dataset_flag': 'VAL',
      'dataset_offset': val_offset
    },{
      'dataset_flag': 'TEST',
      'dataset_offset': test_offset
    }]

    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'TRAIN' in self.context.params.browser['offset']:
        train_offset = self.context.params.browser['offset']['TRAIN']
        offset_config = {
          'dataset_flag': 'TRAIN',
          'dataset_offset': train_offset
        }
        offset_configs[0] = offset_config

    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'VAL' in self.context.params.browser['offset']:
        val_offset = self.context.params.browser['offset']['VAL']
        offset_config = {
          'dataset_flag': 'VAL',
          'dataset_offset': val_offset
        }
        offset_configs[1] = offset_config

    if self.context.params.browser is not None and 'offset' in self.context.params.browser:
      if 'TEST' in self.context.params.browser['offset']:
        test_offset = self.context.params.browser['offset']['TEST']
        offset_config = {
          'dataset_flag': 'TEST',
          'dataset_offset': test_offset
        }
        offset_configs[2] = offset_config

    dataset_flag = 'train'
    if self.context.params.browser is not None and 'dataset_flag' in self.context.params.browser:
      if self.context.params.browser['dataset_flag'].lower() in ['train', 'val', 'test']:
        dataset_flag = self.context.params.browser['dataset_flag'].lower()

    train_dataset = dataset_cls(dataset_flag, os.path.join(self.ant_data_source, self.dataset_name))
    self.context.recorder.dataset_flag = dataset_flag.upper()
    self.context.recorder.dataset_size = train_dataset.size
    self.context.recorder.dump_dir = os.path.join(self.dump_dir, 'browser', 'static', 'data')
    self.context.recorder.tag_dir = os.path.join(self.dump_dir, 'record')

    sample_offset = train_offset
    if dataset_flag == 'test':
      sample_offset = test_offset
    elif dataset_flag == 'val':
      sample_offset = val_offset

    if train_dataset.size == 0:
      logger.warn("Dont have waiting browser dataset(%s)."%dataset_flag)
      return

    logger.info('Browser %s dataset.' % dataset_flag)

    # 设置数据基本信息
    profile_config = {
      'dataset_flag': dataset_flag.upper(),
      'samples_num': train_dataset.size,
      'samples_num_checked': sample_offset
    }

    # 设置记录器偏移
    self.context.recorder.sample_index = sample_offset

    # 3.3.step 在线程中启动数据处理
    def _run_datagenerator_process():
      try:
        count = 0
        for data in self.context.data_generator(train_dataset):
          logger.info('Record data %d for browser.' % count)
          self.context.recorder.record(data)
          count += 1
      except:
        logger.info('Finish all records in browser %s dataset.' % dataset_flag)

    process = threading.Thread(target=_run_datagenerator_process)
    process.daemon = True
    process.start()

    # 3.4.step 启动web服务
    browser_mode = 'browser'
    if self.context.params.browser is not None and \
        'mode' in self.context.params.browser:
      browser_mode = self.context.params.browser['mode']

    browser_server_start(os.path.join(self.ant_data_source, self.dataset_name),
                         self.dump_dir,
                         self.context.recorder.queue,
                         tags,
                         self.host_port,
                         offset_configs,
                         profile_config,
                         browser_mode)
