# -*- coding: UTF-8 -*-
# @Time    : 2020/10/26 10:11 上午
# @File    : watch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.task.task import *
from antgo.dataflow.common import *
from antgo.dataflow.basic import *
from antgo.dataflow.recorder import *
# from antgo.crowdsource.watch_server import *
from antvis.client.httprpc import *
import json
from antgo.dataflow.dataset.spider_dataset import *
try:
    import queue
except:
    import Queue as queue


class WatchDataRecorder(object):
  def __init__(self, maxsize=30):
    self.queue = queue.Queue()  # 不设置队列最大缓冲
    self.dump_dir = ''
    self.dataset_size = 0
    self.sample_index = 0
    self.tag_dir = ''

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

      body['dataset_size'] = self.dataset_size
      web_data.append(body)

    # # 加入队列，如果队列满，将阻塞
    # self.queue.put(web_data)
    # rpc 调用

    # 增加样本索引编号
    self.sample_index += 1


class AntWatch(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_host_ip,
               ant_host_port,
               token,
               data_factory,
               ant_dump_dir,
               ant_task_config, **kwargs):
    super(AntWatch, self).__init__(ant_name, ant_context, token)
    self.ant_data_source = data_factory
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config
    self.context.devices = [int(d) for d in kwargs.get('devices', '').split(',') if d != '']
    self.restore_experiment = kwargs.get('restore_experiment', None)
    self.host_ip = ant_host_ip
    self.host_port = ant_host_port
    self.rpc = None

  def start(self):
    # 1.step 加载挑战任务
    running_ant_task = None
    if self.token is not None:
      # 1.1.step 从平台获取挑战任务配置信息
      response = mlogger.info.challenge.get(command=type(self).__name__)
      if response['status'] == 'ERROR':
        logger.error('couldnt load challenge task')
        self.token = None
      elif response['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('prohibit submit challenge task frequently')
        exit(-1)
      elif response['status'] == 'OK':
        content = response['content']

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
      # 1.2.step 加载自定义任务配置信息
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(-1)
      running_ant_task = custom_task

    assert (running_ant_task is not None)

    # 2.step 获得实验ID
    experiment_uuid = self.context.experiment_uuid
    if self.restore_experiment is not None:
      experiment_uuid = self.restore_experiment

    # 3.step make experiment folder
    logger.info('build experiment folder')
    if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
      os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))
    experiment_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid)

    # 4.step 配置web服务基本信息
    # 选择端口
    self.host_port = _pick_idle_port(self.host_port)

    # 配置调用
    self.rpc = HttpRpc("v1", "watch-api", "127.0.0.1", self.host_port)

    # 准备素材资源
    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    batch_static_dir = os.path.join(experiment_dump_dir, 'watch')
    if not os.path.exists(batch_static_dir):
      shutil.copytree(os.path.join(static_folder, 'resource', 'watch'), batch_static_dir)

    # 5.step 配置运行
    self.context.recorder = WatchDataRecorder()
    def _run_watch_process():
      # 5.1.step prepare ablation blocks
      logger.info('prepare model ablation blocks')
      ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
      if ablation_blocks is None:
        ablation_blocks = []
      for b in ablation_blocks:
        self.ant_context.deactivate_block(b)

      # 5.2.step 创建数据源
      logger.info('build spider data source')
      ant_watch_dataset = SpiderDataset(queue)
      data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_watch_dataset))

      # 5.3.step 启动推断过程
      logger.info('launch predict process')
      output_dir = os.path.join(experiment_dump_dir, 'batch', 'static', 'data', 'dump')
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      with safe_recorder_manager(self.context.recorder):
        self.context.call_infer_process(data_annotation_branch.output(0), dump_dir=output_dir)

    process = threading.Thread(target=_run_watch_process)
    process.daemon = True
    process.start()

    # 6.step 启动bach server
    watch_server_start(experiment_dump_dir, self.host_port, queue)