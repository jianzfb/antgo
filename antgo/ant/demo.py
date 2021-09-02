# -*- coding: UTF-8 -*-
# @Time    : 18-4-27
# @File    : demo.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.dataflow.dataset.queue_dataset import *
from antgo.dataflow.recorder import *
from antgo.crowdsource.demo_server import *
from antgo.utils import logger
from antvis.client.httprpc import *
try:
    import queue
except:
    import Queue as queue
import traceback

class AntDemo(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_dump_dir,
               ant_token,
               ant_task_config,
               **kwargs):
    super(AntDemo, self).__init__(ant_name, ant_context, ant_token, **kwargs)

    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config

    self.html_template = kwargs.get('html_template', None)
    self.demo_port = kwargs.get('port', None)
    self.demo_port = int(self.demo_port) if self.demo_port is not None else None
    # self.support_user_upload = kwargs.get('support_user_upload', False)
    # self.support_user_input = kwargs.get('support_user_input', False)
    # self.support_user_interaction = kwargs.get('support_user_interaction', False)
    # self.support_user_constraint = kwargs.get('support_user_constraint', None)
    self.context.devices = [int(d) for d in kwargs.get('devices', '').split(',') if d != '']

  def start(self):
    # 1.step loading demo task
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

    if running_ant_task is None:
      # 1.2.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(-1)
      running_ant_task = custom_task

    assert (running_ant_task is not None)

    # 2.step 注册实验
    experiment_uuid = self.context.experiment_uuid
    # 3.step 配置数据传输管道
    dataset_queue = queue.Queue()

    demo_dataset = QueueDataset(dataset_queue)
    demo_dataset._force_inputs_dirty()

    # 3.step 配置dump路径
    infer_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid, 'inference')
    if not os.path.exists(infer_dump_dir):
      os.makedirs(infer_dump_dir)
    else:
      shutil.rmtree(infer_dump_dir)
      os.makedirs(infer_dump_dir)

    # 4.step 启动web服务
    # 4.1.step 选择合适端口
    if self.demo_port is None:
      self.demo_port = 10000
    self.demo_port = _pick_idle_port(self.demo_port)
    logger.info('Demo prepare using port %d.'%self.demo_port)

    # 后台api调用
    self.rpc = HttpRpc("v1", "api", "127.0.0.1", self.demo_port)

    demo_name = running_ant_task.task_name
    demo_type = running_ant_task.task_type
    demo_config = {
      'interaction':{
        'support_user_upload': True,
        'support_user_input': True,
        'support_user_interaction': False,
        'support_user_constraint': {}
      },
      'description_config': self.description_config,
      'port': self.demo_port,
      'html_template': self.html_template,
      'dump_dir': infer_dump_dir
    }

    request_waiting_time = 30
    if self.context.params.demo is not None:
      if 'support_user_upload' in self.context.params.demo:
        demo_config['interaction']['support_user_upload'] = self.context.params.demo['support_user_upload']

      if 'support_user_input' in self.context.params.demo:
        demo_config['interaction']['support_user_input'] = self.context.params.demo['support_user_input']

      if 'support_user_interaction' in self.context.params.demo:
        demo_config['interaction']['support_user_interaction'] = self.context.params.demo['support_user_interaction']

      if 'support_user_constraint' in self.context.params.demo:
        demo_config['interaction']['support_user_constraint'] = self.context.params.demo['support_user_constraint']

      if 'request_waiting_time' in self.context.params.demo:
        request_waiting_time = self.context.params.demo['request_waiting_time']

    # 5.step 启动后台线程运行预测过程
    def _run_infer_process():
      # prepare ablation blocks
      logger.info('Prepare model ablation blocks.')
      ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
      if ablation_blocks is None:
        ablation_blocks = []
      for b in ablation_blocks:
        self.ant_context.deactivate_block(b)

      # infer
      logger.info('Running inference process.')
      try:
        def _callback_func(data):
            id = demo_dataset.now()['id'] if 'id' in demo_dataset.now() else ''
            record_content = {
                        'experiment_uuid': experiment_uuid, 
                        'data': data,
                        'id': id
                    }
            for data_group in record_content['data']:
              for item in data_group:
                  if item['type'] in ['IMAGE', 'VIDEO', 'FILE']:
                      item['data'] = '%s/record/%s'%(infer_dump_dir, item['data'])
                
            self.rpc.response.post(response=json.dumps(record_content))
                    
        self.context.recorder =  LocalRecorderNodeV2(_callback_func)
        self.context.call_infer_process(demo_dataset, dump_dir=infer_dump_dir)
      except Exception as e:
        print(e)
        traceback.print_exc()

    process = threading.Thread(target=_run_infer_process)
    process.daemon = True
    process.start()

    # 6.step 启动web服务
    demo_server_start(demo_name,demo_type,demo_config,os.getpid(),dataset_queue, request_waiting_time)
