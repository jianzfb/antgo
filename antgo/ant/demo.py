# -*- coding: UTF-8 -*-
# @Time    : 18-4-27
# @File    : demo.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.ant.base import *
from antgo.dataflow.dataset.queue_dataset import *
from antgo.dataflow.recorder import *
from antgo.crowdsource.demo_server import *
from antgo.utils import logger
from multiprocessing import Process, Queue
import socket


def _is_open(check_ip, port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((check_ip, int(port)))
    s.shutdown(2)
    return True
  except:
    return False


def _pick_idle_port(from_port=40000, check_count=100):
  check_port = from_port
  while check_count:
    if not _is_open('127.0.0.1', check_port):
      break

    logger.warn('port %d is occupied, try to use %d port'%(int(check_port), int(check_port + 1)))

    check_port += 1
    check_count -= 1

    if check_count == 0:
      check_port = None

  if check_port is None:
    logger.warn('couldnt find valid free port')
    exit(-1)

  return check_port


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
      response = self.context.dashboard.challenge.get(command=type(self).__name__)
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
    dataset_queue = Queue()

    demo_dataset = QueueDataset(dataset_queue)
    demo_dataset._force_inputs_dirty()

    recorder_queue = Queue()
    self.context.recorder = QueueRecorderNode(((), None), recorder_queue)

    self.context.recorder.dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid, 'recorder')
    if not os.path.exists(self.context.recorder.dump_dir):
      os.makedirs(self.context.recorder.dump_dir)

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
    logger.info('demo prepare using port %d'%self.demo_port)

    demo_name = running_ant_task.task_name
    demo_type = running_ant_task.task_type
    demo_config = {
      'interaction':{
        'support_user_upload': False,
        'support_user_input': False,
        'support_user_interaction': False,
        'support_user_constraint': {}
      },
      'description_config': self.description_config,
      'port': self.demo_port,
      'html_template': self.html_template,
      'dump_dir': infer_dump_dir
    }

    if self.context.params.demo is not None:
      if 'support_user_upload' in self.context.params.demo:
        demo_config['interaction']['support_user_upload'] = self.context.params.demo['support_user_upload']

      if 'support_user_input' in self.context.params.demo:
        demo_config['interaction']['support_user_input'] = self.context.params.demo['support_user_input']

      if 'support_user_interaction' in self.context.params.demo:
        demo_config['interaction']['support_user_interaction'] = self.context.params.demo['support_user_interaction']

      if 'support_user_constraint' in self.context.params.demo:
        demo_config['interaction']['support_user_constraint'] = self.context.params.demo['support_user_constraint']

    process = Process(target=demo_server_start,
                                      args=(demo_name,
                                            demo_type,
                                            demo_config,
                                            os.getpid(),
                                            dataset_queue,
                                            recorder_queue))
    process.daemon = True
    process.start()

    # 5.step 启动推断服务，等待客户端请求
    logger.info('start model infer background process')
    ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
    if ablation_blocks is None:
      ablation_blocks = []
    for b in ablation_blocks:
      self.ant_context.deactivate_block(b)

    try:
      self.context.call_infer_process(demo_dataset, dump_dir=infer_dump_dir)
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]
