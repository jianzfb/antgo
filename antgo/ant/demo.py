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
import socket
import zmq

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
    self.support_user_upload = kwargs.get('support_user_upload', False)
    self.support_user_input = kwargs.get('support_user_input', False)
    self.support_user_interaction = kwargs.get('support_user_interaction', False)
    self.support_upload_formats = kwargs.get('support_upload_formats', None)

  def start(self):
    # 0.step loading demo task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      challenge_task_config = self.rpc("TASK-CHALLENGE")
      if challenge_task_config is None:
        # invalid token
        logger.error('couldnt load challenge task')
        self.token = None
      elif challenge_task_config['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('prohibit submit challenge task frequently')
        exit(-1)
      elif challenge_task_config['status'] == 'OK':
        # maybe user token or task token
        if 'task' in challenge_task_config:
          challenge_task = create_task_from_json(challenge_task_config)
          if challenge_task is None:
            logger.error('couldnt load challenge task')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('unknow error')
        exit(-1)

    if running_ant_task is None:
      # 0.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(0)
      running_ant_task = custom_task

    assert (running_ant_task is not None)

    # 1.step prepare datasource and infer recorder
    demo_dataset = QueueDataset()
    demo_dataset._force_inputs_dirty()
    self.context.recorder = QueueRecorderNode(((), None), demo_dataset)

    # 2.step prepare dump dir
    now_time_stamp = datetime.fromtimestamp(self.time_stamp).strftime('%Y%m%d.%H%M%S.%f')
    infer_dump_dir = os.path.join(self.ant_dump_dir, now_time_stamp, 'inference')
    if not os.path.exists(infer_dump_dir):
      os.makedirs(infer_dump_dir)
    else:
      shutil.rmtree(infer_dump_dir)
      os.makedirs(infer_dump_dir)

    # 3.step start http server until it has been launched
    # 3.1.step check whether demo port has been occupied
    if self.demo_port is None:
      self.demo_port = 10000
    self.demo_port = _pick_idle_port(self.demo_port)
    logger.info('demo prepare using port %d'%self.demo_port)

    demo_name = running_ant_task.task_name
    demo_type = running_ant_task.task_type
    process = multiprocessing.Process(target=demo_server_start,
                                      args=(demo_name,
                                            demo_type,
                                            self.description_config,
                                            self.support_user_upload,
                                            self.support_user_input,
                                            self.support_user_interaction,
                                            self.support_upload_formats,
                                            infer_dump_dir,
                                            self.html_template,
                                            self.demo_port,
                                            os.getpid()))
    process.daemon = True
    process.start()

    # 4.step listening queue, wait client requery data
    logger.info('start model infer background process')
    ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
    for b in ablation_blocks:
      self.ant_context.deactivate_block(b)

    try:
      self.context.call_infer_process(demo_dataset, dump_dir=infer_dump_dir)
    except:
      logger.error('model infor error, please check your code')
