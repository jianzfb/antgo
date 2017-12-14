# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import imp
import os
import sys
import getopt
import yaml
from antgo.ant.train import *
from antgo.ant.deploy import *
from antgo.ant.workflow import *
from antgo.ant.challenge import *
from antgo.ant.generate import *
from antgo.ant.cmd import *
from antgo.utils import logger
from antgo.ant import flags
from antgo import config
from antgo.ant.utils import *
from antgo.sandbox.sandbox import *
from antgo.dataflow.dataflow_server import *
from datetime import datetime
from antgo.utils.utils import *
if sys.version > '3':
    PY3 = True
else:
    PY3 = False


def _check_environment():
  is_in_mltalker = True if os.environ.get('ANT_ENVIRONMENT', '') != '' else False
  return is_in_mltalker

_ant_support_commands = ["train", "challenge", "compose", "deploy", "dataset", "frozen"]

flags.DEFINE_string('main_file', None, 'main file')
flags.DEFINE_string('main_param', None, 'model parameters')
flags.DEFINE_string('main_folder', None, 'resource folder')
flags.DEFINE_string('task', None, 'task file')
flags.DEFINE_string('dataset', None, 'dataset')
flags.DEFINE_string('dump', None, 'dump dir')
flags.DEFINE_string('token', None, 'token')
flags.DEFINE_string('platform', 'local', 'local or cloud')
flags.DEFINE_string('sandbox_time', None, 'max running time')
flags.DEFINE_string('from_experiment', None, 'load model from experiment')
flags.DEFINE_string('input_nodes', '', 'input node names in graph')
flags.DEFINE_string('output_nodes', '', 'output node names in graph')

FLAGS = flags.AntFLAGS
Config = config.AntConfig


def main():
  if len(sys.argv) == 1:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    sys.exit(-1)

  # 0.step antgo global config
  config_xml = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'config.xml')
  Config.parse_xml(config_xml)

  # check data_factory and task_factory config
  data_factory = getattr(Config, 'data_factory', None)
  task_factory = getattr(Config, 'task_factory', None)

  if data_factory is None:
    logger.error('must set data factory in config.xml')
    sys.exit(-1)

  if task_factory is None:
    logger.error('must set task factory in config.xml')
    sys.exit(-1)
  
  if not os.path.exists(data_factory):
    os.makedirs(data_factory)

  if not os.path.exists(task_factory):
    os.makedirs(task_factory)

  # 1.step parse running params
  if sys.argv[1].startswith('--') or sys.argv[1].startswith('-'):
    flags.cli_param_flags(sys.argv[1:])
  else:
    flags.cli_param_flags(sys.argv[2:])

  # 2.step
  # 2.1 step key 1: antgo token (secret)
  token = FLAGS.token()
  if not PY3 and token is not None:
    token = unicode(token)

  # 2.2 step key 2: antgo daemon (data server)
  # dataflow_server_host = getattr(Config, 'dataflow_server_host', 'tcp://127.0.0.1:9999')
  # dataflow_server_threads = getattr(Config, 'dataflow_server_threads', 1)
  # dfs_daemon = DataflowServerDaemon(int(dataflow_server_threads), dataflow_server_host, 'antgo-data-server.pid')
  # dfs_daemon.start()

  # 3.step parse execute command
  if sys.argv[1].startswith('--') or sys.argv[1].startswith('-'):
    # interactive control
    cmd_process = AntCmd(token)
    cmd_process.start()
    return

  # other command (train, challenge, compose, deploy, server)
  ant_cmd = sys.argv[1]
  if ant_cmd not in _ant_support_commands:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    return

  # # 4.step check related params
  # # 4.1 step check name, if None, set it as current time automatically
  # name = FLAGS.name()
  # if name is None:
  #   name = datetime.now().strftime('%Y%m%d.%H%M%S.%f')
  # time_stamp = datetime.now().timestamp()
  time_stamp = timestamp()
  name = datetime.fromtimestamp(time_stamp).strftime('%Y%m%d.%H%M%S.%f')
  if not PY3:
    name = unicode(name)
  
  # 4.2 check main folder (all related model code, includes main_file and main_param)
  main_folder = FLAGS.main_folder()
  if main_folder is None:
    main_folder = os.path.abspath(os.curdir)
  
  # 4.3 check dump dir (all running data is stored here)
  dump_dir = FLAGS.dump()
  if dump_dir is None:
    dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

  # 4.4. step tools (option)
  if ant_cmd == 'frozen':
    # tensorflow tools
    import antgo.utils.tftools as tftools
    tftools.tftool_frozen_graph(dump_dir,
                                FLAGS.from_experiment(),
                                FLAGS.input_nodes(),
                                FLAGS.output_nodes())
    return
  
  # 4.5 check main file
  main_file = FLAGS.main_file()
  if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
    logger.error('main executing file dont exist')
    sys.exit(-1)

  # 5. step custom workflow
  if ant_cmd == 'compose':
    # user custom workflow
    work_flow = WorkFlow(name,
                         token,
                         yaml.load(open(os.path.join(main_folder, FLAGS.main_param()), 'r')),
                         main_file,
                         main_folder,
                         dump_dir,
                         data_factory)
    work_flow.start()
    return

  # 6 step ant running
  # 6.1 step what is task
  task = FLAGS.task()
  if task is not None:
    task = os.path.join(task_factory, task)
  dataset = FLAGS.dataset()
  if task is None and dataset is not None:
    # build default task
    with open(os.path.join(task_factory, '%s.xml'%name), 'w') as fp:
      task_content = '<task><task-name>%s</task-name>' \
                     '<input>' \
                     '<source_data><data_set_name>%s</data_set_name>' \
                     '</source_data>' \
                     '</input>' \
                     '</task>'%('default-task', dataset)
      fp.write(task_content)
    task = os.path.join(task_factory, '%s.xml'%name)
  
  # 6.2 step load ant context
  ant_context = main_context(main_file, main_folder)
  if FLAGS.from_experiment() is not None:
    if os.path.exists(os.path.join(dump_dir, FLAGS.from_experiment(), 'train')):
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'train')
    else:
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'inference')

  # 6.3 step load model config
  main_param = FLAGS.main_param()
  if main_param is not None:
    main_config_path = os.path.join(main_folder, main_param)
    params = yaml.load(open(main_config_path, 'r'))
    ant_context.params = params
  
  if ant_cmd == "train":
    sandbox_time = FLAGS.sandbox_time()
    with running_sandbox(sandbox_time=sandbox_time):
      running_process = AntTrain(ant_context,
                                 name,
                                 data_factory,
                                 dump_dir,
                                 token,
                                 task,
                                 main_file=main_file,
                                 main_folder=main_folder,
                                 main_param=main_param,
                                 time_stamp=time_stamp)
      running_process.start()
  elif ant_cmd == 'challenge':
    running_process = AntChallenge(ant_context,
                                   name,
                                   data_factory,
                                   dump_dir,
                                   token,
                                   task,
                                   main_file=main_file,
                                   main_folder=main_folder,
                                   main_param=main_param,
                                   time_stamp=time_stamp)
    running_process.start()
  elif ant_cmd == "deploy":
    pass
  elif ant_cmd == 'dataset':
    running_process = AntGenerate(ant_context,
                                  name,
                                  data_factory,
                                  token)
    running_process.start()

  # 7.step clear context
  ant_context.wait_until_clear()

if __name__ == '__main__':
  main()
