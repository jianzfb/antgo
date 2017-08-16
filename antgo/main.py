#encoding=utf-8
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
from antgo.utils import logger
from antgo.ant import flags
from antgo.config import *


def _main_context(main_file, source_paths):
  # filter .py
  key_model = main_file
  dot_pos = key_model.rfind(".")
  if dot_pos != -1:
    key_model = key_model[0:dot_pos]

  sys.path.append(source_paths)
  f, p, d = imp.find_module(key_model, [source_paths])
  module = imp.load_module('mm', f, p, d)
  return module.get_global_context()


def _check_environment():
  is_in_mltalker = True if os.environ.get('ANT_ENVIRONMENT','') != '' else False
  return is_in_mltalker

_ant_support_commands = ["train", "challenge", "compose","deploy"]

flags.DEFINE_string('main_file', None, 'main file')
flags.DEFINE_string('main_param', None, 'model parameters')
flags.DEFINE_string('main_folder', None, 'resource folder')
flags.DEFINE_string('task', None, 'task file')
flags.DEFINE_string('dump', None, 'dump dir')
flags.DEFINE_string('token', None, 'id')
flags.DEFINE_string('name', None, 'app name')
flags.DEFINE_string('config', 'config.xml', 'antgo config')

FLAGS = flags.AntFLAGS


def main():
  if len(sys.argv) == 1:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    sys.exit(-1)

  # 1.step parse running params
  flags.cli_param_flags()
  ant_cmd = sys.argv[1]

  # 2.step check parameters
  if ant_cmd == "":
    logger.error('antgo cli only support( %s )command'%",".join(_ant_support_commands))
    sys.exit(-1)

  main_folder = FLAGS.main_folder()
  if main_folder is None:
    main_folder = os.path.abspath(os.curdir)

  main_file = FLAGS.main_file()
  if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
    logger.error('main executing file dont exist')
    sys.exit(-1)

  dump_dir = FLAGS.dump()
  if dump_dir is None:
    dump_dir = os.path.join(os.curdir, 'dump')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

  # load antgo server config
  config_xml = os.path.join(os.path.split(os.path.realpath(__file__))[0], FLAGS.config())
  antgo_config = Config(config_xml)
  data_factory = getattr(antgo_config, 'data_factory', None)
  task_factory = getattr(antgo_config, 'task_factory', '')

  if data_factory is None:
    logger.error('must set data factory')
    sys.exit(-1)

  # client token
  token = FLAGS.token()

  # name
  name = FLAGS.name()
  if name is None:
    name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

  # 5.0 step custom workflow
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

  # 5.1 step ant running
  # 5.1.1 step what is task
  task = FLAGS.task()
  if task is not None:
    task = os.path.join(task_factory, task)

  # 5.1.2 step load ant context
  ant_context = _main_context(main_file, main_folder)

  # 5.1.3 step load model config
  main_param = FLAGS.main_param()
  if main_param is not None:
    main_config_path = os.path.join(main_folder, main_param)
    params = yaml.load(open(main_config_path, 'r'))
    ant_context.params = params

  if ant_cmd == "train":
    running_process = AntRun(ant_context,
                             name,
                             data_factory,
                             dump_dir,
                             token,
                             task)
    running_process.start()
  elif ant_cmd == 'challenge':
    running_process = AntChallenge(ant_context,
                                   name,
                                   data_factory,
                                   dump_dir,
                                   token,
                                   task)
    running_process.start()
  elif ant_cmd == "deploy":
    pass

  # 6.step clear context
  ant_context.wait_until_clear()

if __name__ == '__main__':
  main()
