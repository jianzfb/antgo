# -*- coding: UTF-8 -*-
# Time: 10/8/17
# File: shell.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.ant.base import *
from antgo.ant import flags
from antgo.ant.challenge import *
from antgo.ant.train import *
from antgo.ant.utils import *
from antgo.resource.html import *
from antgo import config
from antgo.task.task import *
from antgo.measures import *
from antgo.measures.repeat_statistic import *
from antgo.utils import logger
from antvis.client.httprpc import *
from multiprocessing import Process
from prettytable import PrettyTable
import subprocess
import getopt
import json
import shutil
import sys
if sys.version > '3':
  PY3 = True
else:
  PY3 = False


FLAGS = flags.AntFLAGS
Config = config.AntConfig


class AntShell(AntBase):
  def __init__(self, ant_context, ant_token):
    flags.DEFINE_string('experiment_uuid', None, 'uuid')
    flags.DEFINE_string('dataset_name', None, 'dataset name')
    flags.DEFINE_string('apply_name', None, 'apply name')
    flags.DEFINE_string('task_name', None, 'task name')

    super(AntShell, self).__init__('SHELL', ant_context=ant_context, ant_token=ant_token)

  def process_task_command(self):
    task_name = FLAGS.task_name() # task name
    if task_name is None:
      response = self.context.dashboard.task.get()

      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      content = response['content']
      task_table = PrettyTable(["task", "time", "dataset", "apply", "publisher"])
      for task in content:
        task_table.add_row([task['task_name'],
                            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(task['task_time'])),
                            task['task_dataset'],
                            task['task_apply_num'],
                            task['task_publisher']])

      print(task_table)
    else:
      response = self.context.dashboard.apply.get(apply_name=task_name)

      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      content = response['content']
      task_table = PrettyTable(["uuid", "experiment", "time", "optimum", "report", "model"])
      for experiment in content['experiments']:
        task_table.add_row([experiment['experiment_uuid'],
                            experiment['experiment_name'],
                            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(experiment['experiment_time'])),
                            experiment['experiment_optimum'],
                            experiment['experiment_report'],
                            experiment['experiment_model']])

      print(task_table)

  def process_experiment_command(self):
    experiment_uuid = FLAGS.experiment_uuid()
    if experiment_uuid is None:
      response = self.context.dashboard.experiment.get()
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      content = response['content']

      table = PrettyTable(["uuid", "task", "experiment", "time", "optimum", "report", "model"])
      for experiment in content:
        table.add_row([experiment['experiment_uuid'],
                       experiment['experiment_task'],
                       experiment['experiment_name'],
                       time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment['experiment_time'])),
                       experiment['experiment_optimum'],
                       experiment['experiment_report'],
                       experiment['experiment_model']])

      print(table)
    else:
      response = self.context.dashboard.experiment.get(experiment_uuid=experiment_uuid)
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      experiment = response['content']
      if experiment['experiment_report']:
        # download
        # # to html
        # # read report and transform to resource
        # if os.path.exists(os.path.join(target_path, experiment_name)):
        #   fp = open(os.path.join(target_path, experiment_name), 'rb')
        #   report_data = fp.read()
        #   report_data = loads(report_data)
        #   fp.close()
        #   # clear temp report file
        #   os.remove(os.path.join(target_path, experiment_name))
        #
        #   if len(report_data) == 0:
        #     print('no experiment reports')
        #     return
        #   for stage, stage_report in report_data.items():
        #     target_path = os.path.join(os.curdir, 'experiment', experiment_name, 'report', stage)
        #     if not os.path.exists(target_path):
        #       os.makedirs(target_path)
        #
        #     everything_to_html(stage_report, target_path)
        pass

      if experiment['experiment_model']:
        # download experiment model
        pass

  def process_dataset_command(self):
    dataset_name = FLAGS.dataset_name()
    if dataset_name is None:
      response = self.context.dashboard.dataset.get()
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      datasets = response['content']
      table = PrettyTable(["name", "time", "task", "url", "publisher"])
      for dataset in datasets:
          table.add_row([
            dataset['dataset_name'],
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dataset['dataset_time'])),
            dataset['task_num'],
            dataset['dataset_url'],
            dataset['dataset_publisher']
          ])

      print(table)
    else:
      response = self.context.dashboard.dataset.get(dataset_name=dataset_name)
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      dataset = response['content']
      table = PrettyTable(["name", "time", "task", "url", "publisher"])
      table.add_row([
        dataset['dataset_name'],
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dataset['dataset_time'])),
        dataset['task_num'],
        dataset['dataset_url'],
        dataset['dataset_publisher']
      ])
      print(table)

  def process_apply_command(self):
    task_name = FLAGS.task_name()
    apply_name = FLAGS.apply_name()
    if task_name is None:
      # list all applied task
      response = self.context.dashboard.apply.get()
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      content = response['content']
      table = PrettyTable(["task", "apply", "time", "dataset", "experiment", "token"])

      for task in content:
        table.add_row([
          task['task_name'],
          task['apply_name'],
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['task_time'])),
          task['task_dataset'],
          task['task_experiment'],
          task['task_token']
        ])
      print(table)
    else:
      # apply a task
      if apply_name is None:
        apply_name = task_name

      response = self.context.dashboard.apply.post(apply_name=apply_name,
                               task_name=task_name)
      if response['status'] == "ERROR":
        logger.error(response['message'])
        return

      content = response['content']
      table = PrettyTable(["task", "apply", "time", "token"])
      table.add_row([
        content['task_name'],
        content['apply_name'],
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(content['apply_time'])),
        content['apply_token']
      ])
      print(table)

  def process_del_command(self):
    apply_name = FLAGS.apply_name()
    task_name = FLAGS.task_name()
    dataset_name = FLAGS.dataset_name()
    uuid = FLAGS.experiment_uuid()

    if apply_name is not None:
      response = self.context.dashboard.apply.delete(apply_name=apply_name)
      if response['status'] == 'ERROR':
        logger.error(response['message'])
        return

      logger.info(response['message'])

    if dataset_name is not None:
      response = self.context.dashboard.dataset.delete(dataset_name=dataset_name)
      if response['status'] == 'ERROR':
        logger.error(response['message'])
        return

      logger.info(response['message'])


    if task_name is not None:
      response = self.context.dashboard.task.delete(task_name=task_name)
      if response['status'] == 'ERROR':
        logger.error(response['message'])
        return

      logger.info(response['message'])

    if uuid is not None:
      response = self.context.dashboard.experiment.delete(experiment_uuid=uuid)
      if response['status'] == 'ERROR':
        logger.error(response['message'])
        return

      logger.info(response['message'])

  def _key_params(self):
    # related parameters
    # 0.step token
    token = FLAGS.token()
    if not PY3 and token is not None:
      token = unicode(token)
    token = self.app_token if token is None else token

    # 1.step check name, if None, set it as current time automatically
    name = FLAGS.name()
    if name is None:
      name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    if not PY3:
      name = unicode(name)

    # 2.step check main folder (all related model code, includes main_file and main_param)
    main_folder = FLAGS.main_folder()
    if main_folder is None:
      main_folder = os.path.abspath(os.curdir)

    main_file = FLAGS.main_file()
    if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
      logger.error('main executing file dont exist')
      return

    # 3.step check dump dir (all running data is stored here)
    dump_dir = FLAGS.dump()
    if dump_dir is None:
      dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # 4.step what is task
    task = FLAGS.task()

    # 5.step model params
    main_param = FLAGS.main_param()

    return token, name, main_file, main_folder, dump_dir, main_param, task

  def process_challenge_command(self):
    token, name, main_file, main_folder, dump_dir, params, task = self._key_params()
    challenge_cmd = 'antgo challenge'
    if token is not None:
      challenge_cmd += ' --token=%s'%token

    if name is not None:
      challenge_cmd += ' --name=%s'%name

    if main_file is not None:
      challenge_cmd += ' --main_file=%s'%main_file

    if main_folder is not None:
      challenge_cmd += ' --main_folder=%s'%main_folder

    if params is not None:
      challenge_cmd += ' --main_param=%s'%params

    if task is not None:
      challenge_cmd += ' --task=%s' % task

    if dump_dir is not None:
      challenge_cmd += ' --dump=%s'%dump_dir

    challenge_cmd += ' > %s-challenge.log 2>&1 &' % name

    # launch background process
    subprocess.call("nohup %s"%challenge_cmd, shell=True)

  def process_train_command(self):
    token, name, main_file, main_folder, dump_dir, params, task = self._key_params()
    train_cmd = 'antgo train'
    if token is not None:
      train_cmd += ' --token=%s' % token

    if name is not None:
      train_cmd += ' --name=%s' % name

    if main_file is not None:
      train_cmd += ' --main_file=%s' % main_file

    if main_folder is not None:
      train_cmd += ' --main_folder=%s' % main_folder

    if params is not None:
      train_cmd += ' --main_param=%s' % params

    if task is not None:
      train_cmd += ' --task=%s' % task

    if dump_dir is not None:
      train_cmd += ' --dump=%s' % dump_dir

    train_cmd += ' > %s-train.log 2>&1 &' % name

    # launch background process
    subprocess.call("nohup %s" % train_cmd, shell=True)

  def process_cmd(self, command):
    try:
      if command == 'task':
        self.process_task_command()
      elif command == 'experiment':
        self.process_experiment_command()
      elif command == 'dataset':
        self.process_dataset_command()
      elif command == 'apply':
        self.process_apply_command()
      elif command == 'del':
        self.process_del_command()
      elif command == 'challenge':
        self.process_challenge_command()
      elif command == 'train':
        self.process_train_command()
    except:
      logger.error('error response from server')

  def start(self):
    cmd = ''
    if PY3:
      cmd = input('antgo > ')
    else:
      cmd = raw_input('antgo > ')

    while cmd != 'quit':
      try:
        command = cmd.split(' ')
        assert(command[0] in ['task',
                              'dataset',
                              'experiment',
                              'apply',
                              'del',
                              'challenge',
                              'train'])

        flags.cli_param_flags(command[1:])

        # process user command
        self.process_cmd(command[0])

        # clear flags
        flags.clear_cli_param_flags()
      except:
        logger.error('error antgo command\n')

      if PY3:
        cmd = input('antgo > ')
      else:
        cmd = raw_input('antgo > ')