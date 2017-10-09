# -*- coding: UTF-8 -*-
# Time: 10/8/17
# File: cmd.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.ant.base import *
from antgo.ant import flags
from antgo.html.html import *
import getopt
import json

FLAGS = flags.AntFLAGS


class AntCmd(AntBase):
  def __init__(self, ant_token):
    flags.DEFINE_integer('id', None, 'task or experiment id')
    flags.DEFINE_boolean('model', None, 'experiment main_file and main_param')
    flags.DEFINE_boolean('report', None, 'experiment report')
    flags.DEFINE_boolean('optimum', None, 'whether experiment is optimum')

    super(AntCmd, self).__init__('CMD', ant_token=ant_token)

  def process_challenge_command(self):
    task_id = FLAGS.id()        # task id
    if task_id is None:
      task_id = -1

    remote_api = 'hub/api/terminal/task/%d'%task_id
    response = self.remote_api_request(remote_api)

    if task_id == -1:
      print('%-5s %-10s %-30s %-10s %-10s'%('id','name','time','dataset','experiments'))

      for task in response:
        task_id = task['task-id']
        task_name = '-' if len(task['task-name']) == 0 else task['task-name']
        task_time = task['task-time']
        task_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task_time))
        task_dataset = task['task-dataset']
        task_experiments = task['task-experiment']
        print('%-5d %-10s %-30s %-10s %-10d'%(task_id,task_name,task_time,task_dataset,task_experiments))
    else:
      print('%-5s %-10s %-30s %-10s %-10s %-10s'%('id','name','time','optimum','report','model'))
      for experiment in response:
        experiment_id = experiment['experiment-id']
        experiment_name = experiment['experiment-name']
        experiment_time = experiment['experiment-time']
        experiment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment_time))
        experiment_optimum = experiment['experiment-optimum']
        experiment_report = experiment['experiment-report']
        experiment_model = experiment['experiment-model']

        print('%-5d %-10s %-30s %-10d %-10d %-10d'%
              (experiment_id,
               experiment_name,
               experiment_time,
               experiment_optimum,
               experiment_report,
               experiment_model))

  def process_experiment_command(self):
    experiment_id = FLAGS.id()
    experiment_new_name = FLAGS.name()
    experiment_download_model = FLAGS.model()
    experiment_download_report = FLAGS.report()
    experiment_set_optimum = FLAGS.optimum()

    if experiment_id is None:
      print('must set experiment id')
      return

    # remote api
    remote_api = 'hub/api/terminal/experiment/%d' % experiment_id
    response = self.remote_api_request(remote_api,
                                       action='get')
    if response is None or len(response) == 0:
      print('no experiment')
      return

    experiment_name = response['experiment-name']
    experiment_time = response['experiment-time']
    experiment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment_time))
    experiment_task = response['experiment-task']
    experiment_report = response['experiment-report']
    experiment_model = response['experiment-model']
    experiment_optimum = response['experiment-optimum']

    if experiment_new_name is not None:
      # rename experiment name request
      remote_api = 'hub/api/terminal/experiment/%d' % experiment_id
      response = self.remote_api_request(remote_api,
                                         data={'experiment-name':experiment_new_name},
                                         action='patch')
      if response is not None and len(response) > 0:
        print('experiment id="%d" name has been changed from %s to %s'%
              (experiment_id,response['old-experiment-name'], response['experiment-name']))
    elif experiment_download_model is not None:
      # download experiment model request
      remote_api = 'hub/api/terminal/download/experiment/%d/model' % experiment_id
      url = '%s://%s:%s/%s' % (self.http_prefix, self.root_ip, self.http_port, remote_api)

      target_path = os.path.join(os.curdir,'experiment', experiment_name)
      if not os.path.exists(target_path):
        os.makedirs(target_path)

      self.download(url,
                    target_path=target_path,
                    target_name='%s.tar.gz'%experiment_name,
                    archive='model',
                    data=None)
    elif experiment_download_report is not None:
      # experiment report at every stage of experiment
      if len(experiment_report) == 0:
        print('no experiment reports')
        return
      for stage, stage_report in experiment_report.items():
        target_path = os.path.join(os.curdir, 'experiment', experiment_name,'report', stage)
        if not os.path.exists(target_path):
          os.makedirs(target_path)

        everything_to_html(stage_report, target_path)
    elif experiment_set_optimum is not None:
      # set experiment optimum
      remote_api = 'hub/api/terminal/experiment/%d' % experiment_id
      response = self.remote_api_request(remote_api,
                                         data={'experiment-optimum':1},
                                         action='patch')
      if response is not None and len(response) > 0:
        print('experiment id="%s" optimum status has been changed from %d to %d'%
              (experiment_id,response['old-experiment-optimum'], response['experiment-optimum']))
    else:
      print('%-5s %-10s %-10s %-30s %-10s %-10s %-10s' % ('id', 'name', 'task','time', 'optimum', 'report', 'model'))
      experiment_report_num = len(experiment_report)
      print('%-5d %-10s %-10s %-30s %-10d %-10d %-10d' %
            (experiment_id,
             experiment_name,
             experiment_task,
             experiment_time,
             experiment_optimum,
             experiment_report_num,
             experiment_model))

  def process_cmd(self, command):
    try:
      if command == 'challenge':
        self.process_challenge_command()
      elif command == 'experiment':
        self.process_experiment_command()
    except:
      print('error response from server')

  def start(self):
    cmd = raw_input('antgo > ')
    while cmd != 'quit':
      try:
        command = cmd.split(' ')
        assert(command[0] in ['challenge', 'experiment'])

        flags.cli_param_flags(command[1:])

        # process user command
        self.process_cmd(command[0])

        # clear flags
        flags.clear_cli_param_flags()
      except:
        print('error antgo command\n')
      cmd = raw_input('antgo > ')