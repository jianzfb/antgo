# -*- coding: UTF-8 -*-
# @Time    : 2018/11/27 2:46 PM
# @File    : train_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from tornado import web, gen
from tornado import httpclient
from antgo.utils import logger
from antgo.utils.utils import *
from datetime import datetime
import tornado.web
import multiprocessing
import json
import os
import uuid
import time
import yaml
import subprocess
import traceback
import sys
import shutil
import functools
import zmq
from zmq.eventloop import future
import requests


class BaseHandler(tornado.web.RequestHandler):
  @property
  def signature(self):
    return self.settings.get('signature', '')

  @property
  def main_folder(self):
    return self.settings.get('main_folder', '')

  @property
  def main_param(self):
    return self.settings.get('main_param', None)

  @property
  def main_file(self):
    return self.settings.get('main_file', '')

  @property
  def experiment_records(self):
    # {'id': {'study_name':, 'trail_name': , 'start_time': ,'stop_time':, 'measure': }}
    return self.settings.get('experiment_records', {})

  @property
  def file_records(self):
    return self.settings.get('file_records', {})

  @property
  def server_records(self):
    return self.settings.get('server_records', {})

  @property
  def device_list(self):
    return self.settings.get('device_list', [])

  @property
  def max_time(self):
    return self.settings.get('max_time', '10d')

  @property
  def server_port(self):
    return self.settings.get('server_port', 0)

  @property
  def is_worker(self):
    return self.settings.get('is_worker', True)

  @property
  def client_socket(self):
    return self.settings.get('client_socket', None)

  @property
  def html_template(self):
    return self.settings['html_template']


def update_suggestion_process(server_records, experiment_records):
  running_experiments = []
  for experiment_id, experiment_record in experiment_records.items():
    if experiment_record['status'] != 'stop':
      # update trail status
      running_experiments.append(experiment_id)

  if len(running_experiments) == 0:
    return

  requests.post('http://%s/update/' % server_records['master_ip'],
                data={'experiments': json.dumps(running_experiments),
                      'signature': server_records['signature']})


def launch_train_process(server_records, experiment_records, content):
  # 1.step check device resource
  if 'occupied_devices' not in server_records:
    server_records['occupied_devices'] = []

  free_devices = [n for n in server_records['devices'] if n not in server_records['occupied_devices']]
  if len(free_devices) == 0:
    logger.error('have no free device resource')
    return

  # record
  study_name = content['study_name']
  trail_name = content['trail_name']

  experiment_id = trail_name
  start_time = time.time()
  main_param = json.loads(content['hyperparameter'])
  structure = content['structure']
  max_runtime = content['max_time']

  if experiment_id in experiment_records:
    logger.error('has existed experiment')

  # task token
  experiment_records[experiment_id] = {'start_time': start_time,
                                       'study_name': study_name,
                                       'trail_name': trail_name,
                                       'automl': {'graph': structure},
                                       'main_config': main_param,
                                       'max_time': max_runtime,
                                       'main_file': '',
                                       'main_param': '',
                                       'task': '',
                                       'token': '',
                                       'status': 'prepare',
                                       'devices': [],
                                       'evaluation_value': [],
                                       'evaluation_time': [],
                                       'pid': None}

  # apply devices
  apply_devices = 1
  if apply_devices == 0:
    apply_devices = 1

  if 'num_clones' in main_param:
    apply_devices = int(main_param['num_clones'])

  if apply_devices > len(free_devices):
    logger.error('have no free device resource')

    # remove experiment record
    experiment_records.pop(experiment_id)
    return False

  # 2.step prepare running environment
  # prepare workspace
  os.makedirs(os.path.join(server_records['main_folder'], experiment_id))

  # prepare main param
  if server_records['main_param'] is not None and server_records['main_param'] != '':
    with open(os.path.join(server_records['main_folder'], server_records['main_param']), 'r') as fp:
      # load basic parameter
      main_param.update(yaml.load(fp))

  experiment_devices = free_devices[0:apply_devices]
  server_records['occupied_devices'].extend(experiment_devices)
  experiment_records[experiment_id]['devices'] = experiment_devices
  main_param.update({'devices': experiment_devices})
  main_param.update({'automl': {'graph': structure}})

  with open(os.path.join(server_records['main_folder'], experiment_id, 'main_param.yaml'), 'w') as fp:
    fp.write(yaml.dump(main_param))

  experiment_records[experiment_id]['main_param'] = os.path.join(server_records['main_folder'],
                                                                 experiment_id,
                                                                 '%s.yaml' % experiment_id)

  # prepare main file
  main_file = 'main_file.py'
  shutil.copy(os.path.join(server_records['main_folder'], server_records['main_file']),
              os.path.join(server_records['main_folder'], experiment_id, 'main_file.py'))

  experiment_records[experiment_id]['main_file'] = os.path.join(server_records['main_folder'],
                                                                experiment_id,
                                                                'main_file.py')

  # run script
  cmd_shell = 'antgo train --main_file=%s --main_param=%s' % (main_file, 'main_param.yaml')
  cmd_shell += ' --main_folder=%s' % os.path.join(server_records['main_folder'], experiment_id)
  cmd_shell += ' --dump=%s' % os.path.join(server_records['main_folder'], experiment_id, 'dump')
  cmd_shell += ' --max_time=%s' % max_runtime
  cmd_shell += ' --signature=%s' % server_records['signature']
  cmd_shell += ' --proxy=%s' % ('127.0.0.1:%d' % server_records['server_port'])
  cmd_shell += ' --name=%s' % experiment_id

  # prepare task xml file
  if server_records['token'] is None:
    shutil.copy(server_records['task'], os.path.join(server_records['main_folder'], experiment_id, 'task.xml'))
    cmd_shell += ' --task=task.xml'
  else:
    cmd_shell += ' --token=%s' % server_records['token']

  # start running
  p = subprocess.Popen('%s > %s.log' % (cmd_shell, experiment_id), shell=True)
  experiment_records[experiment_id]['pid'] = p
  return True


def request_suggestion_process(experiment_records, server_records):
  # based free devices request suggestion
  if server_records['master_ip'] is not None:
    # 1.step check device resource
    if 'occupied_devices' not in server_records:
      server_records['occupied_devices'] = []

    study_name = None
    if 'study_name' in server_records:
      study_name = server_records['study_name']

    if study_name is None:
      return

    # 2.step check completed experiment
    new_experiments = {}
    for experiment_id, experiment_record in experiment_records.items():
      if experiment_record['status'] != 'stop':
        train_p = experiment_record['pid']

        if train_p.poll() == 0:
          train_p.wait()
          # release occupied devices
          free_devices = experiment_record['devices']
          server_records['occupied_devices'] = \
            [n for n in server_records['occupied_devices'] if n not in free_devices]

          # modify experiment status
          experiment_record['status'] = 'stop'

          # launch new trail
          trail_name = experiment_record['trail_name']
          objective_value = experiment_record['evaluation_value'][-1] if len(experiment_record['evaluation_value']) > 0 else -1.0
          objective_value = float(objective_value)
          logger.info('experiment id %s stop evaluation value %f' % (experiment_id, objective_value))

          response = requests.post('http://%s/server/' % server_records['master_ip'],
                                   data={'study_name': study_name,
                                         'trail_name': trail_name,
                                         'objective_value': objective_value,
                                         'signature': server_records['signature']})
          content = json.loads(response.content)
          if len(content) == 0:
            continue

          if content['status'] == 'running':
            temp = {}
            result = launch_train_process(server_records, temp, content)
            if result:
              new_experiments.update(temp)

    experiment_records.update(new_experiments)

    # 3.step cold start
    free_devices = [n for n in server_records['devices'] if n not in server_records['occupied_devices']]
    if len(free_devices) == 0:
      return

    if server_records['study_name'] is None:
      return

    response = requests.post('http://%s/server/' % server_records['master_ip'],
                             data={'study_name': study_name,
                                   'trail_name': None,
                                   'objective_value': None,
                                   'signature': server_records['signature']})
    content = json.loads(response.content)
    if len(content) == 0:
      return

    if content['status'] == 'running':
      launch_train_process(server_records, experiment_records, content)


class UpdateModelHandler(BaseHandler):
  @gen.coroutine
  def post(self, experiment_id):
    # check signature
    signature = self.get_argument('signature', '')
    if self.signature != signature:
      logger.error('signature not consistent %s'%signature)
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSignature'}))
      self.finish()
      return

    if experiment_id not in self.experiment_records:
      logger.error('no experiemnt %s here'%experiment_id)
      self.set_status(404)
      self.write(json.dumps({'code': 'InvalidInput', 'message': 'dont have experiment %s'%experiment_id}))
      self.finish()
      return

    self.experiment_records[experiment_id]['status'] = 'running'

    # update model evaluate value
    evaluation_val = self.get_argument('evaluation_value', None)
    if evaluation_val is not None:
      self.experiment_records[experiment_id]['evaluation_value'].append(evaluation_val)
      self.experiment_records[experiment_id]['evaluation_time'].append(time.time())

    # do other things
    status = self.get_argument('status', None)
    if status is not None:
      self.experiment_records[experiment_id]['status'] = status

      if status == 'stop':
        free_devices = self.experiment_records[experiment_id]['devices']
        self.server_records['occupied_devices'] = [n for n in self.server_records['occupied_devices'] if n not in free_devices]


class IndexHanlder(BaseHandler):
  @gen.coroutine
  def get(self):
    if self.is_worker:
      self.write('hello worker')
      self.finish()
    else:
      self.client_socket.send_json({'cmd': 'study/all'})
      response = yield self.client_socket.recv_json()

      if len(response) == 0:
        self.set_status(500)
        self.finish()
        return

      studys = response['result']

      study_infos = []
      for s_i, s in enumerate(studys):
        study_name, study_created_time, study_id, study_status = s
        self.client_socket.send_json({'cmd': 'study/get',
                                      'study_name': study_name})
        rr = yield self.client_socket.recv_json()
        if len(rr) == 0:
          self.set_status(500)
          self.finish(500)
          return

        if rr['status'] != 'ok':
          self.set_status(500)
          self.finish(500)
          return

        trials = rr['result']

        study_info = {}
        # get completed_trial list
        study_info['completed_trial'] = len(list(filter(lambda x: x[2] == 'Completed', trials)))

        # get error_trial list
        study_info['error_trial'] = len(list(filter(lambda x: x[2] == 'Failed', trials)))

        # get uncompleted_trial list
        study_info['uncompleted_trial'] = len(trials) - study_info['completed_trial'] - study_info['error_trial']

        study_info['name'] = study_name
        study_info['index'] = s_i
        study_info['objective_value'] = trials[0][-1] if len(trials) > 0 else -1
        study_info['status'] = study_status
        study_info['created_time'] = '-' if study_created_time is None else datetime.fromtimestamp(study_created_time).strftime('%Y-%m-%d')

        study_infos.append(study_info)

      self.render(self.html_template,automl={'study': study_infos})


class StudyStartorStopHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    study_name = self.get_argument('study_name', '')

    self.client_socket.send_json({'cmd': 'study/startorstop',
                                  'study_name': study_name})

    response = yield self.client_socket.recv_json()
    if response['status'] == 'ok':
      self.write(json.dumps({'status': 'ok', 'study_status': response['study_status']}))
      self.finish()
    else:
      self.write(json.dumps({'status': 'fail'}))
      self.finish()


class StudyGetHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    study_name = self.get_argument('study_name', '')

    self.client_socket.send_json({'cmd': 'study/get',
                                  'study_name': study_name})

    response = yield self.client_socket.recv_json()
    trials = response['result']

    trials_list = [{'trial_name': t[0],
                    'trial_created_time': datetime.fromtimestamp(t[1]).strftime('%Y-%m-%d %H:%M:%S'),
                    'trial_status': t[2],
                    'trial_objective_value': t[3]} for t in trials]
    self.write(json.dumps(trials_list))
    self.finish()


class StudyAddHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    study_name = self.get_argument('study_name', '')
    study_max_trials = int(self.get_argument('study_max_trials', '10'))
    study_max_time = self.get_argument('study_max_time','10')
    study_hyperparameter_search = self.get_argument('study_hyperparameter_search', '')
    study_hyperparameters = self.get_argument('study_hyperparameters', '')
    study_architecture_search = self.get_argument('study_architecture_search', '')
    study_default_architecture = self.get_argument('study_default_architecture', '')
    study_flops = self.get_argument('study_flops', 0)

    if len(study_hyperparameters) == 0 and len(study_default_architecture) == 0:
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidInput', 'message': 'must set study_hyperparameters or study_default_architecture'}))
      self.finish()
      return

    upload_file_path = ''
    if len(study_default_architecture) > 0:
      upload_file_path = os.path.join(self.main_folder, 'static', 'upload', study_default_architecture)

    if not os.path.exists(upload_file_path):
      self.set_status(404)
      self.write(json.dumps({'code': 'InvaildUploadFile'}))
      self.finish()
      return

    study_max_time = '%dd'%int(study_max_time)

    self.client_socket.send_json({'cmd': 'study/add',
                                  'study_name': study_name,
                                  'study_max_trials': int(study_max_trials),
                                  'study_max_time': study_max_time,
                                  'study_hyperparameter_search': study_hyperparameter_search,
                                  'study_hyperparameters': study_hyperparameters,
                                  'study_architecture_search': study_architecture_search,
                                  'study_default_architecture': upload_file_path,
                                  'study_flops': study_flops})

    response = yield self.client_socket.recv_json()

    if len(response) == 0 or response['status'] != 'ok':
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidServer'}))
      self.finish()
      return

    self.finish()


class TrialInfoHanlder(BaseHandler):
  @gen.coroutine
  def post(self, trial_name):
    self.finish()


class ServerHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    if self.is_worker:
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidServer', 'message': 'not server server'}))
      self.finish()
      return

    signature = self.get_argument('signature', '')
    if self.signature != signature:
      logger.error('signature not consistent %s'%signature)
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSignature'}))
      self.finish()
      return

    study_name = self.get_argument('study_name', '')
    trail_name = self.get_argument('trail_name', None)
    objective_value = self.get_argument('objective_value', -1.0)
    created_time = self.get_argument('created_time', None)
    updated_time = self.get_argument('updated_time', None)

    self.client_socket.send_json({'cmd': 'suggestion/make',
                                  'study_name': study_name,
                                  'trail_name': trail_name,
                                  'objective_value': objective_value,
                                  'created_time': created_time,
                                  'updated_time': updated_time,})
    server_response = yield self.client_socket.recv_json()
    self.write(json.dumps(server_response))


class UpdateServerHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    if self.is_worker:
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidServer', 'message': 'not server server'}))
      self.finish()
      return

    signature = self.get_argument('signature', '')
    if self.signature != signature:
      logger.error('signature not consistent %s'%signature)
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSignature'}))
      self.finish()
      return

    running_experiments_str = self.get_argument('experiments', '')
    running_experiments = json.loads(running_experiments_str)

    self.client_socket.send_json({'cmd': 'suggestion/update',
                                  'experiments': running_experiments})
    yield self.client_socket.recv_json()

    self.finish()


class FileHanlder(BaseHandler):
  @gen.coroutine
  def post(self):
    file_metas = self.request.files.get('file', None)
    if not file_metas:
      self.set_status(400)
      self.write(json.dumps({'code': 'InvalidUploadFile', 'message': 'The input file is not uploaded correctly'}))
      self.finish()
      return

    upload_file_path = os.path.join(self.main_folder, 'static', 'upload')
    if not os.path.exists(upload_file_path):
      os.makedirs(upload_file_path)

    _file_name = ''
    _file_path = ''
    for meta in file_metas:
      _file_name = '%s-%s-%s'%(str(uuid.uuid4()),
                               datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'),
                               meta['filename'])
      _file_path = os.path.join(upload_file_path, _file_name)

      with open(_file_path, 'wb') as fp:
        fp.write(meta['body'])

      break

    self.file_records[_file_name] = _file_path
    self.write(json.dumps({'file': _file_name}))


class TrainHanlder(BaseHandler):
  @gen.coroutine
  def post(self):
    # 1.step check call permission
    # 1.1.step check signature
    signature = self.get_argument('signature', '')
    if self.signature != signature:
      logger.error('signature not consistent %s'%signature)
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSignature'}))
      self.finish()
      return

    # 1.2.step check device resource
    if 'occupied_devices' not in self.server_records:
      self.server_records['occupied_devices'] = []

    free_devices = [n for n in self.device_list if n not in self.server_records['occupied_devices']]
    if len(free_devices) == 0:
      logger.error('have no free device resource')
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSupport', 'message': 'not enough devices'}))
      self.finish()
      return

    # record
    try_config = self.get_argument('AUTOML', {})
    experiment_id = '%s-%s'%(str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))
    start_time = time.time()
    main_param = self.get_argument('MAINPARAM', {})
    max_runtime = self.get_argument('MAX_RUNTIME', None)
    if max_runtime is None:
      max_runtime = self.max_time

    # task token
    token = self.get_argument('TOKEN', None)
    self.experiment_records[experiment_id] = {'start_time': start_time,
                                              'try_config': try_config,
                                              'main_config': main_param,
                                              'max_time': max_runtime,
                                              'main_file': '',
                                              'main_param': '',
                                              'task': '',
                                              'status': 'prepare',
                                              'token': token,
                                              'devices': [],
                                              'evaluation_value': [],
                                              'evaluation_time': [],
                                              'pid': None}

    # 2.step prepare running environment
    # prepare workspace
    os.makedirs(os.path.join(self.main_folder, experiment_id))

    # prepare main param
    if self.main_param is not None and self.main_param != '':
      with open(os.path.join(self.main_folder, self.main_param), 'r') as fp:
        # load basic parameter
        main_param.update(yaml.load(fp))

    # update automl config
    main_param.update({'automl': try_config})

    # apply devices
    apply_devices = int(self.get_argument('APPLY_DEVICES', 1))
    if apply_devices == 0:
      apply_devices = 1

    if 'num_clones' in main_param:
      apply_devices = int(main_param['num_clones'])

    if apply_devices > len(free_devices):
      logger.error('have no free device resource')
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSupport', 'message': 'not enough devices'}))
      self.finish()

      # remove experiment record
      self.experiment_records.pop(experiment_id)
      return

    experiment_devices = free_devices[0:apply_devices]
    self.server_records['occupied_devices'].extend(experiment_devices)
    self.experiment_records[experiment_id]['device'] = experiment_devices
    main_param.update({'devices': experiment_devices})

    with open(os.path.join(self.main_folder, experiment_id, '%s.yaml'%experiment_id), 'w') as fp:
      fp.write(yaml.dump(main_param))

    main_param = '%s.yaml'%experiment_id

    self.experiment_records[experiment_id]['main_param'] = os.path.join(self.main_folder, experiment_id, '%s.yaml'%experiment_id)

    # prepare main file
    main_file = 'main_file.py'
    if self.main_file is None or self.main_file == '':
      file_id = self.get_argument('MAINFILE', None)
      if file_id is None or file_id not in self.file_records:
        self.set_status(400)
        self.write(json.dumps({'code': 'InvalidUploadFile','message': 'The input file is not uploaded correctly'}))
        self.finish()

        # remove experiment record
        self.experiment_records.pop(experiment_id)
        return

      shutil.copy(self.file_records[file_id], os.path.join(self.main_folder, experiment_id, 'main_file.py'))
    else:
      shutil.copy(os.path.join(self.main_folder, self.main_file), os.path.join(self.main_folder, experiment_id, 'main_file.py'))

    self.experiment_records[experiment_id]['main_file'] = os.path.join(self.main_folder, experiment_id, 'main_file.py')

    # run script
    cmd_shell = 'antgo train --main_file=%s --main_param=%s'%(main_file, main_param)
    cmd_shell += ' --main_folder=%s'%os.path.join(self.main_folder, experiment_id)
    cmd_shell += ' --dump=%s'%os.path.join(self.main_folder, experiment_id, 'dump')
    cmd_shell += ' --max_time=%s'%max_runtime
    cmd_shell += ' --signature=%s'%self.signature
    cmd_shell += ' --proxy=%s'%('127.0.0.1:%d'%self.server_port)

    # prepare task xml file
    if token is None:
      file_id = self.get_argument('TASKXML', None)
      if file_id is None:
        self.set_status(400)
        self.write(json.dumps({'code': 'InvalidUploadFile','message': 'The task xml is not uploaded correctly'}))
        self.finish()

        # remove experiment record
        self.experiment_records.pop(experiment_id)
        return

      shutil.copy(self.file_records[file_id], os.path.join(self.main_folder, experiment_id, 'task.xml'))
      self.experiment_records[experiment_id]['task'] = os.path.join(self.main_folder, experiment_id, 'task.xml')
      cmd_shell += ' --task=task.xml'
    else:
      cmd_shell += ' --token=%s'%token

    # start running
    p = subprocess.Popen('nohup %s > %s.log 2>&1 &'%(cmd_shell, experiment_id), shell=True)
    self.experiment_records[experiment_id]['pid'] = p.pid

    self.finish()


def train_server_start(main_file,
                       main_param,
                       main_folder,
                       token,
                       task,
                       devices,
                       max_time,
                       is_worker,
                       signature,
                       server_port,
                       master_ip,
                       parent_id):
  try:
    define('port', default=server_port, help='run on port')

    # records db
    experiment_records = {}
    study_name = None
    if token is not None or task is not None:
      study_name = token if token is not None else task.split('/')[-1]

    server_records = {'signature': signature,
                      'study_name': study_name,
                      'main_folder': main_folder,
                      'main_param': main_param,
                      'main_file': main_file,
                      'devices': devices,
                      'master_ip': master_ip,
                      'task': task,
                      'token': token,
                      'server_port': server_port}
    file_records = {}

    client_socket = None
    if not is_worker:
      zmq_ctx = future.Context.instance()
      client_socket = zmq_ctx.socket(zmq.REQ)
      client_socket.bind('ipc://%s'%str(parent_id))

    if not os.path.exists(os.path.join(main_folder, 'template')):
      os.makedirs(os.path.join(main_folder, 'template'))

    train_server_template_dir = os.path.join(main_folder, 'template')

    if not os.path.exists(os.path.join(main_folder, 'static')):
      os.makedirs(os.path.join(main_folder, 'static'))

    train_server_static_dir = os.path.join(main_folder, 'static')

    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
      if static_file[0] == '.':
        continue

      shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file),
                  train_server_static_dir)

    html_template = 'trainworker.html' if is_worker else 'trainmaster.html'
    shutil.copy(os.path.join(static_folder, 'resource', 'templates', html_template),
                os.path.join(train_server_template_dir, html_template))

    settings = {'main_file': main_file,
                'main_param': main_param,
                'main_folder': main_folder,
                'device_list': devices,
                'max_time': max_time,
                'signature': signature,
                'experiment_records': experiment_records,
                'server_records': server_records,
                'file_records': file_records,
                'server_port': server_port,
                'is_worker': is_worker,
                'client_socket': client_socket,
                'template_path': train_server_template_dir,
                'static_path': train_server_static_dir,
                'html_template': html_template,
                }

    if is_worker and master_ip is not None and master_ip != '':
      app = tornado.web.Application(handlers=[('/', IndexHanlder),
                                              ('/train/', TrainHanlder),
                                              ('/update/model/([^/]+)/', UpdateModelHandler),
                                              ('/submit/', FileHanlder),],
                                    **settings)

      http_server = tornado.httpserver.HTTPServer(app)
      http_server.listen(options.port)
      tornado.ioloop.PeriodicCallback(functools.partial(request_suggestion_process,
                                                        experiment_records=experiment_records,
                                                        server_records=server_records,
                                                        ), 10000).start()
      tornado.ioloop.PeriodicCallback(functools.partial(update_suggestion_process,
                                                        experiment_records=experiment_records,
                                                        server_records=server_records,
                                                        ), 10*60*1000).start()
    else:
      app = tornado.web.Application(handlers=[('/', IndexHanlder),
                                              ('/server/', ServerHandler),
                                              ('/update/', UpdateServerHandler),
                                              ('/study/startorstop/', StudyStartorStopHandler),
                                              ('/study/get/', StudyGetHandler),
                                              ('/study/add/', StudyAddHandler),
                                              ('/trial/([^/]+)/', TrialInfoHanlder),
                                              ('/submit/', FileHanlder),],
                                    **settings)
      http_server = tornado.httpserver.HTTPServer(app)
      http_server.listen(options.port)

    logger.info('train server is launch on port %d'%server_port)
    tornado.ioloop.IOLoop.instance().start()

  except:
    traceback.print_exc()
    raise sys.exc_info()[0]