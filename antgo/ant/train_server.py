# -*- coding: UTF-8 -*-
# @Time    : 2018/11/27 7:08 PM
# @File    : train_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant import flags
from antgo.ant.base import *
from antgo.crowdsource.train_server import *
from antgo.utils import logger
from antgo.automl.suggestion.models import *
from antgo.automl.suggestion.algorithm.grid_search import *
from antgo.automl.suggestion.search_space import *
from antgo.automl.graph import *
import json

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


class AntTrainServer(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_token,
               main_file,
               main_param,
               main_folder,
               dump_dir,
               **kwargs):
    super(AntTrainServer, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.main_file = main_file
    self.main_param = main_param
    self.main_folder = main_folder
    self.dump_dir = dump_dir

    self.is_worker = kwargs.get('is_worker', False)
    self.is_master = kwargs.get('is_master', False)
    if len(kwargs.get('devices', '')) > 0:
      self.devices = [int(n) for n in kwargs.get('devices', '').split(',')]
    else:
      self.devices = []

    self.max_time = kwargs.get('max_time', '10d')
    self.servers = kwargs.get('servers', None)
    if self.servers is not None and len(self.servers) > 0:
      self.servers = self.servers.split(',')[0]

    self.task = kwargs.get('task', '')

  def suggestion_algorithm(self, study):
    if study.algorithm == 'grid_search':
      return GridSearchAlgorithm()

    return None

  def search_space_algorithm(self, study):
    study_configuration = json.loads(study.study_configuration)
    graph_content = study_configuration['graph']
    graph = Decoder().decode(graph_content)
    graph.layer_factory = LayerFactory()
    if study.algorithm == 'Dense Search':
      return DenseArchitectureSearchSpace(graph, study.flops)

  def add_study(self, query):
    study_name = query.get('study_name', '')
    study_max_trials = query.get('study_max_trials', 100)
    study_max_time = query.get('study_max_time', '1d')
    study_hyperparameter_search = query.get('study_hyperparameter_search', '')
    study_hyperparameters = query.get('study_hyperparameters','')
    study_architecture_search = query.get('study_architecture_search', '')
    study_default_architecture = query.get('study_default_architecture', '')
    study_flops = query.get('study_flops', 0)

    s = Study.get(key='name', value=study_name)
    if s is not None:
      return {'status': 'fail'}

    study_mode = 'structure'
    if len(study_hyperparameters) > 0:
      study_mode = 'hyperparameter'
    if len(study_default_architecture) > 0:
      study_mode = 'structure'

    if study_mode == 'structure':
      with open(study_default_architecture, 'r') as fp:
        graph_content = fp.read()

      study_configuration = {
        "goal": "MAXIMIZE",
        "maxTrials": study_max_trials,
        "maxTime": study_max_time,
        "graph": graph_content,
      }

      ss = Study(name=study_name,
                 study_configuration=json.dumps(study_configuration),
                 algorithm=study_architecture_search,
                 mode=study_mode,
                 flops=study_flops,
                 created_time=time.time())
      Study.create(ss)

      return {'status': 'ok'}


    return {'status': 'fail'}

  def delete_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(key="name", value=study_name)
    if study is None:
      return {'status': 'fail'}

    if study.status == 'running':
      return {'status': 'fail'}

    Study.delete(study)
    return {'status': 'ok'}

  def start_or_stop_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(key="name", value=study_name)
    if study is None:
      return {'status': 'fail'}

    if study.status == 'running':
      study.status = 'stop'
    elif study.status == 'stop':
      study.status = 'running'
    else:
      return {'status': 'fail'}

    return {'status': 'ok', 'study_status': study.status}

  def get_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(key="name", value=study_name)
    if study is None:
      return {'status': 'fail'}

    all_trails = Trial.filter(key='study_name', value=study_name)
    all_trails_result = [(trail.name, trail.created_time, trail.status, trail.objective_value) for trail in all_trails]
    all_trails_result = sorted(all_trails_result, key=lambda x: x[3], reverse=True)

    return{'status': 'ok', 'result': all_trails_result}

  def get_studys(self, query):
    studys = [(s.name, s.created_time, s.id, s.status) for s in Study.contents]
    return {'status': 'ok', 'result': studys}

  def get_trial(self, query):
    study_name = query.get('study_name', None)
    trial_name = query.get('trial_name', None)
    if study_name is None or trial_name is None:
      return {'status': 'fail'}

    triails = Trial.filter(key='name', value=trial_name)
    trial = [t for t in triails if t.study_name == study_name]

    if len(trial) == 0:
      return {'status': 'fail'}

    trial = trial[0]
    return {'status': 'ok', 'result': (trial.name, trial.created_time, trial.status, trial.objective_value)}

  def listening_and_command_dispatch(self):
    self._backend = zmq.Context().socket(zmq.REP)
    self._backend.connect('ipc://%s'%str(os.getpid()))

    while True:
      client_query = self._backend.recv_json()
      cmd = client_query['cmd']
      try:
        response = {}
        if cmd == 'study/add':
          response = self.add_study(client_query)
        elif cmd == 'study/delete':
          response = self.delete_study(client_query)
        elif cmd == 'study/startorstop':
          response = self.start_or_stop_study(client_query)
        elif cmd == 'study/all':
          response = self.get_studys(client_query)
        elif cmd == 'study/get':
          response = self.get_study(client_query)
        elif cmd == 'trial/get':
          response = self.get_trial(client_query)
        elif cmd == 'suggestion/make':
          response = self.make_suggestion(client_query)
        elif cmd == 'suggestion/update':
          response = self.update_suggestion(client_query)

        self._backend.send_json(response)
      except:
        traceback.print_exc()

        self._backend.send_json({})
        continue

  def make_suggestion(self, query):
    # 0.step study is ok
    if 'study_name' not in query:
      return {}

    study = Study.get('name', query['study_name'])
    if study is None:
      return {}

    if study.status != 'running':
      return {'status': 'stop', 'message': 'study does not start'}

    study_name = study.name
    trail_name = query.get('trail_name', None)
    objective_value = query.get('objective_value', None)
    objective_value = float(objective_value)
    created_time = query.get('created_time', None)
    updated_time = query.get('updated_time', None)

    # 1.step trail result
    if trail_name is not None and trail_name != '' and objective_value is not None:
      trail = Trial.get('name', trail_name)
      if trail is not None:
        if objective_value > 0.0:
          trail.status = 'Completed'
          trail.objective_value = objective_value
        else:
          trail.status = 'Failed'

    # 2.step check whether arrive max trials or max time
    study_configuration = json.loads(study.study_configuration)
    if 'maxTime' in study_configuration and len(study_configuration['maxTime']) > 0:
      max_time = study_configuration['maxTime']
      if max_time[-1] == 'd':
        max_time = 24*60*60*float(max_time[0:-1])
      elif max_time[-1] == 'h':
        max_time = 60*60*float(max_time[0:-1])
      else:
        max_time = 0

      if max_time > 0:
        if (time.time() - study.created_time) > max_time:
          study.status = 'completed'
          return {'status': 'completed', 'message': 'arrive study max time'}

    if 'maxTrials' in study_configuration:
      max_trials = study_configuration['maxTrials']
      max_trials = int(max_trials)
      trials = Trial.filter(key='study_name', value=study_name)
      if len(trials) > max_trials:
        study.status = 'completed'
        return {'status': 'completed', 'message': 'arrive study max trials'}

    # parameter_values  {"name": ([], superscript, target), ...}
    # generate hyper-parameter config
    trail = None
    if study.mode == 'hyperparameter':
      # 2.step get new suggestion
      suggestion_algorithm = self.suggestion_algorithm(study)
      trail, = suggestion_algorithm.get_new_suggestions(study_name, number=1)

      if trail is None:
        # have no new trial
        study.status = 'completed'
        return {'status': 'completed', 'message': 'no new trial'}
    else:
      # 3.step generate graph structure config
      search_space_algorithm = self.search_space_algorithm(study)
      trail, = search_space_algorithm.get_new_suggestions(study_name, number=1)

      if trail is None:
        # have no new trial
        study.status = 'completed'
        return {'status': 'completed', 'message': 'no new trial'}

    if trail is None:
      return {'status': 'completed', 'message': 'no new trial'}

    response = {'study_name': study_name,
                'trail_name': trail.name,
                'created_time': trail.created_time,
                'updated_time': trail.updated_time,
                'hyperparameter': trail.parameter_values if study.mode == 'hyperparameter' else json.dumps({}),
                'structure': trail.parameter_values if study.mode == 'structure' else '',
                'max_time': study_configuration['maxTime'],
                'status': study.status}
    return response

  def update_suggestion(self, query):
    experiments = query['experiments']
    for experiment in experiments:
      mm = Trial.get(key='name', value=experiment)
      if mm is not None:
        mm.updated_time = time.time()

    return {'status': 'ok'}

  def start(self):
    server_port = 10000
    server_port = _pick_idle_port(server_port)

    logger.info('launch train %s server'%'worker' if self.is_worker else 'master')
    pid = None
    if not self.is_worker:
      process = multiprocessing.Process(target=self.listening_and_command_dispatch)
      process.start()
      pid = process.pid

    train_server_start(self.main_file if self.is_worker else None,
                       self.main_param if self.is_worker else None,
                       self.main_folder,
                       self.app_token if self.is_worker else None,
                       self.task if self.is_worker else None,
                       self.devices if self.is_worker else None,
                       self.max_time,
                       self.is_worker,
                       self.signature,
                       server_port,
                       self.servers if self.is_worker else None,
                       pid)
