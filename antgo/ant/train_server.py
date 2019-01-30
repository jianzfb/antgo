# -*- coding: UTF-8 -*-
# @Time    : 2018/11/27 7:08 PM
# @File    : train_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *
from antgo.crowdsource.train_server import *
from antgo.utils import logger
from antgo.automl.suggestion.algorithm.grid_search import *
from antgo.automl.suggestion.searchspace.dpc import *
from antgo.automl.suggestion.searchspace.searchspace_factory import *
from antgo.automl.suggestion.algorithm.hyperparameters_factory import *
from antgo.automl.graph import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    self.port = int(kwargs.get('port', 10000))

  def suggestion_algorithm(self, study):
    if study.algorithm == 'grid_search':
      return GridSearchAlgorithm()

    return None

  def search_space_algorithm(self, study):
    if SearchspaceFactory.get(study.search_space) is None:
      return None

    study_configuration = json.loads(study.study_configuration)
    return SearchspaceFactory.get(study.search_space)(study, **study_configuration['searchSpace'])

  def add_study(self, query, status='stop'):
    study_name = query.get('study_name', '')
    study_goal = query.get('study_goal', 'MAXIMIZE')
    study_max_trials = query.get('study_max_trials', 100)
    study_max_time = query.get('study_max_time', '1d')
    study_hyperparameter_search = query.get('study_hyperparameter_search', '')
    study_hyperparameters = query.get('study_hyperparameters', [])
    study_architecture_search = query.get('study_architecture_search', '')
    study_architecture_parameters = query.get('study_architecture_parameters', {})

    s = Study.get(name=study_name)
    if s is not None:
      return {'status': 'fail'}

    if study_hyperparameter_search == '' and study_architecture_search == '':
      return {'status': 'fail'}

    study_configuration = {
      "goal": study_goal,
      "maxTrials": study_max_trials,
      "maxTime": study_max_time,
      "searchSpace": {},
      "params": []
    }

    if study_architecture_search is not None and study_architecture_search != '':
      searchspace_cls = SearchspaceFactory.get(study_architecture_search)
      if searchspace_cls is None:
        return {'status': 'fail'}

    if study_architecture_parameters is not None and study_architecture_search != '':
      for k, v in study_architecture_parameters.items():
        if k == 'graph':
          if os.path.exists(v):
            with open(v, 'r') as fp:
              graph_content = fp.read()
              study_architecture_parameters[k] = graph_content
      study_configuration['searchSpace'] = study_architecture_parameters

    study_configuration["params"] = study_hyperparameters

    ss = Study(name=study_name,
               study_configuration=json.dumps(study_configuration),
               algorithm=study_hyperparameter_search,
               search_space=study_architecture_search,
               created_time=time.time(),
               status=status)
    Study.create(ss)
    return {'status': 'ok'}

  def delete_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(name=study_name)
    if study is None:
      return {'status': 'fail'}

    Study.delete(study)
    return {'status': 'ok'}

  def start_or_stop_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(name=study_name)
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
    study = Study.get(name=study_name)
    if study is None:
      return {'status': 'fail'}

    all_trails = Trial.filter(study_name=study_name)
    all_trails_result = [(trail.name, trail.created_time, trail.status, trail.objective_value) for trail in all_trails]
    all_trails_result = sorted(all_trails_result, key=lambda x: x[3], reverse=True)

    return{'status': 'ok', 'result': all_trails_result}

  def download_study(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(name=study_name)
    if study is None:
      return {'status': 'fail'}

    all_trails = Trial.filter(study_name=study_name, status='Completed')
    study_configuration = json.loads(study.study_configuration)
    study_content = {'study': study_name, 'goal': study_configuration['goal'], 'trials_num': len(all_trails),'trials': []}
    for trail in all_trails:
      trial_result = {}
      trial_result['graph'] = trail.structure[0]
      trial_result['graph_info'] = trail.structure[1]
      trial_result['tag'] = trail.tag
      trial_result['created_time'] = trail.created_time
      trial_result['updated_time'] = trail.updated_time
      trial_result['name'] = trail.name
      trial_result['address'] = trail.address
      trial_result['objective_value'] = trail.objective_value

      study_content['trials'].append(trial_result)
    return {'status': 'ok', 'result': json.dumps(study_content)}

  def study_visualization(self, query):
    study_name = query.get('study_name', None)
    study = Study.get(name=study_name)
    if study is None:
      return {'status': 'fail'}

    dump_dir = query.get('dump_dir', None)
    if dump_dir is None:
      return {'status': 'fail'}

    time_str = datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f')
    all_trails = Trial.filter(study_name=study_name, status='Completed')
    all_trails = sorted(all_trails, key=lambda x: x.created_time)

    plt.subplot(2, 1, 1)
    plt.title('study visualization')
    plt.xlabel('time(hours)')
    plt.ylabel('accuracy')
    x = [(m.created_time - study.created_time) / 3600.0 for m in all_trails]
    y = [m.objective_value for m in all_trails]
    plt.scatter(x=x, y=y, c='r', marker='o')

    plt.subplot(2, 1, 2)
    plt.xlabel('ADDMUL/FLOPS')
    plt.ylabel('accuracy')
    x = [m.multi_objective_value[0] for m in all_trails]
    y = [m.objective_value for m in all_trails]
    plt.scatter(x=x, y=y, c='r', marker='o')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig('%s/study_%s_%s.png'%(dump_dir, study_name, time_str))
    return {'status': 'ok', 'result': '%s/study_%s_%s.png'%(dump_dir, study_name, time_str)}

  def get_studys(self, query):
    studys = [(s.name, s.created_time, s.id, s.status) for s in Study.contents]
    return {'status': 'ok', 'result': studys}

  def get_trial(self, query):
    study_name = query.get('study_name', None)
    trial_name = query.get('trial_name', None)
    if study_name is None or trial_name is None:
      return {'status': 'fail'}

    triails = Trial.filter(name=trial_name)
    trial = [t for t in triails if t.study_name == study_name]

    if len(trial) == 0:
      return {'status': 'fail'}

    trial = trial[0]
    return {'status': 'ok', 'result': (trial.name,
                                       trial.created_time,
                                       trial.status,
                                       trial.objective_value,
                                       trial.structure,
                                       trial.parameter_values,
                                       trial.address)}

  def get_searchspace(self, query):
    searchspace_name = query.get('searchspace', None)
    if searchspace_name is None:
      return {'status': 'fail'}

    searchspace_cls = SearchspaceFactory.get(searchspace_name)
    if searchspace_cls is None:
      return {'status': 'fail'}

    searchspace_params = searchspace_cls.default_params
    return {'status': 'ok', 'result': searchspace_params}

  def get_searchspace_all(self, query):
    return {'status': 'ok', 'result': SearchspaceFactory.all()}

  def get_hyperparameter_all(self, query):
    return {'status': 'ok', 'result': HyperparametersFactory.all()}

  def listening_and_command_dispatch(self, *args, **kwargs):
    # initialize
    if len(kwargs) > 0:
      self.add_study(kwargs, 'running')

    # listening and dispatch
    self._backend = zmq.Context().socket(zmq.REP)
    self._backend.connect('ipc://%s' % str(os.getpid()))

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
        elif cmd == 'study/download':
          response = self.download_study(client_query)
        elif cmd == 'trial/get':
          response = self.get_trial(client_query)
        elif cmd == 'suggestion/make':
          response = self.make_suggestion(client_query)
        elif cmd == 'suggestion/update':
          response = self.update_suggestion(client_query)
        elif cmd == 'searchspace/get':
          response = self.get_searchspace(client_query)
        elif cmd == 'searchspace/all':
          response = self.get_searchspace_all(client_query)
        elif cmd == 'hyperparameter/all':
          response = self.get_hyperparameter_all(client_query)
        elif cmd == "study/visualization":
          response = self.study_visualization(client_query)

        self._backend.send_json(response)
      except:
        traceback.print_exc()

        self._backend.send_json({})
        continue

  def make_suggestion(self, query):
    # 0.step study is ok
    if 'study_name' not in query:
      return {}

    study = Study.get(name=query['study_name'])
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
    trail_address = query.get('address', None)

    # 1.step trail result
    if trail_name is not None and trail_name != '' and objective_value is not None:
      trail = Trial.get(name=trail_name)
      if trail is not None:
        if objective_value > 0.0:
          trail.status = 'Completed'
          trail.objective_value = objective_value
          trail.address = trail_address
          trail.updated_time = time.time()
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
      trials = Trial.filter(study_name=study_name)
      if len(trials) > max_trials:
        study.status = 'completed'
        return {'status': 'completed', 'message': 'arrive study max trials'}

    # parameter_values  {"name": ([], superscript, target), ...}
    # generate hyper-parameter config
    suggestion_algorithm = self.suggestion_algorithm(study)
    trail = None
    if suggestion_algorithm is not None:
      trails = suggestion_algorithm.get_new_suggestions(study_name, number=1)
      trail = trails[0] if len(trails) > 0 else None

    search_space_algorithm = self.search_space_algorithm(study)
    if search_space_algorithm is not None:
      search_space_parameters = json.loads(trail.parameter_values) if trail is not None else {}
      trails = search_space_algorithm.get_new_suggestions(number=1, **search_space_parameters)
      if trails is None:
        return {'status': 'waiting', 'message': 'no new trial'}

      trail = trails[0] if len(trails) > 0 else None

    if trail is None:
      return {'status': 'completed', 'message': 'no new trial'}

    logger.info('success to make a suggestion %s for study %s'%(trail.name, study_name))

    response = {'study_name': study_name,
                'trail_name': trail.name,
                'created_time': trail.created_time,
                'updated_time': trail.updated_time,
                'hyperparameter': trail.parameter_values,
                'structure': trail.structure[0] if type(trail.structure) == list or type(trail.structure) == tuple else trail.structure,
                'structure_connection': trail.structure[1] if type(trail.structure) == list or type(trail.structure) == tuple else '',
                'max_time': study_configuration['maxTime'],
                'status': study.status}
    return response

  def update_suggestion(self, query):
    experiments = query['experiments']
    for experiment in experiments:
      mm = Trial.get(name=experiment)
      if mm is not None:
        mm.updated_time = time.time()

    return {'status': 'ok'}

  def start(self):
    server_port = _pick_idle_port(self.port)

    logger.info('launch train %s server'%'worker' if self.is_worker else 'master')
    pid = None
    if not self.is_worker:
      # config train master
      automl_config = {}
      if self.main_param is not None:
        main_config_path = os.path.join(self.main_folder, self.main_param)
        params = yaml.load(open(main_config_path, 'r'))

        if 'automl' in params and 'study' in params['automl']:
          # parse study
          study_name = params['automl']['study'].get('study_name', '')
          automl_config['study_name'] = study_name

          study_goal = params['automl']['study'].get('goal', 'MAXIMIZE')
          assert(study_goal in ['MAXIMIZE', 'MINIMIZE'])
          automl_config['study_goal'] = study_goal

          study_max_trials = params['automl']['study'].get('study_max_trials', 1000)
          automl_config['study_max_trials'] = int(study_max_trials)

          study_max_time = params['automl']['study'].get('study_max_time', '1d')
          automl_config['study_max_time'] = study_max_time

          study_hyperparameter_search = ''
          study_hyperparameters = {}

          study_architecture_search = params['automl']['study'].get('study_architecture_search', 'Evolution')
          automl_config['study_architecture_search'] = study_architecture_search

          study_architecture_parameters = params['automl']['study'].get('study_architecture_parameters', {})
          automl_config['study_architecture_parameters'] = study_architecture_parameters

      # launch command dispatch process
      process = multiprocessing.Process(target=self.listening_and_command_dispatch, kwargs=automl_config)
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
