# -*- coding: UTF-8 -*-
# Time: 1/2/18
# File: crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.task.task import *
from antgo.measures.base import *
from multiprocessing import Process, Queue
import numpy as np
from antgo.utils.encode import *
from antgo.utils import logger
import multiprocessing
import json
import time
import signal
import requests
from antgo.crowdsource.crowdsource_server import *


def _is_open(check_ip, port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((check_ip, int(port)))
    s.shutdown(2)
    return True
  except:
    return False


def _pick_idle_port(from_port=40000, to_port=50000, check_count=20):
  check_port = None
  while check_count:
    check_port = random.randint(from_port, to_port)
    if not _is_open('127.0.0.1', check_port):
      break

    check_count = check_count - 1

  return check_port


class AntCrowdsource(AntMeasure):
  def __init__(self, task, name):
    super(AntCrowdsource, self).__init__(task, name)
    # regular crowdsource evaluation config
    self._min_participants_per_sample = getattr(task, 'min_participants_per_sample', 1)       # (from task)
    self._skip_sample_num = getattr(task, 'skip_sample_num', 2)                               # (from task)
    self._waiting_time_per_sample = getattr(task, 'waiting_time_per_sample', -1)              # unit: second (from task)
    self._max_time_in_session = getattr(task, 'max_time_in_session', -1)                      # unit: second (max time in one session) (from task)
    self._max_samples_in_session = getattr(task, 'max_samples_in_session', -1)                # max samples in one session  (from task)

    self._client_query_data = {}
    self._total_samples = 0
    self._client_html_template = ''
    self._client_keywords_template = {}

    # {CLIENT_ID, {ID: [], RESPONSE: [], RESPONSE_TIME: [], START_TIME:, QUERY_INDEX:, IS_FINISHED:}}
    self._client_response_record = {}           # client response record
    self._sample_finished_count = {}            # sample number which has been finished
    self._dump_dir = ''

    # crowdsource measure flag
    self.crowdsource = True

    self._app_token = None
    self._experiment_id = None

    self._complex_degree = 0                    # 0,1,2,3,4,5 (subclass define)
    self._crowdsource_type = ''                 # (subclass define)
    self._crowdsource_bonum = getattr(task, 'crowdsource_evaluation_bonus', 0)   # (task define)
    self._crowdsource_start_time = time.time()  # (auto)
    self._crowdsource_title = ''                # description (subclass define)
    self._crowdsource_estimated_time = 0.0      # (auto)
    self._crowdsource_complete_degree = 0       # (auto)

  @property
  def client_html_template(self):
    return self._client_html_template

  @client_html_template.setter
  def client_html_template(self, val):
    self._client_html_template = val

  @property
  def client_keywords_template(self):
    return self._client_keywords_template

  @client_keywords_template.setter
  def client_keywords_template(self, val):
    self._client_keywords_template = val

  @property
  def client_query_data(self):
    return self._client_query_data

  @client_query_data.setter
  def client_query_data(self, val):
    self._client_query_data = val
  
  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, val):
    self._dump_dir = os.path.join(val,'static')

  @property
  def app_token(self):
    return self._app_token

  @app_token.setter
  def app_token(self, val):
    self._app_token = val

  @property
  def experiment_id(self):
    return self._experiment_id

  @experiment_id.setter
  def experiment_id(self, val):
    self._experiment_id = val

  @property
  def _is_finished(self):
    update_real_statistic = {}
    if self._total_samples > 0:
      # compute crowdsource complete degree
      complete_degree = 0.0
      for k,v in self._sample_finished_count.items():
        complete_degree = complete_degree + v['COUNT']
      complete_degree = complete_degree / float(self._total_samples * self._min_participants_per_sample + 1e-6)
      complete_degree = min(max(complete_degree, 0.0), 1.0)
      self._crowdsource_complete_degree = complete_degree
      update_real_statistic['complete'] = self._crowdsource_complete_degree

    # compute crodsource estimated time per client
    estimated_time = []
    for _, resposne_record in self._client_response_record.items():
      cc = [c['STOP_TIME']-c['START_TIME'] for c in resposne_record['RESPONSE_TIME'] if c is not None and c['STOP_TIME'] > 0]
      estimated_time.extend(cc)

    if len(estimated_time) > 0:
      estimated_time = np.median(estimated_time)
      self._crowdsource_estimated_time = estimated_time
      update_real_statistic['estimated_time'] = self._crowdsource_estimated_time

    # check is over?
    if len(self._sample_finished_count) == 0:
      return False

    # check is over?
    if len(self._sample_finished_count) < self._total_samples:
      return False
    
    # finish status measure - min participants per sample
    for k, v in self._sample_finished_count.items():
      if v['COUNT'] < self._min_participants_per_sample:
        return False

    return True

  def ground_truth_response(self, client_id, query_index, record_db):
    return {}

  def _prepare_ground_truth_page(self, client_id, query_index, record_db):
    response_data = {'PAGE_STATUS': 'GROUNDTRUTH', 'QUERY_INDEX': query_index}
    response_data['PAGE_DATA'] = {}

    custom_response = self.ground_truth_response(client_id, query_index, record_db)
    response_data['PAGE_DATA'].update(custom_response)
    return response_data

  def _prepare_next_page(self, client_id, query_index, record_db):
    if query_index == len(self._client_response_record[client_id]['ID']) - 1:
      return self._prepare_stop_page()

    now_time = time.time()
    query_index = query_index + 1
    self._client_response_record[client_id]['RESPONSE_TIME'][query_index] = {'START_TIME': now_time, 'STOP_TIME': -1}
    self._client_response_record[client_id]['QUERY_INDEX'] = query_index
    
    # 2.step server response
    tags = [b for a, b in self.client_keywords_template.items()]
    datas = record_db.read(self._client_response_record[client_id]['ID'][query_index], *tags)
    data_map = {}
    for a, b in zip(tags, datas):
      data_map[a] = b

    response_data = {}
    response_data['PAGE_DATA'] = {}
    # 2.1.step prepare page data
    for k, v in self.client_query_data['QUERY'].items():
      # k is name, v is type(IMAGE, SOUND, TEXT)
      data = data_map[k]

      if v == 'IMAGE':
        # save to .png
        data_en = png_encode(data)
        with open(os.path.join(self.dump_dir, '%s-%d.png' % (k,self._client_response_record[client_id]['ID'][query_index])), 'wb') as fp:
          fp.write(data_en)

        response_data['PAGE_DATA'][k] = {'DATA': '%s-%d.png' % (k,self._client_response_record[client_id]['ID'][query_index]), 'TYPE': 'IMAGE'}
        pass
      elif v == 'SOUND':
        # save
        pass
      else:
        response_data['PAGE_DATA'][k] = {'DATA': data, 'TYPE': 'TEXT'}

    # 2.3.step prepare sample index
    response_data['QUERY_INDEX'] = query_index
    # 2.4.step condition
    response_data['WAITING_TIME'] = self._waiting_time_per_sample
    # 2.5.step remained num
    response_data['REMAINED_NUM'] = len(self._client_response_record[client_id]['ID']) - query_index - 1

    return response_data

  def _prepare_stop_page(self):
    return {'PAGE_STATUS': 'STOP'}

  def _start_click_branch(self, client_query, record_db):
    client_id = client_query['CLIENT_ID']
    if client_id in self._client_response_record and self._client_response_record[client_id]['QUERY_INDEX'] != 0:
      # contiue unfinished session
      logger.info('client id %s restart unfinished session'%client_id)
      # user reenter crowdsource evaluation
      # 1.step update start time
      last_query_index = self._client_response_record[client_id]['QUERY_INDEX']
      
      last_active_time = -1
      first_sample_time = None
      for sample_response in self._client_response_record[client_id]['RESPONSE_TIME'][0:last_query_index]:
        if sample_response is not None:
          sample_start_time = sample_response['START_TIME']
          if first_sample_time is None:
            first_sample_time = sample_start_time
          
          if first_sample_time > sample_start_time:
            first_sample_time = sample_start_time
          
          sample_stop_time = sample_response['STOP_TIME']
          if sample_stop_time < -1:
            break
          
          if last_active_time < sample_stop_time:
            last_active_time = sample_stop_time

      now_time = time.time()
      if last_active_time == -1 or first_sample_time is None:
        self._client_response_record[client_id]['START_TIME'] = now_time
      else:
        self._client_response_record[client_id]['START_TIME'] = now_time - (last_active_time - first_sample_time)
      client_query['QUERY_INDEX'] = self._client_response_record[client_id]['QUERY_INDEX'] - 1
      client_query['CLIENT_RESPONSE'] = None
      # client_query['QUERY_INDEX'] may be -1
      return self._next_click_branch(client_query, record_db, True)
    else:
      # clear
      self._client_response_record[client_id] = {}
      # add client_id record
      logger.info('enter client id %s "START" response'%client_id)
      # start a new session
      # 1.step update client response_record
      ids = list(range(int(record_db.count)))
      np.random.shuffle(ids)
      client_record = {}
      # 1.1.step prepare data index order
      client_record['ID'] = ids
      client_record['RESPONSE'] = [None for _ in ids]
      # 1.2.step record start task time
      now_time = time.time()
      client_record['START_TIME'] = now_time
      # 1.3.step record first sample time
      client_record['RESPONSE_TIME'] = [None for _ in ids]
      client_record['RESPONSE_TIME'][0] = {'START_TIME': now_time, 'STOP_TIME': -1}
      client_record['QUERY_INDEX'] = 0
      self._client_response_record[client_id] = client_record

      # 2.step server response
      tags = [b for a,b in self.client_keywords_template.items()]
      datas = record_db.read(client_record['ID'][0], *tags)
      data_map = {}
      for a,b in zip(tags, datas):
        data_map[a] = b

      response_data = {}
      response_data['PAGE_STATUS'] = 'PREDICT'
      response_data['PAGE_DATA'] = {}
      # 2.1.step prepare page data
      for k, v in self.client_query_data['QUERY'].items():
        # k is name, v is type(IMAGE, SOUND, TEXT)
        data = data_map[k]

        if v == 'IMAGE':
          # save to .png
          data_en = png_encode(data)
          with open(os.path.join(self.dump_dir, '%s-%d.png'%(k, client_record['ID'][0])), 'wb') as fp:
            fp.write(data_en)

          response_data['PAGE_DATA'][k] = {'DATA': '%s-%d.png'%(k, client_record['ID'][0]), 'TYPE': 'IMAGE'}
          pass
        elif v == 'SOUND':
          # save
          pass
        else:
          response_data['PAGE_DATA'][k] = {'DATA': data, 'TYPE': 'TEXT'}

      # 2.3.step prepare sample index
      response_data['QUERY_INDEX'] = 0
      # 2.4.step condition
      response_data['WAITING_TIME'] = self._waiting_time_per_sample
      # 2.5.step total sample number
      self._total_samples = len(self._client_response_record[client_id]['ID'])
      # 2.6.step remained num in this session
      total_num = len(self._client_response_record[client_id]['ID']) \
        if self._max_samples_in_session < 0 else self._max_samples_in_session
      response_data['REMAINED_NUM'] = total_num - 1
      logger.info('leave client id %s "START" response'%client_id)
      return response_data

  def _next_click_branch(self, client_query, record_db, to_next_page=False):
    # continue unfinished session
    client_id = client_query['CLIENT_ID']
    if client_id not in self._client_response_record:
      return {}

    logger.info('enter client id %s "NEXT" response' % client_id)
    now_time = time.time()

    # 1.step client response content
    query_index = client_query['QUERY_INDEX']

    if client_query['QUERY_STATUS'] == 'CLOSE_ECHO' or to_next_page:
      logger.info('prepare next page data')
      # enter next sample
      return self._prepare_next_page(client_id, query_index, record_db)

    # make sure only once access
    # assert(self._client_response_record[client_id]['RESPONSE'][query_index] is None)
    
    if client_query['CLIENT_RESPONSE'] is not None:
      try:
        logger.info('record client id %s response'%client_id)
        user_response = client_query['CLIENT_RESPONSE']
        
        # 1.2.step record client response
        self._client_response_record[client_id]['RESPONSE'][query_index] = user_response
        self._client_response_record[client_id]['RESPONSE_TIME'][query_index]['STOP_TIME'] = now_time
  
        query_data_index = self._client_response_record[client_id]['ID'][query_index]
        if query_data_index not in self._sample_finished_count:
          self._sample_finished_count[query_data_index] = {'CLIENT': [], 'COUNT': 0}
        
        if client_id not in self._sample_finished_count[query_data_index]['CLIENT']:
          self._sample_finished_count[query_data_index]['COUNT'] = \
            self._sample_finished_count[query_data_index]['COUNT'] + 1
          self._sample_finished_count[query_data_index]['CLIENT'].append(client_id)
  
        logger.info('finish record client id %s response'%client_id)
      except:
        logger.error('failed to record client id %s response'%client_id)
        self._client_response_record[client_id]['RESPONSE'][query_index] = None
        self._client_response_record[client_id]['RESPONSE_TIME'][query_index]['STOP_TIME'] = -1
        return {}

    # 2.step server response
    # 2.1.step whether over predefined max time in one session
    if self._max_time_in_session > 0:
      # user has been defined max time in session
      if now_time - self._client_response_record[client_id]['START_TIME'] > self._max_time_in_session:
        # return stop flag
        logger.info('client id %s session time is larger than max time'%client_id)
        return self._prepare_stop_page()

    # 2.2.step whether over predefined max samples in one session
    if self._max_samples_in_session > 0:
      # user has been defined max samples in session
      if (query_index + 1) >= self._max_samples_in_session:
        # return stop flag
        logger.info('client id %s max samples is larger than task defined'%client_id)
        return self._prepare_stop_page()

    # 2.3.step whether need skip first some samples (in first some samples, we must return ground truth)
    if self._skip_sample_num > 0:
      # user has been defined skip first some samples
      if query_index < self._skip_sample_num:
        # user would see the ground truth of his last judgement
        logger.info('prepare grouth truth page')
        return self._prepare_ground_truth_page(client_id, query_index, record_db)

    # 2.4.step enter next sample
    logger.info('prepare next page')
    return self._prepare_next_page(client_id, query_index, record_db)

  def _query_is_legal(self, client_query):
    if 'CLIENT_ID' not in client_query:
      logger.error('client query must contain key "CLIENT_ID"')
      return False

    if 'QUERY' not in client_query:
      logger.error('client query must contain key "QUERY"')
      return False

    if client_query['QUERY'] not in ['START', 'NEXT']:
      logger.error('client query must be "START" or "NEXT"')
      return False
    
    if client_query['QUERY'] != 'START':
      if client_query['CLIENT_ID'] in self._client_response_record:
        # 1.step check query index consistent
        if int(client_query['QUERY_INDEX']) != self._client_response_record[client_query['CLIENT_ID']]['QUERY_INDEX']:
          logger.error('client_id %s query index %d not '
                       'consistent with server query index %d'%(client_query['CLIENT_ID'],
                                                                int(client_query['QUERY_INDEX']),
                                                                int(self._client_response_record[client_query['CLIENT_ID']]['QUERY_INDEX'])))
          return False
  
        # 2.step check client session has been finished
        if self._client_response_record[client_query['CLIENT_ID']]['QUERY_INDEX'] == \
                len(self._client_response_record[client_query['CLIENT_ID']]['ID']):
          logger.error('client_id %s session has been finished'%client_query['CLIENT_ID'])
          return False

    return True

  def crowdsource_server(self, record_db):
    # 0.step search idel server port
    idle_server_port = _pick_idle_port(from_port=40000, to_port=50000, check_count=20)
    if idle_server_port is None:
      logger.error('couldnt find idle port for crowdsoure server')
      return False

    # select crowdsource info
    crowdsource_info = {'title': self._crowdsource_title,
                        'type': self._crowdsource_type,
                        'complex': self._complex_degree,
                        'time': self._crowdsource_start_time,
                        'complete': 0.0,
                        'bonus': self._crowdsource_bonum,
                        'estimated_time': self._crowdsource_estimated_time}

    # 1.step launch crowdsource server (independent process)
    request_queue = Queue()
    response_queue = Queue()

    process = multiprocessing.Process(target=crowdsrouce_server_start,
                                      args=(os.getpid(),
                                            self.experiment_id,
                                            self.app_token,
                                            '/'.join(os.path.normpath(self.dump_dir).split('/')[0:-1]),
                                            self.name,
                                            self.client_html_template,
                                            self.client_keywords_template,
                                            idle_server_port,
                                            crowdsource_info,
                                            request_queue,
                                            response_queue))
    process.start()

    # 2.step listening crowdsource server is OK
    waiting_time = 10
    is_connected = False
    now_time = time.time()
    while not is_connected and (time.time() - now_time) < 60 * 5:
      # waiting crowdsource server in 5 minutes
      try:
        res = requests.get('http://127.0.0.1:%d/heartbeat'%idle_server_port)
        heatbeat_content = json.loads(res.content)
        if 'ALIVE' in heatbeat_content:
          is_connected = True
          break
      except:
        time.sleep(waiting_time)

    if not is_connected:
      logger.error('fail to connect local crowdsource server, couldnt start crowdsource crowdsource server')
      return False

    # 3.step listening client query until crowdsource server is finished
    while not self._is_finished:
      client_id = ''
      try:
        # 2.1 step receive request
        client_query = request_queue.get()

        # 2.2 step check query is legal
        if not self._query_is_legal(client_query):
          if self._client_response_record[client_query['CLIENT_ID']]['QUERY_INDEX'] == \
                  len(self._client_response_record[client_query['CLIENT_ID']]['ID']) - 1:
              # send stop page
              response_queue.put(self._prepare_stop_page())
          else:
            # send unknown page
            response_queue.put({})
          continue
          
        # client id
        client_id = client_query['CLIENT_ID']

        # 2.3 step response client query
        # QUERY: 'START', 'NEXT'
        ########################################################
        #############            START           ###############
        ########################################################
        if client_query['QUERY'] == 'START':
          logger.info('response client_id %s %s query'%(client_id, 'START'))
          response = self._start_click_branch(client_query, record_db)
          response_queue.put(response)
          continue

        ########################################################
        #############            NEXT            ###############
        ########################################################
        if client_query['QUERY'] == 'NEXT':
          logger.info('response client_id %s %s query'%(client_id, 'NEXT'))
          response = self._next_click_branch(client_query, record_db)
          response_queue.put(response)
          continue

        logger.error('client_id %s unknow error'%client_id)
        response_queue.put({})
        continue
      except:
        logger.error('client_id %s unknow error'%client_id)
        response_queue.put({})
    
    # save crowdsource client response
    with open(os.path.join(self.dump_dir, 'crowdsource_record.txt'), 'w') as fp:
      fp.write(json.dumps(self._client_response_record))

    # kill crowdsource server
    # suspend 5 minutes, crowdsource abort suddenly
    time.sleep(30)
    # kill crowdsource server
    os.kill(process.pid, signal.SIGTERM)
    return True