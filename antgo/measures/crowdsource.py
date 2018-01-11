# -*- coding: UTF-8 -*-
# Time: 1/2/18
# File: crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.task.task import *
from antgo.measures.base import *
from multiprocessing import Process, Pipe
import numpy as np
from antgo.utils.encode import *
from antgo.utils import logger
import multiprocessing
import json
import time
import signal
from antgo.http.crowdsource_server import *


class AntCrowdsource(AntMeasure):
  def __init__(self, task, name):
    super(AntCrowdsource, self).__init__(task, name)
    self._server, self._client = Pipe()

    self._min_participants_per_sample = getattr(task, 'min_participants_per_sample', 1)       # (from task)
    self._skip_sample_num = getattr(task, 'skip_sample_num', -1)                              # (from task)
    self._waiting_time_per_sample = getattr(task, 'waiting_time_per_sample', 5)               # unit: second (from task)
    self._max_time_in_session = getattr(task, 'max_time_in_session', -1)                      # unit: second (max time in one session) (from task)
    self._max_samples_in_session = getattr(task, 'max_samples_in_session', -1)                # max samples in one session  (from task)
    self._client_query_data = {}
    self._client_response_data = {}             # measure designer define (key(data id): value(data type))
                                                # socre, list

    self._task_html = getattr(task, 'html', '<img id="PREDICT">') # task html
    self._total_samples = 0
    self._client_query_html = ''
    self._client_query_js = ''

    # {CLIENT_ID, {ID: [], RESPONSE: [], RESPONSE_TIME: [], START_TIME:, QUERY_INDEX:, IS_FINISHED:}}
    self._client_response_record = {}           # client response record
    self._sample_finished_count = {}
    self._dump_dir = ''
    
    self._crowdsource_helper = 'welcome mltalker crowdsource evaluation'
    self.crowdsource = True

  @property
  def client_response_data(self):
    return self._client_response_data
  @client_response_data.setter
  def client_response_data(self, val):
    self._client_response_data = val

  @property
  def client_query_html(self):
    return self._client_query_html
  @client_query_html.setter
  def client_query_html(self, val):
    self._client_query_html = val

  @property
  def client_query_js(self):
    return self._client_query_js
  @client_query_js.setter
  def client_query_js(self, val):
    self._client_query_js = val

  @property
  def client_query_data(self):
    return self._client_query_data
  @client_query_data.setter
  def client_query_data(self, val):
    self._client_query_data = val

  @property
  def crowdsource_helper(self):
    return self._crowdsource_helper
  @crowdsource_helper.setter
  def crowdsource_helper(self, val):
    self._crowdsource_helper = val
  
  @property
  def _is_finished(self):
    if len(self._sample_finished_count) == 0:
      return False
    
    if len(self._sample_finished_count) < self._total_samples:
      return False
    
    # finish status measure - min participants per sample
    for k, v in self._sample_finished_count.items():
      if v['COUNT'] < self._min_participants_per_sample:
        return False
      
    return True

  def prepare_custom_response(self, client_id, query_index, record_db):
    return {}

  def _prepare_ground_truth_page(self, client_id, query_index, record_db):
    gt, predict = record_db.read(self._client_response_record[client_id]['ID'][query_index])
    response_data = {'PAGE_STATUS': 'GROUNDTRUTH', 'QUERY_INDEX': query_index}
    response_data['PAGE_DATA'] = {}
    # 2.1.step prepare page data
    for k, v in self.client_query_data['GROUNDTRUTH'].items():
      # k is name, v is type(IMAGE, SOUND, TEXT)
      data = None
      if k.startswith('GROUNDTRUTH_'):
        k = k.replace('GROUNDTRUTH_', '')
        data = gt[k]
      elif k.startswith('PREDICT_'):
        k = k.replace('PREDICT_', '')
        data = predict[k]
        
      if v == 'IMAGE':
        # save to .png
        data_en = png_encode(data)
        with open(os.path.join(self._dump_dir, '%d.png' % self._client_response_record['ID'][query_index]), 'wb') as fp:
          fp.write(data_en)

        response_data['PAGE_DATA'][k] = {'DATA': '%d.png' % self._client_response_record['ID'][query_index],
                                         'TYPE': 'IMAGE'}
        pass
      elif v == 'SOUND':
        # save
        pass
      else:
        response_data['PAGE_DATA'][k] = {'DATA': data, 'TYPE': 'TEXT'}

    custom_response = self.prepare_custom_response(client_id, query_index, record_db)
    response_data['PAGE_DATA'].update(custom_response)
    return response_data

  def _prepare_next_page(self, client_id, query_index,record_db):
    if query_index == len(self._client_response_record[client_id]['ID']) - 1:
      return self._prepare_stop_page()

    now_time = time.time()
    query_index = query_index + 1
    self._client_response_record[client_id]['RESPONSE_TIME'][query_index] = {'START_TIME': now_time, 'STOP_TIME': -1}
    self._client_response_record[client_id]['QUERY_INDEX'] = query_index
    
    # 2.step server response
    gt, predict = record_db.read(self._client_response_record[client_id]['ID'][query_index], 'groundtruth', 'predict')
    response_data = {}
    response_data['PAGE_DATA'] = {}
    # 2.1.step prepare page data
    for k, v in self.client_query_data['QUERY'].items():
      # k is name, v is type(IMAGE, SOUND, TEXT)
      data = None
      if k.startswith('GROUNDTRUTH_'):
        k = k.replace('GROUNDTRUTH_', '')
        data = gt[k]
      elif k.startswith('PREDICT_'):
        k = k.replace('PREDICT_', '')
        data = predict[k]

      if v == 'IMAGE':
        # save to .png
        data_en = png_encode(data)
        with open(os.path.join(self._dump_dir, '%d.png' % self._client_response_record[client_id]['ID'][query_index]), 'wb') as fp:
          fp.write(data_en)

        response_data['PAGE_DATA'][k] = {'DATA': '%d.png' % self._client_response_record[client_id]['ID'][query_index], 'TYPE': 'IMAGE'}
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
    if client_id in self._client_response_record:
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
      # client_query['QUERY_INDEX'] may be -1
      return self._next_click_branch(client_query, record_db)
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
      gt, predict = record_db.read(client_record['ID'][0], 'groundtruth', 'predict')
      response_data = {}
      response_data['PAGE_STATUS'] = 'PREDICT'
      response_data['PAGE_DATA'] = {}
      # 2.1.step prepare page data
      for k, v in self.client_query_data['QUERY'].items():
        # k is name, v is type(IMAGE, SOUND, TEXT)
        data = None
        if k.startswith('GROUNDTRUTH_'):
          k = k.replace('GROUNDTRUTH_', '')
          data = gt[k]
        elif k.startswith('PREDICT_'):
          k = k.replace('PREDICT_', '')
          data = predict[k]
          
        if v == 'IMAGE':
          # save to .png
          data_en = png_encode(data)
          with open(os.path.join(self._dump_dir, '%d.png'%client_record['ID'][0]), 'wb') as fp:
            fp.write(data_en)

          response_data['PAGE_DATA'][k] = {'DATA': '%d.png'%client_record['ID'][0], 'TYPE': 'IMAGE'}
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

  def _next_click_branch(self, client_query, record_db):
    # continue unfinished session
    client_id = client_query['CLIENT_ID']
    if client_id not in self._client_response_record:
      return {}

    logger.info('enter client id %s "NEXT" response' % client_id)
    now_time = time.time()

    # 1.step client response content
    query_index = client_query['QUERY_INDEX']

    if client_query['QUERY_STATUS'] == 'NEXT_ECHO':
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
      if (query_index + 1) < self._skip_sample_num:
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

  def server(self, record_db, dump_dir):
    # 0.step prepare dump dir
    self._dump_dir = dump_dir

    # 1.step launch http server (independent process)
    process = multiprocessing.Process(target=crowdsrouce_server_start,
                                      args=(self._client,
                                            self._dump_dir,
                                            self.crowdsource_helper,
                                            self.name,
                                            self._task_html,
                                            self.client_response_data))
    process.start()

    # 2.step listening client query
    while not self._is_finished:
      client_id = ''
      try:
        # 2.1 step receive request
        client_query = self._server.recv()

        # 2.2 step check query is legal
        if not self._query_is_legal(client_query):
          if self._client_response_record[client_query['CLIENT_ID']]['QUERY_INDEX'] == \
                  len(self._client_response_record[client_query['CLIENT_ID']]['ID']) - 1:
              # send stop page
              self._server.send(self._prepare_stop_page())
          else:
            # send unknown page
            self._server.send({})
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
          self._server.send(response)
          continue

        ########################################################
        #############            NEXT            ###############
        ########################################################
        if client_query['QUERY'] == 'NEXT':
          logger.info('response client_id %s %s query'%(client_id, 'NEXT'))
          response = self._next_click_branch(client_query, record_db)
          self._server.send(response)
          continue

        logger.error('client_id %s unknow error'%client_id)
        self._server.send({})
        continue
      except:
        logger.error('client_id %s unknow error'%client_id)
        self._server.send({})

    # kill crowdsrouce server process
    # waiting 10 minute
    time.sleep(10 * 60)
    os.kill(process.pid, signal.SIGKILL)