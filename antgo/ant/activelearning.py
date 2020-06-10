# -*- coding: UTF-8 -*-
# @Time    : 2019/1/22 1:16 PM
# @File    : activelearning.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from antgo.resource.html import *
from antgo.ant.base import *
from antgo.activelearning.samplingmethods.kcenter_greedy import *
from antgo.crowdsource.activelearning_server import *
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antvis.client.httprpc import *
from multiprocessing import Process, Queue
from antgo.task.task import *
from scipy.stats import entropy
import subprocess
import os
import socket
import requests
import json


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


class AntActiveLearning(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config=None,
               **kwargs):
    super(AntActiveLearning, self).__init__(ant_name, ant_context, ant_token, **kwargs)

    self.skip_first_training = kwargs.get('skip_training', False)
    self.max_time = kwargs.get('max_time', '1d')
    self.max_iterators = getattr(self.context.params, 'max_iterators', 100)
    self.min_label_ratio = getattr(self.context.params, 'min_label_ratio', 0.1)
    self.dump_dir = ant_dump_dir

    self.web_server_port = kwargs.get('port', None)
    self.web_server_port = int(self.web_server_port) if self.web_server_port is not None else None
    self.html_template = kwargs.get('html_template', None)
    self.keywords_template = {}

    self.ant_task_config = ant_task_config
    self.task = kwargs.get('task', '')
    self.ant_data_source = ant_data_folder
    self.option = kwargs.get('option', '')
    self.root_expeirment = kwargs.get('from_experiment', None)
    self.devices = kwargs.get('devices', '')
    self.rpc = None

  def _core_set_algorithm(self, unlabeled_pool, num):
    feature_data = np.array([data['feature'] for data in unlabeled_pool])
    channels = feature_data.shape[-1]
    kcg = kCenterGreedy(feature_data.reshape(-1, channels))
    next_selected = kcg.select_batch(model=None, already_selected=[], N=num)
    next_selected = [unlabeled_pool[int(s)] for s in next_selected]

    return next_selected

  def _entroy_algorithm(self, unlabeled_pool, num):
    unlabeled_entropy = []
    for index, data in enumerate(unlabeled_pool):
      h = np.histogram(data['feature'].flatten(), 255)
      p = h[0].astype(float) / h[0].sum()   # probability of bins
      unlabeled_entropy.append((entropy(p), index))

    ordered_unlabeled = sorted(unlabeled_entropy, key=lambda x: x[0], reverse=True)
    next_selected = [unlabeled_pool[s[1]] for s in ordered_unlabeled[0:num]]
    return next_selected

  def _smart_select(self, unlabeled_pool, num):
    if self.option.lower() == 'coreset':
      return self._core_set_algorithm(unlabeled_pool, num)
    elif self.option.lower() == 'entropy':
      return self._entroy_algorithm(unlabeled_pool, num)
    else:
      return self._entroy_algorithm(unlabeled_pool, num)

  def start(self):
    # 0.step loading challenge task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      response = self.context.dashboard.challenge.get(command=type(self).__name__)
      if response['status'] is None:
        # invalid token
        logger.error('couldnt load challenge task')
        self.token = None
      elif response['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('prohibit submit challenge task frequently')
        exit(-1)
      elif response['status'] == 'OK':
        # maybe user token or task token
        content = response['content']
        if 'task' in content:
          challenge_task = create_task_from_json(content)
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

    # 配置html模板信息
    self.keywords_template['TASK_TITLE'] = running_ant_task.task_name
    self.keywords_template['TASK_TYPE'] = running_ant_task.task_type
    self.keywords_template['UNLABELED_NUM'] = 0
    self.keywords_template['LABEL_COMPLEX'] = 0

    # dataset
    dataset = running_ant_task.dataset('train',
                                       os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                       running_ant_task.dataset_params)

    # prepare workspace
    if not os.path.exists(os.path.join(self.main_folder, 'web', 'static', 'data')):
      os.makedirs(os.path.join(self.main_folder, 'web', 'static', 'data'))

    annotation_folder = os.path.join(self.main_folder, 'web', 'static', 'data', 'annotations')
    if not os.path.exists(annotation_folder):
      os.makedirs(annotation_folder)

    data_folder = os.path.join(self.main_folder, 'web', 'static', 'data', 'images')
    if not os.path.exists(data_folder):
      os.makedirs(data_folder)

    download_folder = os.path.join(self.main_folder, 'web', 'static', 'data', 'download')
    if not os.path.exists(download_folder):
      os.makedirs(download_folder)

    upload_folder = os.path.join(self.main_folder, 'web', 'static', 'data', 'upload')
    if not os.path.exists(upload_folder):
      os.makedirs(upload_folder)

    # 数据队列
    request_queue = Queue()

    # launch web server
    if self.web_server_port is None:
      self.web_server_port = 10000
    self.web_server_port = _pick_idle_port(self.web_server_port)

    logger.info('launch active learning web server on port %d'%self.web_server_port)
    process = multiprocessing.Process(target=activelearning_web_server,
                                      args=('activelearning',
                                            self.main_folder,
                                            self.html_template,
                                            self.keywords_template,
                                            running_ant_task,
                                            self.web_server_port,
                                            os.getpid(),
                                            download_folder,
                                            upload_folder,
                                            request_queue))
    process.daemon = True
    process.start()
    logger.info('waiting 5 seconds for launching web server')
    time.sleep(5)

    self.rpc = HttpRpc('v1','activelearning','127.0.0.1',self.web_server_port)
    avg_analyze_time = 0

    # prepare waiting unlabeled data
    try_iter = 0
    while try_iter < self.max_iterators:
      unlabeled_dataset_size = dataset.unlabeled_size()
      labeled_dataset_size = dataset.candidates_size()
      if unlabeled_dataset_size == 0:
        logger.info('active learning is over')
        return

      if not os.path.exists(self.dump_dir):
        os.makedirs(self.dump_dir)

      # 1.step training using labeled data
      for f in os.listdir(self.dump_dir):
        if f[0] == '.':
          continue

        if f == self.root_expeirment:
          continue

        shutil.rmtree(os.path.join(self.dump_dir, f))

      # 通知处理状态(开始未标注数据集挖掘)
      self.rpc.state.patch(round=try_iter,
                           process_state="UNLABEL-PREPARE")
      # 开始分析时间
      analyze_start_time = time.time()
      experiment_id = self.root_expeirment
      if not self.skip_first_training or try_iter > 0:
        # shell call
        logger.info('start training using all labeled data (%d iter)'%try_iter)
        cmd_shell = 'antgo train --main_file=%s --main_param=%s' % (self.main_file, self.main_param)
        cmd_shell += ' --max_time=%s' % self.max_time
        cmd_shell += ' --dump=%s' % self.dump_dir
        cmd_shell += ' --main_folder=%s' % self.main_folder
        cmd_shell += ' --task=%s' % self.task
        if self.root_expeirment is not None:
          cmd_shell += ' --from_experiment=%s' % self.root_expeirment
        cmd_shell += ' --devices=%s' % self.devices
        cmd_shell += ' --name=%s' % self.ant_name
        if self.token is not None:
          cmd_shell += ' --token=%s' % self.token
        training_p = subprocess.Popen('%s > %s.log' % (cmd_shell, self.name), shell=True, cwd=self.main_folder)

        # waiting untile finish training
        training_p.wait()

        # 根据返回结果，判断是否正常结束
        if training_p.returncode != 0:
          logger.error('training process exit anomaly')
          exit(-1)

        # 获取训练完成后的实验目录
        experiment_id = None
        for f in os.listdir(self.dump_dir):
          if f[0] == '.':
            continue

          if f == self.root_expeirment:
            continue

          if os.path.isdir(os.path.join(self.dump_dir, f)):
            experiment_id = f

        if experiment_id is None:
          logger.error('couldnt finding training experiment, maybe training and candidate dataset error')
          exit(-1)

      # 2.step inference using unlabeled data
      logger.info('start analyze all unlabeled data distribution (%d iter)'%try_iter)
      cmd_shell = 'antgo batch --main_file=%s --main_param=%s'%(self.main_file, self.main_param)
      cmd_shell += ' --main_folder=%s' % self.main_folder
      cmd_shell += ' --dump=%s' % self.dump_dir
      if experiment_id is not None:
        cmd_shell += ' --from_experiment=%s' % experiment_id
      cmd_shell += ' --task=%s' % self.task
      cmd_shell += ' --unlabel'
      cmd_shell += ' --devices=%s' % self.devices
      cmd_shell += ' --name=%s' % self.ant_name
      if self.token is not None:
        cmd_shell += ' --token=%s' % self.token
      inference_p = subprocess.Popen('%s > %s.log' %(cmd_shell, self.name), shell=True, cwd=self.main_folder)
      # all processed data are saved in dump/self.name/record/

      # waiting untile finish inference
      inference_p.wait()

      # 根据返回结果，判断是否正常结束
      if inference_p.returncode != 0:
        logger.error('inference process exit anomaly')
        exit(-1)

      # 获取推断完成后的实验目录
      inference_experiment_id = None
      for f in os.listdir(self.dump_dir):
        if f[0] == '.':
          continue

        if os.path.isdir(os.path.join(self.dump_dir, f)):
          if f != experiment_id and f != self.root_expeirment:
            inference_experiment_id = f

      if inference_experiment_id is None:
        logger.error('couldnt finding inference result on unlabeled dataset, maybe unlabeled dataset error')
        exit(-1)

      record_reader = RecordReader(os.path.join(self.dump_dir, inference_experiment_id, 'record'))
      unlabeled_pool = []
      for ss in record_reader.iterate_read('groundtruth', 'predict'):
        gt, feature = ss
        data_file_id = gt['file_id']
        unlabeled_pool.append({'file_id': data_file_id, 'feature': feature})

      select_size = int(len(unlabeled_pool) * self.min_label_ratio)
      if select_size == 0:
        select_size = len(unlabeled_pool)

      next_selected = self._smart_select(unlabeled_pool, select_size)
      round_size = len(next_selected)

      # 结束分析时间
      analyze_end_time = time.time()

      # 获得平均分析时间
      avg_analyze_time = (avg_analyze_time * try_iter + (analyze_end_time - analyze_start_time)) / (try_iter + 1)

      next_unlabeled_sample_ids = []
      for f in next_selected:
        next_unlabeled_sample_ids.append(f['file_id'])

      # tar target folder
      tar_file = "round_%d.tar.gz"%try_iter
      tar_path = os.path.join(download_folder, tar_file)
      if os.path.exists(tar_path):
        os.remove(tar_path)

      tar = tarfile.open(tar_path, "w:gz")
      for next_unlabeled_sample_id in next_unlabeled_sample_ids:
        tar.add(os.path.join(dataset.unlabeled_folder, next_unlabeled_sample_id),
                arcname="round_%d/%s"%(try_iter,next_unlabeled_sample_id))
      tar.close()

      # 通知处理状态(已经准备好未标注数据集)
      self.rpc.state.patch(round=try_iter,
                           process_state="UNLABEL-RESET",
                           unlabel_dataset=tar_file,
                           unlabeled_size=unlabeled_dataset_size,
                           labeled_size=labeled_dataset_size,
                           round_size=round_size)

      while True:
        logger.info('waiting label (human in loop) in round %d'%try_iter)
        request_content = request_queue.get()

        # 1.step 解压数据集文件
        logger.info("untar label dataset")
        request_label_dataset = request_content['FILE']
        request_round = request_content['ROUND']
        if request_round != try_iter:
          logger.error('request label round not consistent')
          self.rpc.state.patch(round=try_iter,
                               process_state="LABEL-ERROR")
          continue

        # 2.step 建立文件夹
        if os.path.exists(os.path.join(upload_folder, "round_%d"%request_round)):
          shutil.rmtree(os.path.join(upload_folder, "round_%d"%request_round))

        # uncompress
        tar = tarfile.open(os.path.join(upload_folder, request_label_dataset))
        tar.extractall(upload_folder)
        tar.close()

        # 3.step 检查数据集标准是否符合标准
        logger.info("check labeled dataset format")
        is_ok = dataset.check_candidate(next_unlabeled_sample_ids, os.path.join(upload_folder, "round_%d"%request_round))
        if not is_ok:
          logger.error("dataset format maybe error, need to update label")

          self.rpc.state.patch(round=try_iter,
                               process_state="LABEL-ERROR")
          continue

        break

      self.rpc.state.patch(round=try_iter,
                           process_state='LABEL-FINISH',
                           next_round_waiting=avg_analyze_time)

      logger.info('finish round %d label, start next round'%try_iter)
      for sample_file in next_unlabeled_sample_ids:
        if os.path.exists(os.path.join(upload_folder, 'round_%d'%try_iter, sample_file)):
          dataset.make_candidate(sample_file,
                                 os.path.join(upload_folder, 'round_%d'%try_iter, sample_file),
                                 'OK')

      # increment round
      try_iter += 1
