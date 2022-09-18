# -*- coding: UTF-8 -*-
# @Time    : 2019/1/22 1:16 PM
# @File    : activelearning.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
# from antgo.resource.html import *
# from antgo.ant.base import *
# from antgo.activelearning.samplingmethods.kcenter_greedy import *
# from antgo.crowdsource.activelearning_server import *
# from antgo.dataflow.common import *
# from antgo.dataflow.recorder import *
# from antvis.client.httprpc import *
# from multiprocessing import Process, Queue
# from antgo.task.task import *
# from scipy.stats import entropy
# import traceback
# import subprocess
# import os
# import socket
# import requests
# import json
# import zipfile

# def _is_open(check_ip, port):
#   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#   try:
#     s.connect((check_ip, int(port)))
#     s.shutdown(2)
#     return True
#   except:
#     return False


# def _pick_idle_port(from_port=40000, check_count=100):
#   check_port = from_port
#   while check_count:
#     if not _is_open('127.0.0.1', check_port):
#       break

#     logger.warn('Port %d is occupied, try to use %d port.'%(int(check_port), int(check_port + 1)))

#     check_port += 1
#     check_count -= 1

#     if check_count == 0:
#       check_port = None

#   if check_port is None:
#     logger.warn('Couldnt find valid free port.')
#     exit(-1)

#   return check_port


# class AntActiveLearning(AntBase):
#   def __init__(self,
#                ant_context,
#                ant_name,
#                ant_data_folder,
#                ant_dump_dir,
#                ant_token,
#                ant_task_config=None,
#                **kwargs):
#     super(AntActiveLearning, self).__init__(ant_name, ant_context, ant_token, **kwargs)

#     self.skip_first_training = self.context.params.system['skip_training']
#     self.max_iterators = self.context.params.activelearning.get('max_iterators', 10)
#     if self.max_iterators is None:
#       self.max_iterators = 10

#     self.dump_dir = ant_dump_dir
#     self.web_server_port = self.context.params.system['port']
#     self.web_server_port = int(self.web_server_port) if self.web_server_port is not None else None
#     self.html_template = kwargs.get('html_template', None)
#     self.keywords_template = {}

#     self.ant_task_config = ant_task_config
#     self.ant_data_source = ant_data_folder
#     self.devices = self.context.params.system['devices']
#     self.rpc = None

#   def _core_set_algorithm(self, unlabeled_pool, num):
#     if num > len(unlabeled_pool):
#       logger.warn("Dont have enough unlabeled data(%d), return all."%len(unlabeled_pool))
#       return unlabeled_pool

#     # data: 1XC (C表示特征维度)
#     feature_data = np.array([data['feature'] for data in unlabeled_pool])
#     channels = feature_data.shape[-1]
#     kcg = kCenterGreedy(feature_data.reshape(-1, channels))
#     next_selected = kcg.select_batch(model=None, already_selected=[], N=num)
#     next_selected = [unlabeled_pool[int(s)] for s in next_selected]

#     return next_selected

#   def _entroy_algorithm(self, unlabeled_pool, num):
#     if num > len(unlabeled_pool):
#       logger.warn("Dont have enough unlabeled data(%d), return all."%len(unlabeled_pool))
#       return unlabeled_pool

#     unlabeled_entropy = []
#     for index, data in enumerate(unlabeled_pool):
#       # data: DxC（D表示实验次数，C表示概率信息）
#       assert(len(data['feature'].shape) == 2)
#       p = np.mean(data['feature'], axis=0)
#       unlabeled_entropy.append((entropy(p), index))

#     ordered_unlabeled = sorted(unlabeled_entropy, key=lambda x: x[0], reverse=True)
#     next_selected = [unlabeled_pool[s[1]] for s in ordered_unlabeled[0:num]]
#     return next_selected

#   def _bald_algorithm(self, unlabeled_pool, num):
#     if num > len(unlabeled_pool):
#       logger.warn("Dont have enough unlabeled data(%d), return all."%len(unlabeled_pool))
#       return unlabeled_pool

#     unlabeled_entropy = []
#     for index, data in enumerate(unlabeled_pool):
#       # data: DxC（D表示实验次数，C表示概率信息）
#       assert(len(data['feature'].shape) == 2)
#       part1 = entropy(np.mean(data['feature'], axis=0))

#       experiment_num = data['feature'].shape[0]
#       part2 = []
#       for experiment_i in range(experiment_num):
#         part2.append(entropy(data['feature'][experiment_i:experiment_i+1,:]))
#       part2 = np.mean(part2)
#       unlabeled_entropy.append((part1-part2, index))

#     ordered_unlabeled = sorted(unlabeled_entropy, key=lambda x: x[0], reverse=True)
#     next_selected = [unlabeled_pool[s[1]] for s in ordered_unlabeled[0:num]]
#     return next_selected


#   def _variance_algorithm(self, unlabeled_pool, num):
#     if num > len(unlabeled_pool):
#       logger.warn("Dont have enough unlabeled data(%d), return all."%len(unlabeled_pool))      
#       return unlabeled_pool

#     unlabeled_var = []
#     for index, data in enumerate(unlabeled_pool):
#       # data: DxC (D表示试验次数，C表示概率信息)
#       assert(len(data['feature'].shape) == 2)
#       var_val = np.var(data['feature'], axis=0)
#       unlabeled_var.append(((float)(np.mean(var_val)), index))
    
#     ordered_unlabeled = sorted(unlabeled_var, key=lambda x: x[0], reverse=True)
#     next_selected = [unlabeled_pool[s[1]] for s in ordered_unlabeled[0:num]]
#     return next_selected

#   def _unform_sampling_algorithm(self, unlabeled_pool, num):
#     if num > len(unlabeled_pool):
#       logger.warn("Dont have enough unlabeled data(%d), return all."%len(unlabeled_pool))
#       return unlabeled_pool
    
#     return np.random.choice(unlabeled_pool, num, False)

#   def _waiting_label_sample_select(self, unlabeled_pool, num):
#     sampling_strategy = self.context.params.activelearning.get('sampling_strategy', 'entropy')
#     logger.info('Using %s Strategy to select waiting label data.'%sampling_strategy)
#     if sampling_strategy == 'coreset':
#       return self._core_set_algorithm(unlabeled_pool, num)
#     elif sampling_strategy == 'entropy':
#       return self._entroy_algorithm(unlabeled_pool, num)
#     elif sampling_strategy == 'bald':
#       return self._bald_algorithm(unlabeled_pool, num)
#     elif sampling_strategy == 'variance':
#       return self._variance_algorithm(unlabeled_pool, num)
#     elif sampling_strategy == 'random':
#       return self._unform_sampling_algorithm(unlabeled_pool, num)
#     else:
#       return self._unform_sampling_algorithm(unlabeled_pool, num)

#   def start(self):
#     # 0.step loading challenge task
#     running_ant_task = None
#     if self.token is not None:
#       # 0.step load challenge task
#       response = self.context.dashboard.challenge.get(command=type(self).__name__)
#       if response['status'] is None:
#         # invalid token
#         logger.error('Couldnt load challenge task.')
#         self.token = None
#       elif response['status'] == 'SUSPEND':
#         # prohibit submit challenge task frequently
#         # submit only one in one week
#         logger.error('Prohibit submit challenge task frequently.')
#         exit(-1)
#       elif response['status'] == 'OK':
#         # maybe user token or task token
#         content = response['content']
#         if 'task' in content:
#           challenge_task = create_task_from_json(content)
#           if challenge_task is None:
#             logger.error('Couldnt load challenge task.')
#             exit(-1)
#           running_ant_task = challenge_task
#       else:
#         # unknow error
#         logger.error('Unknow error.')
#         exit(-1)

#     if running_ant_task is None:
#       # 0.step load custom task
#       custom_task = create_task_from_xml(self.ant_task_config, self.context)
#       if custom_task is None:
#         logger.error('Couldnt load custom task.')
#         exit(0)
#       running_ant_task = custom_task

#     assert (running_ant_task is not None)

#     # 配置html模板信息
#     self.keywords_template['TASK_TITLE'] = running_ant_task.task_name
#     self.keywords_template['TASK_TYPE'] = running_ant_task.task_type
#     self.keywords_template['UNLABELED_NUM'] = 0
#     self.keywords_template['LABEL_COMPLEX'] = 0

#     # dataset
#     dataset = \
#         running_ant_task.dataset('train',os.path.join(self.ant_data_source, running_ant_task.dataset_name), running_ant_task.dataset_params)

#     # 如果处于研究状态，自动生成未标注数据（等价于训练集）
#     if self.context.params.system['research']:
#       unlabeled_folder = os.path.join(dataset.dir, dataset.unlabeled_tag)
#       if not os.path.exists(unlabeled_folder):
#         os.makedirs(unlabeled_folder)
        
#         # 生成空文件
#         for index in range(dataset.size):
#           with open('%s/%d'%(unlabeled_folder,index), 'w') as fp:
#               fp.write('')

#     use_local_candidate_file = False
#     if os.path.exists(os.path.join(self.main_folder, 'candidates.txt')):
#       use_local_candidate_file = True
#       logger.info('Using candidate file %s'%os.path.join(self.main_folder, 'candidates.txt'))
#       dataset.candidate_file = os.path.join(self.main_folder, 'candidates.txt')
  
#     # prepare workspace
#     if not os.path.exists(os.path.join(self.main_folder, 'show', 'static', 'data')):
#       os.makedirs(os.path.join(self.main_folder, 'show', 'static', 'data'))

#     annotation_folder = os.path.join(self.main_folder, 'show', 'static', 'data', 'annotations')
#     if not os.path.exists(annotation_folder):
#       os.makedirs(annotation_folder)

#     data_folder = os.path.join(self.main_folder, 'show', 'static', 'data', 'images')
#     if not os.path.exists(data_folder):
#       os.makedirs(data_folder)

#     download_folder = os.path.join(self.main_folder, 'show', 'static', 'data', 'download')
#     if not os.path.exists(download_folder):
#       os.makedirs(download_folder)

#     upload_folder = os.path.join(self.main_folder, 'show', 'static', 'data', 'upload')
#     if not os.path.exists(upload_folder):
#       os.makedirs(upload_folder)

#     # 数据队列
#     request_queue = Queue()

#     # launch show server
#     if self.web_server_port is None:
#       self.web_server_port = 10000
#     self.web_server_port = _pick_idle_port(self.web_server_port)

#     logger.info('Launch active learning show server on port %d.'%self.web_server_port)
#     process = multiprocessing.Process(target=activelearning_web_server,
#                                       args=('activelearning',
#                                             self.main_folder,
#                                             self.html_template,
#                                             self.keywords_template,
#                                             running_ant_task,
#                                             self.web_server_port,
#                                             os.getpid(),
#                                             download_folder,
#                                             upload_folder,
#                                             request_queue))
#     process.daemon = True
#     process.start()
#     logger.info('Waiting 5 seconds for launching show server.')
#     time.sleep(5)

#     self.rpc = HttpRpc('v1','activelearning','127.0.0.1',self.web_server_port)
#     avg_analyze_time = 0

#     # prepare waiting unlabeled data
#     try_iter = 0
#     experiment_id = None
#     while try_iter < self.max_iterators:
#       logger.info('Finding unlabeled and cnadidates size.')
#       unlabeled_dataset_size = dataset.unlabeled_size()
#       labeled_dataset_size = dataset.candidates_size()
      
#       logger.info('Unlabeled data size %d, Labeled data size %d'%(unlabeled_dataset_size, labeled_dataset_size))
#       min_sampling_num = self.context.params.activelearning.get('min_sampling_num', None)
#       if min_sampling_num is not None:
#         min_sampling_num = (int)(min_sampling_num)
#         if unlabeled_dataset_size < min_sampling_num:
#           logger.info('Active learning is over. (unlabeled sample size %d < %d).'%(unlabeled_dataset_size, min_sampling_num))
#           return

#       if unlabeled_dataset_size < 10:
#         logger.info('Active learning is over. (unlabeled sample size %d < 10).'%unlabeled_dataset_size)
#         return

#       logger.info("Round %d, unlabeled dataset size %d, labeled dataset size %d."%(try_iter, unlabeled_dataset_size, labeled_dataset_size))
#       if not os.path.exists(self.dump_dir):
#         os.makedirs(self.dump_dir)

#       # 当前阶段
#       self.stage = "ACTIVELEARNING-TRAIN-ROUND-%d"%try_iter

#       # 通知处理状态(开始未标注数据集挖掘)
#       self.rpc.state.patch(round=try_iter, process_state="UNLABEL-PREPARE")
#       # 开始分析时间
#       analyze_start_time = time.time()
#       if try_iter == 0:
#         experiment_id = self.context.from_experiment

#       if (not self.skip_first_training or try_iter > 0) and labeled_dataset_size > 0:
#         # shell call
#         logger.info('Start training using all labeled data (%d iter).'%try_iter)

#         if os.path.exists(os.path.join(self.dump_dir, 'try_round_train_%d'%try_iter)):
#           shutil.rmtree(os.path.join(self.dump_dir, 'try_round_train_%d'%try_iter))
#         os.makedirs(os.path.join(self.dump_dir, 'try_round_train_%d'%try_iter))

#         cmd_shell = 'antgo train --main_file=%s --main_param=%s' % (self.main_file, self.main_param)
#         cmd_shell += ' --dump=%s/%s' % (self.dump_dir, 'try_round_train_%d'%try_iter)
#         cmd_shell += ' --main_folder=%s' % self.main_folder
#         cmd_shell += ' --task=%s' % self.ant_task_config.split('/')[-1]
#         if experiment_id is not None:
#           cmd_shell += ' --from_experiment=%s' % experiment_id
#         cmd_shell += ' --candidate'
#         cmd_shell += ' --devices=%s' % self.devices
#         cmd_shell += ' --dataset=%s/train' % running_ant_task.dataset_name
#         cmd_shell += ' --name=%s_train_round_%d' % (self.ant_name, try_iter)
#         if use_local_candidate_file:
#           cmd_shell += ' --param=candidate_file:%s/candidates.txt'%self.main_folder
#         training_p = \
#             subprocess.Popen('%s > %s.log' % (cmd_shell, '%s_try_rounnd_train_%d'%(self.name, try_iter)), 
#                               shell=True, 
#                               cwd=self.main_folder)

#         # waiting untile finish training
#         training_p.wait()

#         # 根据返回结果，判断是否正常结束
#         if training_p.returncode != 0:
#           logger.error('Training process exit anomaly.')
#           exit(-1)

#         # 获取训练完成后的实验目录ß
#         experiment_prefix = 'try_round_train_%d'%try_iter
#         experiment_id = None
#         for k in os.listdir(os.path.join(self.dump_dir, experiment_prefix)):
#           if k[0] == '.':
#             continue

#           if os.path.isdir(os.path.join(self.dump_dir, experiment_prefix, k)):
#             experiment_id = os.path.join(self.dump_dir, experiment_prefix, k)
#             break
      
#       # 2.step inference using unlabeled data
#       # 当前阶段
#       self.stage = "ACTIVELEARNING-ANALYZE-ROUND-%d"%try_iter
#       if os.path.exists(os.path.join(self.dump_dir, 'try_round_analyze_%d'%try_iter)):
#         shutil.rmtree(os.path.join(self.dump_dir, 'try_round_analyze_%d'%try_iter))
#       os.makedirs(os.path.join(self.dump_dir, 'try_round_analyze_%d'%try_iter))        

#       logger.info('Start analyze all unlabeled data distribution (%d iter).'%try_iter)
#       cmd_shell = 'antgo predict --main_file=%s --main_param=%s'%(self.main_file, self.main_param)
#       cmd_shell += ' --main_folder=%s' % self.main_folder
#       cmd_shell += ' --dump=%s/%s' % (self.dump_dir, 'try_round_analyze_%d'%try_iter)
#       if experiment_id is not None and experiment_id != '':
#         cmd_shell += ' --from_experiment=%s' % experiment_id
#       cmd_shell += ' --task_t=%s' % running_ant_task.task_type
#       cmd_shell += ' --unlabel'
#       cmd_shell += ' --devices=%s' % self.devices
#       cmd_shell += ' --dataset=%s/train' % running_ant_task.dataset_name
#       cmd_shell += ' --name=%s_predict_round_%d' % (self.ant_name, try_iter)
#       if use_local_candidate_file:
#           cmd_shell += ' --param=candidate_file:%s/candidates.txt'%self.main_folder      
#       inference_p = subprocess.Popen('%s > %s.log' %(cmd_shell, '%s_try_rounnd_analyze_%d'%(self.name, try_iter)), 
#                                       shell=True, 
#                                       cwd=self.main_folder)

#       # waiting untile finish inference
#       inference_p.wait()

#       # 根据返回结果，判断是否正常结束
#       if inference_p.returncode != 0:
#         logger.error('Inference process exit anomaly.')
#         exit(-1)

#       # 获取推断完成后的实验目录
#       inference_experiment_prefix = 'try_round_analyze_%d'%try_iter
#       inference_experiment_id = ''
#       for k in os.listdir(os.path.join(self.dump_dir, inference_experiment_prefix)):
#         if k[0] == '.':
#           continue

#         if os.path.isdir(os.path.join(self.dump_dir, inference_experiment_prefix, k)):
#           inference_experiment_id = k
#           break
      
#       record_reader = RecordReader(os.path.join(self.dump_dir, inference_experiment_prefix, inference_experiment_id, 'record'))
#       unlabeled_pool = []
#       for ss in record_reader.iterate_read('groundtruth', 'predict'):
#         gt, feature = ss
#         unlabeled_pool.append({'file_id': gt['file_id'], 'feature': feature, 'id': gt['id']})

#       select_size = self.context.params.activelearning.get('min_sampling_num', None)
#       if select_size is None:
#         min_sampling_ratio = self.context.params.activelearning.get('min_sampling_ratio', None)
#         if min_sampling_ratio is None:
#           min_sampling_ratio = 0.1
#         select_size = int(len(unlabeled_pool) * min_sampling_ratio)
#         if select_size == 0:
#           select_size = len(unlabeled_pool)

#       if select_size == 0:
#         logger.info('Active learning is over. (selecting size == 0.')
#         return

#       next_selected = self._waiting_label_sample_select(unlabeled_pool, select_size)
#       if len(next_selected) == 0:
#         logger.info("Active learning is over. (selecting size == 0).")
#         return

#       logger.info("Round %d, selecting size %d by %s method."%(try_iter, select_size, self.context.params.activelearning.get('sampling_strategy', 'entropy')))

#       # 结束分析时间
#       analyze_end_time = time.time()

#       # 获得平均分析时间
#       avg_analyze_time = (avg_analyze_time * try_iter + (analyze_end_time - analyze_start_time)) / (try_iter + 1)

#       next_unlabeled_sample_ids = []
#       for f in next_selected:
#         next_unlabeled_sample_ids.append((f['file_id'], f['id']))

#       # 打包等待下一步进行标注的样本
#       tar_file = "round_%d.tar.gz"%try_iter
#       tar_path = os.path.join(download_folder, tar_file)
#       if os.path.exists(tar_path):
#         os.remove(tar_path)

#       tar = tarfile.open(tar_path, "w:gz")
#       for next_unlabeled_sample_id, _ in next_unlabeled_sample_ids:
#         tar.add(os.path.join(dataset.dir, next_unlabeled_sample_id),
#                 arcname="round_%d/%s"%(try_iter,next_unlabeled_sample_id))
#       tar.close()

#       # 研究模式，在研究模式下，标注结果自动获取
#       if self.context.params.system['research']:
#         # 获取标注数据，并打包保存
#         logger.info("Research: auto get label data.")
#         if not os.path.exists(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter)):
#           os.makedirs(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter))

#         for file_id, id in next_unlabeled_sample_ids:
#           _, label = dataset.at(id, file_id)
#           label.update({'file_id': file_id, 'id': id})
#           # 去除不可序列化数据
#           filter_label = {}
#           for a, b in label.items():
#             if type(b) != float and type(b) != int and type(b) != list and type(b) != dict and type(b) != tuple and type(b) != str:
#               # 不可序列化数据
#               continue

#             filter_label[a] = b 
          
#           label = filter_label
#           # 自动生成子目录
#           if '/' in file_id:
#             if not os.path.exists(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter, file_id.split('/')[0])):
#               os.makedirs(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter, file_id.split('/')[0]))

#           # label 写成文件
#           with open(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter, file_id), 'w') as fp:
#             json.dump(label, fp)

#         # 使用tar,打包
#         logger.info("Research: warp label data.")
#         tar = tarfile.open(os.path.join(self.dump_dir, 'try_round_auto_label_%d.tar.gz'%try_iter), "w:gz")
#         for file_id, _ in next_unlabeled_sample_ids:
#           tar.add(os.path.join(self.dump_dir, 'try_round_auto_label_%d'%try_iter, file_id),
#                   arcname="try_round_auto_label_%d/%s"%(try_iter, file_id))
#         tar.close()

#         logger.info("Research: copy to %s."%upload_folder)
#         shutil.copy(os.path.join(self.dump_dir, 'try_round_auto_label_%d.tar.gz'%try_iter), upload_folder)
#         request_queue.put({'FILE': 'try_round_auto_label_%d.tar.gz'%try_iter, 'ROUND': try_iter})

#       # 通知处理状态(已经准备好未标注数据集)
#       self.rpc.state.patch(round=try_iter,
#                            process_state="UNLABEL-RESET",
#                            unlabel_dataset=tar_file,
#                            unlabeled_size=unlabeled_dataset_size,
#                            labeled_size=labeled_dataset_size,
#                            round_size=len(next_selected))

#       while True:
#         logger.info('Waiting label (human in loop) in round %d.'%try_iter)
#         request_content = request_queue.get()

#         # 1.step 解压数据集文件
#         logger.info("Untar label dataset.")
#         request_label_dataset = request_content['FILE']
#         request_round = request_content['ROUND']
#         if request_round != try_iter:
#           logger.error('Request label round not consistent.')
#           self.rpc.state.patch(round=try_iter, process_state="LABEL-ERROR")
#           continue

#         # 2.step 建立文件夹
#         if os.path.exists(os.path.join(upload_folder, "round_%d"%request_round)):
#           shutil.rmtree(os.path.join(upload_folder, "round_%d"%request_round))

#         # uncompress
#         if request_label_dataset.endswith('tar') or request_label_dataset.endswith('tar.gz'): 
#           tar = tarfile.open(os.path.join(upload_folder, request_label_dataset))
#           tar_list = tar.getnames()
#           tar.extractall(upload_folder)
#           tar.close()

#           if len(tar_list) == 0:
#             logger.error('Untar file error.')
#             continue

#           if not os.path.exists(os.path.join(upload_folder, "round_%d"%request_round)):
#             # 将解压后的文件夹，修改为 "round_%d"%request_round
#             file_name = os.path.normpath(tar_list[0]).split('/')[0]
#             shutil.move(os.path.join(upload_folder, file_name), os.path.join(upload_folder, "round_%d"%request_round))
#         elif request_label_dataset.endswith('zip'):
#           zip_file = zipfile.ZipFile(os.path.join(upload_folder, request_label_dataset))
#           zip_list = zip_file.namelist()
#           for f in zip_list:
#               zip_file.extract(f, upload_folder)
#           zip_file.close()

#           if len(zip_list) == 0:
#             logger.error('Unzip file error.')
#             continue

#           if not os.path.exists(os.path.join(upload_folder, "round_%d"%request_round)):
#             # 将解压后的文件夹，修改为 "round_%d"%request_round
#             file_name = os.path.normpath(zip_list[0]).split('/')[0]
#             shutil.move(os.path.join(upload_folder, file_name), os.path.join(upload_folder, "round_%d"%request_round))
#         else:
#           logger.error('Dont support upload type.')
#           continue
        
#         # 3.step 检查数据集标准是否符合标准
#         logger.info("Check labeled dataset format.")
#         is_ok = dataset.check_candidate(next_unlabeled_sample_ids, os.path.join(upload_folder, "round_%d"%request_round))
#         if not is_ok:
#           logger.error("Dataset format maybe error, need to update label.")
#           self.rpc.state.patch(round=try_iter, process_state="LABEL-ERROR")
#           continue

#         break

#       self.rpc.state.patch(round=try_iter,
#                            process_state='LABEL-FINISH',
#                            next_round_waiting=avg_analyze_time)

#       logger.info('Prepare candidates.txt by using new labeled data.')
#       for sample_file, sample_id in next_unlabeled_sample_ids:
#         if os.path.exists(os.path.join(upload_folder, 'round_%d'%try_iter, sample_file)):
#           dataset.make_candidate(sample_id,
#                                   sample_file,
#                                   os.path.join(upload_folder, 'round_%d'%try_iter, sample_file),
#                                   'OK')

#       logger.info('Finish round %d label, start next round.'%try_iter)
#       # increment round
#       try_iter += 1
