# -*- coding: UTF-8 -*-
# @Time    : 17-6-22
# @File    : train.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import tarfile
from datetime import datetime
from multiprocessing import Process
from antgo.ant import flags
from antgo.ant.base import *
from antgo.ant.utils import *
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antgo.measures.statistic import *
from antgo.task.task import *
from antgo.utils.netvis import *
from antgo.utils.serialize import *
from antgo.resource.html import *
from antgo.utils.concurrency import *
from antgo.measures.statistic import *
from antgo.measures.repeat_statistic import *
from antgo.measures.deep_analysis import *
import signal
import traceback
if sys.version > '3':
    PY3 = True
else:
    PY3 = False

FLAGS = flags.AntFLAGS


class AntTrain(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config,
               **kwargs):
    super(AntTrain, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.ant_data_source = ant_data_folder
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config
    self.skip_training = kwargs.get('skip_training', False)
    self.context.devices = [int(d) for d in kwargs.get('devices', '').split(',') if d != '']

  def error_analysis(self, running_ant_task, running_ant_dataset, task_running_statictic):
    # error analysis
    logger.info('Start error analysis.')
    # task_running_statictic={self.ant_name:
    #                           {'measure':[
    #                             {'statistic': {'name': 'MESR',
    #                                            'value': [{'name': 'MESR', 'value': 0.4, 'type':'SCALAR'}]},
    #                                            'info': [{'id':0,'score':0.8,'category':1},
    #                                                     {'id':1,'score':0.3,'category':1},
    #                                                     {'id':2,'score':0.9,'category':1},
    #                                                     {'id':3,'score':0.5,'category':1},
    #                                                     {'id':4,'score':1.0,'category':1}]},
    #                             {'statistic': {'name': "SE",
    #                                            'value': [{'name': 'SE', 'value': 0.5, 'type': 'SCALAR'}]},
    #                                            'info': [{'id':0,'score':0.4,'category':1},
    #                                                     {'id':1,'score':0.2,'category':1},
    #                                                     {'id':2,'score':0.1,'category':1},
    #                                                     {'id':3,'score':0.5,'category':1},
    #                                                     {'id':4,'score':0.23,'category':1}]}]}}
    
    # stage-1 error statistic analysis
    for measure_result in task_running_statictic[self.ant_name]['measure']:
      if 'info' in measure_result and len(measure_result['info']) > 0:
        measure_name = measure_result['statistic']['name']
        measure_data = measure_result['info']
    
        # independent analysis per category for classification problem
        measure_data_list = []
        if running_ant_task.class_label is not None and len(running_ant_task.class_label) > 1:
          if running_ant_task.class_label is not None:
            for cl_i, cl in enumerate(running_ant_task.class_label):
              measure_data_list.append([md for md in measure_data if md['category'] == cl or md['category'] == cl_i])
      
        if len(measure_data_list) == 0:
          measure_data_list.append(measure_data)
    
        for category_id, category_measure_data in enumerate(measure_data_list):
          if len(category_measure_data) == 0:
            continue
      
          if 'analysis' not in task_running_statictic[self.ant_name]:
            task_running_statictic[self.ant_name]['analysis'] = {}
      
          if measure_name not in task_running_statictic[self.ant_name]['analysis']:
            task_running_statictic[self.ant_name]['analysis'][measure_name] = {}
      
          # reorganize as list
          method_samples_list = [{'name': self.ant_name, 'data': category_measure_data}]

          # reorganize data as score matrix
          method_num = len(method_samples_list)
          # samples_num are the same among methods
          samples_num = len(method_samples_list[0]['data'])
          # samples_num = ant_test_dataset.size
          method_measure_mat = np.zeros((method_num, samples_num))
          samples_map = []
      
          for method_id, method_measure_data in enumerate(method_samples_list):
            # reorder data by index
            order_key = 'id'
            if 'index' in method_measure_data['data'][0]:
              order_key = 'index'
            method_measure_data_order = sorted(method_measure_data['data'], key=lambda x: x[order_key])
        
            if method_id == 0:
              # record sample id
              for sample_id, sample in enumerate(method_measure_data_order):
                samples_map.append(sample)
        
            # order consistent
            for sample_id, sample in enumerate(method_measure_data_order):
              method_measure_mat[method_id, sample_id] = sample['score']
      
          is_binary = False
          # collect all score
          test_score = [td['score'] for td in method_samples_list[0]['data']
                        if td['score'] > -float("inf") and td['score'] < float("inf")]
          hist, x_bins = np.histogram(test_score, 100)
          if len(np.where(hist > 0.0)[0]) <= 2:
            is_binary = True
      
          # score matrix analysis
          if not is_binary:
            s, ri, ci, lr_samples, mr_samples, hr_samples = \
              continuous_multi_model_measure_analysis(method_measure_mat, samples_map, running_ant_dataset)
        
            analysis_tag = 'Global'
            if len(measure_data_list) > 1:
              analysis_tag = 'Global-Category-' + str(running_ant_task.class_label[category_id])
        
            model_name_ri = [method_samples_list[r]['name'] for r in ri]
            task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = \
              {'value': s.tolist() if type(s) != list else s,
               'type': 'MATRIX',
               'x': ci,
               'y': model_name_ri,
               'sampling': [{'name': 'High Score Region', 'data': hr_samples},
                            {'name': 'Middle Score Region', 'data': mr_samples},
                            {'name': 'Low Score Region', 'data': lr_samples}]}
        
            # group by tag
            tags = getattr(running_ant_dataset, 'tag', None)
            if tags is not None:
              for tag in tags:
                g_s, g_ri, g_ci, g_lr_samples, g_mr_samples, g_hr_samples = \
                  continuous_multi_model_measure_analysis(method_measure_mat,
                    samples_map,
                    running_ant_dataset,
                    filter_tag=tag)
            
                analysis_tag = 'Group'
                if len(measure_data_list) > 1:
                  analysis_tag = 'Group-Category-' + str(running_ant_task.class_label[category_id])
            
                if analysis_tag not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                  task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = []
            
                model_name_ri = [method_samples_list[r]['name'] for r in g_ri]
                tag_data = {'value': g_s.tolist() if type(g_s) != list else g_s,
                            'type': 'MATRIX',
                            'x': g_ci,
                            'y': model_name_ri,
                            'sampling': [{'name': 'High Score Region', 'data': g_hr_samples},
                                         {'name': 'Middle Score Region', 'data': g_mr_samples},
                                         {'name': 'Low Score Region', 'data': g_lr_samples}]}
            
                task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag].append((tag, tag_data))
          else:
            s, ri, ci, region_95, region_52, region_42, region_13, region_one, region_zero = \
              discrete_multi_model_measure_analysis(method_measure_mat,
                samples_map,
                running_ant_dataset)
        
            analysis_tag = 'Global'
            if len(measure_data_list) > 1:
              analysis_tag = 'Global-Category-' + str(running_ant_task.class_label[category_id])
        
            model_name_ri = [method_samples_list[r]['name'] for r in ri]
            task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = \
              {'value': s.tolist() if type(s) != list else s,
               'type': 'MATRIX',
               'x': ci,
               'y': model_name_ri,
               'sampling': [{'name': '95%', 'data': region_95},
                            {'name': '52%', 'data': region_52},
                            {'name': '42%', 'data': region_42},
                            {'name': '13%', 'data': region_13},
                            {'name': 'best', 'data': region_one},
                            {'name': 'zero', 'data': region_zero}]}
        
            # group by tag
            tags = getattr(running_ant_dataset, 'tag', None)
            if tags is not None:
              for tag in tags:
                g_s, g_ri, g_ci, g_region_95, g_region_52, g_region_42, g_region_13, g_region_one, g_region_zero = \
                  discrete_multi_model_measure_analysis(method_measure_mat,
                    samples_map,
                    running_ant_dataset,
                    filter_tag=tag)
                # if 'group' not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                #   task_running_statictic[self.ant_name]['analysis'][measure_name]['group'] = []
                #
                analysis_tag = 'Group'
                if len(measure_data_list) > 1:
                  analysis_tag = 'Group-Category-' + str(running_ant_task.class_label[category_id])
            
                if analysis_tag not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                  task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = []
            
                model_name_ri = [method_samples_list[r]['name'] for r in g_ri]
                tag_data = {'value': g_s.tolist() if type(g_s) != list else g_s,
                            'type': 'MATRIX',
                            'x': g_ci,
                            'y': model_name_ri,
                            'sampling': [{'name': '95%', 'data': region_95},
                                         {'name': '52%', 'data': region_52},
                                         {'name': '42%', 'data': region_42},
                                         {'name': '13%', 'data': region_13},
                                         {'name': 'best', 'data': region_one},
                                         {'name': 'zero', 'data': region_zero}]}
            
                task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag].append((tag, tag_data))
    
    # stage-2 error eye analysis (all or subset in wrong samples)
    eye_analysis_set_size = int(getattr(running_ant_task, 'badcase_size', 100))
    focus_category = int(getattr(running_ant_task, 'badcase_category', 1))
    
    for measure_result in task_running_statictic[self.ant_name]['measure']:
      # analyze measure result
      measure_name = None
      measure_data = None
      if 'info' in measure_result and len(measure_result['info']) > 0:
        measure_name = measure_result['statistic']['name']
        measure_data = measure_result['info']
      
      if measure_name is None or measure_data is None:
        continue

      focus_measure_data = [x for x in measure_data if x['category'] == focus_category]
      if len(focus_measure_data) == 0:
        continue

      measure_obj, = running_ant_task.evaluation_measure(measure_name)
      eye_analysis_error = []
      eye_analysis_all = []
      for sample_result_info in focus_measure_data:
        sample_id = sample_result_info['id']
        sample_score = sample_result_info['score']
        sample_category = sample_result_info['category'] if 'category' in sample_result_info else '-'
        eye_analysis_all.append((sample_id, sample_score, sample_category))
      
      if len(eye_analysis_all) == 0:
        continue
      eye_analysis_all_sort = sorted(eye_analysis_all, key=lambda x: x[1])

      eye_analysis_set_size = min(eye_analysis_set_size, len(eye_analysis_all_sort))
      eye_analysis_set = []
      if measure_obj.larger_is_better:
        eye_analysis_set = eye_analysis_all_sort[0:eye_analysis_set_size]
      else:
        eye_analysis_set = eye_analysis_all_sort[-eye_analysis_set_size:]

      for id, score, category in eye_analysis_set:
        data_attribute_info = {}
        data_attribute_info['id'] = '%d-(c %s)'%(id, str(category))
        data_attribute_info['score'] = score
        
        data_attribute_info['tag'] = []
        sample, label = running_ant_dataset.at(int(id))
        if 'tag' in label:
          if type(label['tag']) == list or type(label['tag']) == tuple:
            data_attribute_info['tag'].extend(list(label['tag']))
          else:
            data_attribute_info['tag'].append(label['tag'])

        if type(sample) == np.ndarray:
          if len(sample.shape) == 2:
            image = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255).astype(np.uint8)
            data_attribute_info['data_type'] = 'IMAGE'
            data_attribute_info['data'] = base64.b64encode(png_encode(image, True)).decode('utf-8')
          elif len(sample.shape) == 3:
            image = sample.astype(np.uint8)
            data_attribute_info['data_type'] = 'IMAGE'
            data_attribute_info['data'] = base64.b64encode(png_encode(image, True)).decode('utf-8')
          elif type(sample) == str:
            data_attribute_info['data_type'] = 'STRING'
            data_attribute_info['data'] = sample
          else:
            # TODO support multi images(gif)
            pass

        data_attribute_info['category'] = category
        data_attribute_info['category_type'] = 'STRING'
        if running_ant_task.task_type == 'SEGMENTATION':
          label_map = label['segmentation_map']
          label_map = ((label_map - np.min(label_map)) / (np.max(label_map) - np.min(label_map)) * 255).astype(np.uint8)
          data_attribute_info['category'] = base64.b64encode(png_encode(label_map, True)).decode('utf-8')
          data_attribute_info['category_type'] = 'IMAGE'
        eye_analysis_error.append(data_attribute_info)
      
      if 'eye' not in task_running_statictic[self.ant_name]:
          task_running_statictic[self.ant_name]['eye'] = {}

      task_running_statictic[self.ant_name]['eye'][measure_name] = eye_analysis_error
    return task_running_statictic

  def start(self):
    # 1.step 加载训练任务
    running_ant_task = None
    if self.token is not None:
      # 1.1.step load train task
      response = mlogger.getEnv().dashboard.challenge.get(command=type(self).__name__)
      if response['status'] == 'ERROR':
        # invalid token
        logger.error('Couldnt load train task.')
        self.token = None
      elif response['status'] in ['OK', 'SUSPEND']:
        content = response['content']

        # maybe user token or task token
        if 'task' in content:
          # task token
          challenge_task = create_task_from_json(content)
          if challenge_task is None:
            logger.error('Couldnt load train task.')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('Unknow error.')
        exit(-1)

    if running_ant_task is None:
      # 1.2.step load custom task
      if self.ant_task_config is not None:
        custom_task = create_task_from_xml(self.ant_task_config, self.context)
        if custom_task is None:
          logger.error('Couldnt load custom task.')
          exit(-1)
        running_ant_task = custom_task

    assert (running_ant_task is not None)

    # 2.step 注册实验
    experiment_uuid = self.context.experiment_uuid

    # 3.step 分析消除配置
    # analyze ablation blocks
    # user custom devices
    apply_devices = getattr(self.context.params, 'devices', [])
    if apply_devices is None:
      apply_devices = []

    # ablation experiment
    ablation_blocks = getattr(self.context.params, 'ablation', None)
    ablation_method = getattr(self.context.params, 'ablation_method', 'regular')
    if ablation_method is None:
      ablation_method = 'regular'
    assert(ablation_method in ['regular', 'inregular', 'accumulate', 'any', 'fixed'])
    ablation_experiments_devices_num = 0
    if ablation_blocks is not None:
      ablation_experiments_devices_num = 0
      if ablation_method in ['regular', 'inregular', 'accumulate']:
        ablation_experiments_devices_num = len(ablation_blocks)
        if ablation_method == 'inregular' and len(ablation_blocks) == 1:
          logger.warn('Only exists one ablation block %s, couldnt set inregular ablation method.' % ablation_blocks[0])
          ablation_method = 'regular'
      elif ablation_method == 'any':
        for i in range(len(ablation_blocks)):
          ablation_experiments_devices_num += len(list(itertools.combinations(ablation_blocks, i + 1)))
      else:
        # fixed ablation method
        ablation_experiments_devices_num = 0

    # 4.step  备份实验基本信息
    # 4.1.step 打包代码，并上传至云端
    self.stage = 'TRAIN'
    # - backup in dump_dir
    main_folder = self.main_folder
    if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
      os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))
    goldcoin = os.path.join(self.ant_dump_dir, experiment_uuid, 'code.tar.gz')
    
    if os.path.exists(goldcoin):
      os.remove(goldcoin)

    logger.info('Prepare package model files.')
    tar = tarfile.open(goldcoin, 'w:gz')

    # 排除dump目录，将当前文件夹下所有数据上传
    for root, dirs, files in os.walk(main_folder):
      if os.path.commonprefix([root, self.ant_dump_dir]) == self.ant_dump_dir:
        continue

      rel_root = os.path.relpath(root, main_folder)
      for f in files:
        tar.add(os.path.join(root, f), arcname=os.path.join(rel_root, f))

    tar.close()
    logger.info('Finish package process.')

    # TODO: 在下一个版本支持模型文件上传
    # # 上传模型代码
    # mlogger.getEnv().dashboard.experiment.upload(MODEL=goldcoin, APP_STAGE=self.stage)
    # logger.info('finish upload model files')

    # 4.2.step 更新实验配置
    if self.app_token is not None or self.user_token is not None:
      mlogger.getEnv().dashboard.experiment.patch(experiment_uuid=experiment_uuid,
                                                  experiment_hyper_parameter=json.dumps(self.ant_context.params.content))

    # 5.step 加载训练数据集
    logger.info('Loading train dataset %s.'%running_ant_task.dataset_name)
    ant_train_dataset = running_ant_task.dataset('train',
                                                 os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                 running_ant_task.dataset_params)

    # 6.step 消除实验
    # if ablation_blocks is not None:
    #   if len(apply_devices) >= ablation_experiments_devices_num + 1:
    #     ablation_experiments = []
    #     if ablation_method != 'fixed':
    #       ablation_experiments_devices = apply_devices[:ablation_experiments_devices_num]
    #       apply_devices = apply_devices[ablation_experiments_devices_num:]
    #
    #       # assign device to every ablation experiment
    #       ablation_experiments = self.start_ablation_train_proc(ant_train_dataset,
    #                                                             running_ant_task,
    #                                                             ablation_blocks,
    #                                                             ablation_method,
    #                                                             experiment_uuid,
    #                                                             ablation_experiments_devices)
    #       # launch all ablation experiments
    #       logger.info('waiting until all ablation experiments finish')
    #       for ablation_experiment in ablation_experiments:
    #         ablation_experiment.start()
    #
    #     if ablation_method == 'fixed':
    #       for b in ablation_blocks:
    #         self.ant_context.deactivate_block(b)
    #
    #     # launch complete model training process
    #     self.context.params.devices = apply_devices
    #     num_clones = getattr(self.context.params, 'num_clones', None)
    #     if num_clones is not None:
    #       self.context.params.num_clones = len(apply_devices)
    #
    #     self.stage = "TRAIN"
    #     train_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid, 'train')
    #     if not os.path.exists(train_dump_dir):
    #       os.makedirs(train_dump_dir)
    #
    #     if not self.skip_training:
    #       logger.info('start training process with complete model')
    #       with safe_recorder_manager(ant_train_dataset):
    #         ant_train_dataset.reset_state()
    #         self.context.call_training_process(ant_train_dataset, train_dump_dir)
    #       logger.info('stop main training process with complete model')
    #
    #     # start evaluation and error analysis
    #     logger.info('start evaluation process with complete model')
    #     try:
    #       _, validation_dataset = ant_train_dataset.split(split_params={}, split_method='holdout')
    #       data_annotation_branch = DataAnnotationBranch(Node.inputs(validation_dataset))
    #       self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
    #
    #       intermediate_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid, 'record')
    #       with safe_recorder_manager(self.context.recorder):
    #         self.context.recorder.dump_dir = intermediate_dump_dir
    #         with performance_statistic_region(self.ant_name):
    #           self.context.call_infer_process(data_annotation_branch.output(0), train_dump_dir)
    #       self.context.recorder.close()
    #
    #       task_running_statictic = get_performance_statistic(self.ant_name)
    #       task_running_statictic = {self.ant_name: task_running_statictic}
    #       task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
    #       task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
    #         task_running_elapsed_time / float(validation_dataset.size)
    #
    #       evaluation_measure_result = []
    #       with safe_recorder_manager(RecordReader(intermediate_dump_dir)) as record_reader:
    #         for measure in running_ant_task.evaluation_measures:
    #           if not measure.crowdsource:
    #             # evaluation
    #             record_generator = record_reader.iterate_read('predict', 'groundtruth')
    #             result = measure.eva(record_generator, None)
    #             if measure.is_support_rank:
    #               evaluation_measure_result.append(result)
    #
    #         task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result
    #
    #       # error analysis
    #       task_running_statictic = \
    #         self.error_analysis(running_ant_task, validation_dataset, task_running_statictic)
    #
    #       if task_running_statictic is not None and len(task_running_statictic) > 0:
    #         logger.info('generate model evaluation report')
    #         self.stage = 'EVALUATION-HOLDOUT-REPORT'
    #
    #         # 更新实验统计信息
    #         mlogger.getEnv().dashboard.experiment.patch(experiment_data=
    #                                                 json.dumps({'REPORT': task_running_statictic,
    #                                                              'APP_STAGE': self.stage}))
    #         # 生成实验报告
    #         everything_to_html(task_running_statictic,
    #                            os.path.join(self.ant_dump_dir, experiment_uuid))
    #
    #         # 上传实验报告
    #         logger.info('uploading experiment report')
    #         mlogger.getEnv().dashboard.experiment.upload(
    #           REPORT_HTML=os.path.join(self.ant_dump_dir, experiment_uuid, 'statistic-report.html'),
    #           APP_STAGE=self.stage)
    #
    #       logger.info('stop evaluation process with complete model')
    #     except:
    #       logger.warn('could not find val dataset, and finish evaluation and error analysis')
    #
    #     # join (waiting until all experiments stop)
    #     for ablation_experiment in ablation_experiments:
    #       ablation_experiment.join()
    #     logger.info('all ablation experiments complete')
    #     return
    #
    #   logger.warn('couldnt enable ablation experiment until set devices in *.yaml')

    # 7.step 模型训练实验
    with safe_manager(ant_train_dataset):
      # 7.1.step 模型训练及评估
      # 四种主流估计方法，holdout; repeated-holdout; bootstrap; kfold
      if running_ant_task.estimation_procedure is not None and \
              running_ant_task.estimation_procedure.lower() in ["holdout",
                                                                "repeated-holdout",
                                                                "bootstrap",
                                                                "kfold"]:
        logger.info('Start model training and evaluation process.')

        estimation_procedure = running_ant_task.estimation_procedure.lower()
        estimation_procedure_params = running_ant_task.estimation_procedure_params
        evaluation_measures = running_ant_task.evaluation_measures

        if estimation_procedure == 'holdout':
          evaluation_statistic = self._holdout_validation(ant_train_dataset, running_ant_task, experiment_uuid)
          
          if evaluation_statistic is not None and len(evaluation_statistic) > 0:
            logger.info('Generate model evaluation report.')
            self.stage = 'EVALUATION-HOLDOUT-REPORT'

            # 更新实验统计信息
            if self.app_token is not None or self.user_token is not None:
              mlogger.getEnv().dashboard.experiment.patch(
                experiment_data=json.dumps({'REPORT': evaluation_statistic,
                                            'APP_STAGE': self.stage}))

            # 生成实验报告
            everything_to_html(evaluation_statistic,
                               os.path.join(self.ant_dump_dir, experiment_uuid))

            # 上传实验报告
            if self.app_token is not None or self.user_token is not None:
              logger.info('Uploading experiment report.')
              mlogger.getEnv().dashboard.experiment.upload(
                REPORT_HTML=os.path.join(self.ant_dump_dir, experiment_uuid, 'statistic-report.html'),
                APP_STAGE=self.stage)
        elif estimation_procedure == "repeated-holdout":
          number_repeats = 2              # default value
          is_stratified_sampling = True   # default value
          split_ratio = 0.7               # default value (andrew ng, machine learning yearning)
          if estimation_procedure_params is not None:
            number_repeats = int(estimation_procedure_params.get('number_repeats', number_repeats))
            is_stratified_sampling = int(estimation_procedure_params.get('stratified_sampling', is_stratified_sampling))
            split_ratio = float(estimation_procedure_params.get('split_ratio', split_ratio))

          # start model estimation procedure
          evaluation_statistic = self._repeated_holdout_validation(number_repeats,
                                                                   ant_train_dataset,
                                                                   split_ratio,
                                                                   is_stratified_sampling,
                                                                   evaluation_measures,
                                                                   experiment_uuid)
          logger.info('Generate model evaluation report.')
          self.stage = 'EVALUATION-REPEATEDHOLDOUT-REPORT'

          # 更新实验统计信息
          if self.app_token is not None or self.user_token is not None:
            mlogger.getEnv().dashboard.experiment.patch(experiment_data=
                                                        json.dumps({'REPORT': evaluation_statistic,
                                                                    'APP_STAGE': self.stage}))
          # 生成实验报告
          everything_to_html(evaluation_statistic,
                             os.path.join(self.ant_dump_dir, experiment_uuid))

          # 上传实验报告
          if self.app_token is not None or self.user_token is not None:
            logger.info('Uploading experiment report.')
            mlogger.getEnv().dashboard.experiment.upload(
              REPORT_HTML=os.path.join(self.ant_dump_dir, experiment_uuid, 'statistic-report.html'),
              APP_STAGE=self.stage)
        elif estimation_procedure == "bootstrap":
          bootstrap_counts = 5
          if estimation_procedure_params is not None:
            bootstrap_counts = int(estimation_procedure_params.get('bootstrap_counts', bootstrap_counts))
          evaluation_statistic = self._bootstrap_validation(bootstrap_counts,
                                                            ant_train_dataset,
                                                            evaluation_measures,
                                                            experiment_uuid)
          logger.info('Generate model evaluation report.')
          self.stage = 'EVALUATION-BOOTSTRAP-REPORT'

          # 更新实验统计信息
          if self.app_token is not None or self.user_token is not None:
            mlogger.getEnv().dashboard.experiment.patch(experiment_data=
                                                    json.dumps({'REPORT': evaluation_statistic,
                                                                'APP_STAGE': self.stage}))
          # 生成实验报告
          everything_to_html(evaluation_statistic,
                             os.path.join(self.ant_dump_dir, experiment_uuid))

          # 上传实验报告
          if self.app_token is not None or self.user_token is not None:
            logger.info('Uploading experiment report.')
            mlogger.getEnv().dashboard.experiment.upload(
              REPORT_HTML=os.path.join(self.ant_dump_dir, experiment_uuid, 'statistic-report.html'),
              APP_STAGE=self.stage)
        else:
          kfolds = 5
          if estimation_procedure_params is not None:
            kfolds = int(estimation_procedure_params.get('kfold', kfolds))
          evaluation_statistic = self._kfold_cross_validation(kfolds,
                                                              ant_train_dataset,
                                                              evaluation_measures,
                                                              experiment_uuid)

          logger.info('Generate model evaluation report.')
          self.stage = 'EVALUATION-KFOLD-REPORT'

          # 更新实验统计信息
          if self.app_token is not None or self.user_token is not None:
            mlogger.getEnv().dashboard.experiment.patch(experiment_data=
                                                    json.dumps({'REPORT': evaluation_statistic,
                                                                'APP_STAGE': self.stage}))
          # 生成实验报告
          everything_to_html(evaluation_statistic,
                             os.path.join(self.ant_dump_dir, experiment_uuid))

          # 上传实验报告
          if self.app_token is not None or self.user_token is not None:
            logger.info('Uploading experiment report.')
            mlogger.getEnv().dashboard.experiment.upload(
              REPORT_HTML=os.path.join(self.ant_dump_dir, experiment_uuid, 'statistic-report.html'),
              APP_STAGE=self.stage)

        return

      # 7.2.step 模型训练 (忽略 --skip_training)
      self.stage = "TRAIN"
      train_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid, 'train')
      if not os.path.exists(train_dump_dir):
          os.makedirs(train_dump_dir)

      logger.info('Start training process.')
      ant_train_dataset.reset_state()
      try:
        self.context.call_training_process(ant_train_dataset, train_dump_dir)
      except Exception as e:
        if type(e.__cause__) != StopIteration:
          print(e)
          traceback.print_exc()
      
      self.context.from_experiment = train_dump_dir
      logger.info('Stop training process.')

  def _holdout_validation(self, train_dataset, running_ant_task, now_time):
    # 1.step split train set and validation set
    part_train_dataset, part_validation_dataset = \
      train_dataset.split(split_params={},
                          split_method='holdout')
    part_train_dataset.reset_state()

    # dump_dir
    dump_dir = os.path.join(self.ant_dump_dir, now_time, 'train')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

    self.stage = 'EVALUATION-HOLDOUT-TRAIN'
    if not self.skip_training:
      # training model
      logger.info("Traning process.")
      try:
        self.context.call_training_process(part_train_dataset, dump_dir)
      except Exception as e:
          if type(e.__cause__) != StopIteration:
            print(e)
            traceback.print_exc()

      self.context.from_experiment = dump_dir
    
    logger.info("Infer process.")
    # data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
    self.context.recorder = RecorderNode2()
    intermediate_dump_dir = os.path.join(self.ant_dump_dir, dump_dir, 'record')
    
    with safe_recorder_manager(self.context.recorder):
      # self.context.recorder.dump_dir = intermediate_dump_dir
      with performance_statistic_region(self.ant_name):
        try:
          self.context.call_infer_process(part_validation_dataset, dump_dir)
        except Exception as e:
          if type(e.__cause__) != StopIteration:
            print(e)
            traceback.print_exc()

    task_running_statictic = get_performance_statistic(self.ant_name)
    task_running_statictic = {self.ant_name: task_running_statictic}
    task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
    task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
      task_running_elapsed_time / float(part_validation_dataset.size)

    evaluation_measure_result = []
    no_rank_measure = []
    with safe_recorder_manager(RecordReader(intermediate_dump_dir)) as record_reader:
      for measure in running_ant_task.evaluation_measures:
        if not measure.crowdsource:
          # evaluation
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          if measure.is_support_rank:
            evaluation_measure_result.append(result)
          else:
            no_rank_measure.append(result)
  
      task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

    ################
    # 临时代码
    # voc_measure = {'statistic': {'name': 'voc',
    #                              'value': [{'name': 'MAP', 'value': [23.0, 11.0, 12.0], 'type': 'SCALAR', 'x': 'class',
    #                                         'y': 'Mean Average Precision'},
    #                                        {'name': 'Mean-MAP', 'value': 0.13, 'type': 'SCALAR'}]}}
    #
    # roc_auc_measure = {'statistic': {'name': 'roc_auc',
    #                                  'value': [{'name': 'ROC', 'value': [[[0, 0], [1, 1], [2, 3]],
    #                                                                      [[0, 3], [2, 5], [3, 0]]],
    #                                             'type': 'CURVE', 'x': 'FP', 'y': 'TP',
    #                                             'legend': ['class-0', 'class-1']},
    #                                            {'name': 'AUC', 'value': [0.3, 0.4], 'type': 'SCALAR', 'x': 'class',
    #                                             'y': 'AUC'}]}}
    #
    # pr_f1_measure = {'statistic': {'name': 'pr_f1',
    #                                'value': [{'name': 'Precision-Recall',
    #                                           'value': [[[0, 0], [1, 1], [2, 3]],
    #                                                     [[0, 3], [2, 5], [3, 0]]],
    #                                           'type': 'CURVE', 'x': 'precision', 'y': 'recall'},
    #                                          {'name': 'F1', 'value': [1.0, 2.0], 'type': 'SCALAR', 'x': 'class',
    #                                           'y': 'F1'}]}}
    #
    # confusion_m = {'statistic': {'name': 'cm',
    #                              'value': [{'name': 'ccmm', 'value': (np.ones((3, 4)) * 3).tolist(), 'type': 'MATRIX',
    #                                         'x': ['a', 'b', 'c', 'd'], 'y': ['x', 'y', 'z']}]}}
    #
    # random_img = np.random.random((100, 100))
    # random_img = random_img * 255
    # random_img = random_img.astype(np.uint8)
    # image_m = {'statistic': {'name': 'image',
    #                          'value': [{'name': 'image', 'value': random_img, 'type': 'IMAGE'}]}}
    # task_running_statictic[self.ant_name]['measure'].extend([voc_measure, roc_auc_measure,pr_f1_measure, confusion_m, image_m])
    #
    # task_running_statictic[self.ant_name]['significant_diff'] = {}
    # task_running_statictic[self.ant_name]['significant_diff']['QAS'] = [{'name': '20180429.161714.105706', 'score': 0},
    #                                                                    {'name': '20180429.220804.516647', 'score': 1},
    #                                                                    {'name': '20180429.222627.393376', 'score': -1},
    #                                                                    {'name': '20180429.222721.264422', 'score': 1}]
    # ################

    # error analysis
    task_running_statictic = self.error_analysis(running_ant_task, part_validation_dataset, task_running_statictic)
    task_running_statictic[self.ant_name]['measure'].extend(no_rank_measure)
    return task_running_statictic

  def _repeated_holdout_validation(self,
                                   repeats,
                                   train_dataset,
                                   split_ratio,
                                   is_stratified_sampling,
                                   evaluation_measures,
                                   nowtime):
    
    from_experiment = getattr(self.context, 'from_experiment', None)
    if from_experiment is not None:
      self.context.from_experiment = None
      
    repeated_running_statistic = []
    for repeat in range(repeats):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = \
        train_dataset.split(split_params={'ratio': split_ratio,
                                          'is_stratified': is_stratified_sampling},
                            split_method='repeated-holdout')
      part_train_dataset.reset_state()

      # dump_dir
      dump_dir = os.path.join(self.ant_dump_dir, nowtime, 'train', 'repeated-holdout-evaluation', 'repeat-%d'%repeat)
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

      # 2.step training model
      self.stage = 'EVALUATION-REPEATEDHOLDOUT-TRAIN-%d' % repeat
      self.context.from_experiment = from_experiment
      logger.info('Start training process at repeathold %d round.'%repeat)
      try:
        self.context.call_training_process(part_train_dataset, dump_dir)
      except Exception as e:
          if type(e.__cause__) != StopIteration:
            print(e)
            traceback.print_exc()

      self.context.from_experiment = dump_dir
      logger.info('Stop training process at repeathold %d round.' % repeat)
      
      # 3.step evaluation measures
      # split data and label
      logger.info('Start infer process at repeathold %d round.' % repeat)
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
      
      self.context.from_experiment = None
      self.stage = 'EVALUATION-REPEATEDHOLDOUT-EVALUATION-%d' % repeat
      with safe_recorder_manager(self.context.recorder):
        with performance_statistic_region(self.ant_name):
          try:
            self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)
          except Exception as e:
            if type(e.__cause__) != StopIteration:
              print(e)
              traceback.print_exc()

      logger.info('Start infer process at repeathold %d round.' % repeat)
      
      # clear
      self.context.recorder = None

      task_running_statictic = get_performance_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('Start evaluation process at repeathold %d round.'%repeat)
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result
      
      logger.info('Stop evaluation process at repeathold %d round.'%repeat)
      repeated_running_statistic.append(task_running_statictic)

    evaluation_result = \
      multi_repeats_measures_statistic(repeated_running_statistic,
                                       method='repeated-holdout')
    return evaluation_result

  def _bootstrap_validation(self, bootstrap_rounds, train_dataset, evaluation_measures, nowtime):
    bootstrap_running_statistic = []
    from_experiment = getattr(self.context, 'from_experiment', None)
    if from_experiment is not None:
      self.context.from_experiment = None
      
    for bootstrap_i in range(bootstrap_rounds):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = \
        train_dataset.split(split_params={},
                            split_method='bootstrap')

      # dump_dir
      dump_dir = os.path.join(self.ant_dump_dir,
                              nowtime,
                              'train',
                              'bootstrap-evaluation',
                              'bootstrap-%d-evaluation' % bootstrap_i)
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

      # 2.step training model
      self.stage = 'EVALUATION-BOOTSTRAP-TRAIN-%d' % bootstrap_i
      self.context.from_experiment = from_experiment
      logger.info('Start training process at bootstrap %d round.'%bootstrap_i)
      try:
        self.context.call_training_process(part_train_dataset, dump_dir)
      except Exception as e:
          if type(e.__cause__) != StopIteration:
            print(e)
            traceback.print_exc()

      self.context.from_experiment = dump_dir
      logger.info('Stop training process at bootstrap %d round.' % bootstrap_i)
      
      # 3.step evaluation measures
      # split data and label
      logger.info('Start infer process at bootstrap %d round.' % bootstrap_i)
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
      self.context.from_experiment = None
      self.stage = 'EVALUATION-BOOTSTRAP-EVALUATION-%d' % bootstrap_i
      with safe_recorder_manager(self.context.recorder):
        with performance_statistic_region(self.ant_name):
          try:
            self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)
          except Exception as e:
            if type(e.__cause__) != StopIteration:
              print(e)
              traceback.print_exc()

      logger.info('Stop infer process at bootstrap %d round.' % bootstrap_i)
      
      # clear
      self.context.recorder = None

      task_running_statictic = get_performance_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('Start evaluation process at bootstrap %d round.' % bootstrap_i)
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result
      
      logger.info('Stop evaluation process at bootstrap %d round.' % bootstrap_i)
      
      bootstrap_running_statistic.append(task_running_statictic)

    evaluation_result = \
      multi_repeats_measures_statistic(bootstrap_running_statistic,
                                       method='bootstrap')
    return evaluation_result

  def _kfold_cross_validation(self, kfolds, train_dataset, evaluation_measures, nowtime):
    # assert (kfolds in [5, 10])
    kfolds_running_statistic = []
    from_experiment = getattr(self.context, 'from_experiment', None)
    if from_experiment is not None:
      self.context.from_experiment = None

    for k in range(kfolds):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = \
        train_dataset.split(split_params={'kfold': kfolds,
                                          'k': k},
                            split_method='kfold')

      part_train_dataset.reset_state()
      # dump_dir
      dump_dir = os.path.join(self.ant_dump_dir, nowtime, 'train', 'kfold-evaluation', 'fold-%d-evaluation' % k)
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

      # 2.step training model
      self.stage = 'EVALUATION-KFOLD-TRAIN-%d' % k
      logger.info('Start training process at kfold %d round.'%k)
      self.context.from_experiment = from_experiment
      try:
        self.context.call_training_process(part_train_dataset, dump_dir)
      except Exception as e:
          if type(e.__cause__) != StopIteration:
            print(e)
            traceback.print_exc()

      self.context.from_experiment = dump_dir
      logger.info('Stop training process at kfold %d round.'%k)

      # 3.step evaluation measures
      # split data and label
      logger.info('Start infer process at kfold %d round.'%k)
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
      self.context.from_experiment = None
      
      self.stage = 'EVALUATION-KFOLD-EVALUATION-%d' % k
      with safe_recorder_manager(self.context.recorder):
        with performance_statistic_region(self.ant_name):
          try:
            self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)
          except Exception as e:
            if type(e.__cause__) != StopIteration:
              print(e)
              traceback.print_exc()

      logger.info('Stop infer process at kfold %d round.'%k)
      # clear
      self.context.recorder = None

      task_running_statictic = get_performance_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('Start evaluation process at kfold %d round.'%k)
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result
      logger.info('Stop evaluation process at kfold %d round.'%k)
      
      kfolds_running_statistic.append(task_running_statictic)

    evaluation_result = multi_repeats_measures_statistic(kfolds_running_statistic, method='kfold')
    return evaluation_result

  # def start_ablation_train_proc(self,
  #                               data_source,
  #                               challenge_task,
  #                               ablation_blocks,
  #                               ablation_method,
  #                               time_stamp,
  #                               spare_devices=None):
  #   if ablation_method is None:
  #     ablation_method = 'regular'
  #   # check ablation method
  #   assert(ablation_method in ['regular', 'inregular', 'accumulate', 'any'])
  #
  #   # child func
  #   def proc_func(handle,
  #                 experiment_data_source,
  #                 experiment_challenge_task,
  #                 ablation_block,
  #                 root_time_stamp,
  #                 spare_device,
  #                 skip_training):
  #     # 1.step proc_func is running in a independent process (clone running environment)
  #     handle.reset()
  #     # reassign running device
  #     handle.context.params.devices = [spare_device]
  #     # only one clone
  #     handle.context.params.num_clones = 1
  #
  #     part_train_dataset = experiment_data_source
  #     part_validation_dataset = None
  #     try:
  #       part_train_dataset, part_validation_dataset = \
  #                                       experiment_data_source.split(split_params={}, split_method='holdout')
  #     except:
  #       logger.warn('could not find val dataset for ablation experiment')
  #
  #     # shuffle
  #     part_train_dataset.reset_state()
  #
  #     # deactivate ablation_block
  #     if type(ablation_block) == list or type(ablation_block) == tuple:
  #       for bb in ablation_block:
  #         handle.context.deactivate_block(bb)
  #       ablation_block = '_'.join(ablation_block)
  #     else:
  #       handle.context.deactivate_block(ablation_block)
  #
  #     logger.info('start ablation experiment %s on device %s' % (ablation_block, str(spare_device)))
  #
  #     # dump_dir for ablation experiment
  #     ablation_dump_dir = os.path.join(handle.ant_dump_dir, root_time_stamp, 'train', 'ablation', ablation_block)
  #     if not os.path.exists(ablation_dump_dir):
  #       os.makedirs(ablation_dump_dir)
  #
  #     # 2.step start training process
  #     handle.stage = 'ABLATION-%s-TRAIN' % ablation_block
  #     logger.info('on {}'.format(handle.stage))
  #     if not skip_training:
  #       handle.context.call_training_process(part_train_dataset, ablation_dump_dir)
  #
  #     # 3.step start evaluation process
  #     if part_validation_dataset is not None:
  #       # split data and label
  #       data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
  #       handle.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
  #
  #       handle.stage = 'ABLATION-%s-EVALUATION' % ablation_block
  #       logger.info('on {}'.format(handle.stage))
  #       with safe_recorder_manager(handle.context.recorder):
  #         handle.context.call_infer_process(data_annotation_branch.output(0), ablation_dump_dir)
  #       handle.context.recorder.close()
  #
  #       # clear
  #       handle.context.recorder = None
  #
  #       ablation_running_statictic = {handle.ant_name: {}}
  #       ablation_evaluation_measure_result = []
  #
  #       logger.info('compute measure')
  #       with safe_recorder_manager(RecordReader(ablation_dump_dir)) as record_reader:
  #         for measure in experiment_challenge_task.evaluation_measures:
  #           record_generator = record_reader.iterate_read('predict', 'groundtruth')
  #           result = measure.eva(record_generator, None)
  #           ablation_evaluation_measure_result.append(result)
  #
  #       ablation_running_statictic[handle.ant_name]['measure'] = ablation_evaluation_measure_result
  #       handle.stage = 'ABLATION-%s-REPORT' % ablation_block
  #
  #       logger.info('on {}'.format(handle.stage))
  #       # 更新实验统计信息
  #       logger.info('update experiment report')
  #       mlogger.getEnv().dashboard.experiment.patch(experiment_data=
  #                                                 json.dumps({'REPORT': ablation_running_statictic,
  #                                                              'APP_STAGE': handle.stage}))
  #
  #       # 生成消除实验报告
  #       everything_to_html(ablation_running_statictic, ablation_dump_dir)
  #
  #       # 上传实验报告
  #       logger.info('uploading experiment report')
  #       mlogger.getEnv().dashboard.experiment.upload(
  #         REPORT_HTML=os.path.join(ablation_dump_dir, 'statistic-report.html'),
  #         APP_STAGE=handle.stage)
  #
  #     handle.context.wait_until_clear()
  #
  #   if ablation_method is None:
  #     ablation_method = 'regular'
  #
  #   traverse_ablation_blocks = []
  #   if ablation_method == 'regular':
  #     traverse_ablation_blocks.extend(ablation_blocks)
  #   elif ablation_method == 'inregular':
  #     for b in ablation_blocks:
  #       traverse_ablation_blocks.append([m for m in ablation_blocks if m != b])
  #   elif ablation_method == 'accumulate':
  #     accumulate_blocks = []
  #     for block in ablation_blocks:
  #       accumulate_blocks.append(block)
  #       traverse_ablation_blocks.append(copy.deepcopy(accumulate_blocks))
  #   else:
  #     for i in range(len(ablation_blocks)):
  #       aa = list(itertools.combinations(ablation_blocks, i+1))
  #       traverse_ablation_blocks.extend(aa)
  #
  #     traverse_ablation_blocks = [list(m) for m in traverse_ablation_blocks]
  #
  #   ablation_experiments = []
  #   for try_i, try_ablation_blocks in enumerate(traverse_ablation_blocks):
  #     # apply independent process
  #     if type(try_ablation_blocks) != list and type(try_ablation_blocks) != tuple:
  #       try_ablation_blocks = [try_ablation_blocks]
  #
  #     try_ablation_blocks = sorted(try_ablation_blocks)
  #
  #     # launch ablation experiment process
  #     block_ablation_process = Process(target=proc_func,
  #                                      args=(self,
  #                                            data_source,
  #                                            challenge_task,
  #                                            copy.deepcopy(try_ablation_blocks),
  #                                            time_stamp,
  #                                            spare_devices[try_i],
  #                                            self.skip_training),
  #                                      name='%s_ablation_block_%s'%(self.ant_name, '_'.join(copy.deepcopy(try_ablation_blocks))))
  #     ablation_experiments.append(block_ablation_process)
  #
  #   return ablation_experiments