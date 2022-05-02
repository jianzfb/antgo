# -*- coding: UTF-8 -*-
# @Time    : 17-5-9
# @File    : challenge.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from antgo.resource.html import *
from antgo.ant.base import *
from antgo.dataflow.common import *
from antgo.measures.statistic import *
from antgo.task.task import *
from antgo.utils import logger
from antgo.dataflow.recorder import *
from antgo.measures.deep_analysis import *
from antgo.measures.significance import *
import shutil
import tarfile
from datetime import datetime
from antgo.measures import *
from antgo.measures.yesno_crowdsource import *
import traceback
import zlib
from antgo.dataflow.dataset.proxy_dataset import *


class AntChallenge(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config=None,
               ant_dataset=None,
               ant_task_benchmark=None,
               **kwargs):
    super(AntChallenge, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.ant_data_source = ant_data_folder
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config
    self.ant_dataset = ant_dataset
    self.ant_task_benchmark = ant_task_benchmark
    self.context.devices = self.context.params.system['devices']
    self._running_dataset = None
    self._running_task = None
    self.task_running_statictic = {}

  @property
  def running_dataset(self):
    return self._running_dataset

  @property
  def running_task(self):
    return self._running_task

  def evaluate(self):
    if not self.context.recorder.is_measure:
      return

    logger.info('Start evaluation process.')
    evaluation_measure_result = []
    with safe_recorder_manager(RecordReader(self.context.recorder.dump_dir)) as record_reader:
      for measure in self.running_task.evaluation_measures:
        if measure.crowdsource:
          # start crowdsource server
          measure.dump_dir = os.path.join(infer_dump_dir, measure.name, 'static')
          if not os.path.exists(measure.dump_dir):
            os.makedirs(measure.dump_dir)

          measure.experiment_id = self.context.experiment_uuid
          measure.app_token = self.token
          logger.info('Launch crowdsource evaluation server.')
          crowdsrouce_evaluation_status = measure.crowdsource_server(record_reader)
          if not crowdsrouce_evaluation_status:
            logger.error('Couldnt finish crowdsource evaluation server.')
            continue

          # using crowdsource evaluation
          result = measure.eva()
          # TODO: support bootstrap confidence interval for crowdsource evaluation
        else:
          # evaluation
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)

          # compute confidence interval
          if measure.is_support_rank and getattr(self.running_task, 'confidence_interval', False):
            confidence_interval = bootstrap_confidence_interval(record_reader, time.time(), measure, 10)
            result['statistic']['value'][0]['interval'] = confidence_interval
          elif measure.is_support_rank:
            result['statistic']['value'][0]['interval'] = (
            result['statistic']['value'][0]['value'], result['statistic']['value'][0]['value'])

        evaluation_measure_result.append(result)

      # #########################
      # roc_auc_measure = {'statistic': {'name': 'roc_auc',
      #                                  'value': [{'name': 'ROC', 'value': [[[0, 3], [1, 0], [2, 6]],
      #                                                                      [[0, 8], [2, 3], [3, 7]]],
      #                                             'type': 'CURVE', 'x': 'FP', 'y': 'TP',
      #                                             'legend': ['class-0', 'class-1']},
      #                                            {'name': 'AUC', 'value': [0.1, 0.2], 'type': 'SCALAR', 'x': 'class',
      #                                             'y': 'AUC'}]}}
      #
      # voc_measure = {'statistic': {'name': 'voc',
      #                              'value': [{'name': 'MAP', 'value': [18.0, 9.0, 20.0], 'type': 'SCALAR', 'x': 'class',
      #                                         'y': 'Mean Average Precision'},
      #                                        {'name': 'Mean-MAP', 'value': 0.14, 'type': 'SCALAR'}]}}
      #
      #
      # evaluation_measure_result.append(roc_auc_measure)
      # evaluation_measure_result.append(voc_measure)
      # #########################

      self.task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

    # if self.is_non_mltalker_task:
    #   # generate report resource
    #   logger.info('generate model evaluation report')
    #   everything_to_html(task_running_statictic, os.path.join(self.ant_dump_dir, now_time_stamp))
    #   return

    # significance statistic
    logger.info('Significance difference compare and rank.')
    # benchmark record
    benchmark_model_data = {}
    if self.token is not None:
      response = mlogger.info.benchmark.get()
      if response['status'] == 'OK':
        benchmark_info = response['content']
        for bmd in benchmark_info:
          benchmark_name = bmd['benchmark_name']  # benchmark name (experiment_uuid)
          benchmark_record = bmd['benchmark_record']  # url
          benchmark_report = bmd['benchmark_report']  # 统计数据

          # download benchmark record from url
          logger.info('Download benchmark %s.' % benchmark_name)
          mlogger.info.experiment.download(file_name=benchmark_record,
                                           file_folder=os.path.join(self.ant_dump_dir,
                                                                    self.context.experiment_uuid,
                                                                    'benchmark',
                                                                    benchmark_name),
                                           experiment_uuid=benchmark_name)

          if 'record' not in benchmark_model_data:
            benchmark_model_data['record'] = {}
          benchmark_model_data['record'][benchmark_name] = os.path.join(self.ant_dump_dir,
                                                                        self.context.experiment_uuid,
                                                                        'benchmark',
                                                                        benchmark_name,
                                                                        'record')

          if 'report' not in benchmark_model_data:
            benchmark_model_data['report'] = {}

          for benchmark_experiment_name, benchmark_experiment_report in benchmark_report.items():
            benchmark_model_data['report'][benchmark_name] = benchmark_experiment_report
    elif self.ant_task_benchmark is not None:
      for experiment in self.ant_task_benchmark.split(','):
        if os.path.exists(os.path.join(self.ant_dump_dir, experiment)):
          if 'record' not in benchmark_model_data:
            benchmark_model_data['record'] = {}
          benchmark_model_data['record'][experiment] = os.path.join(self.ant_dump_dir, experiment, 'record')

    if benchmark_model_data is not None and 'record' in benchmark_model_data:
      benchmark_model_record = benchmark_model_data['record']

      self.task_running_statictic[self.ant_name]['significant_diff'] = {}
      for meature_index, measure in enumerate(self.running_task.evaluation_measures):
        if measure.is_support_rank and not measure.crowdsource:
          significant_diff_score = []
          for benchmark_model_name, benchmark_model_address in benchmark_model_record.items():
            if getattr(self.running_task, 'confidence_interval', False):
              with safe_recorder_manager(RecordReader(self.context.recorder.dump_dir)) as record_reader:
                with safe_recorder_manager(RecordReader(benchmark_model_address)) as benchmark_record_reader:
                  s = bootstrap_ab_significance_compare([record_reader, benchmark_record_reader], time.time(), measure,
                                                        10)

                  significant_diff_score.append({'name': benchmark_model_name, 'score': s})
            else:
              compare_value = \
                self.task_running_statictic[self.ant_name]['measure'][meature_index]['statistic']['value'][0]['value'] - \
                benchmark_model_data['report'][benchmark_model_name]['measure'][meature_index]['statistic']['value'][0][
                  'value']
              if compare_value > 0:
                if getattr(measure, 'larger_is_better', 0) == 1:
                  significant_diff_score.append({'name': benchmark_model_name, 'score': 1})
                else:
                  significant_diff_score.append({'name': benchmark_model_name, 'score': -1})
              elif compare_value < 0:
                if getattr(measure, 'larger_is_better', 0) == 1:
                  significant_diff_score.append({'name': benchmark_model_name, 'score': -1})
                else:
                  significant_diff_score.append({'name': benchmark_model_name, 'score': 1})
              else:
                significant_diff_score.append({'name': benchmark_model_name, 'score': 0})

          self.task_running_statictic[self.ant_name]['significant_diff'][measure.name] = significant_diff_score
        elif measure.is_support_rank and measure.crowdsource:
          # TODO: support model significance compare for crowdsource evaluation
          pass

    # error analysis
    logger.info('Error analysis.')
    # benchmark report
    benchmark_model_statistic = None
    if benchmark_model_data is not None and 'report' in benchmark_model_data:
      benchmark_model_statistic = benchmark_model_data['report']

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

    for measure_result in self.task_running_statictic[self.ant_name]['measure']:
      if 'info' in measure_result and len(measure_result['info']) > 0:
        measure_name = measure_result['statistic']['name']
        measure_data = measure_result['info']

        # independent analysis per category for classification problem
        measure_data_list = []
        if self.running_task.class_label is not None and len(self.running_task.class_label) > 1:
          if self.running_task.class_label is not None:
            for cl_i, cl in enumerate(self.running_task.class_label):
              measure_data_list.append([md for md in measure_data if md['category'] == cl or md['category'] == cl_i])

        if len(measure_data_list) == 0:
          measure_data_list.append(measure_data)

        for category_id, category_measure_data in enumerate(measure_data_list):
          if len(category_measure_data) == 0:
            continue

          if 'analysis' not in self.task_running_statictic[self.ant_name]:
            self.task_running_statictic[self.ant_name]['analysis'] = {}

          if measure_name not in self.task_running_statictic[self.ant_name]['analysis']:
            self.task_running_statictic[self.ant_name]['analysis'][measure_name] = {}

          # reorganize as list
          method_samples_list = [{'name': self.ant_name, 'data': category_measure_data}]
          if benchmark_model_statistic is not None:
            # extract statistic data from benchmark
            for benchmark_name, benchmark_statistic_data in benchmark_model_statistic.items():
              # finding corresponding measure
              for benchmark_measure_result in benchmark_statistic_data['measure']:
                if benchmark_measure_result['statistic']['name'] == measure_name:
                  benchmark_measure_data = benchmark_measure_result['info']

                  # finding corresponding category
                  sub_benchmark_measure_data = None
                  if self.running_task.class_label is not None and len(self.running_task.class_label) > 1:
                    sub_benchmark_measure_data = \
                      [md for md in benchmark_measure_data if
                       md['category'] == self.running_task.class_label[category_id] or md['category'] == category_id]
                  if sub_benchmark_measure_data is None:
                    sub_benchmark_measure_data = benchmark_measure_data

                  method_samples_list.append({'name': benchmark_name, 'data': sub_benchmark_measure_data})

                  break
              break

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
              continuous_multi_model_measure_analysis(method_measure_mat, samples_map, self.running_dataset)

            analysis_tag = 'Global'
            if len(measure_data_list) > 1:
              analysis_tag = 'Global-Category-' + str(self.running_task.class_label[category_id])

            model_name_ri = [method_samples_list[r]['name'] for r in ri]
            self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = \
              {'value': s.tolist() if type(s) != list else s,
               'type': 'MATRIX',
               'x': ci,
               'y': model_name_ri,
               'sampling': [{'name': 'High Score Region', 'data': hr_samples},
                            {'name': 'Middle Score Region', 'data': mr_samples},
                            {'name': 'Low Score Region', 'data': lr_samples}]}

            # group by tag
            tags = getattr(self.running_dataset, 'tag', None)
            if tags is not None:
              for tag in tags:
                g_s, g_ri, g_ci, g_lr_samples, g_mr_samples, g_hr_samples = \
                  continuous_multi_model_measure_analysis(method_measure_mat,
                                                          samples_map,
                                                          self.running_dataset,
                                                          filter_tag=tag)

                analysis_tag = 'Group'
                if len(measure_data_list) > 1:
                  analysis_tag = 'Group-Category-' + str(self.running_task.class_label[category_id])

                if analysis_tag not in self.task_running_statictic[self.ant_name]['analysis'][measure_name]:
                  self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = []

                model_name_ri = [method_samples_list[r]['name'] for r in g_ri]
                tag_data = {'value': g_s.tolist() if type(g_s) != list else g_s,
                            'type': 'MATRIX',
                            'x': g_ci,
                            'y': model_name_ri,
                            'sampling': [{'name': 'High Score Region', 'data': g_hr_samples},
                                         {'name': 'Middle Score Region', 'data': g_mr_samples},
                                         {'name': 'Low Score Region', 'data': g_lr_samples}]}

                self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag].append((tag, tag_data))
          else:
            s, ri, ci, region_95, region_52, region_42, region_13, region_one, region_zero = \
              discrete_multi_model_measure_analysis(method_measure_mat,
                                                    samples_map,
                                                    self.running_dataset)

            analysis_tag = 'Global'
            if len(measure_data_list) > 1:
              analysis_tag = 'Global-Category-' + str(self.running_task.class_label[category_id])

            model_name_ri = [method_samples_list[r]['name'] for r in ri]
            self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = \
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
            tags = getattr(self.running_dataset, 'tag', None)
            if tags is not None:
              for tag in tags:
                g_s, g_ri, g_ci, g_region_95, g_region_52, g_region_42, g_region_13, g_region_one, g_region_zero = \
                  discrete_multi_model_measure_analysis(method_measure_mat,
                                                        samples_map,
                                                        self.running_dataset,
                                                        filter_tag=tag)
                # if 'group' not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                #   task_running_statictic[self.ant_name]['analysis'][measure_name]['group'] = []
                #
                analysis_tag = 'Group'
                if len(measure_data_list) > 1:
                  analysis_tag = 'Group-Category-' + str(self.running_task.class_label[category_id])

                if analysis_tag not in self.task_running_statictic[self.ant_name]['analysis'][measure_name]:
                  self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = []

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

                self.task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag].append((tag, tag_data))

    # 更新实验统计信息
    if self.app_token is not None or self.user_token is not None:
      mlogger.info.experiment.patch(
        experiment_data=zlib.compress(json.dumps({'REPORT': self.task_running_statictic,
                                                  'APP_STAGE': self.stage}).encode()))

    # 生成实验报告
    logger.info('Save experiment report.')
    everything_to_html(self.task_running_statictic,
                       os.path.join(self.ant_dump_dir, self.context.experiment_uuid))

  def start(self):
    # 1.step 加载挑战任务
    running_ant_task = None
    if self.token is not None:
      # 1.1.step 从平台获取挑战任务配置信息
      response = mlogger.info.challenge.get(command=type(self).__name__)
      if response['status'] == 'ERROR':
        logger.error('Couldnt load challenge task.')
        self.token = None
      elif response['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('Prohibit submit challenge task frequently.')
        exit(-1)
      elif response['status'] == 'OK':
        content = response['content']

        if 'task' in content:
          challenge_task = create_task_from_json(content)
          if challenge_task is None:
            logger.error('Couldnt load challenge task.')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('Unknow error.')
        exit(-1)

    if running_ant_task is None:
      # 1.2.step 加载自定义任务配置信息
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('Couldnt load custom task.')
        exit(-1)
      running_ant_task = custom_task

    assert (running_ant_task is not None)
    self._running_task = running_ant_task

    # 3.step  备份实验基本信息
    # 3.1.step 打包代码，并上传至云端
    self.stage = 'CHALLENGE'
    # - backup in dump_dir
    main_folder = self.main_folder
    if not os.path.exists(os.path.join(self.ant_dump_dir, self.context.experiment_uuid)):
      os.makedirs(os.path.join(self.ant_dump_dir, self.context.experiment_uuid))

    # goldcoin = os.path.join(self.ant_dump_dir, experiment_uuid, 'code.tar.gz')

    # if os.path.exists(goldcoin):
    #   os.remove(goldcoin)

    # logger.info('Prepare package model files.')
    # tar = tarfile.open(goldcoin, 'w:gz')

    # # 排除dump目录，将当前文件夹下所有数据上传
    # for root, dirs, files in os.walk(main_folder):
    #   if os.path.commonprefix([root, self.ant_dump_dir]) == self.ant_dump_dir:
    #     continue

    #   rel_root = os.path.relpath(root, main_folder)
    #   for f in files:
    #     tar.add(os.path.join(root, f), arcname=os.path.join(rel_root, f))

    # tar.close()
    # logger.info('Finish package process.')

    # TODO: 在下一个版本支持模型文件上传
    # # 上传模型代码
    # mlogger.getEnv().dashboard.experiment.upload(MODEL=goldcoin,
    #                                              APP_STAGE=self.stage)

    # 3.2.step 更新基本配置
    if self.app_token is not None or self.user_token is not None:
      for k, v in self.context.params.items():
        setattr(mlogger.tag, k, str(v))

    # 4.step 加载测试数据集
    if running_ant_task.dataset_name is None or running_ant_task.dataset_name == '':
      logger.info('Missing dataset set in task config, switch to use --dataset=xxx')
      if self.ant_dataset is None or self.ant_dataset == '':
        logger.error('Sorry, --dataset= is empty.')
        return
      running_ant_task.config(dataset_name=self.ant_dataset)

    logger.info('Loading test dataset %s.'%running_ant_task.dataset_name)

    ant_test_dataset = None
    if self.context.register_at('test') is not None:
      ant_test_dataset = ProxyDataset('test')
      ant_test_dataset.register(test=self.context.register_at('test'))
    else:
      ant_test_dataset = running_ant_task.dataset('test',
                                                  os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                  running_ant_task.dataset_params)

    self._running_dataset = ant_test_dataset
    with safe_manager(ant_test_dataset):
      self.context.recorder = RecorderNode2()

      self.stage = "CHALLENGE"
      logger.info('Start infer process.')
      infer_dump_dir = os.path.join(self.ant_dump_dir, self.context.experiment_uuid, 'inference')
      if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)
      else:
        shutil.rmtree(infer_dump_dir)
        os.makedirs(infer_dump_dir)

      if not os.path.exists(os.path.join(infer_dump_dir, 'record')):
        os.makedirs(os.path.join(infer_dump_dir, 'record'))

      intermediate_dump_dir = os.path.join(self.ant_dump_dir, self.context.experiment_uuid, 'record')
      with safe_recorder_manager(self.context.recorder):
        self.context.recorder.dump_dir = intermediate_dump_dir
        # from ablation experiment
        ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
        if ablation_blocks is None:
          ablation_blocks = []
        for b in ablation_blocks:
          self.ant_context.deactivate_block(b)

        if self.context.is_interact_mode:
          logger.info('Running on interact mode.')
          return

        with performance_statistic_region(self.ant_name):
          try:
            self.context.call_infer_process(ant_test_dataset, infer_dump_dir)
          except Exception as e:
            if type(e.__cause__) != StopIteration:
              print(e)
              traceback.print_exc()

        task_running_statictic = get_performance_statistic(self.ant_name)
        self.task_running_statictic = {self.ant_name: task_running_statictic}
        task_running_elapsed_time = self.task_running_statictic[self.ant_name]['time']['elapsed_time']
        self.task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
            task_running_elapsed_time / float(ant_test_dataset.size)

        if not self.context.recorder.is_measure:
          # has no annotation to continue to meausre
          # 更新实验统计信息
          if self.app_token is not None or self.user_token is not None:
            mlogger.info.experiment.patch(
              experiment_data=zlib.compress(
                json.dumps({'REPORT': self.task_running_statictic,
                            'APP_STAGE': self.stage
                            }).encode()))

          # 生成实验报告
          logger.info('Save experiment report.')
          everything_to_html(self.task_running_statictic,
                             os.path.join(self.ant_dump_dir, self.context.experiment_uuid))
          return

        # 模型评估
        self.evaluate()