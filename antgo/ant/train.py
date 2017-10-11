# encoding=utf-8
# @Time    : 17-6-22
# @File    : train.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

from antgo.html.html import *
from antgo.ant.base import *
from antgo.dataflow.common import *
from antgo.measures.statistic import *
from antgo.task.task import *
from antgo.utils import logger
from antgo.dataflow.recorder import *
import tarfile
import sys
from datetime import datetime
from antgo.ant import flags
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
               ant_task_config):
    super(AntTrain, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_source = ant_data_folder
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config

  def start(self):
    # 0.step loading challenge task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      challenge_task_config = self.rpc("TASK-CHALLENGE")
      if challenge_task_config is None:
        # invalid token
        logger.error('couldnt load challenge task')
        self.token = None
      elif challenge_task_config['status'] in ['OK', 'SUSPEND']:
        # maybe user token or task token
        if 'task' in challenge_task_config:
          # task token
          challenge_task = create_task_from_json(challenge_task_config)
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
      if self.ant_task_config is not None:
        custom_task = create_task_from_xml(self.ant_task_config, self.context)
        if custom_task is None:
          logger.error('couldnt load custom task')
          exit(-1)
        running_ant_task = custom_task

    assert(running_ant_task is not None)
    
    # now time stamp
    # train_time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(self.time_stamp))
    train_time_stamp = datetime.now().strftime('%Y%m%d.%H%M%S.%f')

    # 0.step warp model (main_file and main_param)
    self.stage = 'MODEL'
    # - backup in dump_dir
    main_folder = FLAGS.main_folder()
    main_param = FLAGS.main_param()
    main_file = FLAGS.main_file()

    if not os.path.exists(os.path.join(self.ant_dump_dir, train_time_stamp)):
      os.makedirs(os.path.join(self.ant_dump_dir, train_time_stamp))

    goldcoin = os.path.join(self.ant_dump_dir, train_time_stamp, '%s-goldcoin.tar.gz'%self.ant_name)
    
    if os.path.exists(goldcoin):
      os.remove(goldcoin)

    tar = tarfile.open(goldcoin, 'w:gz')
    tar.add(os.path.join(main_folder, main_file), arcname=main_file)
    if main_param is not None:
      tar.add(os.path.join(main_folder, main_param), arcname=main_param)
    tar.close()

    # - backup in cloud
    if os.path.exists(goldcoin):
      file_size = os.path.getsize(goldcoin) / 1024.0
      if file_size < 500:
        if not PY3 and sys.getdefaultencoding() != 'utf8':
          reload(sys)
          sys.setdefaultencoding('utf8')
        # model file shouldn't too large (500KB)
        with open(goldcoin, 'rb') as fp:
          self.context.job.send({'DATA': {'MODEL': fp.read()}})

    # 1.step loading training dataset
    logger.info('loading train dataset %s'%running_ant_task.dataset_name)
    ant_train_dataset = running_ant_task.dataset('train',
                                                 os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                 running_ant_task.dataset_params)
    
    with safe_recorder_manager(ant_train_dataset):
      # 2.step model evaluation (optional)
      if running_ant_task.estimation_procedure is not None and \
              running_ant_task.estimation_procedure.lower() in ["holdout","repeated-holdout","bootstrap","kfold"]:
        logger.info('start model evaluation')

        estimation_procedure = running_ant_task.estimation_procedure.lower()
        estimation_procedure_params = running_ant_task.estimation_procedure_params
        evaluation_measures = running_ant_task.evaluation_measures

        evaluation_statistic = None
        if estimation_procedure == 'holdout':
          evaluation_statistic = self._holdout_validation(ant_train_dataset, evaluation_measures, train_time_stamp)

          logger.info('generate model evaluation report')
          self.stage = 'EVALUATION-HOLDOUT-REPORT'
          # send statistic report
          self.context.job.send({'DATA': {'REPORT': evaluation_statistic}})
          everything_to_html(evaluation_statistic, os.path.join(self.ant_dump_dir, train_time_stamp))
        elif estimation_procedure == "repeated-holdout":
          number_repeats = 2              # default value
          is_stratified_sampling = True   # default value
          split_ratio = 0.6               # default value
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
                                                                   train_time_stamp)
          logger.info('generate model evaluation report')
          self.stage = 'EVALUATION-REPEATEDHOLDOUT-REPORT'
          # send statistic report
          self.context.job.send({'DATA': {'REPORT': evaluation_statistic}})
          everything_to_html(evaluation_statistic, os.path.join(self.ant_dump_dir, train_time_stamp))
        elif estimation_procedure == "bootstrap":
          bootstrap_counts = 5
          if estimation_procedure_params is not None:
            bootstrap_counts = int(estimation_procedure_params.get('bootstrap_counts', bootstrap_counts))
          evaluation_statistic = self._bootstrap_validation(bootstrap_counts,
                                                            ant_train_dataset,
                                                            evaluation_measures,
                                                            train_time_stamp)
          logger.info('generate model evaluation report')
          self.stage = 'EVALUATION-BOOTSTRAP-REPORT'
          # send statistic report
          self.context.job.send({'DATA': {'REPORT': evaluation_statistic}})
          everything_to_html(evaluation_statistic, os.path.join(self.ant_dump_dir, train_time_stamp))
        elif estimation_procedure == "kfold":
          kfolds = 5
          if estimation_procedure_params is not None:
            kfolds = int(estimation_procedure_params.get('kfold', kfolds))
          evaluation_statistic = self._kfold_cross_validation(kfolds, ant_train_dataset, evaluation_measures, train_time_stamp)

          logger.info('generate model evaluation report')
          self.stage = 'EVALUATION-KFOLD-REPORT'
          # send statistic report
          self.context.job.send({'DATA':{'REPORT': evaluation_statistic}})
          everything_to_html(evaluation_statistic, os.path.join(self.ant_dump_dir, train_time_stamp))

      # 3.step model training
      self.stage = "TRAIN"
      train_dump_dir = os.path.join(self.ant_dump_dir, train_time_stamp, 'train')
      if not os.path.exists(train_dump_dir):
          os.makedirs(train_dump_dir)

      logger.info('start training process')
      ant_train_dataset.reset_state()
      self.context.call_training_process(ant_train_dataset, train_dump_dir)

      # 4.step ablation experiment(optional)
      if len(self.context.ablation.blocks) > 0:
        part_train_dataset, part_validation_dataset = ant_train_dataset.split(split_method='holdout')
        part_train_dataset.reset_state()

        for block in self.context.ablation.blocks:
          self.context.ablation.disable(block)
          logger.info('start ablation experiment %s'%block)

          # dump_dir for ablation experiment
          ablation_dump_dir = os.path.join(self.ant_dump_dir, train_time_stamp, 'train', 'ablation', block)
          if not os.path.exists(ablation_dump_dir):
            os.makedirs(ablation_dump_dir)

          # 2.step training model
          self.stage = 'ABLATION(%s)-TRAIN'%block
          self.context.call_training_process(part_train_dataset, ablation_dump_dir)

          # 3.step evaluation measures
          # split data and label
          data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
          self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

          self.stage = 'ABLATION(%s)-EVALUATION'%block
          with safe_recorder_manager(self.context.recorder):
            self.context.call_infer_process(data_annotation_branch.output(0), ablation_dump_dir)

          # clear
          self.context.recorder = None

          ablation_running_statictic = {self.ant_name: {}}
          ablation_evaluation_measure_result = []

          with safe_recorder_manager(RecordReader(ablation_dump_dir)) as record_reader:
            for measure in running_ant_task.evaluation_measures:
              record_generator = record_reader.iterate_read('predict', 'groundtruth')
              result = measure.eva(record_generator, None)
              ablation_evaluation_measure_result.append(result)

          ablation_running_statictic[self.ant_name]['measure'] = ablation_evaluation_measure_result
          self.stage = 'ABLATION(%s)-REPORT'%block
          # send statistic report
          self.context.job.send({'DATA':{'REPORT': ablation_running_statictic}})
          everything_to_html(ablation_evaluation_measure_result, ablation_dump_dir)

  def _holdout_validation(self, train_dataset, evaluation_measures, now_time):
    # 1.step split train set and validation set
    part_train_dataset, part_validation_dataset = train_dataset.split(split_method='holdout')
    part_train_dataset.reset_state()

    # dump_dir
    dump_dir = os.path.join(self.ant_dump_dir, now_time, 'train', 'holdout-evaluation')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

    # 2.step training model
    self.stage = 'EVALUATION-HOLDOUT-TRAIN'
    self.context.call_training_process(part_train_dataset, dump_dir)

    # 3.step evaluation measures
    # split data and label
    data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
    self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

    self.stage = 'EVALUATION-HOLDOUT-EVALUATION'
    with safe_recorder_manager(self.context.recorder):
      with running_statistic(self.ant_name):
        self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

    # clear
    self.context.recorder = None

    task_running_statictic = get_running_statistic(self.ant_name)
    task_running_statictic = {self.ant_name: task_running_statictic}
    task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
    task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
        task_running_elapsed_time / float(part_validation_dataset.size)

    logger.info('start evaluation process')
    evaluation_measure_result = []

    with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
      for measure in evaluation_measures:
        record_generator = record_reader.iterate_read('predict', 'groundtruth')
        result = measure.eva(record_generator, None)
        evaluation_measure_result.append(result)
      task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

    return task_running_statictic

  def _repeated_holdout_validation(self, repeats,
                                   train_dataset,
                                   split_ratio,
                                   is_stratified_sampling,
                                   evaluation_measures,
                                   nowtime):
    repeated_running_statistic = []
    for repeat in range(repeats):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = train_dataset.split(split_params={'ratio': split_ratio,
                                                                                      'is_stratified': is_stratified_sampling},
                                                                        split_method='repeated-holdout')
      part_train_dataset.reset_state()
      # dump_dir
      dump_dir = os.path.join(self.ant_dump_dir, nowtime, 'train', 'repeated-holdout-evaluation', 'repeat-%d'%repeat)
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

      # 2.step training model
      self.stage = 'EVALUATION-REPEATEDHOLDOUT-TRAIN-%d' % repeat
      self.context.call_training_process(part_train_dataset, dump_dir)

      # 3.step evaluation measures
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

      self.stage = 'EVALUATION-REPEATEDHOLDOUT-EVALUATION-%d' % repeat
      with safe_recorder_manager(self.context.recorder):
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

      # clear
      self.context.recorder = None

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('start evaluation process')
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

      repeated_running_statistic.append(task_running_statictic)

    evaluation_result = multi_repeats_measures_statistic(repeated_running_statistic, method='repeated-holdout')
    return evaluation_result

  def _bootstrap_validation(self, bootstrap_rounds, train_dataset, evaluation_measures, nowtime):
    bootstrap_running_statistic = []
    for bootstrap_i in range(bootstrap_rounds):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = train_dataset.split(split_params={},
                                                                        split_method='bootstrap')
      part_train_dataset.reset_state()
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
      self.context.call_training_process(part_train_dataset, dump_dir)

      # 3.step evaluation measures
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

      self.stage = 'EVALUATION-BOOTSTRAP-EVALUATION-%d' % bootstrap_i
      with safe_recorder_manager(self.context.recorder):
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

      # clear
      self.context.recorder = None

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('start evaluation process')
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

      bootstrap_running_statistic.append(task_running_statictic)

    evaluation_result = multi_repeats_measures_statistic(bootstrap_running_statistic, method='bootstrap')
    return evaluation_result

  def _kfold_cross_validation(self, kfolds, train_dataset, evaluation_measures, nowtime):
    assert (kfolds in [5, 10])
    kfolds_running_statistic = []
    for k in range(kfolds):
      # 1.step split train set and validation set
      part_train_dataset, part_validation_dataset = train_dataset.split(split_params={'kfold': kfolds,
                                                                                      'k': k},
                                                                        split_method='kfold')
      part_train_dataset.reset_state()
      # dump_dir
      dump_dir = os.path.join(self.ant_dump_dir, nowtime, 'train', 'kfold-evaluation', 'fold-%d-evaluation' % k)
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

      # 2.step training model
      self.stage = 'EVALUATION-KFOLD-TRAIN-%d' % k
      self.context.call_training_process(part_train_dataset, dump_dir)

      # 3.step evaluation measures
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

      self.stage = 'EVALUATION-KFOLD-EVALUATION-%d' % k
      with safe_recorder_manager(self.context.recorder):
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

      # clear
      self.context.recorder = None

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(part_validation_dataset.size)

      logger.info('start evaluation process')
      evaluation_measure_result = []

      with safe_recorder_manager(RecordReader(dump_dir)) as record_reader:
        for measure in evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

      kfolds_running_statistic.append(task_running_statictic)

    evaluation_result = multi_repeats_measures_statistic(kfolds_running_statistic, method='kfold')
    return evaluation_result