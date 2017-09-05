# encoding=utf-8
# @Time    : 17-6-22
# @File    : work.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import yaml
import os
import time
import shutil
from antgo.ant.basework import *
from multiprocessing import Process, Lock
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antgo.context import *
from antgo.task.task import *
from antgo.measures.measure import *
from antgo.html.html import *
from antgo.utils.cpu import *
from antgo.utils.gpu import *


class Training(BaseWork):
  def __init__(self, name, config_parameters, code_path, code_main_file,
               port='',
               ant_name=None,
               ant_token=None):
    super(Training, self).__init__(name=name,
                                   code_path=code_path,
                                   code_main_file=code_main_file,
                                   config_parameters=config_parameters,
                                   port=port,
                                   ant_name=ant_name,
                                   ant_token=ant_token)

  def run(self, *args, **kwargs):
    # 0.step load context
    self.context = self.load_context()
    self.stage = self.name

    # 1.step load config file
    loaded_training_config = self.load_config()

    # update by loaded training config
    dataset_name = None
    dataset_train_or_test = 'train'
    dataset_params = {}
    if 'dataset' in loaded_training_config:
      dataset_name = loaded_training_config['dataset'].get('name', None)
      dataset_params = loaded_training_config['dataset'].get('params', {})

      # how to split dataset as training dataset
      how_to_split = loaded_training_config['dataset'].get('split', {})
      if len(how_to_split) > 0:
        if 'train' in how_to_split:
          if type(how_to_split['train']) == str:
            # dataset flag (train)
            dataset_train_or_test = how_to_split['train']
          else:
            assert(type(how_to_split['train']) == list)
            # id list
            dataset_params['filter'] = how_to_split['train']

    model_parameters = copy.deepcopy(loaded_training_config)
    # if 'dataset' in model_parameters:
    #   model_parameters.pop('dataset')

    if 'dataset' in self.config_parameters:
      dn = self.config_parameters['dataset'].get('name', None)
      dataset_name = dn if dn is not None else dataset_name
      dataset_params.update(self.config_parameters['dataset'].get('params', {}))

    if 'model' in self.config_parameters:
      model_parameters.update(self.config_parameters.get('model', {}))

    continue_condition = None
    if 'continue' in self.config_parameters:
      continue_condition = {}
      continue_condition['key'] = self.config_parameters['continue']['key']
      continue_condition['value'] = self.config_parameters['continue']['value']
      continue_condition['condition'] = self.config_parameters['continue']['condition']

    if self.gpu is not None:
      model_parameters['gpu'] = self.gpu
    elif self.cpu is not None:
      model_parameters['cpu'] = self.cpu

    # update config file
    loaded_training_config.update(model_parameters)
    # loaded_training_config.update(
    #     {'dataset': {'name': dataset_name, 'train_or_test': dataset_train_or_test, 'params': dataset_params}})
    self.save_config(loaded_training_config)

    # 2.step registry trigger
    if continue_condition is not None:
      self.context.registry_trainer_callback(continue_condition['key'],
                                         continue_condition['value'],
                                         continue_condition['condition'],
                                         self.notify_func)

    # 3.step start running
    self.context.params = model_parameters
    assert(dataset_name is not None)
    dataset_cls = self.context.dataset_factory(dataset_name)
    dataset = dataset_cls(dataset_train_or_test, os.path.join(self.data_factory, dataset_name), dataset_params)
    dataset.reset_state()
    with safe_recorder_manager(dataset):
      self.context.call_training_process(dataset, self.dump_dir)

      # 4.step work is done
      self.trigger(self.dump_dir, 'DONE')
      self.context.wait_until_clear()

  def notify_func(self):
    # 1.step notify
    self.trigger(self.dump_dir, 'CONTINUE')


class Inference(BaseWork):
  def __init__(self, name, config_parameters, code_path, code_main_file,
               port='',
               ant_name=None,
               ant_token=None):
    super(Inference, self).__init__(name=name,
                                    code_path=code_path,
                                    code_main_file=code_main_file,
                                    config_parameters=config_parameters,
                                    port=port,
                                    ant_name=ant_name,
                                    ant_token=ant_token)

  def run(self, *args, **kwargs):
    # 0.step load ctx
    self.context = self.load_context()
    self.stage = self.name

    # 1.step load config file
    dataset_name = None
    dataset_train_or_test = 'test'
    dataset_params = {}
    loaded_infer_config = self.load_config()

    model_parameters = {}
    if loaded_infer_config is not None:
      if 'dataset' in loaded_infer_config:
        dataset_name = loaded_infer_config['dataset'].get('name', None)
        dataset_params = loaded_infer_config['dataset'].get('params', {})

        # how to split dataset as training dataset
        how_to_split = loaded_infer_config['dataset'].get('split', {})
        if len(how_to_split) > 0:
          if 'test' in how_to_split:
            if type(how_to_split['test']) == str:
              # dataset flag (train)
              dataset_train_or_test = how_to_split['test']
            else:
              assert (type(how_to_split['test']) == list)
              # id list
              dataset_params['filter'] = how_to_split['test']
              dataset_train_or_test = 'train'

      model_parameters = copy.deepcopy(loaded_infer_config)
      # if 'dataset' in model_parameters:
      #   model_parameters.pop('dataset')

    # custom config
    if 'dataset' in self.config_parameters:
      dn = self.config_parameters['dataset'].get('name', None)
      dataset_name = dn if dn is not None else dataset_name
      dataset_params.update(self.config_parameters['dataset'].get('params', {}))

    if 'model' in self.config_parameters:
      model_parameters.update(self.config_parameters.get('model', {}))

    if self.gpu is not None:
      model_parameters['gpu'] = self.gpu
    elif self.cpu is not None:
      model_parameters['cpu'] = self.cpu

    # update config file
    loaded_infer_config.update(model_parameters)
    # loaded_infer_config.update(
    #     {'dataset': {'name': dataset_name, 'train_or_test': dataset_train_or_test, 'params': dataset_params}})

    assert(dataset_name is not None)
    self.save_config(loaded_infer_config)

    # 2.step start running
    self.context.params = model_parameters
    dataset_cls = self.context.dataset_factory(dataset_name)
    dataset = dataset_cls(dataset_train_or_test, os.path.join(self.data_factory, dataset_name), dataset_params)
    
    with safe_recorder_manager(dataset):
      data_annotation_branch = DataAnnotationBranch(Node.inputs(dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
      with safe_recorder_manager(self.context.recorder):
        self.context.call_infer_process(data_annotation_branch.output(0), self.dump_dir)
      
      self.context.recorder = None
  
      # work is done
      self.trigger(os.path.join(self.dump_dir), 'DONE')
      self.context.wait_until_clear()


class Evaluating(BaseWork):
  def __init__(self, name, config_parameters, code_path, code_main_file,
               port='',
               ant_name=None,
               ant_token=None):
    super(Evaluating, self).__init__(name=name,
                                     code_path=code_path,
                                     code_main_file=code_main_file,
                                     config_parameters=config_parameters,
                                     port=port,
                                     ant_name=ant_name,
                                     ant_token=ant_token)

  def run(self, *args, **kwargs):
    # 0.step load context
    self.context = self.load_context()
    self.stage = self.name

    assert('task' in self.config_parameters)
    assert('type' in self.config_parameters['task'])
    class_label = []
    if 'class_label' in self.config_parameters['task']:
      class_label = self.config_parameters['task']['class_label']

    method = 'repeated-holdout'
    if 'method' in self.config_parameters:
      method = self.config_parameters['method']

    dummy_ant_task = AntTask(task_id=-1, task_name=None, task_type_id=-1,
                             task_type=self.config_parameters['task']['type'],
                             dataset_id=-1, dataset_name=None, dataset_params=None,
                             estimation_procedure_type=None, estimation_procedure_params=None,
                             evaluation_measure=None, cost_matrix=None,
                             class_label=class_label)

    temp = os.listdir(self.dump_dir)
    experiment_folders = []
    for f in temp:
      if os.path.isdir(os.path.join(self.dump_dir, f)):
        experiment_folders.append(os.path.join(self.dump_dir, f))

    if len(experiment_folders) == 0:
      task_running_statictic = {}
      task_running_statictic[self.name] = {}
      evaluation_measure_result = []
      
      with safe_recorder_manager(RecordReader(self.dump_dir)) as record_reader:
        for measure in dummy_ant_task.evaluation_measures:
          if measure.name in self.config_parameters['measure']:
            record_generator = record_reader.iterate_read('predict', 'groundtruth')
            result = measure.eva(record_generator, None)
            evaluation_measure_result.append(result)
      task_running_statictic[self.name]['measure'] = evaluation_measure_result
      # evaluation report
      everything_to_html(task_running_statictic, self.dump_dir)
    else:
      multi_expriments = []
      for experiment_folder in experiment_folders:
        task_running_statictic = {}
        task_running_statictic[self.name] = {}

        evaluation_measure_result = []
        with safe_recorder_manager(RecordReader(experiment_folder)) as record_reader:
          for measure in dummy_ant_task.evaluation_measures:
            if measure.name in self.config_parameters['measure']:
              record_generator = record_reader.iterate_read('predict', 'groundtruth')
              result = measure.eva(record_generator, None)
              evaluation_measure_result.append(result)
        task_running_statictic[self.name]['measure'] = evaluation_measure_result

        multi_expriments.append(task_running_statictic)
      evaluation_result = multi_repeats_measures_statistic(multi_expriments, method=method)
      # evaluation report
      everything_to_html(evaluation_result, self.dump_dir)

    self.trigger('', 'DONE')
    self.context.wait_until_clear()


class DataSplit(BaseWork):
  def __init__(self, name, config_parameters, code_path, code_main_file,
               port='',
               ant_name=None,
               ant_token=None):
    super(DataSplit, self).__init__(name=name,
                                    code_path=code_path,
                                    code_main_file=code_main_file,
                                    config_parameters=config_parameters,
                                    port=port,
                                    ant_name=ant_name,
                                    ant_token=ant_token)

    # custom experiment
    self.split_method = self.config_parameters['method']
    self.split_params = self.config_parameters.get('params', {})

    assert (self.split_method in ['holdout', 'repeated-holdout', 'bootstrap', 'kfold'])

    self.dataset_name = self.config_parameters['dataset']['name']
    self.dataset_params = self.config_parameters['dataset'].get('params', {})
    self.dataset_train_or_test = self.config_parameters['dataset'].get('train_or_test', 'train')

    self.need_waiting_feedback = True

    self.trials = 0
    if self.split_method == 'repeated-holdout':
      self.repeated_times = self.split_params['number_repeats']
    elif self.split_method == 'bootstrap':
      self.bootstrap_counts = self.split_params['bootstrap_counts']
    elif self.split_method == 'kfold':
      self.kfold = self.split_params['kfold']

  def run(self, *args, **kwargs):
    if self.context is None:
      self.context = self.load_context()
      self.stage = self.name

    # 0.step load model
    if self.split_method == 'holdout':
      if self.trials == 1:
        self.trigger('', 'FEEDBACK-DONE')
        return

      dataset_cls = self.context.dataset_factory(self.dataset_name)
      dataset = dataset_cls(self.dataset_train_or_test,
                            os.path.join(self.data_factory, self.dataset_name),
                            self.dataset_params)
      
      with safe_recorder_manager(dataset):
        t, v = dataset.split(self.split_params, self.split_method)
        dataset_config = copy.deepcopy(self.config_parameters)
        dataset_config['dataset']['split'] = {}
        dataset_config['dataset']['split']['train'] = t.ids
        dataset_config['dataset']['split']['test'] = v.ids
  
        with open(os.path.join(self.dump_dir, 'experiment-holdout-config.yaml'), 'w') as fp:
          yaml.dump(dataset_config, fp)

      self.trials += 1
      # task is done
      self.trigger(os.path.join(self.dump_dir, 'experiment-holdout-config.yaml'), 'DONE')
    elif self.split_method == 'repeated-holdout':
      if self.trials == self.repeated_times:
        self.trigger('', 'FEEDBACK-DONE')
        return

      dataset_cls = self.context.dataset_factory(self.dataset_name)
      dataset = dataset_cls(self.dataset_train_or_test,
                            os.path.join(self.data_factory, self.dataset_name),
                            self.dataset_params)
      
      with safe_recorder_manager(dataset):
        t, v = dataset.split(self.split_params, self.split_method)
        dataset_config = copy.deepcopy(self.config_parameters)
        dataset_config['dataset']['split'] = {}
        dataset_config['dataset']['split']['train'] = t.ids
        dataset_config['dataset']['split']['test'] = v.ids
  
        with open(os.path.join(self.dump_dir, 'experiment-repeated-holdout-%d-config.yaml' % self.trials), 'w') as fp:
          yaml.dump(dataset_config, fp)

      self.trials += 1
      # trigger next task
      self.trigger(os.path.join(self.dump_dir, 'experiment-repeated-holdout-%d-config.yaml' % (self.trials - 1)), 'DONE')
    elif self.split_method == 'bootstrap':
      if self.trials == self.bootstrap_counts:
        self.trigger('', 'FEEDBACK-DONE')
        return

      dataset_cls = self.context.dataset_factory(self.dataset_name)
      dataset = dataset_cls(self.dataset_train_or_test,
                            os.path.join(self.data_factory, self.dataset_name),
                            self.dataset_params)
      
      with safe_recorder_manager(dataset):
        t, v = dataset.split(self.split_params, self.split_method)
        dataset_config = copy.deepcopy(self.config_parameters)
        dataset_config['dataset']['split'] = {}
        dataset_config['dataset']['split']['train'] = t.ids
        dataset_config['dataset']['split']['test'] = v.ids
  
        with open(os.path.join(self.dump_dir, 'experiment-bootstrap-%d-config.yaml' % self.trials), 'w') as fp:
          yaml.dump(dataset_config, fp)

      self.trials += 1
      # trigger next task
      self.trigger(os.path.join(self.dump_dir, 'experiment-bootstrap-%d-config.yaml' % (self.trials - 1)), 'DONE')
    elif self.split_method == 'kfold':
      if self.trials == self.kfold:
        self.trigger('', 'FEEDBACK-DONE')
        return

      dataset_cls = self.context.dataset_factory(self.dataset_name)
      dataset = dataset_cls(self.dataset_train_or_test,
                            os.path.join(self.data_factory, self.dataset_name),
                            self.dataset_params)
      
      with safe_recorder_manager(dataset):
        self.split_params['k'] = self.trials
    
        t, v = dataset.split(self.split_params, self.split_method)
        dataset_config = copy.deepcopy(self.config_parameters)
        dataset_config['dataset']['split'] = {}
        dataset_config['dataset']['split']['train'] = t.ids
        dataset_config['dataset']['split']['test'] = v.ids
    
        with open(os.path.join(self.dump_dir, 'experiment-kfold-%d-config.yaml' % self.trials), 'w') as fp:
          yaml.dump(dataset_config, fp)

      self.trials += 1
      # trigger next task
      self.trigger(os.path.join(self.dump_dir, 'experiment-kfold-%d-config.yaml' % (self.trials - 1)), 'DONE')
    else:
        raise NotImplementedError()