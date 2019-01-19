# -*- coding: UTF-8 -*-
# Time: 10/8/17
# File: cmd.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.ant.base import *
from antgo.ant import flags
from antgo.ant.challenge import *
from antgo.ant.train import *
from antgo.ant.utils import *
from antgo.resource.html import *
from antgo import config
from antgo.task.task import *
from antgo.measures import *
from antgo.measures.repeat_statistic import *
from antgo.utils import logger
from multiprocessing import Process
import subprocess
import getopt
import json
import shutil
import sys
if sys.version > '3':
  PY3 = True
else:
  PY3 = False


FLAGS = flags.AntFLAGS
Config = config.AntConfig


class AntCmd(AntBase):
  def __init__(self, ant_token):
    flags.DEFINE_integer('id', None, 'task or experiment id')
    flags.DEFINE_boolean('model', None, 'experiment main_file and main_param')
    flags.DEFINE_boolean('report', None, 'experiment report')
    flags.DEFINE_boolean('optimum', None, 'whether experiment is optimum')
    flags.DEFINE_boolean('is_local', None, 'whether store in local')
    flags.DEFINE_boolean('is_public',None, 'whether public in cloud')
    flags.DEFINE_string('dataset_name',None, 'dataset name')
    flags.DEFINE_string('dataset_path',None, 'dataset path')
    flags.DEFINE_string('dataset_url', None, 'dataset url')
    flags.DEFINE_string('dataset_train_or_test', 'train', 'dataset train or test')
    flags.DEFINE_string('dataset_params', None, 'dataset parse parameter')
    flags.DEFINE_string('task_name',None,'task name')
    flags.DEFINE_string('task_type',None, 'task type')
    flags.DEFINE_string('task_measure',None, 'task measure')
    flags.DEFINE_string('task_est',None,'task estimation procedure')
    flags.DEFINE_string('task_est_params',None,'task estimation procedure params')
    flags.DEFINE_string('task_params', None, 'task extent parameter')
    flags.DEFINE_string('task_class_label',None, 'classification task label')
    flags.DEFINE_string('experiment_name', None, 'experiment name')
    flags.DEFINE_string('group', None, 'group name')
    flags.DEFINE_string('user', None, 'user name')

    super(AntCmd, self).__init__('CMD', ant_token=ant_token)

  def process_task_command(self):
    task_id = FLAGS.id()        # task id
    if task_id is None:
      task_id = -1

    remote_api = 'hub/api/terminal/task/%d'%task_id
    response = self.remote_api_request(remote_api)

    if task_id == -1:
      print('%-5s %-10s %-30s %-10s %-20s %-30s'%('id','name','time','dataset','experiments','token'))

      for task in response:
        task_id = task['task-id']
        task_name = '-' if len(task['task-name']) == 0 else task['task-name']
        task_time = task['task-time']
        task_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task_time))
        task_dataset = task['task-dataset']
        task_experiments = task['task-experiment']
        task_token = task['task-token']
        print('%-5d %-10s %-30s %-10s %-20d %-30s'%(task_id,task_name,task_time,task_dataset,task_experiments, task_token))
    else:
      print('%-5s %-10s %-30s %-10s %-10s %-10s'%('id','name','time','optimum','report','model'))
      for experiment in response:
        experiment_id = experiment['experiment-id']
        experiment_name = experiment['experiment-name']
        experiment_time = experiment['experiment-time']
        experiment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment_time))
        experiment_optimum = experiment['experiment-optimum']
        experiment_report = experiment['experiment-report']
        experiment_model = experiment['experiment-model']

        print('%-5d %-10s %-30s %-10d %-10d %-10d'%
              (experiment_id,
               experiment_name,
               experiment_time,
               experiment_optimum,
               experiment_report,
               experiment_model))

  def process_experiment_command(self):
    experiment_id = FLAGS.id()
    experiment_download_model = FLAGS.model()
    experiment_download_report = FLAGS.report()

    if experiment_id is None:
      experiment_id = -1

    # remote api
    remote_api = 'hub/api/terminal/experiment/%d' % experiment_id
    response = self.remote_api_request(remote_api, action='get')
    if response is None or len(response) == 0:
      print('no experiment')
      return

    if experiment_id == -1:
      print('%-5s %-10s %-10s %-30s %-10s %-10s %-10s' % ('id', 'name', 'task','time', 'optimum', 'report', 'model'))
      for experiment in response:
        experiment_id = experiment['experiment-id']
        experiment_name = experiment['experiment-name']
        experiment_time = experiment['experiment-time']
        experiment_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment_time))
        experiment_task = experiment['experiment-task']
        experiment_report = experiment['experiment-report']
        experiment_model = experiment['experiment-model']
        experiment_optimum = experiment['experiment-optimum']

        print('%-5d %-10s %-10s %-30s %-10d %-10d %-10d' %
              (experiment_id,
               experiment_name,
               experiment_task,
               experiment_time,
               experiment_optimum,
               experiment_report,
               experiment_model))

      return

    experiment_name = response['experiment-name']
    if experiment_download_model is not None:
      # download experiment model request
      remote_api = 'hub/api/terminal/download/%s/experiment/%d/model' %(self.app_token, experiment_id)
      url = '%s://%s:%s/%s' % (self.http_prefix, self.server_ip, self.http_port, remote_api)

      target_path = os.path.join(os.curdir,'experiment', experiment_name)
      if not os.path.exists(target_path):
        os.makedirs(target_path)

      self.download(url,
                    target_path=target_path,
                    target_name='%s.tar.gz'%experiment_name,
                    archive='model')

    if experiment_download_report is not None:
      # experiment report at every stage of experiment
      # download experiment model request
      remote_api = 'hub/api/terminal/download/%s/experiment/%d/report' % (self.app_token, experiment_id)
      url = '%s://%s:%s/%s' % (self.http_prefix, self.server_ip, self.http_port, remote_api)

      target_path = os.path.join(os.curdir, 'experiment', experiment_name)
      if not os.path.exists(target_path):
        os.makedirs(target_path)

      self.download(url,
                    target_path=target_path,
                    target_name='%s' % experiment_name)

      # read report and transform to resource
      if os.path.exists(os.path.join(target_path,experiment_name)):
        fp = open(os.path.join(target_path,experiment_name), 'rb')
        report_data = fp.read()
        report_data = loads(report_data)
        fp.close()
        # clear temp report file
        os.remove(os.path.join(target_path,experiment_name))

        if len(report_data) == 0:
          print('no experiment reports')
          return
        for stage, stage_report in report_data.items():
          target_path = os.path.join(os.curdir, 'experiment', experiment_name,'report', stage)
          if not os.path.exists(target_path):
            os.makedirs(target_path)

          everything_to_html(stage_report, target_path)

  def process_dataset_command(self):
    dataset_id = FLAGS.id()  # dataset id
    if dataset_id is None:
      dataset_id = -1

    remote_api = 'hub/api/terminal/dataset/%d' % dataset_id
    response = self.remote_api_request(remote_api)
    print('%-5s %-10s %-30s %-10s' % ('id', 'name', 'time', 'tasks'))
    for dataset in response:
      dataset_id = dataset['dataset-id']
      dataset_name = '-' if len(dataset['dataset-name']) == 0 else dataset['dataset-name']
      dataset_time = dataset['dataset-time']
      dataset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dataset_time))
      dataset_tasks = dataset['dataset-tasks']
      print('%-5d %-10s %-30s %-10d' % (dataset_id, dataset_name, dataset_time, dataset_tasks))

  def process_apply_command(self):
    task_id = FLAGS.id()
    if task_id is None:
      task_id = -1

    if task_id == -1:
      remote_api = 'hub/api/terminal/apply/task/-1'
      response = self.remote_api_request(remote_api, action='get')

      # task_id, task_name, task_time, task_dataset, task_applicants(apllicants)
      print('%-5s %-10s %-30s %-10s %-10s'%('id','name','time','dataset','applicants'))
      for item in response:
        task_id = item['task-id']
        task_name = item['task-name']
        task_time = item['task-time']
        task_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task_time))
        task_dataset = item['task-dataset']
        task_applicants = item['task-applicants']
        print('%-5d %-10s %-30s %-10s %-10d'%(task_id,task_name,task_time,task_dataset,task_applicants))

    else:
      remote_api = 'hub/api/terminal/apply/task/%d'%task_id
      response = self.remote_api_request(remote_api, action='post')

      if response is None or len(response) == 0:
        print('task (id=%d) is not existed or you dont have permission'%task_id)
        return

      print('task (id=%d) has been applied successfully (token = %s), now challenge go!!'%(task_id, response['token']))

  def process_create_command(self):
    ################################################
    ###########   stage.1 create group #############
    ################################################
    if FLAGS.group() is not None and FLAGS.task_name() is None:
      # creat group
      res = self.remote_api_request('hub/api/terminal/groups/%s'%str(FLAGS.group()), None, 'post')
      if res is not None:
        logger.info('success to create %s group'%str(FLAGS.group()))
      else:
        logger.error('fail to create %s group'%str(FLAGS.group()))
      return

    ################################################
    ###########   stage.2 crate task   #############
    ################################################
    if  FLAGS.task_name() is None:
      logger.error('if you want to create task, you must set task name')
      return

    # every task must be related with one dataset
    # 0.step check paramters
    is_public = FLAGS.is_public()
    if is_public is None:
      is_public = 0

    dataset_name = FLAGS.dataset_name()   # must be unique at cloud
    if dataset_name is None:
      logger.error('dataset name must be set')
      return

    # check task type
    task_type = FLAGS.task_type()
    if task_type is None:
      logger.error('must set task type')
      return
    if task_type not in AntTask.support_task_types():
      logger.error('task type must be in [%s]'%','.join(AntTask.support_task_types()))
      return

    # check task measures
    task_measures = FLAGS.task_measure()
    if task_measures is None:
      logger.error('must set task measure')
      return

    task_measures = task_measures.split(',')
    dummy_task = create_dummy_task(task_type)
    dummy_task_measures = AntMeasuresFactory(dummy_task)
    task_support_measures = [measure.name for measure in dummy_task_measures.measures()]
    for measure in task_measures:
      if measure not in task_support_measures:
        logger.error('task measure %s not supported by "%s" task'%(measure, task_type))
        return
    task_measures = json.dumps(task_measures)

    # check task estimation procedure
    task_est = FLAGS.task_est()
    if task_est is not None:
      if task_est not in ['holdout','repeated-holdout','bootstrap','kfold']:
        logger.error('task estimation procedure %s not in [%s]'%(task_est,','.join(['holdout','repeated-holdout','bootstrap','kfold'])))
        return

    # check task estimation procedure parameters
    task_est_params = FLAGS.task_est_params()
    task_est_params_dict = {}
    task_est_params_dict['params'] = {}
    if task_est_params is not None:
      task_est_params = task_est_params.split(';')
      for param in task_est_params:
        param_key_value = param.split(':')
        if len(param_key_value) != 2:
          logger.error('task estimation procedure params must be format like "key:value;key:value;..."')
          return
        else:
          task_est_params_dict['params'][param_key_value[0]] = param_key_value[1]
    task_est_params = json.dumps(task_est_params_dict)

    # check task extent parameters(some closed measures different parameters)
    task_params = FLAGS.task_params()
    task_params_dict = {}
    if task_params is not None:
      task_params_splits = task_params.split(';')
      for param in task_params_splits:
        param_key_value = param.split(':')
        if len(param_key_value) != 2:
          logger.error('task extent params must be format like "key:value;key:value;..."')
          return
        else:
          task_params_dict[param_key_value[0]] = param_key_value[1]
      task_params = json.dumps(task_params_dict)
    else:
      task_params = {}

    # check task class label
    task_class_label = FLAGS.task_class_label()
    if task_class_label is not None:
      if ',' in task_class_label:
        task_class_label = json.dumps(task_class_label.split(','))
      else:
        task_class_label = [i for i in range(int(task_class_label))]
        task_class_label = json.dumps(task_class_label)

    # create task binded with dataset
    task_name = FLAGS.task_name()
    remote_api = 'hub/api/terminal/create/task'
    response = self.remote_api_request(remote_api,
                                       action='post',
                                       data={'task-name': task_name,
                                             'task-dataset-name': dataset_name,
                                             'task-type': task_type,
                                             'task-measures': task_measures,
                                             'task-estimation-procedure':task_est,
                                             'task-estimation-procedure-params':task_est_params,
                                             'task-params':task_params,
                                             'task-class-label':task_class_label,
                                             'task-is-public':int(is_public),
                                             'group':FLAGS.group()})

    if response['status'] != 'OK':
      logger.error('task name has been existed at cloud, please reset')
      return

    logger.info('task has been created successfully, please enjoy...')

  def process_add_command(self):
    ######################################################
    ############  stage.1 add user  ######################
    ######################################################
    if FLAGS.user() is not None and FLAGS.group() is not None:
      users_info = {'users': [FLAGS.user()]}
      res = self.remote_api_request('hub/api/terminal/groups/%s'%str(FLAGS.group()),
                                   json.dumps(users_info),
                                   'patch')
      if res is not None:
        logger.info('add user into group %s'%str(FLAGS.group()))
      else:
        logger.error('fail to add user')
      return

    ######################################################
    ############  stage.2 add new task type   ############
    ############  and task measure            ############
    ######################################################
    task_type = FLAGS.task_type()
    task_measure = FLAGS.task_measure()

    if task_type is None or task_measure is None:
      logger.error('need set task_type and task_measure simutaneously')
      return

    task_measures = task_measure.split(',')
    task_measures = json.dumps(task_measures)

    remote_api = 'hub/api/terminal/task/type/%s'%task_type
    response = self.remote_api_request(remote_api,
                                       action='post',
                                       data={'task-measures': task_measures})

    if response is None:
      logger.error('fail to add task type')
      return

    logger.info('success to add task type')
    print(response)

  def process_del_command(self):
    dataset_name = FLAGS.dataset_name()
    task_name = FLAGS.task_name()
    experiment_name = FLAGS.experiment_name()
    task_type = FLAGS.task_type()
    task_measure = FLAGS.task_measure()
    group_name = FLAGS.group()
    user_name = FLAGS.user()

    if group_name is not None and user_name is not None:
      # delete user from group
      users_info = {'users': [user_name]}
      res = self.remote_api_request('hub/api/terminal/groups/%s'%str(group_name),
                                   json.dumps(users_info),
                                   'delete')
      if res is not None:
        logger.info('success to del user %s from group %s'%(str(user_name),str(group_name)))
      else:
        logger.error('fail to del user %s from group %s'%(str(user_name),str(group_name)))
      return

    if group_name is not None and user_name is None:
      # detele group
      logger.error('dont support del group')
      return

    if user_name is not None:
      logger.warn('ignore user_name set')

    if dataset_name is None and \
            task_name is None and \
            experiment_name is None and \
            task_type is None:
      logger.error('must set delete object [%s]' % ','.join(['dataset',
                                                             'task',
                                                             'experiment',
                                                             'task-type']))
      return

    if dataset_name is not None:
      delete_remote_api = 'hub/api/terminal/delete/dataset'
      response = self.remote_api_request(delete_remote_api,
                                         action='delete',
                                         data={'dataset-name': dataset_name})
      if response['status'] != 'OK':
        logger.error('delete error')
        return
    elif task_name is not None:
      delete_remote_api = 'hub/api/terminal/delete/task'
      response = self.remote_api_request(delete_remote_api,
                                         action='delete',
                                         data={'task-name': task_name})

      if response['status'] != 'OK':
        logger.error('delete error')
        return
    elif experiment_name is not None:
      delete_remote_api = 'hub/api/terminal/delete/experiment'
      response = self.remote_api_request(delete_remote_api,
                                         action='delete',
                                         data={'experiment-name': experiment_name})

      if response['status'] != 'OK':
        logger.error('delete error, maybe experiment name not unique')
        return
    elif task_type is not None:
      data = {}
      if task_measure is not None:
        task_measures = task_measure.split(',')
        task_measures = json.dumps(task_measures)
        data['task-measures'] = task_measures

      remote_api = 'hub/api/terminal/task/type/%s'%task_type
      response = self.remote_api_request(remote_api,
                                         action='delete',
                                         data=data)

      if response is None:
        logger.error('fail to delete task type')
        return

      logger.info('success to delete task type')
      print(response)

  def process_update_command(self):
    task_name = FLAGS.task_name()
    experiment_name = FLAGS.experiment_name()

    if task_name is None and experiment_name is None:
      logger.error('must set update object [%s]'%','.join(['task', 'experiment']))
      return

    if task_name is not None:
      # update body
      update_dict = {'task-name': task_name}

      # parameter-1: check task type
      task_type = FLAGS.task_type()
      if task_type is not None:
        if task_type not in AntTask.support_task_types():
          logger.error('task type must be in [%s]' % ','.join(AntTask.support_task_types()))
          return

        update_dict['task-type'] = task_type

      # parameter-2: check task measures
      task_measures = FLAGS.task_measure()
      if task_measures is None:
        logger.error('must set task measure')
        return

      if task_measures is not None:
        task_measures = task_measures.split(',')
        dummy_task = create_dummy_task(task_type)
        dummy_task_measures = AntMeasuresFactory(dummy_task)
        task_support_measures = [measure.name for measure in dummy_task_measures.measures()]
        for measure in task_measures:
          if measure not in task_support_measures:
            logger.error('task measure %s not supported by "%s" task' % (measure, task_type))
            return
        task_measures = json.dumps(task_measures)

        update_dict['task-measures'] = task_measures

      # parameter-3: check task estimation procedure
      task_est = FLAGS.task_est()
      if task_est is not None:
        if task_est not in ['holdout', 'repeated-holdout', 'bootstrap', 'kfold']:
          logger.error('task estimation procedure %s not in [%s]' % (
          task_est, ','.join(['holdout', 'repeated-holdout', 'bootstrap', 'kfold'])))
          return

        update_dict['task-estimation-procedure'] = task_est

      # parameter-4: check task estimation procedure parameters
      task_est_params = FLAGS.task_est_params()
      if task_est_params is not None:
        task_est_params_dict = {}
        task_est_params_dict['params'] = {}

        task_est_params = task_est_params.split(';')
        for param in task_est_params:
          param_key_value = param.split(':')
          if len(param_key_value) != 2:
            logger.error('task estimation procedure params must be format like "key:value;key:value;..."')
            return
          else:
            task_est_params_dict['params'][param_key_value[0]] = param_key_value[1]

        task_est_params = json.dumps(task_est_params_dict)

        update_dict['task-estimation-procedure-params'] = task_est_params

      # parameter-5: check task extent parameters(some closed measures different parameters)
      task_params = FLAGS.task_params()
      if task_params is not None:
        task_params_dict = {}
        task_params_splits = task_params.split(';')
        for param in task_params_splits:
          param_key_value = param.split(':')
          if len(param_key_value) != 2:
            logger.error('task extent params must be format like "key:value;key:value;..."')
            return
          else:
            task_params_dict[param_key_value[0]] = param_key_value[1]
        task_params = json.dumps(task_params_dict)

        update_dict['task-params'] = task_params

      # parameter-6: check task class label
      task_class_label = FLAGS.task_class_label()
      if task_class_label is not None:
        if ',' in task_class_label:
          task_class_label = json.dumps(task_class_label.split(','))
        else:
          task_class_label = [i for i in range(int(task_class_label))]
          task_class_label = json.dumps(task_class_label)

        update_dict['task-class-label'] = task_class_label

      # parameter-7: check is public
      is_public = FLAGS.is_public()
      if is_public is not None:
        update_dict['task-is-public'] = 1 if is_public else 0

      remote_api = 'hub/api/terminal/update/task'
      response = self.remote_api_request(remote_api,
                                         action='patch',
                                         data=update_dict)

      if response is None:
        logger.error('update error')
        return
    elif experiment_name is not None:
      # rename experiment name request
      experiment_optimum = FLAGS.optimum()

      data = {'experiment-name': experiment_name}
      if experiment_optimum is not None:
        data['experiment-optimum'] = int(experiment_optimum)

      if len(data) > 1:
        remote_api = 'hub/api/terminal/update/experiment'
        response = self.remote_api_request(remote_api, data=data, action='patch')

        if response is None:
          logger.error('update error')
          return

  def process_upload_command(self):
    # dataset name
    dataset_name = FLAGS.dataset_name()   # must be unique at cloud
    dataset_is_local = FLAGS.is_local()
    dataset_is_public = FLAGS.is_public()
    if dataset_name is None or dataset_is_local is None or dataset_is_public is None:
      logger.error('must set dataset_name, is_local and is_public')
      return

    #################### stage 1 - lookup and create dataset ########################
    # lookup dataset record at cloud
    create_dataset_remote_api = 'hub/api/terminal/create/dataset'
    response = self.remote_api_request(create_dataset_remote_api, data={'dataset-name': dataset_name,
                                                                        'dataset-is-local': int(dataset_is_local),
                                                                        'dataset-is-public': int(dataset_is_public)},action='post')

    if 'status' not in response:
      logger.error('fail to create dataset')
      return

    if response['status'] != 'OK':
      logger.error('fail to create dataset')
      return

    # dataset valid name (dataset name maybe added prefix automatically)
    dataset_name = response['dataset-name']

    ################# stage 2 - upload and update dataset config #####################
    # only public or self created dataset is allowed
    if dataset_is_local:
      # copy dataset to local datafactory
      data_factory = getattr(Config, 'data_factory', None)
      dataset_path = FLAGS.dataset_path()
      if dataset_path is None:
        logger.error('dataset_path must be set')
        return

      if not os.path.exists(dataset_path):
        logger.error('dataset path dont exist')
        return

      if not os.path.isdir(dataset_path):
        logger.error('dataset path must be folder')
        return

      if os.path.exists(os.path.join(data_factory, dataset_name)):
        logger.error('dataset has existed')
        return

      # 1.step config dataset parameter
      dataset_params = FLAGS.dataset_params()
      dataset_params_dict = {}
      if dataset_params is not None:
        for kv_pair in dataset_params.split(';'):
          param_key_value = kv_pair.split(':')
          if len(param_key_value) != 2:
            logger.error('task extent params must be format like "key:value;key:value;..."')
            return

          k, v = param_key_value
          dataset_params_dict[k] = v

      if len(dataset_params_dict) > 0:
        create_dataset_remote_api = 'hub/api/terminal/update/dataset'
        response = self.remote_api_request(create_dataset_remote_api,
                                           action='patch',
                                           data={'dataset-name': dataset_name,
                                                 'dataset-param': json.dumps(dataset_params_dict)})
        if response['status'] != 'OK':
          logger.error('fail to update dataset parameter')
          return

      # 2.step move dataset to datafactory physically
      shutil.copytree(dataset_path, os.path.join(data_factory, dataset_name))
      return
    else:
      # upload dataset to cloud
      dataset_path = FLAGS.dataset_path()
      dataset_url = FLAGS.dataset_url()

      if dataset_url is None and dataset_path is None:
        logger.error('must set dataset url or dataset local path')
        return

      if dataset_url is not None:
        # dataset is provided by 3rdpart
        # check dataset url address is valid
        create_dataset_remote_api = 'hub/api/terminal/update/dataset'
        response = self.remote_api_request(create_dataset_remote_api,
                                           action='patch',
                                           data={'dataset-name': dataset_name,
                                                 'dataset-url': dataset_url})
        if response['status'] == 'OK':
          logger.info('dataset address has been config successfully')
        else:
          logger.error('dataset address upload error')
      else:
        if not os.path.isfile(dataset_path):
          logger.error('only support single file')
          return

        # dataset train or test ('train, test, val or sample)
        dataset_train_or_test = FLAGS.dataset_train_or_test()
        if not PY3:
          dataset_train_or_test = unicode(dataset_train_or_test)
        if dataset_train_or_test not in ['train', 'test', 'val', 'sample']:
          logger.error('dataset_train_or_test must be in "%s"' % ",".join(['train', 'test', 'val', 'sample']))
          return

        # 1.step config dataset parameter
        dataset_params = FLAGS.dataset_params()
        dataset_params_dict = {}
        if dataset_params is not None:
          for kv_pair in dataset_params.split(';'):
            param_key_value = kv_pair.split(':')
            if len(param_key_value) != 2:
              logger.error('task extent params must be format like "key:value;key:value;..."')
              return

            k, v = param_key_value
            dataset_params_dict[k] = v

        if len(dataset_params_dict) > 0:
          create_dataset_remote_api = 'hub/api/terminal/update/dataset'
          response = self.remote_api_request(create_dataset_remote_api,
                                             action='patch',
                                             data={'dataset-name': dataset_name,
                                                   'dataset-param': json.dumps(dataset_params_dict)})
          if response['status'] != 'OK':
            logger.error('fail to update dataset parameter')
            return

        # 2.step upload dataset file
        dataset_path = dataset_path.replace('\\','/')
        file_name = dataset_path.split('/')[-1]
        if not PY3:
          dataset_path = unicode(dataset_path)
          file_name = unicode(file_name)
        flag = self.send_file(dataset_path, dataset_name, dataset_train_or_test, 'DATASET-FILE', file_name)
        if not flag:
          logger.error('dataset uploaded error')
          return

  def _key_params(self):
    # related parameters
    # 0.step token
    token = FLAGS.token()
    if not PY3 and token is not None:
      token = unicode(token)
    token = self.app_token if token is None else token

    # 1.step check name, if None, set it as current time automatically
    name = FLAGS.name()
    if name is None:
      name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    if not PY3:
      name = unicode(name)

    # 2.step check main folder (all related model code, includes main_file and main_param)
    main_folder = FLAGS.main_folder()
    if main_folder is None:
      main_folder = os.path.abspath(os.curdir)

    main_file = FLAGS.main_file()
    if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
      logger.error('main executing file dont exist')
      return

    # 3.step check dump dir (all running data is stored here)
    dump_dir = FLAGS.dump()
    if dump_dir is None:
      dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
      if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # 4.step what is task
    task = FLAGS.task()

    # 5.step model params
    main_param = FLAGS.main_param()

    return token, name, main_file, main_folder, dump_dir, main_param, task

  def process_challenge_command(self):
    token, name, main_file, main_folder, dump_dir, params, task = self._key_params()
    challenge_cmd = 'antgo challenge'
    if token is not None:
      challenge_cmd += ' --token=%s'%token

    if name is not None:
      challenge_cmd += ' --name=%s'%name

    if main_file is not None:
      challenge_cmd += ' --main_file=%s'%main_file

    if main_folder is not None:
      challenge_cmd += ' --main_folder=%s'%main_folder

    if params is not None:
      challenge_cmd += ' --main_param=%s'%params

    if task is not None:
      challenge_cmd += ' --task=%s' % task

    if dump_dir is not None:
      challenge_cmd += ' --dump=%s'%dump_dir

    challenge_cmd += ' > %s-challenge.log 2>&1 &' % name

    # launch background process
    subprocess.call("nohup %s"%challenge_cmd, shell=True)

  def process_train_command(self):
    token, name, main_file, main_folder, dump_dir, params, task = self._key_params()
    train_cmd = 'antgo train'
    if token is not None:
      train_cmd += ' --token=%s' % token

    if name is not None:
      train_cmd += ' --name=%s' % name

    if main_file is not None:
      train_cmd += ' --main_file=%s' % main_file

    if main_folder is not None:
      train_cmd += ' --main_folder=%s' % main_folder

    if params is not None:
      train_cmd += ' --main_param=%s' % params

    if task is not None:
      train_cmd += ' --task=%s' % task

    if dump_dir is not None:
      train_cmd += ' --dump=%s' % dump_dir

    train_cmd += ' > %s-train.log 2>&1 &' % name

    # launch background process
    subprocess.call("nohup %s" % train_cmd, shell=True)

  def process_cmd(self, command):
    try:
      if command == 'task':
        self.process_task_command()
      elif command == 'experiment':
        self.process_experiment_command()
      elif command == 'dataset':
        self.process_dataset_command()
      elif command == 'apply':
        self.process_apply_command()
      elif command == 'create':
        self.process_create_command()
      elif command == 'add':
        self.process_add_command()
      elif command == 'del':
        self.process_del_command()
      elif command == 'update':
        self.process_update_command()
      elif command == 'upload':
        self.process_upload_command()
      elif command == 'challenge':
        self.process_challenge_command()
      elif command == 'train':
        self.process_train_command()
    except:
      logger.error('error response from server')

  def start(self):
    cmd = ''
    if PY3:
      cmd = input('antgo > ')
    else:
      cmd = raw_input('antgo > ')

    while cmd != 'quit':
      try:
        command = cmd.split(' ')
        assert(command[0] in ['task',
                              'dataset',
                              'experiment',
                              'apply',
                              'create',
                              'del',
                              'add',
                              'update',
                              'upload',
                              'challenge',
                              'train'])

        flags.cli_param_flags(command[1:])

        # process user command
        self.process_cmd(command[0])

        # clear flags
        flags.clear_cli_param_flags()
      except:
        logger.error('error antgo command\n')

      if PY3:
        cmd = input('antgo > ')
      else:
        cmd = raw_input('antgo > ')