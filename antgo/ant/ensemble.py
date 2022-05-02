# -*- coding: UTF-8 -*-
# @Time    : 2019/1/22 1:16 PM
# @File    : ensemble.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.resource.html import *
from antgo.ant.base import *
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antvis.client.httprpc import *
from antgo.task.task import *
import traceback
import subprocess
import os
import socket
import requests
import json
import zipfile
import multiprocessing
from antgo.crowdsource.ensemble_server import *
from antgo.dataflow.dataset.proxy_dataset import *
import requests
from io import BytesIO
import numpy as np
import msgpack
import msgpack_numpy as ms


class EnsembleMergeRecorder(object):
    def __init__(self, role='master',
                 ip='', port='',
                 experiment_uuid='',
                 dump_dir='',
                 ensemble_method='',
                 dataset_name='',
                 dataset_flag='train',
                 dataset_num=0,
                 model_weight=1.0,
                 prefix='',
                 feedback=True):
        self.role = role        # master, worker
        self._dump_dir = ''
        self.url = 'http://%s:%s'%(ip, (str)(port))
        self.experiment_uuid = experiment_uuid
        self.record_variable_name = []
        self.record_sample_list = {}
        self.dataset_name = dataset_name
        self.dataset_num = dataset_num
        self.dataset_flag = dataset_flag
        self.ensemble_method = ensemble_method
        self.dump_dir = dump_dir
        self.model_weight = model_weight
        self.feedback = feedback
        self.prefix = prefix
        self.worker_number = f'{uuid.uuid4()}'

    def avg(self, data):
        assert('id' in data)
        id = data['id']['data']
        data.pop('id')

        result = {}
        for k, v in data.items():
            if type(v['data']) != np.ndarray:
                continue

            proxies = {
                "http": None,
                'https': None
            }

            # 将数据以文件流模式上传
            data = msgpack.packb(v['data'], default=ms.encode)
            response = \
                requests.post('%s/ensemble-api/avg/'%(self.url),
                              headers={'file_id': f'{id}',
                                       'worker_number': self.worker_number,
                                       'worker_prefix': f'{self.prefix}',
                                       'id': f'{id}',
                                       'name': f'{k}',
                                       'weight': f"{self.model_weight}",
                                       'feedback': f"{self.feedback}"},
                              data=data,
                              proxies=proxies, stream=True)

            if response.status_code != 200:
                continue

            result[k] = {
                'data': None
            }
            if self.feedback:
                with BytesIO() as pdf:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            pdf.write(chunk)
                    data = msgpack.unpackb(pdf.getvalue(), object_hook=ms.decode)

                result[k] = {
                    'data': data,
                }

        return result

    def record(self, data):
        if self.role == 'master':
            assert('id' in data)
            id = str(data['id']['data'])
            data.pop('id')

            # 记录样本ID
            if id not in self.record_sample_list:
                self.record_sample_list[id] = {}

            # 记录样本记录的数据
            for k in data.keys():
                if k not in self.record_variable_name:
                    self.record_variable_name.append(k)
                if k not in self.record_sample_list[id]:
                    self.record_sample_list[id][k] = 0

                self.record_sample_list[id][k] += 1

            # 记录样本
            for k, v in data.items():
                name = k
                value = v['data']
                if type(value) != np.ndarray:
                    logger.warn('Record %s only support numpy array.'%name)
                    continue

                # 本地保存数据
                key = '%s/%s/%s.data'%(self.experiment_uuid, id, name)
                if not os.path.exists(os.path.join(self.dump_dir, '%s/%s'%(self.experiment_uuid, id))):
                    os.makedirs(os.path.join(self.dump_dir, '%s/%s'%(self.experiment_uuid, id)))

                with open(os.path.join(self.dump_dir, key), 'wb') as fp:
                    fp.write(msgpack.packb(value, default=ms.encode))

    def close(self):
        if self.experiment_uuid == '':
            logger.error('No experiment uuid, couldnt upload result.')
            return

        # 提交完成 ensemble 模型创建
        # 变量列表，样本数，数据类型(train,val,test)，方法
        record_sample_num = len(self.record_sample_list)
        dataset_sample_num = self.dataset_num
        # 检查记录的样本数是否一致
        if record_sample_num != dataset_sample_num:
            logger.error("Record sample number not equal to dataset sample number")
            return
        # 检查样本的变量是否一致
        is_ok = True
        for index in range(record_sample_num):
            if len(self.record_sample_list[str(index)]) != len(self.record_variable_name):
                is_ok = False
                break
        if not is_ok:
            logger.error("Record sample missing variables.")
            return

        # 上传数据（COS）
        logger.info('Upload ensemble model data.')
        finish_num = 0
        for key, key_content in self.record_sample_list.items():
            sample_id = key
            for sample_attr in key_content.keys():
                key = '%s/%s/%s.data' % (self.experiment_uuid, sample_id, sample_attr)
                sample_attr_file = os.path.join(self.dump_dir, key)
                mlogger.file.upload('ensemble/variable', key, sample_attr_file, is_file=True)
            finish_num += 1
            if finish_num % 100 == 0:
                logger.info('Finish %d/%d'%(finish_num, record_sample_num))

        # 记录信息（MLTALKER）
        logger.info('Update ensemble experiment record.')
        info = {
            'record_sample_num': record_sample_num,
            'dataset_num': dataset_sample_num,
            'dataset_name': self.dataset_name,
            'dataset_flag': self.dataset_flag,
            'ensemble_method': self.ensemble_method,
            'record_variable_name': self.record_variable_name
        }

        response = mlogger.info.ensemble.merge.post(info=json.dumps(info))
        if response['status'] == 'OK':
            logger.info('Finish ensemble model submit.')
        else:
            logger.error('Fail ensemble model submit')

    @property
    def dump_dir(self):
        return self._dump_dir

    @dump_dir.setter
    def dump_dir(self, val):
        self._dump_dir = val


class EnsembleReleaseRecorder(object):
    def __init__(self, ensemble_uuid):
        # 聚合名字
        self.ensemble_uuid = ensemble_uuid

        # 获得实验列表
        response = mlogger.info.ensemble.release.get(ensemble_uuid=ensemble_uuid)
        content = response['content']
        # [{'experiment_uuid': '', 'experiment_weight': 1.0, 'experiment_user'},{}]
        self.ensemble_models = self.parse_ensemble_model(content)

        self._dump_dir = ''

    def recursive_parse(self, root, root_weight, root_developer, data):
        if 'experiment_uuid' in root:
            for model, weight, developer in zip(root['experiment_uuid'], root['experiment_weight'], root['experiment_user']):
                data.append({
                    'experiment_uuid': model,
                    'experiment_weight': weight * root_weight,
                    'experiment_user': developer
                })
            return

        for model, weight, developer in zip(root['ensemble_uuid'], root['ensemble_weight'], root['ensemble_user']):
            self.recursive_parse(model, weight, developer, data)

    def parse_ensemble_model(self, content):
        # {'experiment_uuid': [], 'experiment_weight': []}
        # {'ensemble_uuid': [{'ensemble_uuid':[], 'ensemble_weight': []}], 'ensemble_weight': [1.0]}
        if 'experiment_uuid' in content:
            data = []
            for model, weight, developer in zip(content['experiment_uuid'],
                                                content['experiment_weight'],
                                                content['experiment_user']):
                data.append({
                    'experiment_uuid': model,
                    'experiment_weight': weight,
                    'experiment_user': developer
                })

            return data

        data = []
        for model, weight, developer in zip(content['ensemble_uuid'], content['ensemble_weight'], content['ensemble_user']):
            self.recursive_parse(model, weight, developer, data)
        return data

    def get(self, **kwargs):
        assert ('id' in kwargs)
        id = str(kwargs['id'])
        kwargs.pop('id')

        result = {}
        for k, v in kwargs.items():
            name = k

            merge_data = None
            for ensemble_mode in self.ensemble_models:
                experiment_uuid = ensemble_mode['experiment_uuid']
                model_developer = ensemble_mode['experiment_user']
                model_weight = ensemble_mode['experiment_weight']

                key = '%s/%s/%s.data' % (experiment_uuid, id, name)
                data = mlogger.file.download('ensemble/variable', key, model_developer)
                if data is None:
                    continue
                data = msgpack.unpackb(data, object_hook=ms.decode)

                # data_collect.append({
                #     'data': data,
                #     'weight': model_weight
                # })
                if merge_data is None:
                    merge_data = data * model_weight
                else:
                    merge_data = merge_data + data * model_weight

            result[name] = merge_data

        return result

    def close(self):
        pass

    @property
    def dump_dir(self):
        return self._dump_dir

    @dump_dir.setter
    def dump_dir(self, val):
        self._dump_dir = val


class AntEnsemble(AntBase):
    def __init__(self,
                 ant_context,
                 ant_name,
                 ant_data_folder,
                 ant_dump_dir,
                 ant_token,
                 ant_task_config,
                 dataset,
                 ensemble_stage, **kwargs):
        super().__init__(ant_name, ant_context=ant_context, ant_token=ant_token, **kwargs)
        # master, slave
        # bagging, stacking
        self.ensemble_stage = ensemble_stage            # train, merge, release
        assert(self.ensemble_stage in ['train', 'merge', 'release'])

        self.mode = self.context.params.ensemble['mode']       # online/offline
        self.method = self.context.params.ensemble['method']   # bagging,stacking,blending,bosting
        self.ant_dump_dir = ant_dump_dir

        self.ant_task_config = ant_task_config
        self.ant_data_source = ant_data_folder
        self.dataset = dataset
        self.model_weight = 1.0

    def start(self):
        # 1.step 加载训练任务
        running_ant_task = None
        if self.token is not None:
            # 1.1.step load train task
            response = mlogger.info.challenge.get(command=type(self).__name__)
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

        # 启动服务器
        if self.context.params.ensemble['role'] == 'master':
            worker_num = (int)(self.context.params.ensemble['worker'])
            server_ip = self.context.params.system['ip']
            server_port = (int)(self.context.params.system['port'])
            p = multiprocessing.Process(target=ensemble_server_start,
                                        args=(self.ant_dump_dir,
                                              server_port,
                                              worker_num,
                                              self.context.params.ensemble.get('uncertain_vote', None),
                                              self.context.params.ensemble.get('enable_data_record', False)))
            p.daemon = True
            p.start()

        # 等待master服务开启
        while True:
            try:
                server_ip = self.context.params.system['ip']
                server_port = (int)(self.context.params.system['port'])
                proxies = {
                    "http": None,
                    'https': None
                }
                result = requests.get('http://%s:%d/ensemble-api/live/'%(server_ip, server_port), proxies=proxies)
                if result.status_code == 200:
                    break
            except:
                logger.info("Waiting ensemble server launch.")
                time.sleep(10)

        # 数据集
        dataset_flag = 'train'
        if self.ensemble_stage == 'train':
            dataset_flag = 'train'

        dataset_name = ''
        if self.dataset is not None and self.dataset != '':
            if len(self.dataset.split('/')) == 2:
                dataset_name, dataset_flag = self.dataset.split('/')
            else:
                dataset_name = self.dataset
                if self.ensemble_stage in ['merge', 'release']:
                    dataset_flag = 'test'
                else:
                    dataset_flag = 'train'
        elif 'dataset' in self.context.params.ensemble:
            if len(self.context.params.ensemble['dataset'].split('/')) == 2:
                dataset_name, dataset_flag = self.context.params.ensemble['dataset'].split('/')
            else:
                dataset_name = self.context.params.ensemble['dataset']
                if self.ensemble_stage in ['merge', 'release']:
                    dataset_flag = 'test'
                else:
                    dataset_flag = 'train'
        else:
            dataset_name = running_ant_task.dataset_name
            if len(dataset_name.split('/')) == 2:
                dataset_name = dataset_name.split('/')[0]

            if self.ensemble_stage in ['merge', 'release']:
                dataset_flag = 'test'
            else:
                dataset_flag = 'train'

        # 模型权重
        if 'weight' in self.context.params.ensemble:
            self.model_weight = self.context.params.ensemble['weight']

        logger.info('Using dataset %s/%s.'%(dataset_name, dataset_flag))

        ant_dataset = None
        if self.context.register_at(dataset_flag) is not None:
            ant_dataset = ProxyDataset(dataset_flag)
            kwargs = {dataset_flag: self.context.register_at(dataset_flag)}
            ant_dataset.register(**kwargs)
        else:
            ant_dataset = \
                running_ant_task.dataset(dataset_flag,
                                         os.path.join(self.ant_data_source, dataset_name),
                                         running_ant_task.dataset_params)

        self.stage = 'ENSEMBLE-%s' % self.ensemble_stage
        if self.ensemble_stage == 'merge':
            dump_dir = os.path.join(self.ant_dump_dir, 'ensemble', 'merge')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            logger.info('Start ensemble merge process.')
            self.context.recorder = \
                EnsembleMergeRecorder(role=self.context.params.ensemble['role'],
                                      ip=self.context.params.system['ip'],
                                      port=self.context.params.system['port'],
                                      experiment_uuid=self.experiment_uuid,
                                      dump_dir=dump_dir,
                                      ensemble_method=self.method,
                                      dataset_name=dataset_name,
                                      dataset_flag=dataset_flag,
                                      dataset_num=ant_dataset.size,
                                      prefix=f"{self.ant_name}-{self.context.params.ensemble.get('model_name', 'model')}",
                                      feedback=self.context.params.ensemble['feedback'])

            if self.context.is_interact_mode:
                return

            with safe_recorder_manager(self.context.recorder):
                try:
                    self.context.call_infer_process(ant_dataset, dump_dir)
                except Exception as e:
                    if type(e.__cause__) != StopIteration:
                        print(e)
                        traceback.print_exc()
        elif self.ensemble_stage == 'release':
            dump_dir = os.path.join(self.ant_dump_dir, 'ensemble', 'release')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            logger.info('Start ensemble release process.')
            ensemble_uuid = self.context.params.ensemble['uuid'] if 'uuid' in self.context.params.ensemble else ''
            self.context.recorder = EnsembleReleaseRecorder(ensemble_uuid)

            if self.context.is_interact_mode:
                return

            with safe_recorder_manager(self.context.recorder):
                try:
                    self.context.call_infer_process(ant_dataset, dump_dir)
                except Exception as e:
                    if type(e.__cause__) != StopIteration:
                        print(e)
                        traceback.print_exc()
        else:
            # 数据重采样
            dump_dir = os.path.join(self.ant_dump_dir, 'ensemble', 'train')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            logger.info('Start ensemble train process.')
            if self.context.is_interact_mode:
                return

            try:
                self.context.call_training_process(ant_dataset, dump_dir)
            except Exception as e:
                if type(e.__cause__) != StopIteration:
                    print(e)
                    traceback.print_exc()