# -*- coding: UTF-8 -*-
# @Time    : 18-9-22
# @File    : batch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from antgo.dataflow.dataset.spider_dataset import SpiderDataset
from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.task.task import *
from antgo.dataflow.common import *
from antgo.dataflow.basic import *
from antgo.dataflow.recorder import *
from antgo.crowdsource.batch_server import *
from antvis.client.httprpc import *
import json
import traceback
from antgo.dataflow.dataset.proxy_dataset import *
import threading


class AntBatch(AntBase):
    def __init__(self,
                 ant_context,
                 ant_name,
                 token,
                 data_factory,
                 ant_dump_dir,
                 ant_task_config,
                 dataset,
                 **kwargs):
        super(AntBatch, self).__init__(ant_name, ant_context, token)

        self.ant_data_source = data_factory
        self.ant_dump_dir = ant_dump_dir
        self.ant_context.ant = self
        self.unlabel = kwargs.get('unlabel', False)
        self.ant_task_config = ant_task_config
        self.restore_experiment = kwargs.get('restore_experiment', None)
        self.host_ip = self.context.params.system.get('ip', '127.0.0.1')
        self.host_port = self.context.params.system.get('port', -1)
        self.dataset = dataset
        self.rpc = None
        self._running_dataset = None
        self._running_task = None

    def ping_until_ok(self):
        while True:
            content = self.rpc.ping.get()
            if content['status'] != 'ERROR':
                break
            # 暂停5秒钟，再进行尝试
            time.sleep(5)

    @property
    def running_dataset(self):
        return self._running_dataset

    @property
    def running_task(self):
        return self._running_task

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

        # 2.step 获得实验ID
        experiment_uuid = self.context.experiment_uuid
        if self.restore_experiment is not None:
            experiment_uuid = self.restore_experiment

        # 3.step make experiment folder
        logger.info('Build experiment folder.')
        if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
            os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))
        experiment_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid)

        # 获得是否加载web server标记
        is_launch_web_server = True
        if self.host_port < 0:
            is_launch_web_server = False

        # 4.step 设置记录器
        # 设置记录器
        # 数据标签
        batch_params = getattr(self.context.params, 'predict', None)
        tags = []
        if batch_params is not None:
            tags = batch_params.get('tags', [])

        def _callback_func(dataset_stage, data):
            record_content = {
                'experiment_uuid': experiment_uuid,
                'dataset': {
                    dataset_stage: [{'data': single_d, 'tag': []} for single_d in data]
                },
                'tags': tags
            }
            for data_group in record_content['dataset'][dataset_stage]:
                for item in data_group['data']:
                    if item['type'] in ['IMAGE', 'VIDEO', 'FILE']:
                        item['data'] = 'static/data/%s/record/%s' % (dataset_stage, item['data'])

            # 添加当前进度
            record_content['waiting'] = \
                self.running_dataset.waiting_process_num() if self.running_dataset is not None else \
                    self.context.params.predict.get('size', 0)
            record_content['finished'] = \
                self.running_dataset.finish_process_num() if self.running_dataset is not None else 0
            self.rpc.predict.config.post(config_data=json.dumps(record_content))

            # 保存执行信息
            # 每执行完一次预测并记录，会将结果同时记录到json中
            if not os.path.exists(os.path.join(experiment_dump_dir, '%s.json' % experiment_uuid)):
                with open(os.path.join(experiment_dump_dir, '%s.json' % experiment_uuid), 'w') as fp:
                    json.dump(record_content, fp)
            else:
                history_running_info = {}
                with open(os.path.join(experiment_dump_dir, '%s.json' % experiment_uuid), 'r') as fp:
                    history_running_info = json.load(fp)
                history_running_info['dataset'][dataset_stage].extend(record_content['dataset'][dataset_stage])

                with open(os.path.join(experiment_dump_dir, '%s.json' % experiment_uuid), 'w') as fp:
                    json.dump(history_running_info, fp)

        output_dir = experiment_dump_dir
        if is_launch_web_server:
            output_dir = \
                os.path.join(experiment_dump_dir, 'predict', 'static', 'data', 'test')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        self.context.recorder = \
            LocalRecorderNodeV2(lambda data: _callback_func('test', data) if is_launch_web_server else None)
        self.context.recorder.dump_dir = os.path.join(output_dir, 'record')
        if not os.path.exists(self.context.recorder.dump_dir):
            os.makedirs(self.context.recorder.dump_dir)

        # 4.step 配置web服务基本信息
        if is_launch_web_server:
            # 选择端口
            self.host_port = _pick_idle_port(self.host_port)

            # 配置调用
            self.rpc = HttpRpc("v1", "antgo/api", self.host_ip, self.host_port)

            # 准备素材资源
            static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
            batch_static_dir = os.path.join(experiment_dump_dir, 'predict', 'static')
            if os.path.exists(batch_static_dir):
                shutil.rmtree(batch_static_dir)

            shutil.copytree(os.path.join(static_folder, 'resource', 'app'), batch_static_dir)
            # 启动服务器
            white_users = self.context.params.predict.get('white_users', None)
            p = \
                multiprocessing.Process(
                    target=batch_server_start,
                    args=(
                        experiment_dump_dir,
                        self.host_port,
                        None,
                        white_users
                    ))
            p.start()

            # 等待web服务启动
            self.ping_until_ok()

        if self.context.is_interact_mode:
          logger.info('Running on interact mode.')
          return

        # 5.step 配置运行
        def _run_batch_process():
            # 是否来自已有实验
            if self.restore_experiment is not None and \
                os.path.exists(
                    os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment)):
                # 加载保存的执行信息
                running_info = {}
                with open(os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment), 'r') as fp:
                    running_info = json.load(fp)

                # 更新web页面
                if is_launch_web_server:
                    self.rpc.predict.config.post(config_data=json.dumps(running_info))
                return

            # 4.step load dataset
            logger.info('Load task dataset and split.')
            
            dataset_name = ''
            selected_dataset_stages = []
            if len(running_ant_task.dataset_name.split('/')) == 2:
                dataset_name, dataset_stage = \
                    running_ant_task.dataset_name.split('/')
                selected_dataset_stages.append(dataset_stage)
            else:
                dataset_name = running_ant_task.dataset_name
                selected_dataset_stages = ['test']

            if self.dataset is not None and self.dataset != '':
                logger.info('')
                if len(self.dataset.split('/')) == 2:
                    dataset_name, dataset_stage = self.dataset.split('/')
                    selected_dataset_stages.append(dataset_stage)
                else:
                    dataset_name = self.dataset
                    selected_dataset_stages = ['test']

            logger.info('Using dataset %s/%s'%(dataset_name, selected_dataset_stages[0]))

            # 5.step prepare ablation blocks
            logger.info('Prepare model ablation blocks.')
            ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
            if ablation_blocks is None:
                ablation_blocks = []
            for b in ablation_blocks:
                self.ant_context.deactivate_block(b)

            # 6.step infer
            logger.info('Running inference process.')
            for dataset_stage in selected_dataset_stages:
                logger.info('Process dataset: %s.' % dataset_stage)
                # 获得数据集对象
                ant_test_dataset = None
                if dataset_name.startswith('spider'):
                    # 使用SpiderDataset
                    ant_test_dataset = \
                        SpiderDataset(self.command_queue,
                                        os.path.join(self.ant_data_source, dataset_name), None)
                else:
                    if self.context.register_at(dataset_stage) is not None:
                        ant_test_dataset = ProxyDataset(dataset_stage)
                        ant_test_dataset.register(**{dataset_stage:self.context.register_at(dataset_stage)})
                    else:
                        ant_test_dataset = \
                            running_ant_task.dataset(dataset_stage,
                                                        os.path.join(self.ant_data_source, dataset_name),
                                                        running_ant_task.dataset_params)

                # using unlabel
                if self.unlabel:
                    if 'candidate_file' in self.context.params.system['ext_params']:
                        logger.info('Using candidate file %s'%self.context.params.system['ext_params']['candidate_file'])
                        ant_test_dataset.candidate_file = self.context.params.system['ext_params']['candidate_file']
                    if 'unlabeled_list_file' in self.context.params.system['ext_params']:
                        logger.info('Using unlabeled list file %s'%self.context.params.system['ext_params']['unlabeled_list_file'])
                        ant_test_dataset.unlabeled_list_file = self.context.params.system['ext_params']['unlabeled_list_file']
                    ant_test_dataset = UnlabeledDataset(ant_test_dataset)

                # 重新设置输出目录
                output_dir = experiment_dump_dir
                if is_launch_web_server:
                    output_dir = \
                        os.path.join(experiment_dump_dir, 'predict', 'static', 'data', dataset_stage)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                # 重新设置记录器
                self.context.recorder = \
                    LocalRecorderNodeV2(lambda data: _callback_func(dataset_stage, data) if is_launch_web_server else None)

                # 调用预测的回调函数
                self._running_dataset = ant_test_dataset
                with safe_recorder_manager(self.context.recorder):
                    # 完成推断过程
                    try:
                        self.context.call_infer_process(ant_test_dataset, dump_dir=output_dir)
                    except Exception as e:
                        if type(e.__cause__) != StopIteration:
                            print(e)
                            traceback.print_exc()

                self.context.recorder.close()
            return

        # 5.step 启动bach server
        _run_batch_process()