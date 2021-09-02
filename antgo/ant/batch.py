# -*- coding: UTF-8 -*-
# @Time    : 18-9-22
# @File    : batch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cv2 import log
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


class AntBatch(AntBase):
    def __init__(self,
                 ant_context,
                 ant_name,
                 ant_host_ip,
                 ant_host_port,
                 token,
                 data_factory,
                 ant_dump_dir,
                 ant_task_config, **kwargs):
        super(AntBatch, self).__init__(ant_name, ant_context, token)

        self.ant_data_source = data_factory
        self.ant_dump_dir = ant_dump_dir
        self.ant_context.ant = self
        self.unlabel = kwargs.get('unlabel', False)
        self.ant_task_config = ant_task_config
        self.context.devices = [int(d) for d in kwargs.get('devices', '').split(',') if d != '']
        self.restore_experiment = kwargs.get('restore_experiment', None)
        self.host_ip = ant_host_ip
        self.host_port = ant_host_port
        self.rpc = None
        self.command_queue = None

    def ping_until_ok(self):
        while True:
            content = self.rpc.ping.get()
            if content['status'] != 'ERROR':
                break
            # 暂停5秒钟，再进行尝试
            time.sleep(5)

    def start(self):
        # 1.step 加载挑战任务
        running_ant_task = None
        if self.token is not None:
            # 1.1.step 从平台获取挑战任务配置信息
            response = mlogger.getEnv().dashboard.challenge.get(command=type(self).__name__)
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

        # 2.step 获得实验ID
        experiment_uuid = self.context.experiment_uuid
        if self.restore_experiment is not None:
            experiment_uuid = self.restore_experiment

        # 3.step make experiment folder
        logger.info('Build experiment folder.')
        if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
            os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))
        experiment_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid)

        # 4.step 配置web服务基本信息
        is_launch_web_server = True
        if self.host_port < 0:
            is_launch_web_server = False

        if is_launch_web_server:
            # 选择端口
            self.host_port = _pick_idle_port(self.host_port)

            # 配置调用
            self.rpc = HttpRpc("v1", "batch-api", "127.0.0.1", self.host_port)

            # 准备素材资源
            static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
            batch_static_dir = os.path.join(experiment_dump_dir, 'batch')
            if not os.path.exists(batch_static_dir):
                shutil.copytree(os.path.join(static_folder, 'resource', 'batch'), batch_static_dir)

            # 命令队列
            self.command_queue = queue.Queue()

        # 数据标签
        batch_params = getattr(self.context.params, 'predict', None)
        tags = []
        if batch_params is not None:
            tags = batch_params.get('tags', [])

        # 5.step 配置运行
        def _run_batch_process():
            # 等待web服务启动
            if is_launch_web_server:
                self.ping_until_ok()

            # 是否来自已有实验
            if self.restore_experiment is not None and \
                os.path.exists(os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment)):
                # 加载保存的执行信息
                running_info = {}
                with open(os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment), 'r') as fp:
                    running_info = json.load(fp)

                # 更新web页面
                if is_launch_web_server:
                    self.rpc.config.post(config_data=json.dumps(running_info))
                return

            # 4.step load dataset
            logger.info('Load task dataset and split.')
            
            dataset_name = ''
            selected_dataset_stages = []
            if len(running_ant_task.dataset_name.split('/')) == 2:
                dataset_name, dataset_stage = running_ant_task.dataset_name.split('/')
                selected_dataset_stages.append(dataset_stage)
            else:
                dataset_name = running_ant_task.dataset_name
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
                    ant_test_dataset = \
                        running_ant_task.dataset(dataset_stage,
                                                    os.path.join(self.ant_data_source, dataset_name),
                                                    running_ant_task.dataset_params)

                # using unlabel
                if self.unlabel:
                    ant_test_dataset = UnlabeledDataset(ant_test_dataset)

                output_dir = experiment_dump_dir
                if is_launch_web_server:
                    output_dir = os.path.join(experiment_dump_dir, 'batch', 'static', 'data', dataset_stage)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                # 设置记录器
                def _callback_func(data):
                    record_content = {
                        'experiment_uuid': experiment_uuid, 
                        'dataset': {
                            dataset_stage: [{'data':single_d, 'tag':[]} for single_d in data]
                        },
                        'tags': tags
                    }
                    for data_group in record_content['dataset'][dataset_stage]:
                        for item in data_group['data']:
                            if item['type'] in ['IMAGE','VIDEO','FILE']:
                                item['data'] = 'static/data/%s/record/%s' % (dataset_stage, item['data'])

                    # 添加当前进度
                    record_content['waiting'] = ant_test_dataset.waiting_process_num()
                    record_content['finished'] = ant_test_dataset.finish_process_num()
                    self.rpc.config.post(config_data=json.dumps(record_content))
                    
                    # 保存执行信息
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

                self.context.recorder =  LocalRecorderNodeV2(_callback_func if is_launch_web_server else None)
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
        if is_launch_web_server:
            # 在独立线程中启动预测
            process = threading.Thread(target=_run_batch_process)
            process.daemon = True
            process.start()
            
            # 主线程中启动web服务
            batch_server_start(experiment_dump_dir, self.host_port, self.command_queue)
        else:
            # 启动预测
            _run_batch_process()