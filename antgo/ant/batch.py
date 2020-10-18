# -*- coding: UTF-8 -*-
# @Time    : 18-9-22
# @File    : batch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.task.task import *
from antgo.dataflow.common import *
from antgo.dataflow.basic import *
from antgo.dataflow.recorder import *
from antgo.crowdsource.batch_server import *
from antvis.client.httprpc import *
import json


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
        self.web_server_process = None

    def update_web(self, content):
        self.rpc.config.post(config_data=json.dumps(content))

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
                logger.error('couldnt load challenge task')
                self.token = None
            elif response['status'] == 'SUSPEND':
                # prohibit submit challenge task frequently
                # submit only one in one week
                logger.error('prohibit submit challenge task frequently')
                exit(-1)
            elif response['status'] == 'OK':
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
            # 1.2.step 加载自定义任务配置信息
            custom_task = create_task_from_xml(self.ant_task_config, self.context)
            if custom_task is None:
                logger.error('couldnt load custom task')
                exit(-1)
            running_ant_task = custom_task

        assert (running_ant_task is not None)

        # 2.step 获得实验ID
        experiment_uuid = self.context.experiment_uuid
        if self.restore_experiment is not None:
            experiment_uuid = self.restore_experiment

        # 3.step make experiment folder
        logger.info('build experiment folder')
        if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
            os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))
        experiment_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid)

        # 4.step 配置web服务基本信息
        # 选择端口
        self.host_port = _pick_idle_port(self.host_port)

        # 配置调用
        self.rpc = HttpRpc("v1", "batch-api", "127.0.0.1", self.host_port)

        # 准备素材资源
        static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
        batch_static_dir = os.path.join(experiment_dump_dir, 'batch')
        if not os.path.exists(batch_static_dir):
            shutil.copytree(os.path.join(static_folder, 'resource', 'batch'), batch_static_dir)

        # 5.step 配置运行
        def _run_batch_process():
            # 等待web服务启动
            self.ping_until_ok()

            # 是否来自已有实验
            if self.restore_experiment is not None and \
                os.path.exists(os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment)):
                # 加载保存的执行信息
                running_info = {}
                with open(os.path.join(experiment_dump_dir, '%s.json' % self.restore_experiment), 'r') as fp:
                    running_info = json.load(fp)

                # 更新web页面
                self.update_web(running_info)
                return

            # 4.step load dataset
            logger.info('load task dataset and split')

            dataset_name = ''
            selected_dataset_stages = []
            if len(running_ant_task.dataset_name.split('/')) == 2:
                dataset_name, dataset_stage = running_ant_task.dataset_name.split('/')
                selected_dataset_stages.append(dataset_stage)
            else:
                dataset_name = running_ant_task.dataset_name
                selected_dataset_stages = ['test']

            # 5.step prepare ablation blocks
            logger.info('prepare model ablation blocks')
            ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
            if ablation_blocks is None:
                ablation_blocks = []
            for b in ablation_blocks:
                self.ant_context.deactivate_block(b)

            # 6.step infer
            logger.info('running inference process')

            running_info = {'experiment_uuid': experiment_uuid,
                            'dataset': {}}
            for dataset_stage in selected_dataset_stages:
                logger.info('process dataset: %s' % dataset_stage)
                # 获得数据集对象
                ant_test_dataset = running_ant_task.dataset(dataset_stage,
                                                            os.path.join(self.ant_data_source, dataset_name),
                                                            running_ant_task.dataset_params)
                # split data and label
                if self.unlabel:
                    ant_test_dataset = UnlabeledDataset(ant_test_dataset)

                data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))

                output_dir = os.path.join(experiment_dump_dir, 'batch', 'static', 'data', dataset_stage)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 设置记录器
                self.context.recorder = LocalRecorderNodeV2(Node.inputs(data_annotation_branch.output(1)))
                with safe_recorder_manager(self.context.recorder):
                    # 完成推断过程
                    self.context.call_infer_process(data_annotation_branch.output(0), dump_dir=output_dir)

                    # 获得记录信息
                    running_info['dataset'][dataset_stage] = self.context.recorder.content
                    for data_group in running_info['dataset'][dataset_stage]:
                        for item in data_group:
                            item['data'] = 'static/data/%s/record/%s' % (dataset_stage, item['data'])

                    # 清空记录器信息
                    self.context.recorder.clear()

            # 7.step 保存执行信息
            with open(os.path.join(experiment_dump_dir, '%s.json' % experiment_uuid), 'w') as fp:
                json.dump(running_info, fp)

            # 8.step 更新web页面
            self.update_web(running_info)
            return

        process = threading.Thread(target=_run_batch_process)
        process.daemon = True
        process.start()

        # 5.step 启动bach server
        batch_server_start(experiment_dump_dir, self.host_port)
