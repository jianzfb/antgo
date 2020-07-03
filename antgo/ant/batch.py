# -*- coding: UTF-8 -*-
# @Time    : 18-9-22
# @File    : batch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from antgo.ant.base import *
from antgo.task.task import *
from antgo.dataflow.common import *
from antgo.dataflow.basic import *
from antgo.dataflow.recorder import *


class AntBatch(AntBase):
    def __init__(self,
                 ant_context,
                 ant_name,
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

    def start(self):
        # 1.step 加载挑战任务
        running_ant_task = None
        if self.token is not None:
            # 1.1.step 从平台获取挑战任务配置信息
            response = self.context.dashboard.challenge.get(command=type(self).__name__)
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

        # 2.step 注册实验
        experiment_uuid = self.context.experiment_uuid

        # 3.step make experiment folder
        logger.info('build experiment folder')
        if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
            os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))

        experiment_dump_dir = os.path.join(self.ant_dump_dir, experiment_uuid)
        if not os.path.exists(experiment_dump_dir):
            os.makedirs(experiment_dump_dir)
        else:
            shutil.rmtree(experiment_dump_dir)
            os.makedirs(experiment_dump_dir)

        # 4.step load dataset and split
        logger.info('load task dataset and split')

        # 独立处理验证集和测试集

        dataset_name = ''
        dataset_stage = 'test'
        if len(running_ant_task.dataset_name.split('/')) == 2:
            dataset_name, dataset_stage = running_ant_task.dataset_name.split('/')
        else:
            dataset_name = running_ant_task.dataset_name

        ant_test_dataset = running_ant_task.dataset(dataset_stage,
                                                    os.path.join(self.ant_data_source, dataset_name),
                                                    running_ant_task.dataset_params)
        # split data and label
        if self.unlabel:
            ant_test_dataset = UnlabeledDataset(ant_test_dataset)

        data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))

        # 5.step prepare ablation blocks
        logger.info('prepare model ablation blocks')
        ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
        if ablation_blocks is None:
            ablation_blocks = []
        for b in ablation_blocks:
            self.ant_context.deactivate_block(b)

        # 6.step infer
        logger.info('running inference process')
        # intermediate_dump_dir = os.path.join(self.ant_dump_dir, now_time_stamp, 'record')
        # if not os.path.exists(intermediate_dump_dir):
        #     os.makedirs(intermediate_dump_dir)

        with safe_recorder_manager(self.context.recorder):
            # self.context.recorder.dump_dir = intermediate_dump_dir
            if self.unlabel:
                self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
            else:
                self.context.recorder = LocalRecorderNode(Node.inputs(data_annotation_branch.output(1)))

            # infer process
            self.context.call_infer_process(data_annotation_branch.output(0), dump_dir=experiment_dump_dir)
