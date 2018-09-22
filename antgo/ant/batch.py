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

class AntBatch(AntBase):
    def __init__(self,
                 ant_context,
                 ant_name,
                 data_factory,
                 ant_dump_dir,
                 ant_task_config):
        super(AntBatch, self).__init__(ant_name, ant_context, None)

        assert(ant_task_config is not None)

        self.ant_data_source = data_factory
        self.ant_dump_dir = ant_dump_dir
        self.ant_context.ant = self
        self.running_ant_task = create_task_from_xml(ant_task_config, self.context)
        if self.running_ant_task is None:
            logger.error('couldnt load task')
            exit(-1)

    def start(self):
        # 1.step prepare datasource and infer recorder
        now_time_stamp = datetime.fromtimestamp(self.time_stamp).strftime('%Y%m%d.%H%M%S.%f')

        # 2.step make experiment folder
        logger.info('build experiment folder')
        if not os.path.exists(os.path.join(self.ant_dump_dir, now_time_stamp)):
            os.makedirs(os.path.join(self.ant_dump_dir, now_time_stamp))

        infer_dump_dir = os.path.join(self.ant_dump_dir, now_time_stamp, 'inference')
        if not os.path.exists(infer_dump_dir):
            os.makedirs(infer_dump_dir)
        else:
            shutil.rmtree(infer_dump_dir)
            os.makedirs(infer_dump_dir)

        # 3.step load dataset and split
        logger.info('load task dataset and split')
        dataset_name = ''
        dataset_stage = 'test'
        if len(self.running_ant_task.dataset_name.split('/')) == 2:
            dataset_name, dataset_stage = self.running_ant_task.dataset_name.split('/')

        ant_test_dataset = self.running_ant_task.dataset(dataset_stage,
                                                         os.path.join(self.ant_data_source, dataset_name),
                                                         self.running_ant_task.dataset_params)
        # split data and label
        data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))

        # 4.step prepare ablation blocks
        logger.info('prepare model ablation blocks')
        ablation_blocks = getattr(self.ant_context.params, 'ablation', [])
        for b in ablation_blocks:
            self.ant_context.deactivate_block(b)

        # 5.step infer
        logger.info('running inference process')
        self.context.call_infer_process(data_annotation_branch.output(0), dump_dir=infer_dump_dir)
