# encoding=utf-8
# @Time    : 17-6-22
# @File    : train.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

from antgo.html.html import *
from .base import *
from ..dataflow.common import *
from ..measures.statistic import *
from ..task.task import *
from ..utils import logger
from ..dataflow.recorder import *


class AntRun(AntBase):
    def __init__(self, ant_context,
                 ant_name,
                 ant_data_folder,
                 ant_dump_dir,
                 ant_token,
                 ant_task_config):
        super(AntRun, self).__init__(ant_name, ant_context, ant_token)
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
                logger.error('couldnt load challenge task')
                exit(-1)
            elif challenge_task_config['status'] == 'OK':
                challenge_task = create_task_from_json(challenge_task_config)
                if challenge_task is None:
                    logger.error('couldnt load challenge task')
                    exit(-1)
                running_ant_task = challenge_task

        if running_ant_task is None:
            # 0.step load custom task
            if self.ant_task_config is not None:
                custom_task = create_task_from_xml(self.ant_task_config, self.context)
                if custom_task is None:
                    logger.error('couldnt load custom task')
                    exit(-1)
                running_ant_task = custom_task

        assert(running_ant_task is not None)

        # 1.step loading training dataset
        logger.info('loading train dataset %s'%running_ant_task.dataset_name)
        ant_train_dataset = running_ant_task.dataset('train',
                                                     os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                     running_ant_task.dataset_params)

        # 2.step model evaluation (optional)
        if running_ant_task.estimation_procedure is not None:
            self.stage = 'EVALUATION'
            logger.info('start model evaluation')

            estimation_procedure = running_ant_task.estimation_procedure
            estimation_procedure_params = running_ant_task.estimation_procedure_params
            evaluation_measures = running_ant_task.evaluation_measures

            evaluation_statistic = None
            if estimation_procedure == 'holdout':
                evaluation_statistic = self._holdout_validation(ant_train_dataset, evaluation_measures)
            elif estimation_procedure == "repeated-holdout":
                number_repeats = 10             # default value
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
                                                                         evaluation_measures)
            elif estimation_procedure == "bootstrap":
                bootstrap_counts = int(estimation_procedure_params.get('bootstrap_counts', 20))
                evaluation_statistic = self._bootstrap_validation(bootstrap_counts,
                                                                  ant_train_dataset,
                                                                  evaluation_measures)
            elif estimation_procedure == "kfold":
                kfolds = int(estimation_procedure_params.get('kfold', 5))
                evaluation_statistic = self._kfold_cross_validation(kfolds, ant_train_dataset, evaluation_measures)


            logger.info('generate model evaluation report')
            everything_to_html(evaluation_statistic, self.ant_dump_dir)

        # 3.step model training
        self.stage = "TRAINING"
        if not os.path.exists(os.path.join(self.ant_dump_dir, 'train')):
            os.makedirs(os.path.join(self.ant_dump_dir, 'train'))
        logger.info('start training process')
        self.context.call_training_process(ant_train_dataset, os.path.join(self.ant_dump_dir, 'train'))

    def _holdout_validation(self, train_dataset, evaluation_measures):
        # 1.step split train set and validation set
        part_train_dataset, part_validation_dataset = train_dataset.split(split_method='holdout')

        # dump_dir
        dump_dir = os.path.join(self.ant_dump_dir, 'holdout')
        # prepare dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        # 2.step training model
        self.stage = 'EVALUATION-HOLDOUTTRAIN'
        self.context.call_training_process(part_train_dataset, dump_dir)

        # 3.step evaluation measures
        # split data and label
        data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
        self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
        # infer_dump_dir = os.path.join(dump_dir, 'inference')
        # if not os.path.exists(infer_dump_dir):
        #     os.makedirs(infer_dump_dir)

        self.stage = 'EVALUATION-HOLDOUTEVALUATION'
        with running_statistic(self.ant_name):
            self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)
        self.context.recorder.close()

        task_running_statictic = get_running_statistic(self.ant_name)
        task_running_statictic = {self.ant_name: task_running_statictic}
        task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
        task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
            task_running_elapsed_time / float(part_validation_dataset.size)

        logger.info('start evaluation process')
        evaluation_measure_result = []

        record_reader = RecordReader(dump_dir)
        for measure in evaluation_measures:
            record_generator = record_reader.iterate_read('predict', 'groundtruth')
            result = measure.eva(record_generator, None)
            evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

        return task_running_statictic

    def _repeated_holdout_validation(self, repeats, train_dataset, split_ratio, is_stratified_sampling, evaluation_measures):
        repeated_running_statistic = []
        for repeat in range(repeats):
            # 1.step split train set and validation set
            part_train_dataset, part_validation_dataset = train_dataset.split(split_params={'ratio': split_ratio,
                                                                                            'is_stratified': is_stratified_sampling},
                                                                              split_method='repeated-holdout')
            # dump_dir
            dump_dir = os.path.join(self.ant_dump_dir, 'repeated-holdout-evaluation', 'repeat-%d'%repeat)

            # prepare dir
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            # 2.step training model
            self.stage = 'EVALUATION-REPEATEDHOLDOUTTRAIN-%d' % repeat
            self.context.call_training_process(part_train_dataset, dump_dir)
            if self.context.recorder:
                self.context.recorder.close()

            # 3.step evaluation measures
            # split data and label
            data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
            self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
            # infer_dump_dir = os.path.join(dump_dir, 'inference')
            # if not os.path.exists(infer_dump_dir):
            #     os.makedirs(infer_dump_dir)

            self.stage = 'EVALUATION-REPEATEDHOLDOUTEVALUATION-%d' % repeat
            with running_statistic(self.ant_name):
                self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

            if self.context.recorder:
                self.context.recorder.close()

            task_running_statictic = get_running_statistic(self.ant_name)
            task_running_statictic = {self.ant_name: task_running_statictic}
            task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
            task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
                task_running_elapsed_time / float(part_validation_dataset.size)

            logger.info('start evaluation process')
            evaluation_measure_result = []

            record_reader = RecordReader(dump_dir)
            for measure in evaluation_measures:
                record_generator = record_reader.iterate_read('predict', 'groundtruth')
                result = measure.eva(record_generator, None)
                evaluation_measure_result.append(result)
            task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

            repeated_running_statistic.append(task_running_statictic)

        evaluation_result = multi_repeats_measures_statistic(repeated_running_statistic, method='repeated-holdout')
        return evaluation_result

    def _bootstrap_validation(self, bootstrap_rounds, train_dataset, evaluation_measures):
        bootstrap_running_statistic = []
        for bootstrap_i in range(bootstrap_rounds):
            # 1.step split train set and validation set
            part_train_dataset, part_validation_dataset = train_dataset.split(split_params={},
                                                                              split_method='bootstrap')
            # dump_dir
            dump_dir = os.path.join(self.ant_dump_dir, 'bootstrap-%d-evaluation' % bootstrap_i)

            # prepare dir
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            # 2.step training model
            self.stage = 'EVALUATION-BOOTSTRAPTRAIN-%d' % bootstrap_i
            self.context.call_training_process(part_train_dataset, dump_dir)
            if self.context.recorder:
                self.context.recorder.close()

            # 3.step evaluation measures
            # split data and label
            data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
            self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
            # infer_dump_dir = os.path.join(dump_dir, 'inference')
            # if not os.path.exists(infer_dump_dir):
            #     os.makedirs(infer_dump_dir)

            self.stage = 'EVALUATION-BOOTSTRAPEVALUATION-%d' % bootstrap_i
            with running_statistic(self.ant_name):
                self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

            if self.context.recorder:
                self.context.recorder.close()

            task_running_statictic = get_running_statistic(self.ant_name)
            task_running_statictic = {self.ant_name: task_running_statictic}
            task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
            task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
                task_running_elapsed_time / float(part_validation_dataset.size)

            logger.info('start evaluation process')
            evaluation_measure_result = []

            record_reader = RecordReader(dump_dir)
            for measure in evaluation_measures:
                record_generator = record_reader.iterate_read('predict', 'groundtruth')
                result = measure.eva(record_generator, None)
                evaluation_measure_result.append(result)
            task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

            bootstrap_running_statistic.append(task_running_statictic)

        evaluation_result = multi_repeats_measures_statistic(bootstrap_running_statistic, method='bootstrap')
        return evaluation_result

    def _kfold_cross_validation(self, kfolds, train_dataset, evaluation_measures):
        assert (kfolds in [5, 10])
        kfolds_running_statistic = []
        for k in range(kfolds):
            # 1.step split train set and validation set
            part_train_dataset, part_validation_dataset = train_dataset.split(split_params={'kfold': kfolds,
                                                                                            'k': k},
                                                                              split_method='kfold')
            # dump_dir
            dump_dir = os.path.join(self.ant_dump_dir, 'fold-%d-evaluation' % k)

            # prepare dir
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            # 2.step training model
            self.stage = 'EVALUATION-KFOLDTRAIN-%d' % k
            self.context.call_training_process(part_train_dataset, dump_dir)
            if self.context.recorder:
                self.context.recorder.close()

            # 3.step evaluation measures
            # split data and label
            data_annotation_branch = DataAnnotationBranch(Node.inputs(part_validation_dataset))
            self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
            # infer_dump_dir = os.path.join(dump_dir, 'inference')
            # if not os.path.exists(infer_dump_dir):
            #     os.makedirs(infer_dump_dir)

            self.stage = 'EVALUATION-KFOLDEVALUATION-%d' % k
            with running_statistic(self.ant_name):
                self.context.call_infer_process(data_annotation_branch.output(0), dump_dir)

            if self.context.recorder:
                self.context.recorder.close()

            task_running_statictic = get_running_statistic(self.ant_name)
            task_running_statictic = {self.ant_name: task_running_statictic}
            task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
            task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
                task_running_elapsed_time / float(part_validation_dataset.size)

            logger.info('start evaluation process')
            evaluation_measure_result = []

            record_reader = RecordReader(dump_dir)
            for measure in evaluation_measures:
                record_generator = record_reader.iterate_read('predict', 'groundtruth')
                result = measure.eva(record_generator, None)
                evaluation_measure_result.append(result)
            task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

            kfolds_running_statistic.append(task_running_statictic)

        evaluation_result = multi_repeats_measures_statistic(kfolds_running_statistic, method='kfold')
        return evaluation_result