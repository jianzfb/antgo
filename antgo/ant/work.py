# encoding=utf-8
# @Time    : 17-6-22
# @File    : work.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import yaml
import os
import time
import shutil
from antgo.ant.workflow import *
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
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(Training, self).__init__(name=name,
                                       code_path=code_path,
                                       code_main_file=code_main_file,
                                       config_parameters=config_parameters,
                                       port=port)

    def run(self, *args, **kwargs):
        # 0.step load context
        ctx = self.load()

        # 1.step load config file
        loaded_training_config = self.load_config()

        # update by loaded training config
        dataset_name = None
        dataset_train_or_test = 'train'
        dataset_params = {}
        if 'dataset' in loaded_training_config:
            dataset_name = loaded_training_config['dataset'].get('name', dataset_name)
            dataset_params = loaded_training_config['dataset'].get('params', dataset_params)

            # how to split dataset as training dataset
            how_to_split = loaded_training_config['dataset'].get('split', {})
            if len(how_to_split) > 0:
                if 'train' in how_to_split:
                    if type(how_to_split['train']) == str:
                        # dataset flag (train)
                        dataset_train_or_test = how_to_split['train']
                    else:
                        assert(how_to_split['train'] == list)
                        # id list
                        dataset_params['filter'] = how_to_split['train']

        model_parameters = copy.deepcopy(loaded_training_config)
        if 'dataset' in model_parameters:
            model_parameters.pop('dataset')

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
        loaded_training_config.update(
            {'dataset': {'name': dataset_name, 'train_or_test': dataset_train_or_test, 'params': dataset_params}})
        self.update_config(loaded_training_config)

        # 2.step registry trigger
        if continue_condition is not None:
            ctx.registry_trainer_callback(continue_condition['key'],
                                          continue_condition['value'],
                                          continue_condition['condition'],
                                          self.notify_func)

        # 3.step start running
        ctx.params = model_parameters
        assert(dataset_name is not None)
        dataset_cls = ctx.dataset_factory(dataset_name)
        dataset = dataset_cls(dataset_train_or_test, self.datasource, dataset_params)
        dataset.build()
        ctx.call_training_process(dataset, self.dump_dir)

        # 4.step work is done
        if continue_condition is None:
            self.send(self.dump_dir, 'DONE')
        else:
            self.send('', 'DONE')
        ctx.wait_until_clear()

    def notify_func(self):
        # 1.step notify
        self.send(self.dump_dir, 'CONTINUE')


class Inference(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(Inference, self).__init__(name=name,
                                        code_path=code_path,
                                        code_main_file=code_main_file,
                                        config_parameters=config_parameters,
                                        port=port)

    def run(self, *args, **kwargs):
        # 0.step load ctx
        ctx = self.load()

        # 1.step load config file
        dataset_name = None
        dataset_train_or_test = 'test'
        dataset_params = None
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
                            assert (how_to_split['test'] == list)
                            # id list
                            dataset_params['filter'] = how_to_split['test']

            model_parameters = copy.deepcopy(loaded_infer_config)
            if 'dataset' in model_parameters:
                model_parameters.pop('dataset')

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
        loaded_infer_config.update(
            {'dataset': {'name': dataset_name, 'train_or_test':dataset_train_or_test, 'params':dataset_params}})

        assert(dataset_name is not None)
        self.update_config(loaded_infer_config)

        # 2.step start running
        ctx.params = model_parameters
        dataset_cls = ctx.dataset_factory(dataset_name)
        dataset = dataset_cls(dataset_train_or_test, self.datasource, dataset_params)
        dataset.build()
        data_annotation_branch = DataAnnotationBranch(Node.inputs(dataset))
        ctx.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
        ctx.call_infer_process(data_annotation_branch.output(0), self.dump_dir)

        # work is done
        self.send(os.path.join(self.dump_dir, ctx.recorder.recorder_name), 'DONE')
        ctx.wait_until_clear()


class Evaluating(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(Evaluating, self).__init__(name=name,
                                         code_path=code_path,
                                         code_main_file=code_main_file,
                                         config_parameters=config_parameters,
                                         port=port)

    def run(self, *args, **kwargs):
        assert('task' in self.config_parameters)
        assert('type' in self.config_parameters['task'])
        class_label = []
        if 'class_label' in self.config_parameters['task']:
            class_label = self.config_parameters['task']['class_label']

        dummy_ant_task = AntTask(task_id=-1, task_name=None, task_type_id=-1,
                                 task_type=self.config_parameters['task']['type'],
                                 dataset_id=-1, dataset_name=None, dataset_params=None,
                                 estimation_procedure_type=None, estimation_procedure_params=None,
                                 evaluation_measure=None, cost_matrix=None,
                                 class_label=class_label)

        ant_measures = AntMeasures(dummy_ant_task)
        measures_name = self.config_parameters.get('measure', None)
        applied_measures = ant_measures.measures(measures_name)

        flag, _, infer_result, gt = load_records(self.dump_dir)

        statistic_data = []
        if flag == 'single':
            # single process
            for measure_obj in applied_measures:
                statistic_data.append(measure_obj.eva(infer_result, gt))
        else:
            # multi process (need to give confidence interval)
            statistic_experiment_data = []
            for index in range(len(infer_result)):
                p_infer_result = infer_result[index]
                p_gt = gt[index]

                for measure_obj in applied_measures:
                    statistic_experiment_data.append(measure_obj.eva(p_infer_result, p_gt))

            # compute confidence interval
            statistic_data = compute_confidence_interval(statistic_experiment_data)

        # visualization
        everything_to_html(statistic_data, self.dump_dir, True, name=self.name)

        self.send('', 'DONE')


class DataMarket(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(DataMarket, self).__init__(name=name,
                                         code_path=code_path,
                                         code_main_file=code_main_file,
                                         config_parameters=config_parameters,
                                         port=port)

    def run(self, *args, **kwargs):
        dataset_kv = self.config_parameters['dataset']
        for k, v in dataset_kv.items():
            params = {}
            if 'params' in v:
                params = v['params']

            split = {}
            if 'split' in v:
                split = v['split']

            # generate yaml configue file
            dataset_config = {}
            dataset_config['dataset'] = {}
            dataset_config['dataset']['name'] = k
            dataset_config['dataset']['params'] = params
            if len(split) > 0:
                dataset_config['dataset']['split'] = split

            fp = open(os.path.join(self.dump_dir, 'dataset-%s-config.yaml'%k), 'w')
            yaml.dump(dataset_config, fp)
            fp.close()
            # continue
            self.send(os.path.join(self.dump_dir, 'dataset-%s-config.yaml'%k), 'CONTINUE')

        self.send('', 'DONE')


class StatisticExperiment(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(StatisticExperiment, self).__init__(name=name,
                                                  code_path=code_path,
                                                  code_main_file=code_main_file,
                                                  config_parameters=config_parameters,
                                                  port=port)

    def run(self, *args, **kwargs):
        # 0.step load model
        ctx = self.load()

        # 1.step load config (including dataset config)
        loaded_config = self.load_config()
        assert(loaded_config is not None)
        assert('dataset' in loaded_config)

        # 2.step using custom experiment
        split_method = self.config_parameters['method']
        split_params = self.config_parameters.get('params', {})

        assert(split_method in ['holdout', 'repeated-holdout', 'bootstrap', 'kfold'])

        dataset_name = loaded_config['dataset'].get('name', None)
        dataset_params = loaded_config['dataset'].get('params', {})
        dataset_train_or_test = 'train'

        dataset_cls = ctx.dataset_factory(dataset_name)
        dataset = dataset_cls(dataset_train_or_test, self.datasource, dataset_params)
        dataset.build()

        if split_method == 'holdout':
            t, v = dataset.split_index(split_params, split_method)
            dataset_config = copy.deepcopy(loaded_config)
            dataset_config['dataset']['split'] = {}
            dataset_config['dataset']['split']['train'] = t
            dataset_config['dataset']['split']['test'] = v

            fp = open(os.path.join(self.dump_dir, 'experiment-holdout-config.yaml'), 'w')
            yaml.dump(dataset_config, fp)
            fp.close()
            self.send(os.path.join(self.dump_dir, 'experiment-holdout-config.yaml'), 'DONE')
        elif split_method == 'repeated-holdout':
            repeated_times = split_params['repeated_times']
            for index in range(repeated_times):
                t, v = dataset.split_index(split_params, split_method)
                dataset_config = copy.deepcopy(loaded_config)
                dataset_config['dataset']['split'] = {}
                dataset_config['dataset']['split']['train'] = t
                dataset_config['dataset']['split']['test'] = v

                fp = open(os.path.join(self.dump_dir, 'experiment-repeated-holdout-%d-config.yaml'%index), 'w')
                yaml.dump(dataset_config, fp)
                fp.close()
                self.send(os.path.join(self.dump_dir, 'experiment-repeated-holdout-%d-config.yaml'%index), 'CONTINUE')

            self.send('', 'DONE')
        elif split_method == 'bootstrap':
            repeated_times = split_params['repeated_times']
            for index in range(repeated_times):
                t, v = dataset.split_index(split_params, split_method)
                dataset_config = copy.deepcopy(loaded_config)
                dataset_config['dataset']['split'] = {}
                dataset_config['dataset']['split']['train'] = t
                dataset_config['dataset']['split']['test'] = v

                fp = open(os.path.join(self.dump_dir, 'experiment-bootstrap-%d-config.yaml'%index), 'w')
                yaml.dump(dataset_config, fp)
                fp.close()
                self.send(os.path.join(self.dump_dir, 'experiment-bootstrap-%d-config.yaml'%index), 'CONTINUE')

            self.send('', 'DONE')

        elif split_method == 'kfold':
            kfold = split_params['kfold']
            for k in range(kfold):
                split_params['k'] = k

                t, v = dataset.split_index(split_params, split_method)
                dataset_config = copy.deepcopy(loaded_config)
                dataset_config['dataset']['split'] = {}
                dataset_config['dataset']['split']['train'] = t
                dataset_config['dataset']['split']['test'] = v

                fp = open(os.path.join(self.dump_dir, 'experiment-kfold-%d-config.yaml'%k), 'w')
                yaml.dump(dataset_config, fp)
                fp.close()
                self.send(os.path.join(self.dump_dir, 'experiment-kfold-%d-config.yaml'%k), 'CONTINUE')

            self.send('', 'DONE')
        else:
            raise NotImplementedError()


class HyperParameterSelection(object):
    def __init__(self):
        pass

    def start(self):
        pass


class A(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(A, self).__init__(name=name,
                                       code_path=code_path,
                                       code_main_file=code_main_file,
                                       config_parameters=config_parameters,
                                       port=port)

    def run(self, *args, **kwargs):
        for i in range(2):
            print('Im A %d'%i)
            time.sleep(3)
            fp = open(os.path.join(self.dump_dir,'a.txt'),'w')
            fp.write('99')
            fp.close()
            self.send(os.path.join(self.dump_dir,'a.txt'),'CONTINE')

        self.send('', 'DONE')


class B(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(B, self).__init__(name=name,
                                code_path=code_path,
                                code_main_file=code_main_file,
                                config_parameters=config_parameters,
                                port=port)

    def run(self, *args, **kwargs):
        fp = open(os.path.join(self.dump_dir,'a.txt'),'r')
        content = fp.read()
        fp.close()
        num = int(content)

        for i in range(3):
            print('Im B %d'%i)
            time.sleep(15)
            fp = open(os.path.join(self.dump_dir, 'b-%d.txt'%i),'w')
            if i == 0:
                fp.write('%d'%(num-10))
            if i == 1:
                fp.write('%d'%(num*10))
            if i == 2:
                fp.write('%d'%(num + 10))
            fp.close()
            self.send(os.path.join(self.dump_dir, 'b-%d.txt'%i), 'CONTINUE')

        self.send('', 'DONE')


class C(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(C, self).__init__(name=name,
                                code_path=code_path,
                                code_main_file=code_main_file,
                                config_parameters=config_parameters,
                                port=port)

    def run(self, *args, **kwargs):
        content = None
        for ff in os.listdir(self.dump_dir):
            if ff[0] == '.':
                continue

            fp = open(os.path.join(self.dump_dir,ff),'r')
            content = fp.read()
            fp.close()

            break

        num = int(content)

        print('Im C')
        fp = open(os.path.join(self.dump_dir, 'c.txt'), 'w')
        fp.write('%d'%(num*2))
        fp.close()

        self.send(os.path.join(self.dump_dir, 'c.txt'), 'DONE')


class D(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(D, self).__init__(name=name,
                                code_path=code_path,
                                code_main_file=code_main_file,
                                config_parameters=config_parameters,
                                port=port)

    def run(self, *args, **kwargs):
        content = None
        for ff in os.listdir(self.dump_dir):
            if ff[0] == '.':
                continue

            fp = open(os.path.join(self.dump_dir, ff), 'r')
            content = fp.read()
            fp.close()

            break

        num = int(content)

        print('Im D')
        fp = open(os.path.join(self.dump_dir, 'd.txt'), 'w')
        fp.write('%d'%(num*3))
        fp.close()
        time.sleep(10)

        self.send(os.path.join(self.dump_dir, 'd.txt'), 'DONE')


class E(BaseWork):
    def __init__(self, name, config_parameters, code_path, code_main_file, port=''):
        super(E, self).__init__(name=name,
                                code_path=code_path,
                                code_main_file=code_main_file,
                                config_parameters=config_parameters,
                                port=port)

    def run(self, *args, **kwargs):
        content = []
        for ff in os.listdir(self.dump_dir):
            if ff[0] == '.':
                continue

            fp = open(os.path.join(self.dump_dir, ff), 'r')
            content.append(fp.read())
            fp.close()

        num3 = [int(c) for c in content]

        print('Im E')
        fp = open(os.path.join(self.dump_dir, 'e.txt'), 'w')
        fp.write('%d'%(num3[0]+num3[1]+num3[2]))
        fp.close()
        time.sleep(5)

        self.send('','DONE')

WorkNodes = {'Training': Training,
             'Inference': Inference,
             'Evaluating':Evaluating,
             'DataMarket':DataMarket,
             'HyperParameterSelection':HyperParameterSelection,
             'A':A,
             'B':B,
             'C':C,
             'D':D,
             'E':E}


class WorkFlow(object):
    def __init__(self, config_file):
        self.config_content = yaml.load(open(config_file, 'r'))
        self.work_nodes = []
        self.work_acquired_locks = []
        self.nr_cpu = 0
        self.nr_gpu = 0
        # parse work flow
        self._parse_work_flow()
        # analyze work computing resource
        self._analyze_computing_resource()

    class _WorkConfig(object):
        def __init__(self):
            self._config = None
            self._input_bind = []
            self._feedback_bind = []
            self._name = ""
            self._nike_name = ""

        @property
        def config(self):
            return self._config
        @config.setter
        def config(self, val):
            self._config = val

        @property
        def input_bind(self):
            return self._input_bind
        @input_bind.setter
        def input_bind(self, val):
            self._input_bind.extend(val)

        @property
        def feedback_bind(self):
            return self._feedback_bind
        @feedback_bind.setter
        def feedback_bind(self, val):
            self._feedback_bind.extend(val)

        @property
        def name(self):
            return self._name
        @name.setter
        def name(self, val):
            self._name = val

        @property
        def nick_name(self):
            return self._nike_name
        @nick_name.setter
        def nick_name(self, val):
            self._nike_name = val

    def _find_all_root(self, leaf_node, root_list):
        if leaf_node.is_root:
            root_list.append(leaf_node)

        for input_link in leaf_node.input:
            if input_link.link_type == 'NORMAL':
                if input_link.nest.is_root:
                    root_list.append(input_link.nest)
                else:
                    self._find_all_root(input_link.nest, root_list)

    def _parse_work_flow(self):
        works_config = {}
        self._datasource = ""
        self._workspace = ""
        self._code_path = ""
        self._code_main_file = ""

        for k, v in self.config_content.items():
            if type(v) == dict:
                if 'type' in v and v['type'] != 'work':
                    logger.error('type must be work...')
                    return

                work_config = WorkFlow._WorkConfig()
                work_config.name = v['name']
                v.pop('name')
                work_config.nick_name = k
                if 'input-bind' in v:
                    work_config.input_bind = v['input-bind']
                    v.pop('input-bind')
                if 'feedback-bind' in v:
                    work_config.feedback_bind = v['feedback-bind']
                    v.pop('feedback-bind')
                work_config.config = v
                works_config[work_config.nick_name] = work_config
            elif k == 'datasource':
                self._datasource = v
            elif k == 'workspace':
                self._workspace = v
            elif k == 'code_path':
                self._code_path = v
            elif k == 'code_main_file':
                self._code_main_file = v

        # reset worknodes connections
        work_nodes = {}
        for nick_name, cf in works_config.items():
            if cf.name not in WorkNodes:
                logger.error('no exist work')
                return

            work_node = WorkNodes[cf.name](name=cf.nick_name, config_parameters=cf.config, code_path=self._code_path, code_main_file=self._code_main_file)
            work_node.workspace_base = self._workspace
            work_node.datasource = self._datasource
            work_nodes[cf.nick_name] = work_node
            self.work_nodes.append(work_node)

        root_work_nodes = []
        for nick_name, work_node in work_nodes.items():
            if works_config[nick_name].input_bind is not None:
                for mm in works_config[nick_name].input_bind:
                    output_pipe = work_nodes[mm].output
                    work_node.input = (output_pipe, 'NORMAL')
            else:
                work_node.node_type = 'ROOT'
                root_work_nodes.append(work_node)

            # worknode is root
            if work_node.input_num == 0:
                work_node.is_root = True

            # config feedback input
            if works_config[nick_name].feedback_bind is not None:
                for mm in works_config[nick_name].feedback_bind:
                    work_node.input = (work_nodes[mm].output, 'FEEDBACK')

        for nick_name, work_node in work_nodes.items():
            if work_node.output_nonfeedback_num == 0:
                # is leaf
                root_nodes_of_leaf = []
                self._find_all_root(work_node,root_nodes_of_leaf)
                for r in root_nodes_of_leaf:
                    r.input = (work_node.stop_shortcut, 'NORMAL')

    def _analyze_computing_resource(self):
        # cpu number
        self.nr_cpu = get_nr_cpu()
        # gpu number
        self.nr_gpu = get_nr_gpu()
        for work in self.work_nodes:
            work.set_computing_resource(self.nr_cpu, self.nr_gpu)

        # computing resource locks
        locks_pool = {}
        self.work_acquired_locks = [None for _ in range(len(self.work_nodes))]
        for work_i, work in enumerate(self.work_nodes):
            if work.occupy != 'share':
                resource_id = None
                if work.cpu is not None:
                    resource_id = 'cpu:' + '-'.join([str(c) for c in work.cpu])
                if work.gpu is not None:
                    if resource_id is None:
                        resource_id = 'gpu:' + '-'.join([str(g) for g in work.gpu])
                    else:
                        resource_id = resource_id + 'gpu:' + '-'.join([str(g) for g in work.gpu])

                if resource_id is not None:
                    if resource_id not in locks_pool:
                        locks_pool[resource_id] = Lock()

                self.work_acquired_locks[work_i] = locks_pool[resource_id]

    def start(self):
        processes = [Process(target=lambda x,y: x.start(y), args=(self.work_nodes[i], self.work_acquired_locks[i]))
                     for i in range(len(self.work_nodes))]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

if __name__ == '__main__':
    mm = WorkFlow('/home/mi/PycharmProjects/mltalker-antgo/ant-compose-2.yaml')
    mm.start()