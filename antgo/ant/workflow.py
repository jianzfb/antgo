# encoding=utf-8
# @Time    : 17-8-14
# @File    : workflow.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import yaml
import os
import time
from multiprocessing import Process, Lock
from antgo.utils.cpu import *
from antgo.utils.gpu import *
from antgo.ant.work import *
import tarfile
from antgo.ant import flags
FLAGS = flags.AntFLAGS
import sys
if sys.version > '3':
    PY3 = True
else:
    PY3 = False
    
WorkNodes = {'Training': Training,
             'Inference': Inference,
             'Evaluating': Evaluating,
             'DataSplit': DataSplit}


class WorkFlow(AntBase):
  def __init__(self, ant_name,
               ant_token,
               ant_workflow_config,
               main_file,
               main_folder,
               dump_dir,
               data_factory):
    super(WorkFlow, self).__init__(ant_name, None, ant_token)

    self.config_content = ant_workflow_config
    self.work_nodes = []
    self.work_acquired_locks = []
    self.nr_cpu = 0
    self.nr_gpu = 0
    self._main_file = main_file
    self._main_folder = main_folder
    self._dump_dir = dump_dir
    self._data_factory = data_factory

    self._ant_name = ant_name
    self._ant_token = ant_token

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

    # reset worknodes connections
    work_nodes = {}
    for nick_name, cf in works_config.items():
      if cf.name not in WorkNodes:
        logger.error('no exist work')
        return

      work_node = WorkNodes[cf.name](name=cf.nick_name,
                                     config_parameters=cf.config,
                                     code_path=self._main_folder,
                                     code_main_file=self._main_file,
                                     ant_name=self._ant_name,
                                     ant_token=self._ant_token)
      work_node.workspace_base = self._dump_dir
      work_node.data_factory = self._data_factory
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
        self._find_all_root(work_node, root_nodes_of_leaf)
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
    # warp model (main_file and main_param)
    # - backup in dump_dir
    main_folder = FLAGS.main_folder()
    main_param = FLAGS.main_param()
    main_file = FLAGS.main_file()
    goldcoin = os.path.join(self._dump_dir, '%s-goldcoin.tar.gz'%self._ant_name)

    if not os.path.exists(self._dump_dir):
      os.makedirs(self._dump_dir)

    if os.path.exists(goldcoin):
      os.remove(goldcoin)
      
    tar = tarfile.open(goldcoin, 'w:gz')
    tar.add(os.path.join(main_folder, main_file), arcname=main_file)
    if main_param is not None:
      tar.add(os.path.join(main_folder, main_param), arcname=main_param)
    tar.close()
    
    # - backup in cloud
    if os.path.exists(goldcoin):
      file_size = os.path.getsize(goldcoin) / 1024.0
      if file_size < 500:
        if not PY3 and sys.getdefaultencoding() != 'utf8':
          reload(sys)
          sys.setdefaultencoding('utf8')
        # model file shouldn't too large (500KB)
        with open(goldcoin, 'rb') as fp:
          self.send({'MODEL': fp.read()}, 'MODEL')

    # go and enjoy fun
    processes = [Process(target=lambda x, y: x.start(y),
                         args=(self.work_nodes[i], self.work_acquired_locks[i]),
                         name=self.work_nodes[i].name)
                 for i in range(len(self.work_nodes))]
    for p in processes:
      p.start()

    for p in processes:
      p.join()


if __name__ == '__main__':
  # self.config_content =
    config_content = yaml.load(open('/home/mi/antgo-workspace/test/ant-compose-abca.yaml', 'r'))
    mm = WorkFlow(ant_name='ZJ',
                  ant_token=None,
                  ant_workflow_config=config_content,
                  main_file='icnet_example.py',
                  main_folder='/home/mi/antgo/code',
                  dump_dir='/home/mi/antgo-workspace/test',
                  data_factory='/home/mi/antgo/antgo-dataset')
    mm.start()