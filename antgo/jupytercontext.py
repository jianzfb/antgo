# -*- coding: UTF-8 -*-
# @Time    : 2021/1/8 7:23 下午
# @File    : jupytercontext.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.context import *
from antgo.ant.train import *
from antgo.ant.challenge import *
from antgo.ant.demo import *
from antgo.ant.browser import *
from antgo.ant.batch import *


class JupyterContext(Context):
  def __init__(self):
    super(JupyterContext, self).__init__()

  def __prepare_header(self, api_token, name, param, **kwargs):
    # 1.step 获取基本配置信息
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)

    factory = getattr(Config, 'factory', None)
    assert factory is not None

    if not os.path.exists(Config.data_factory):
      os.makedirs(Config.data_factory)
    if not os.path.exists(Config.task_factory):
      os.makedirs(Config.task_factory)

    self.name = name
    self.data_factory = Config.data_factory
    self.task_factory = Config.task_factory
    self.main_folder = os.path.abspath(os.curdir)

    if type(param) == str:
      param = yaml.load(open(param, 'r'))
      self.params = param
    else:
      self.params = param

    # 2.step 设置来自实验名称
    dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

    from_experiment = kwargs.get('from_experiment', None)
    if from_experiment is not None:
      experiment_path = os.path.join(dump_dir, from_experiment)
      if not os.path.exists(experiment_path):
        logger.error('couldnt find experiment %s' % from_experiment)
        raise IOError

      if os.path.exists(os.path.join(experiment_path, 'train')):
        self.from_experiment = os.path.join(experiment_path, 'train')
      elif os.path.exists(os.path.join(experiment_path, 'inference')):
        self.from_experiment = os.path.join(experiment_path, 'inference')
      else:
        self.from_experiment = experiment_path

    return dump_dir

  @contextmanager
  def Train(self, api_token, exp_name, exp_param, **kwargs):
    # 1.step 配置基本信息
    dump_dir = self.__prepare_header(api_token, exp_name, exp_param, **kwargs)
    # 2.step 创建时间戳
    time_stamp = timestamp()
    # 3.step 获得是否制定自定义任务
    task = kwargs.get('task', None)
    # 4.step 是否跳过训练过程
    skip_training = kwargs.get('skip_training', False)
    # 5.step 设置设备
    devices = kwargs.get('devices', '')

    # 6.step 创建训练过程调度对象
    train = AntTrain(self,
                     exp_name,
                     self.data_factory,
                     dump_dir,
                     api_token,
                     task,
                     skip_training=skip_training,
                     time_stamp=time_stamp,
                     devices=devices,
                     main_folder=self.main_folder)

    yield train

    # 日志更新
    mlogger.update()


  @contextmanager
  def Challenge(self, api_token, exp_name, exp_param, benchmark='', **kwargs):
    # 1.step 配置基本信息
    dump_dir = self.__prepare_header(api_token, exp_name, exp_param, **kwargs)
    # 2.step 创建时间戳
    time_stamp = timestamp()
    # 3.step 获得是否制定自定义任务
    task = kwargs.get('task', None)
    # 4.step 设置设备
    devices = kwargs.get('devices', '')

    # 5.step 创建挑战过程调度对象
    challenge = AntChallenge(self,
                             exp_name,
                             self.data_factory,
                             dump_dir,
                             api_token,
                             task,
                             benchmark,
                             time_stamp=time_stamp,
                             devices=devices,
                             main_folder=self.main_folder)
    yield challenge

    # 日志更新
    mlogger.update()

  @contextmanager
  def Demo(self, api_token, exp_name, exp_param, html_template=None, ip=None, port=None, **kwargs):
    # 1.step 配置基本信息
    dump_dir = self.__prepare_header(api_token, exp_name, exp_param, **kwargs)
    # 2.step 创建时间戳
    time_stamp = timestamp()
    # 3.step 获得是否制定自定义任务
    task = kwargs.get('task', None)
    # 4.step 设置设备
    devices = kwargs.get('devices', '')

    demo = AntDemo(self,
                   exp_name,
                   dump_dir,
                   api_token,
                   task,
                   html_template=html_template,
                   ip=ip,
                   port=port,
                   time_stamp=time_stamp,
                   devices=devices)
    yield demo

    # 日志更新
    mlogger.update()


  @contextmanager
  def Browser(self, api_token, exp_name, exp_param, ip=None, port=10000, **kwargs):
    # 1.step 配置基本信息
    dump_dir = self.__prepare_header(api_token, exp_name, exp_param, **kwargs)

    browser = AntBrowser(self,
                         exp_name,
                         api_token,
                         ip,
                         port,
                         self.data_factory,
                         kwargs.get('dataset', ''),
                         dump_dir)

    yield browser

    # 日志更新
    mlogger.update()


  @contextmanager
  def Activelearning(self):
    yield

  @contextmanager
  def Generate(self):
    yield

  @contextmanager
  def Batch(self, api_token, exp_name, exp_param, ip=None, port=10000, unlabel=False, **kwargs):
    # 1.step 配置基本信息
    dump_dir = self.__prepare_header(api_token, exp_name, exp_param, **kwargs)
    # 2.step 自定义任务
    task = kwargs.get('task', None)

    batch = AntBatch(self,
                     exp_name,
                     ip,
                     port,
                     api_token,
                     self.data_factory,
                     dump_dir,
                     task,
                     unlabel=unlabel,
                     restore_experiment=kwargs.get('restore_experiment', None))
    yield batch

    # 日志更新
    mlogger.update()
