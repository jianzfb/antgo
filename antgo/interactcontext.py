# -*- coding: UTF-8 -*-
# @Time    : 2021/1/8 7:23 下午
# @File    : interactcontext.py
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
from antgo.ant.ensemble import *


class InteractContext(Context):
  def __init__(self):
    super(InteractContext, self).__init__(interact_mode=True)

  def __prepare_context(self, name, param, **kwargs):
    # 1.step 获取基本配置信息
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    if not os.path.exists(config_xml):
      # 创建默认配置
      if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
        os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

      env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
      config_template = env.get_template('config.xml')
      config_data = {
        'DATASET_FACTORY': os.path.join(os.environ['HOME'], '.config', 'antgo', 'factory'),
        'USER_TOKEN': ''
      }
      config_content = config_template.render(**config_data)
      with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'), 'w') as fp:
        fp.write(config_content)

      if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'factory')):
        os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo', 'factory'))

      logger.warn('Build default antgo config in ~/.config/antgo/config.xml')

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
    self.api_token = Config.server_user_token
    logger.info(f'Use experiment data_factory {self.data_factory}')
    logger.info(f'Use experiment task_factory {self.task_factory}')
    logger.info(f'Use experiment api token {self.api_token}')

    self.main_folder = os.path.abspath(os.curdir)
    if self.data_factory == os.path.join(os.environ['HOME'], '.config', 'antgo', 'factory'):
      # 当前正在使用默认配置
      logger.warn('Current use default factory, please shell "antgo config --dataset=xxx --token=xxx"')

    self.params = {}
    if type(param) == str:
      param = yaml.load(open(param, 'r'))
      self.params = param
    else:
      self.params = param

    if 'system' not in self.params._params:
      self.params._params.update({'system': {}})
    if 'research' not in self.params._params['system']:
      self.params._params['system']['research'] = ''
    if 'skip_training' not in self.params._params['system']:
      self.params._params['system']['skip_training'] = False
    if 'ip' not in self.params._params['system']:
      self.params._params['system']['ip'] = '127.0.0.1'
    if 'port' not in self.params._params['system']:
      self.params._params['system']['port'] = 8901
    if 'devices' not in self.params._params['system']:
      self.params._params['system']['devices'] = ''
    if 'running_platform' not in self.params._params['system']:
      self.params._params['system']['running_platform'] = ''
    if 'unlabel' not in self.params._params['system']:
      self.params._params['system']['unlabel'] = False
    if 'candidate' not in self.params._params['system']:
      self.params._params['system']['candidate'] = False
    if 'ext_params' not in self.params._params['system']:
      self.params._params['system']['ext_params'] = {}

    # 2.step 设置来自实验名称
    self.dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
    if not os.path.exists(self.dump_dir):
      os.makedirs(self.dump_dir)

    from_experiment = kwargs.get('from_experiment', None)
    if from_experiment is not None:
      experiment_path = os.path.join(self.dump_dir, from_experiment)
      if not os.path.exists(experiment_path):
        logger.error('Couldnt find experiment %s' % from_experiment)
        raise IOError

      if os.path.exists(os.path.join(experiment_path, 'train')):
        self.from_experiment = os.path.join(experiment_path, 'train')
      elif os.path.exists(os.path.join(experiment_path, 'inference')):
        self.from_experiment = os.path.join(experiment_path, 'inference')
      else:
        self.from_experiment = experiment_path

    # 3.step 设置默认任务
    task = kwargs.get('task', None)
    if task is not None:
      if os.path.exists(os.path.join(self.task_factory, task)):
        task = os.path.join(self.task_factory, task)
      else:
        task = os.path.join(self.main_folder, task)

    dataset = kwargs.get('dataset', None)
    if task is None and dataset is not None:
      # build default task
      with open(os.path.join(self.task_factory, '%s.xml' % name), 'w') as fp:
        task_content = '<task><task_name>%s</task_name><task_type>%s</task_type><task_badcase><badcase_num>%d</badcase_num><badcase_category>%d</badcase_category></task_badcase>' \
                       '<input>' \
                       '<source_data><data_set_name>%s</data_set_name>' \
                       '</source_data>' \
                       '<estimation_procedure><type>%s</type></estimation_procedure>' \
                       '<evaluation_measures><evaluation_measure>%s</evaluation_measure></evaluation_measures>' \
                       '</input>' \
                       '</task>' % (
                       name, '', 0, 0,dataset, '', '')
        fp.write(task_content)
      task = os.path.join(self.task_factory, '%s.xml' % name)

    self.dataset = dataset
    self.task = task

  @contextmanager
  def Train(self, exp_name, exp_param, **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)
    # 2.step 创建时间戳
    time_stamp = timestamp()
    try:
      # 6.step 创建训练过程调度对象
      train = AntTrain(self,
                       exp_name,
                       self.data_factory,
                       self.dump_dir,
                       self.api_token,
                       self.task,
                       self.dataset,
                       time_stamp=time_stamp,
                       main_folder=self.main_folder)
      # 训练控制
      train.start()
      yield train

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Challenge(self, exp_name, exp_param, benchmark='', **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)

    try:
      # 5.step 创建挑战过程调度对象
      challenge = AntChallenge(self,
                               exp_name,
                               self.data_factory,
                               self.dump_dir,
                               self.api_token,
                               self.task,
                               benchmark,
                               time_stamp=timestamp(),
                               main_folder=self.main_folder)
      challenge.start()
      yield challenge

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Predict(self, exp_name, exp_param, **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)

    try:
      predict =\
        AntBatch(self,
                 exp_name,
                 self.api_token,
                 self.data_factory,
                 self.dump_dir,
                 self.task,
                 self.dataset)
      predict.start()
      yield predict

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Ensemble(self, exp_name, exp_param, stage, **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)

    try:
      ensemble_handler = \
        AntEnsemble(self,
                    exp_name,
                    self.data_factory,
                    self.dump_dir,
                    self.api_token,
                    self.task,
                    self.dataset,
                    stage)
      # 准备ensemble控制
      ensemble_handler.start()
      yield ensemble_handler

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Demo(self, exp_name, exp_param, html_template=None,  **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)

    try:
      demo = AntDemo(self,
                     exp_name,
                     self.dump_dir,
                     self.api_token,
                     self.task,
                     html_template=html_template,
                     time_stamp=timestamp())
      # demo
      demo.start()
      yield demo

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Browser(self, exp_name, exp_param, ip=None, port=10000, **kwargs):
    # 1.step 配置基本信息
    self.__prepare_context(exp_name, exp_param, **kwargs)

    try:
      browser = AntBrowser(self,
                           exp_name,
                           self.api_token,
                           self.data_factory,
                           kwargs.get('dataset', ''),
                           self.dump_dir)
      # 浏览控制
      browser.start()
      yield browser

      # 日志更新
      mlogger.update()
    except:
      traceback.print_exc()
      raise sys.exc_info()[0]

  @contextmanager
  def Activelearning(self):
    yield
