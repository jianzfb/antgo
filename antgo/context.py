from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
from antgo.dataflow.dataset import *
import antvis.client.mlogger as mlogger
from antgo.ant.train import *
from antgo.ant.challenge import *
from antgo.ant.demo import *
from antgo.ant.browser import *
from antgo.ant.batch import *


class Params(object):
  def __init__(self, params={}):
    if params is not None:
      for k, v in params.items():
        if k != 'self':
          setattr(self, k, v)

    self._params = params

  def define(self, k, v=None):
    setattr(self, k, v)
    self._params[k] = v

  def __getattr__(self, item):
    if item not in object.__dict__:
      return None

    return object.__dict__[item]

  @property
  def content(self):
    return self._params

global_context = None


def get_global_context():
  global global_context
  assert(global_context is not None)
  return global_context


class _Block(object):
  def __init__(self, name):
    self._name = name
    self._activate = True
    
  @property
  def activate(self):
    return self._activate
  @activate.setter
  def activate(self, val):
    self._activate = val
  
  @property
  def name(self):
    return self._name
  
  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    pass
  

class Context(object):
  def __init__(self):
    global global_context
    # assert(global_context == None)

    self.training_process_callback = None
    self.infer_process_callback = None
    self._data_generator = None
    self.running_recorder = None
    self.context_params = None
    
    self.pid = str(os.getpid())
    
    global_context = self
    self.context_ant = None
    self._stage = ""

    self.trainer_callbacks = []
    self.clear_callbacks = []

    self.data_source = None
    self._blocks = []
    self._blocks_status = {}
    self._stoppable_threads = []
    
    self._from_experiment = None
    self._model = None

    self._devices = []

    self._quiet = False
    self._debug = False

    self._name = ''
    self._task_factory = None
    self._data_factory = None
    self._main_folder = None
    self._experiment_uuid = None

  def wait_until_clear(self):
    for stoppable_thread in self._stoppable_threads:
      stoppable_thread.stop()
      if stoppable_thread.stop_condition is not None:
        with stoppable_thread.stop_condition:
          stoppable_thread.stop_condition.notifyAll()
      stoppable_thread.join()
    self._stoppable_threads = []

    for clear_func in self.clear_callbacks:
      clear_func()
    
    global global_context
    global_context = None
    # clear
    self.training_process_callback = None
    self.infer_process_callback = None

    self._data_generator = None

    self.running_recorder = None
    self.context_params = None
    self.context_ant = None
    self._stage = ""
    self.trainer_callbacks = []
    self._data_source = None

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, val):
    self._name = val

  @property
  def ant(self):
    return self.context_ant

  @ant.setter
  def ant(self, val):
    self.context_ant = val

  @property
  def stage(self):
    return self._stage

  @stage.setter
  def stage(self, val):
    # reset experiment stage
    self._stage = val
    # reset experiment dashboard
    mlogger.getEnv().dashboard.experiment_stage = val

  @property
  def devices(self):
    return self._devices

  @devices.setter
  def devices(self, val):
    self._devices = val

  @property
  def model(self):
    return self._model
  
  @model.setter
  def model(self, val):
    self._model = val

  @property
  def training_process(self):
    return self.training_process_callback

  @training_process.setter
  def training_process(self, callback):
    self.training_process_callback = callback

  def call_training_process(self, data_source, dump_dir):
    is_inner_set = False
    if self.recorder is not None and self.recorder.dump_dir == None:
      self.recorder.dump_dir = os.path.join(dump_dir, 'record')
      is_inner_set = True
    
    self.data_source = data_source
    self.training_process(data_source, dump_dir)

    if self.recorder is not None:
      if self.recorder.dump_dir is not None and is_inner_set:
        self.recorder.dump_dir = None

  @property
  def infer_process(self):
    return self.infer_process_callback

  @infer_process.setter
  def infer_process(self, callback):
    self.infer_process_callback = callback

  def call_infer_process(self, data_source, dump_dir):
    is_inner_set = False
    if self.recorder is not None and self.recorder.dump_dir is None:
      if dump_dir != '':
        if not os.path.exists(os.path.join(dump_dir, 'record')):
          os.makedirs(os.path.join(dump_dir, 'record'))

        self.recorder.dump_dir = os.path.join(dump_dir, 'record')
        is_inner_set = True
      
    self.data_source = data_source
    self.infer_process(data_source, dump_dir)

    if self.recorder is not None:
      if self.recorder.dump_dir is not None and is_inner_set:
        self.recorder.dump_dir = None

  @property
  def data_generator(self):
    return self._data_generator

  @data_generator.setter
  def data_generator(self, g):
    self._data_generator = g

  @property
  def recorder(self):
    return self.running_recorder

  @recorder.setter
  def recorder(self, callback):
    self.running_recorder = callback

  @property
  def params(self):
    return self.context_params

  @params.setter
  def params(self, val):
    self.context_params = Params(val)

  def registry_trainer_callback(self, key, value, condition, func):
    # condition: equal, less, greater or mod
    self.trainer_callbacks.append((key, value, condition, func))

  @property
  def registried_trainer_callbacks(self):
    return self.trainer_callbacks

  def registry_clear_callback(self, func):
    self.clear_callbacks.append(func)
  
  @property
  def data_source(self):
    return self._data_source
  
  @data_source.setter
  def data_source(self, val):
    self._data_source = val
  
  def register_stoppable_thread(self, stoppable_thread):
    self._stoppable_threads.append(stoppable_thread)
  
  def block(self, name):
    for b in self._blocks:
      if name == b.name:
        return b
  
    model_block = _Block(name)
    self._blocks.append(model_block)
    model_block.activate = True
    if name in self._blocks_status and not self._blocks_status[name]:
      model_block.activate = False
    return model_block
  
  def blocks(self):
    return self._blocks
    
  def activate_block(self, name):
    self._blocks_status[name] = True
    
  def deactivate_block(self, name):
    self._blocks_status[name] = False
  
  @property
  def from_experiment(self):
    return self._from_experiment

  @from_experiment.setter
  def from_experiment(self, experiment):
    self._from_experiment = experiment

  @property
  def quiet(self):
    return self._quiet

  @quiet.setter
  def quiet(self, val):
    self._quiet = val

  @property
  def debug(self):
    return self._debug

  @debug.setter
  def debug(self, val):
    self._debug = val

  @property
  def data_factory(self):
    return self._data_factory

  @data_factory.setter
  def data_factory(self, val):
    self._data_factory = val

  @property
  def task_factory(self):
    return self._task_factory

  @task_factory.setter
  def task_factory(self, val):
    self._task_factory = val

  @property
  def main_folder(self):
    return self._main_folder

  @main_folder.setter
  def main_folder(self, val):
    self._main_folder = val

  @property
  def experiment_uuid(self):
    return self._experiment_uuid

  @experiment_uuid.setter
  def experiment_uuid(self, val):
    self._experiment_uuid = val

  def __prepare_header(self, api_token, name, param, **kwargs):
    # 1.step 获取基本配置信息
    # parse config file
    if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
      # use default config
      if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
        os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

      shutil.copy(os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-1]), 'config.xml'),
                  os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'))

    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)

    factory = getattr(Config, 'factory', None)
    if factory is None or \
        factory == '' or not os.path.exists(Config.factory):
      # 生成默认配置文件
      shutil.copy(os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-1]), 'config.xml'), "./config.xml")

      # 提示信息
      logger.info('missing config info, please antgo config --config=config.xml with root authority')
      raise IOError

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
    try:
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

      # 日志退出
      mlogger.update()
      mlogger.exit()
      logger.info('finish training process')
    except:
      mlogger.error()
      logger.info('fail training process')

  @contextmanager
  def Challenge(self, api_token, exp_name, exp_param, benchmark='', **kwargs):
    try:
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

      # 日志退出
      mlogger.update()
      mlogger.exit()
      logger.info('finish challenge process')
    except:
      mlogger.error()

  @contextmanager
  def Demo(self, api_token, exp_name, exp_param, html_template=None, ip=None, port=None, **kwargs):
    try:
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

      # 日志退出
      mlogger.update()
      mlogger.exit()
      logger.info('finish demo process')
    except:
      mlogger.error()

  @contextmanager
  def Browser(self, api_token, exp_name, exp_param, ip=None, port=10000, **kwargs):
    try:
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
      mlogger.update()
      mlogger.exit()
      logger.info('finish browser process')
    except:
      mlogger.error()

  @contextmanager
  def Activelearning(self):
    yield

  @contextmanager
  def Generate(self):
    yield

  @contextmanager
  def Batch(self, api_token, exp_name, exp_param, ip=None, port=10000, unlabel=False, **kwargs):
    try:
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
      mlogger.update()
      mlogger.exit()
      logger.info('finish batch process')
    except:
      mlogger.error()