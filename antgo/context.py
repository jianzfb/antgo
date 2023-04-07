from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
from antgo.dataflow.dataset import *
import antvis.client.mlogger as mlogger
import os


class Params(object):
  def __init__(self, params={}):
    assert(type(params) == dict)
    # if params is not None:
    #   for k, v in params.items():
    #     if k != 'self':
    #       setattr(self, k, v)
    # self.__dict__.update(params)
    self._params = params

  def define(self, k, v=None):
    # setattr(self, k, v)
    self._params[k] = v

  def get(self, item=None, default=None):
    if item is None:
      return self._params

    if item not in self._params:
      return default

    return self._params[item]

  def __getattr__(self, item):
    if item not in self._params:
      return None
    if type(self._params[item]) == dict:
      return Params(self._params[item])

    return self._params[item]

  def keys(self):
    return self._params.keys()

  def items(self):
    return self._params.items()


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


class _DataProcessor(object):
  def __init__(self):
    self.processor_sequence = []

  def add(self, *args):
    self.processor_sequence.extend(args)

  def iterator(self, source):
    upper = source
    for processor_cls in self.processor_sequence:
      upper = processor_cls(Node.inputs(upper))

    return upper.iterator_value()


class Context(object):
  def __init__(self, interact_mode=False):
    global global_context
    self.training_process_callback = None
    self.infer_process_callback = None
    self._data_processor = _DataProcessor()
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
    self._is_interact_mode = interact_mode

    # 注册用户数据
    self.register_obj = {}

  def is_interact_mode(self):
    return self._is_interact_mode

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

    self._data_processor = None

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

  def register(self, **kwargs):
    self.register_obj.update(kwargs)

  def register_at(self, key):
    return self.register_obj.get(key, None)

  @property
  def training_process(self):
    return self.training_process_callback

  @training_process.setter
  def training_process(self, callback):
    self.training_process_callback = callback

  def call_training_process(self, data_source, dump_dir):
    is_inner_set = False
    if self.recorder is not None and self.recorder.dump_dir == None:
      if dump_dir != '':
        if not os.path.exists(os.path.join(dump_dir, 'record')):
          os.makedirs(os.path.join(dump_dir, 'record'))

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
  def data_processor(self):
    return self._data_processor

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