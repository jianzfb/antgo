from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
from antgo.job import *
from antgo.dataflow.dataset import *


class Params(object):
  def __init__(self, params=None):
    if params is not None:
      for k, v in params.items():
        if k != 'self':
          setattr(self, k, v)

    self._params = params

  def define(self, k, v=None):
    setattr(self, k, v)
    self._params[k] = v

  @property
  def content(self):
    return self._params

global_context = None


def get_global_context():
  global global_context
  assert(global_context is not None)
  return global_context


class Block(object):
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
    self.dataset_factory_callback = AntDataset
    self.running_recorder = None
    self.context_params = None
    
    self.pid = str(os.getpid())
    
    global_context = self
    self.job = Job(self)
    self.job.start()

    self.context_ant = None
    self.context_stage = ""

    self.trainer_callbacks = []
    self.clear_callbacks = []
    self.init_callbacks = []
  
    self.data_source = None
    self._blocks = []
    self._blocks_status = {}
    self._stoppable_threads = []
    
    self._from_experiment = None

  def wait_until_clear(self):
    if self.job is not None:
      self.job.stop()
      self.job.join()
      self.job = None
    
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
    
    self.dataset_factory_callback = AntDataset
    self.running_recorder = None
    self.context_params = None
    self.context_ant = None
    self.context_stage = ""
    self.trainer_callbacks = []
    self._data_source = None

  @property
  def job(self):
    return self.context_job
  @job.setter
  def job(self, val):
    self.context_job = val
  
  @property
  def ant(self):
    return self.context_ant

  @ant.setter
  def ant(self, val):
    self.context_ant = val

  @property
  def stage(self):
    return self.context_stage

  @stage.setter
  def stage(self, val):
    self.context_stage = val

  def send(self, data, stage):
    if self.ant is not None:
      self.ant.send(data, stage)

  @property
  def training_process(self):
    return self.training_process_callback

  @training_process.setter
  def training_process(self, callback):
    self.training_process_callback = callback

  def call_training_process(self, data_source, dump_dir):
    if self.recorder is not None and self.recorder.dump_dir == None:
      self.recorder.dump_dir = dump_dir
    
    self.data_source = data_source
    self.training_process(data_source, dump_dir)

    # clone charts
    self.job.clone_charts()

  @property
  def infer_process(self):
    return self.infer_process_callback

  @infer_process.setter
  def infer_process(self, callback):
    self.infer_process_callback = callback

  def call_infer_process(self, data_source, dump_dir):
    if self.recorder is not None and self.recorder.dump_dir == None:
      self.recorder.dump_dir = dump_dir
      
    self.data_source = data_source
    self.infer_process(data_source, dump_dir)

    # clone charts
    self.job.clone_charts()
  
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

  @property
  def dataset_factory(self):
    return self.dataset_factory_callback

  def registry_trainer_callback(self, key, value, condition, func):
    # condition: equal, less, greater or mod
    self.trainer_callbacks.append((key, value, condition, func))

  @property
  def registried_trainer_callbacks(self):
    return self.trainer_callbacks

  def registry_clear_callback(self, func):
    self.clear_callbacks.append(func)
  
  def registry_init_callback(self, func):
    self.init_callbacks.append(func)
  @property
  def registried_init_callbacks(self):
    return self.init_callbacks
  
  @property
  def data_source(self):
    return self._data_source
  
  @data_source.setter
  def data_source(self, val):
    self._data_source = val
  
  def register_stoppable_thread(self, stoppable_thread):
    self._stoppable_threads.append(stoppable_thread)
  
  def block(self, name):
    model_block = Block(name)
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
    self._blocks_status = {}
    self._blocks_status[name] = False
  
  @property
  def from_experiment(self):
    return self._from_experiment

  @from_experiment.setter
  def from_experiment(self, experiment):
    self._from_experiment = experiment