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

  def define(self, k, v=None):
    setattr(self, k, v)


global_context = None


def get_global_context():
  global global_context
  assert(global_context is not None)
  return global_context


class _Ablation(object):
  def __init__(self):
    self._is_disable = False

  def disable(self):
    self._is_disable = True
  def enable(self):
    self._is_disable = False

  @property
  def is_ablation(self):
    return self._is_disable

@contextmanager
def ablation(obj):
  yield obj


class _Ablations(object):
  def __init__(self):
    self._blocks = []
    self._ablations = {}

  @property
  def blocks(self):
    return self._blocks

  def disable(self, block):
    for k, _ in self._ablations.items():
      k.enable()
    self._ablations[block].disable()

  def create(self, block):
    if block not in self._blocks:
      self._blocks.append(block)
      self._ablations[block] = _Ablation()

    return ablation(self._ablations[block])


class Context(object):
  def __init__(self):
    global global_context
    assert(global_context == None)

    self.training_process_callback = None
    self.infer_process_callback = None
    self.dataset_factory_callback = AntDataset
    self.running_recorder = None
    self.context_params = None

    global_context = self
    self.context_job = Job(self)
    self.context_job.start()

    self.context_ant = None
    self.context_stage = ""

    self.trainer_callbacks = []
    self.clear_callbacks = []
  
    self.data_source = None
    self._ablation = _Ablations()
    
    self._stoppable_threads = []

  def wait_until_clear(self):
    if self.context_job is not None:
      self.context_job.stop()
      self.context_job.join()
      self.context_job = None
    
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

  @property
  def ant(self):
    return self.context_ant

  @ant.setter
  def ant(self,val):
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

  @dataset_factory.setter
  def dataset_factory(self, callback):
    self.dataset_factory_callback = callback

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

  @property
  def ablation(self):
    return self._ablation
  
  def register_stoppable_thread(self, stoppable_thread):
    self._stoppable_threads.append(stoppable_thread)