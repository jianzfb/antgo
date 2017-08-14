from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from antgo.job import *
from antgo.dataflow.dataset import *


class Params(object):
    def __init__(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def define(self, k, v=None):
        setattr(self, k, v)

global_context = None


def get_global_context():
    global global_context
    return global_context


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

    def wait_until_clear(self):
        if self.context_job is not None:
            self.context_job.stop()
            self.context_job.join()
            self.context_job = None

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
        if self.recorder is not None:
            self.recorder.dump_dir = dump_dir
        self.training_process(data_source, dump_dir)

    @property
    def infer_process(self):
        return self.infer_process_callback

    @infer_process.setter
    def infer_process(self, callback):
        self.infer_process_callback = callback

    def call_infer_process(self, data_source, dump_dir):
        if self.recorder is not None:
            self.recorder.dump_dir = dump_dir
        self.infer_process(data_source, dump_dir)

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