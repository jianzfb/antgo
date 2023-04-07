import numpy as np
import torch
from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper.reader import *
from antgo.ant import environment
import multiprocessing as mp
import pickle


def get_keys(args):
    return environment.KVReader(*args).list_keys()


def worker_init_fn(path, dataset, _):
    dataset.reader = environment.KVReader(path, 2)


@DATASETS.register_module()
class KVDatasetReader(KVReaderBase):
    def __init__(self, data_path_list, pipeline=None, weak_pipeline=None, strong_pipeline=None) -> None:
        super().__init__(pipeline, weak_pipeline, strong_pipeline)
        self.data_path_list = data_path_list

        # TODO, 支持多数据组合

        # 读取所有可以
        with mp.Pool(1) as p:
            self.keys = p.map(get_keys, [(self.data_path_list[0], 1)])[0]

        self.worker_init_fn = lambda worker_id: worker_init_fn(self.data_path_list[0], self, worker_id)

    def reads(self, index):
        if not getattr(self, 'reader', None):
            self.worker_init_fn(0)
            
        # index 是一个list
        index = [self.keys[i] for i in index]
        data_bytes_list = self.reader.read_many(index)

        samples = []
        for data_bytes in data_bytes_list:
            data = pickle.loads(data_bytes)
            samples.append(data)
        
        return samples
