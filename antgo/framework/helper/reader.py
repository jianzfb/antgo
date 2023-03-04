from __future__ import division
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import copy
import cv2
from antgo.framework.helper.utils import build_from_cfg
# from antgo.dataflow.dataset.tfrecord_dataset import *
from antgo.framework.helper.dataset.builder import DATASETS
# import tensorflow as tf


def register(cls):
    class ProxyReader(Reader):
        def __init__(self, train_or_test, dir='', params={}, pipeline=None, inputs_def=None, **kwargs):
            super().__init__(
                cls(train_or_test, dir, params),
                pipeline=pipeline,
                inputs_def=inputs_def,
            )
    from antgo.framework.helper.dataset.builder import DATASETS
    DATASETS.register_module(name=cls.__name__)(ProxyReader)   
    return ProxyReader


class KVReaderBase(torch.utils.data.Dataset):
    def __init__(self, pipeline=None, weak_pipeline=None, strong_pipeline=None) -> None:
        super().__init__()
        self.pipeline = []
        self.weak_pipeline = []
        self.strong_pipeline = []
        self.is_kv = True
        self.keys = []
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')

            if weak_pipeline is not None and strong_pipeline is not None:
                for transform in weak_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.weak_pipeline.append(transform)
                    else:
                        raise TypeError('weak_pipeline must be a dict')
                
                for transform in strong_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.strong_pipeline.append(transform)
                    else:
                        raise TypeError('strong_pipeline must be a dict')
    
    def __len__(self):
        return len(self.keys)

    def reads(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        assert(isinstance(index, list))
        sample_list = []
        for sample in self.reads(index):
            weak_sample = None
            strong_sample = None
            if len(self.weak_pipeline) > 0 or len(self.strong_pipeline) > 0:
                weak_sample = copy.deepcopy(sample)
                for transform in self.weak_pipeline:
                    weak_sample = transform(weak_sample)

                strong_sample = copy.deepcopy(weak_sample)
                for transform in self.strong_pipeline:
                    strong_sample = transform(strong_sample)

            if weak_sample is not None and strong_sample is not None:
                for transform in self.pipeline:
                    weak_sample = transform(weak_sample)

                for transform in self.pipeline:
                    strong_sample = transform(strong_sample)

                sample_list.append(weak_sample)
                sample_list.append(strong_sample)
            else:
                for transform in self.pipeline:
                    sample = transform(sample)
            
                sample_list.append(sample)
        return sample_list


class TFReaderBase(torch.utils.data.IterableDataset):
    def __init__(self, folder, fields2types, pipeline=None, **kwargs) -> None:
        super().__init__()
        self.dataset = TFRecordData(fields2types, folder)
        self.pipeline = []
        self.keys = []
        self.flag = np.zeros(len(self.dataset.tfrecord_dataset), dtype=np.uint8)
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')        
        
    def __iter__(self):
        # do something        
        for data in self.dataset.tfrecord_dataset:
            sample = tf.io.parse_single_example(data, self.dataset.get_features())
            sample = self.dataset.parse(sample)

            # transform
            for transform in self.pipeline:
                sample = transform(sample)

            yield sample
        # for data in range(10):
        #     sample = np.load('/opt/tiger/handdetJ/t.npy')
        #     for transform in self.pipeline:
        #         sample = transform(sample)

        #     yield sample


class Reader(torch.utils.data.Dataset):
    def __init__(self, dataset, pipeline=None, weak_pipeline=None, strong_pipeline=None, inputs_def=None):
        self.proxy_dataset = dataset
        self.pipeline = []
        self.pipeline_types = []
        self.weak_pipeline = []
        self.strong_pipeline = []        
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    self.pipeline_types.append(transform['type'])
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')


            if weak_pipeline is not None and strong_pipeline is not None:
                for transform in weak_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.weak_pipeline.append(transform)
                    else:
                        raise TypeError('weak_pipeline must be a dict')
                
                for transform in strong_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.strong_pipeline.append(transform)
                    else:
                        raise TypeError('strong_pipeline must be a dict')

        self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if hasattr(self.proxy_dataset, 'flag'):
            self.flag = self.proxy_dataset.flag

        self.CLASSES = 1
        if hasattr(self.proxy_dataset, 'CLASSES'):
            self.CLASSES = self.proxy_dataset.CLASSES

    def _arrange(self, sample, fields):
        if type(fields[0]) == list or type(fields[0]) == tuple:
            warp_ins = []
            for field in fields:
                one_ins = {}
                for ff in field:
                    one_ins[ff] = sample[ff]
                
                warp_ins.append(one_ins)
            return warp_ins
        
        warp_ins = {}
        for field in fields:
            warp_ins[field] = sample[field]

        return warp_ins

    def __len__(self):
        return self.proxy_dataset.size
    
    def __getitem__(self, idx):
        sample = None
        try:
            sample = self.proxy_dataset.sample(idx)
        except:
            print(f'sample error {idx}')

        weak_sample = None
        strong_sample = None
        if len(self.weak_pipeline) > 0 or len(self.strong_pipeline) > 0:
            weak_sample = copy.deepcopy(sample)
            for transform in self.weak_pipeline:
                weak_sample = transform(weak_sample)
            strong_sample = copy.deepcopy(weak_sample)
            for transform in self.strong_pipeline:
                strong_sample = transform(strong_sample)

        if weak_sample is not None and strong_sample is not None:
            for transform in self.pipeline:
                weak_sample = transform(weak_sample)

            for transform in self.pipeline:
                strong_sample = transform(strong_sample)

            # arange warp
            weak_sample = self._arrange(weak_sample, self._fields)
            strong_sample = self._arrange(strong_sample, self._fields)
            return [weak_sample, strong_sample]
        else:
            for transform in self.pipeline:
                sample = transform(sample)
            
            # arange warp
            sample = self._arrange(sample, self._fields)
            return sample

    def get_cat_ids(self, idx):
        return self.proxy_dataset.get_cat_ids(idx)

    def get_ann_info(self, idx):
        return self.proxy_dataset.get_ann_info(idx)

    def evaluate(self, preds,**kwargs):
        return self.proxy_dataset.evaluate(preds, **kwargs)

    def set_active_target_size(self, target_size):
        # 从pipeline中，发现resize处理器
        for processor, processor_type in zip(self.pipeline[::-1], self.pipeline_types[::-1]):
            if processor_type.startswith('Resize'):
                processor.reset(target_size)

    def get_active_target_size(self):
        # 从pipeline中，发现resize处理器
        target_size = None
        for processor, processor_type in zip(self.pipeline[::-1], self.pipeline_types[::-1]):
            if processor_type.startswith('Resize'):
                target_size = processor.get()
        
        return target_size
