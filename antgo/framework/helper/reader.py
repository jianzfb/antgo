from __future__ import division
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import logging
import os
import typing
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import copy
from antgo.framework.helper.utils import build_from_cfg


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
        self._alias = None
        if self._fields is not None and 'alias' in inputs_def:
            self._alias = copy.deepcopy(inputs_def['alias'])
        
        if self._fields is not None:
            if self._alias is None:
                self._alias = copy.deepcopy(self._fields)
                
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if hasattr(self.proxy_dataset, 'flag'):
            self.flag = self.proxy_dataset.flag

        self.CLASSES = 1
        if hasattr(self.proxy_dataset, 'CLASSES'):
            self.CLASSES = self.proxy_dataset.CLASSES

    def _arrange(self, sample, fields, alias):
        if fields is None:
            return sample      
          
        if type(fields[0]) == list or type(fields[0]) == tuple:
            warp_ins = []
            for alia, field in zip(alias, fields):
                one_ins = {}
                for aa, ff in zip(alia, field):
                    one_ins[aa] = sample[ff]
                
                warp_ins.append(one_ins)
            return warp_ins
        
        warp_ins = {}
        for alia, field in zip(alias, fields):
            warp_ins[alia] = sample[field]

        return warp_ins

    def __len__(self):
        return self.proxy_dataset.size

    def __getitem__(self, idx):
        sample = None
        try:
            sample = self.proxy_dataset.sample(idx)
            if sample is None:
                fail_count = 0
                while True:
                    # 随机选择，找到有效样本
                    random_i = np.random.randint(0,self.proxy_dataset.size)
                    sample = self.proxy_dataset.sample(random_i)
                    if sample is not None:
                        break

                    fail_count += 1
                    if fail_count > 10:
                        logging.warn(f'Fail find correct sample and exceed count {fail_count}.')
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
            weak_sample = self._arrange(weak_sample, self._fields, self._alias)
            strong_sample = self._arrange(strong_sample, self._fields, self._alias)
            return [weak_sample, strong_sample]
        else:    
            for transform in self.pipeline:
                sample = transform(sample)

            # arange warp
            sample = self._arrange(sample, self._fields, self._alias)
            return sample

    def get_cat_ids(self, idx):
        return self.proxy_dataset.get_cat_ids(idx)

    def get_ann_info(self, idx):
        return self.proxy_dataset.get_ann_info(idx)

    def evaluate(self, preds,**kwargs):
        return self.proxy_dataset.evaluate(preds, **kwargs)


class TVReader(torch.utils.data.Dataset):
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
        self._alias = None
        if self._fields is not None and 'alias' in inputs_def:
            self._alias = copy.deepcopy(inputs_def['alias'])
        
        if self._fields is not None:
            if self._alias is None:
                self._alias = copy.deepcopy(self._fields)

        self.flag = np.zeros(len(self), dtype=np.uint8)
        if hasattr(self.proxy_dataset, 'flag'):
            self.flag = self.proxy_dataset.flag

        self.CLASSES = 1
        if hasattr(self.proxy_dataset, 'CLASSES'):
            self.CLASSES = self.proxy_dataset.CLASSES

    def _arrange(self, sample, fields, alias):
        if fields is None:
            return sample      
          
        if type(fields[0]) == list or type(fields[0]) == tuple:
            warp_ins = []
            for alia, field in zip(alias, fields):
                one_ins = {}
                for aa, ff in zip(alia, field):
                    one_ins[aa] = sample[ff]
                
                warp_ins.append(one_ins)
            return warp_ins
        
        warp_ins = {}
        for alia, field in zip(alias, fields):
            warp_ins[alia] = sample[field]

        return warp_ins
    
    def __len__(self):
        return len(self.proxy_dataset)
    
    def __getitem__(self, idx):
        sample = None
        try:
            sample = self.proxy_dataset[idx]
            if sample is None:
                fail_count = 0
                while True:
                    # 随机选择，找到有效样本
                    random_i = np.random.randint(0,self.proxy_dataset.size)
                    sample = self.proxy_dataset.sample(random_i)
                    if sample is not None:
                        break

                    fail_count += 1
                    if fail_count > 10:
                        logging.warn(f'Fail find correct sample and exceed count {fail_count}.')            

            if isinstance(sample, tuple) or isinstance(sample, list):
                temp = {}
                for data, name in zip(sample, self._alias):
                    temp[name] = data
                sample = temp
            else:
                sample = {self._alias[0]: sample}
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
            weak_sample = self._arrange(weak_sample, self._fields, self._alias)
            strong_sample = self._arrange(strong_sample, self._fields, self._alias)
            return [weak_sample, strong_sample]
        else:
            for transform in self.pipeline:
                sample = transform(sample)

            # arange warp
            sample = self._arrange(sample, self._fields, self._alias)
            return sample
