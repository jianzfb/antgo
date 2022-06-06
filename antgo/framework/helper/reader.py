from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import torch.utils.data
from antgo.framework.helper.utils import build_from_cfg
import torchvision.transforms as transforms
import numpy as np
import copy

class Reader(torch.utils.data.Dataset):
    def __init__(self, dataset, pipeline=None, inputs_def=None):
        self.proxy_dataset = dataset
        self.pipeline = []
        self.pipeline_types = []
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    self.pipeline_types.append(transform['type'])
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')

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
        sample = self.proxy_dataset.sample(idx)
        # transform
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
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