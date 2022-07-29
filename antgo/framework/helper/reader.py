from __future__ import division
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import torch.utils.data
from antgo.framework.helper.utils import build_from_cfg
import torchvision.transforms as transforms
import numpy as np
import copy
import cv2

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
        try:
            sample = self.proxy_dataset.sample(idx)
        except:
            print(f'sample error {idx}')

        # transform
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            try:
                sample = transform(sample)
            except:
                print(f'transform error{transform_type}')

        # arange warp
        sample = self._arrange(sample, self._fields)
        return sample

    def get_cat_ids(self, idx):
        return self.proxy_dataset.get_cat_ids(idx)

    def get_ann_info(self, idx):
        return self.proxy_dataset.get_ann_info(idx)

    def evaluate(self, preds,**kwargs):
        return self.proxy_dataset.evaluate(preds, **kwargs)


class ObjDetReader(Reader):
    def __init__(self, dataset, pipeline, inputs_def, enable_mixup=True, enable_cutmix=True, class_aware_sampling=False, num_classes=1, cache_size=0, file_loader=None):
        super().__init__(dataset, pipeline=pipeline, inputs_def=inputs_def)
        # TODO, support class aware sampling
        self.class_aware_sampling = class_aware_sampling
        self.num_classes = num_classes
        self.cache_size = cache_size
        self.enable_mixup = enable_mixup
        self.enable_cutmix = enable_cutmix
        self.cache_mixup = []
        self.cache_cutmix = []
        self.cache_max_size = 10000
        self.file_loader = file_loader
        if self.file_loader is None:
            self.file_loader = cv2.imread

    def __getitem__(self, idx):
        sample = self.proxy_dataset.sample(idx)

        if self.cache_size > 0:
            # 基于特殊标记，将此样本缓存起来，用于混合使用
            if 'support_mixup' in sample:
                mixup_sample = copy.deepcopy(sample)
                if 'image' in mixup_sample:
                    mixup_sample.pop('image')

                if len(self.cache_mixup) > self.cache_max_size:
                    random_i = np.random.randint(0, len(self.cache_mixup))
                    self.cache_mixup[random_i] = mixup_sample
                else:
                    self.cache_mixup.append(mixup_sample)
            
            if 'support_cutmix' in sample:
                cutmix_sample = copy.deepcopy(sample)
                if 'image' in cutmix_sample:
                    cutmix_sample.pop('image')

                if len(self.cache_cutmix) > self.cache_max_size:
                    random_i = np.random.randint(0, len(self.cache_mixup))
                    self.cache_cutmix[random_i] = cutmix_sample
                else:
                    self.cache_cutmix.append(cutmix_sample)

        if self.enable_mixup:
            # 仅负责mixup的数据准备
            if self.cache_size <= 0:
                num = self.proxy_dataset.size
                mix_idx = np.random.randint(1, num)
                sample['mixup'] = self.proxy_dataset.sample(mix_idx)
            elif len(self.cache_mixup) > self.cache_size:
                mix_idx = np.random.randint(0, len(self.cache_mixup))
                sample['mixup'] = self.cache_mixup[mix_idx]
                if 'image_file' in sample['mixup']:
                    sample['mixup']['image'] = self.file_loader(sample['mixup']['image_file'])
        
        if self.enable_cutmix:
            # 仅负责cutmix的数据准备
            if self.cache_size <= 0:
                num = self.proxy_dataset.size
                mix_idx = np.random.randint(1, num)
                sample['cutmix'] = self.proxy_dataset.sample(mix_idx)
            elif len(self.cache_cutmix) > self.cache_size:
                mix_idx = np.random.randint(0, len(self.cache_cutmix))
                sample['cutmix'] = self.cache_cutmix[mix_idx]
                if 'image_file' in sample['cutmix']:
                    sample['cutmix']['image'] = self.file_loader(sample['cutmix']['image_file'])

        # transform
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            try:
                sample = transform(sample)
            except Exception as e:
                print(f'{transform_type}')
                print(e)
                raise e
        
        # arange warp
        sample = self._arrange(sample, self._fields)
        return sample