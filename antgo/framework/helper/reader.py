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


def objregister(cls):
    class ProxyReader(ObjDetReader):
        def __init__(
            self, 
            train_or_test, 
            dir='', 
            params={},
            pipeline = None, 
            inputs_def=None, 
            enable_mixup=True, 
            enable_cutmix=True, 
            class_aware_sampling=False, 
            num_classes=1, 
            cache_size=0, 
            file_loader=None, **kwargs):
            super().__init__(
                cls(train_or_test, dir, params),
                pipeline=pipeline,
                inputs_def=inputs_def,
            )
    from antgo.framework.helper.dataset.builder import DATASETS
    DATASETS.register_module(name=cls.__name__)(ProxyReader)   
    return ProxyReader



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
        sample = None
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

class ObjDetReader(Reader):
    def __init__(
        self, 
        dataset, 
        pipeline, 
        inputs_def, 
        enable_mixup=True, 
        enable_cutmix=True, 
        class_aware_sampling=False, 
        num_classes=1, 
        cache_size=0, 
        file_loader=None):
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
                    random_i = np.random.randint(0, len(self.cache_cutmix))
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
        
        if self._fields is None:
            return sample

        # arange warp
        sample = self._arrange(sample, self._fields)
        return sample

class SemiReader(torch.utils.data.Dataset):
    def __init__(
        self, 
        reader_cls, 
        dataset,         
        pipeline_teacher, 
        pipeline_student, 
        inputs_def, 
        strategy=None, 
        pipeline=None,
        **kwargs):
        self.reader_base = reader_cls(dataset, None, None, **kwargs)
        self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None

        # 读取策略 
        # {'labeled': 1, 'unlabeled': 1}
        self.strategy = strategy

        self.teacher_pipeline_types = []
        self.teacher_pipeline = []
        if pipeline_teacher is not None:
            from antgo.framework.helper.dataset import PIPELINES
            self.teacher_pipeline = [None for _ in range(len(pipeline_teacher))]
            self.teacher_pipeline_types =  [None for _ in range(len(pipeline_teacher))]

            for teacher_i in range(len(pipeline_teacher)):
                tt = []
                tt_types = []
                for transform in pipeline_teacher[teacher_i]:
                    if isinstance(transform, dict):
                        tt_types.append(transform['type'])
                        transform = build_from_cfg(transform, PIPELINES)
                        tt.append(transform)
                    else:
                        raise TypeError('teacher pipeline must be a dict')
                
                self.teacher_pipeline[teacher_i] = tt
                self.teacher_pipeline_types[teacher_i] = tt_types

        self.student_pipeline_types = []
        self.student_pipeline = [] 
        if pipeline_student is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline_student:
                if isinstance(transform, dict):
                    self.student_pipeline_types.append(transform['type'])
                    transform = build_from_cfg(transform, PIPELINES)
                    self.student_pipeline.append(transform)
                else:
                    raise TypeError('student pipeline must be a dict')
        
        self.pipeline_types = []
        self.pipeline = []
        if pipeline is not None:
            for transform in pipeline:
                if isinstance(transform, dict):
                    self.pipeline_types.append(transform['type'])
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')          
        
        self.flag = np.zeros(len(self), dtype=np.int32)
        if hasattr(self.reader_base, 'flag'):
            self.flag = self.reader_base.flag

        self.CLASSES = 1
        if hasattr(self.reader_base, 'CLASSES'):
            self.CLASSES = self.reader_base.CLASSES

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
        return len(self.reader_base)
    
    def __getitem__(self, idx):
        sample = self.reader_base[idx]

        # 共用管线处理
        processed_sample = copy.deepcopy(sample)
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            try:
                processed_sample = transform(processed_sample)
            except Exception as e:
                print(f'{transform_type}')
                print(e)
                raise e

        # 为teacher构建数据
        teacher_samples = []
        for pipeline_i in range(len(self.teacher_pipeline)):
            teacher_sample = copy.deepcopy(processed_sample)
            for (transform, transform_type) in zip(self.teacher_pipeline[pipeline_i], self.teacher_pipeline_types[pipeline_i]):
                try:
                    teacher_sample = transform(teacher_sample)
                except Exception as e:
                    print(f'{transform_type}')
                    print(e)
                    raise e
            
            # arange warp
            teacher_sample = self._arrange(teacher_sample, self._fields)
            teacher_samples.append(teacher_sample)

        # 为student构建数据
        student_sample = copy.deepcopy(processed_sample)
        for (transform, transform_type) in zip(self.student_pipeline, self.student_pipeline_types):
            try:
                student_sample = transform(student_sample)
            except Exception as e:
                print(f'{transform_type}')
                print(e)
                raise e

        # arange warp
        student_sample = self._arrange(student_sample, self._fields)

        semi_sample = {
            'image': student_sample['image'],
            'image_metas': student_sample['image_metas'],
        }
        semi_sample['image_metas'].update({
            'fields': self._fields
        })

        if sample['image_metas']['labeled']:
            semi_sample.update({
                'labeled': student_sample
            })

        if not sample['image_metas']['labeled']:
            semi_sample.update({
                'unlabeled': {
                    'teacher': teacher_samples,
                    'student': student_sample
                }
            })
        
        return semi_sample
    
    def get_cat_ids(self, idx):
        return self.reader_base.get_cat_ids(idx)

    def get_ann_info(self, idx):
        return self.reader_base.get_ann_info(idx)

    def evaluate(self, preds,**kwargs):
        return self.reader_base.evaluate(preds, **kwargs)
