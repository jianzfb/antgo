from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset, build_kv_dataloader, build_iter_dataloader
from .dataset_wrappers import (ConcatDataset, IterConcatDataset, RepeatDataset)
from .pipelines import *
from .dataset_split import DatasetSamplingByClass
from .dataset_filter import IterableDatasetFilter
from .tfdataset import *
from antgo.dataflow import dataset as local_dataset
from antgo.framework.helper.reader import *
import torchvision


def register_antgo_dataset():
    for dataset_module_name in local_dataset.__all__:
        if dataset_module_name == 'Dataset':
            continue

        dataset_module_reader = \
            type(
                dataset_module_name, 
                (Reader,), 
                {   
                    'name': dataset_module_name,
                    '__doc__': f'{dataset_module_name} reader', 
                    '__init__': lambda self, pipeline=None, weak_pipeline=None, strong_pipeline=None, inputs_def=None, **kwargs: 
                        Reader.__init__(self, getattr(local_dataset, self.name)(**kwargs), pipeline=pipeline, weak_pipeline=weak_pipeline, strong_pipeline=strong_pipeline, inputs_def=inputs_def)
                }
            )
        DATASETS.register_module()(dataset_module_reader)


register_antgo_dataset()


def register_torchvision_dataset():
    needed_torchvision_dataset = ['ImageFolder']
    for dataset_module_name in needed_torchvision_dataset:
        if dataset_module_name.startswith('__'):
            continue

        dataset_module_reader = \
            type(
                dataset_module_name, 
                (TVReader,), 
                {   
                    'name': dataset_module_name,
                    '__doc__': f'{dataset_module_name} reader', 
                    '__init__': lambda self, pipeline=None, weak_pipeline=None, strong_pipeline=None, inputs_def=None, **kwargs: 
                        TVReader.__init__(self, getattr(torchvision.datasets, self.name)(**kwargs), pipeline=pipeline, weak_pipeline=weak_pipeline, strong_pipeline=strong_pipeline, inputs_def=inputs_def)
                }
            )
        DATASETS.register_module()(dataset_module_reader)


register_torchvision_dataset()


__all__ = [
    'DATASETS','build_dataloader','build_dataset','ConcatDataset', 'IterConcatDataset','RepeatDataset', 'TFDataset', 'DatasetSamplingByClass', 'IterableDatasetFilter'
]
