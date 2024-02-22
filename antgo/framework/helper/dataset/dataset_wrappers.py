import bisect
import collections
import copy
import math
from collections import defaultdict
from xml.etree.ElementTree import iselement

import numpy as np
from antgo.framework.helper.utils import build_from_cfg, print_log
import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from .builder import DATASETS, PIPELINES


class InnerProxyDataset(object):
    def __init__(self, obj):
        self.obj = obj
    
    def __len__(self):
        return len(self.obj)

    def __getitem__(self, idx):
        return self.obj._get_data(idx)

@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets, pipeline=None, inputs_def=None):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = getattr(datasets[0], 'CLASSES', None)
        self.PALETTE = getattr(datasets[0], 'PALETTE', None)

        self.flag = []
        for index, dataset in enumerate(datasets):
            self.flag.append(np.ones(len(dataset), dtype=np.int64) * index)
        self.flag = np.concatenate(self.flag)

        self.pipeline = []
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')

        self._fields = None
        self._alias = None
        if inputs_def is not None:
            self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None
            self._alias = None
            if self._fields is not None and 'alias' in inputs_def:
                self._alias = copy.deepcopy(inputs_def['alias'])
            
            if self._fields is not None:
                if self._alias is None:
                    self._alias = copy.deepcopy(self._fields)

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

    def _get_data(self, idx):
        if type(idx) == list:
            d_i_map = {}
            for i in idx:
                if i < 0:
                    if -i > len(self):
                        raise ValueError("absolute value of index should not exceed dataset length")
                    i = len(self) + i

                dataset_idx = bisect.bisect_right(self.cumulative_sizes, i)
                if dataset_idx not in d_i_map:
                    d_i_map[dataset_idx] = []

                if dataset_idx == 0:
                    sample_idx = i
                else:
                    sample_idx = i - self.cumulative_sizes[dataset_idx - 1]
                                    
                d_i_map[dataset_idx].append(sample_idx)

            sample_list = []
            for dataset_idx, sample_idxs in d_i_map.items():
                sample = self.datasets[dataset_idx][sample_idxs]
                if 'dataset' in sample:
                    sample.pop('dataset')
                sample_list.extend(sample)
            return sample_list
        else:
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            sample = self.datasets[dataset_idx][sample_idx]
            if 'dataset' in sample:
                sample.pop('dataset')
            return sample

    def __getitem__(self, idx):
        sample_list = self._get_data(idx)
        if not isinstance(sample_list, list):
            sample_list = [sample_list]

        processed_sample_list = []
        for sample in sample_list:
            print(sample['bbox'].shape)
            # 使用pipeline处理样本
            sample['dataset'] = InnerProxyDataset(self)
            for tranform in self.pipeline:
                sample = tranform(sample)

            # 字段重命名
            if self._fields is not None:
                # arange warp
                sample = self._arrange(sample, self._fields, self._alias)

            processed_sample_list.append(sample)

        if isinstance(idx, list):
            return processed_sample_list
        return processed_sample_list[0]

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def get_ann_info(self, idx):
        """Get annotation of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_ann_info(sample_idx)

    def worker_init_fn(self, *args, **kwargs):
        for dataset in self.datasets:
            if getattr(dataset, 'worker_init_fn', None):
                dataset.worker_init_fn(*args, **kwargs)

    @property
    def is_kv(self):
        return getattr(self.datasets[0], 'is_kv', False)


@DATASETS.register_module()
class IterConcatDataset(torch.utils.data.ChainDataset):
    def __init__(self, datasets, samples_per_gpu=None) -> None:
        super().__init__(datasets)
        self.samples_per_gpu = samples_per_gpu

    def __iter__(self):
        dataset_iters = [iter(d) for d in self.datasets]

        while True:
            data_list = []
            is_exit = False
            for dataset, batch_size in zip(dataset_iters, self.samples_per_gpu):
                for _ in range(batch_size):
                    try:
                        data_list.append(next(dataset))
                        
                    except StopIteration:
                        is_exit = True
                
                if is_exit:
                    break
            if is_exit:
                break
            
            for data in data_list:
                yield data

    def __len__(self):
        total = 0
        for d in self.datasets:
            d_len = len(d)
            if total < d_len:
                total = d_len

        return total


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = getattr(dataset, 'CLASSES', None)
        self.PALETTE = getattr(dataset, 'PALETTE', None)
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def get_ann_info(self, idx):
        """Get annotation of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.dataset.get_ann_info(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len

