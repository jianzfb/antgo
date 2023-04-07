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


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = getattr(datasets[0], 'CLASSES', None)
        self.PALETTE = getattr(datasets[0], 'PALETTE', None)

        self.flag = []
        for index, dataset in enumerate(datasets):
            self.flag.append(np.ones(len(dataset), dtype=np.int64) * index)
        self.flag = np.concatenate(self.flag)

    def __getitem__(self, idx):
        if type(idx) == list:
            # 聚合到每个数据集
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
                sample_list.extend(self.datasets[dataset_idx][sample_idxs])
            
            return sample_list
        else:
            # 
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]

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

