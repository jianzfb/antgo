# Copyright (c) OpenMMLab. All rights reserved.
import math
from os import unlink

import numpy as np
import torch
from antgo.framework.helper.runner import get_dist_info
from torch.utils.data import Sampler
import copy


class SemiSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, strategy={'labeled': 1, 'unlabeled': 1}):
        # flag 标记有标签和无标签
        # 0： 无标签
        # 1： 有标签
        assert hasattr(dataset, 'flag')
        
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.strategy = strategy
        assert(self.strategy is not None)
        assert('labeled' in self.strategy and 'unlabeled' in self.strategy)

        self.label_and_unlabel_group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        labeled_s = int(
                math.ceil(self.label_and_unlabel_group_sizes[0] * 1.0 / self.samples_per_gpu))
        unlabeled_s = int(
                math.ceil(self.label_and_unlabel_group_sizes[1] * 1.0 / self.samples_per_gpu))

        s = max(labeled_s//self.strategy['labeled'], unlabeled_s//self.strategy['unlabeled']) 
        labeled_s = s * self.strategy['labeled']
        unlabeled_s = s * self.strategy['unlabeled']

        self.num_samples += labeled_s * self.samples_per_gpu
        self.num_samples += unlabeled_s * self.samples_per_gpu
        # labeled: 0, unlabeled: 1
        self.labeled_indices = np.where(self.flag == 0)[0]
        self.unlabeled_indices = np.where(self.flag == 1)[0]
        assert(len(self.labeled_indices) > 0 and len(self.unlabeled_indices) > 0)
        self.iter_flag = 0

    def __iter__(self):
        # 基于策略采集样本
        # labeled -> unlabeled
        labeled_indices = self.labeled_indices.copy()
        unlabeled_indices = self.unlabeled_indices.copy()
        
        np.random.shuffle(labeled_indices)
        np.random.shuffle(unlabeled_indices)

        Z = self.strategy['labeled'] + self.strategy['unlabeled']  
        offset_labeled = 0
        offset_unlabeled = 0

        indices = []
        for batch_i in range(self.num_samples // self.samples_per_gpu):
            if batch_i % Z < self.strategy['labeled']:
                # 从有标签数据中采样
                if offset_labeled + self.samples_per_gpu <= len(labeled_indices):
                    indice = labeled_indices[offset_labeled:offset_labeled+self.samples_per_gpu]
                    indices.append(indice)
                    offset_labeled += self.samples_per_gpu
                else:
                    num_extra = offset_labeled + self.samples_per_gpu - len(labeled_indices)
                    indice = labeled_indices[offset_labeled:]

                    if len(indice) != 0:
                        indice = np.concatenate(
                                [indice, np.random.choice(indice, num_extra)])
                    else:
                        indice = np.random.choice(unlabeled_indices, num_extra)

                    indices.append(indice)
                    offset_labeled = len(labeled_indices)
            else:
                # 从无标签数据中采样
                if offset_unlabeled + self.samples_per_gpu <= len(unlabeled_indices):
                    indice = unlabeled_indices[offset_unlabeled:offset_unlabeled+self.samples_per_gpu]
                    indices.append(indice)
                    offset_unlabeled += self.samples_per_gpu
                else:
                    num_extra = offset_unlabeled + self.samples_per_gpu - len(unlabeled_indices)
                    indice = unlabeled_indices[offset_unlabeled:]

                    if len(indice) != 0:
                        indice = np.concatenate(
                                [indice, np.random.choice(indice, num_extra)])
                    else:
                        indice = np.random.choice(unlabeled_indices, num_extra)

                    indices.append(indice)         
                    offset_unlabeled = len(unlabeled_indices)       

        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedSemiSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0, strategy={'labeled': 1, 'unlabeled': 1}):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.strategy = strategy
        assert(self.strategy is not None)
        assert('labeled' in self.strategy and 'unlabeled' in self.strategy)

        self.label_and_unlabel_group_sizes = np.bincount(self.flag)
        self.num_samples = 0

        labeled_s = int(
                math.ceil(self.label_and_unlabel_group_sizes[0] * 1.0 / self.samples_per_gpu /
                          self.num_replicas))
        unlabeled_s = int(
                math.ceil(self.label_and_unlabel_group_sizes[1] * 1.0 / self.samples_per_gpu /
                          self.num_replicas))

        s = max(labeled_s//self.strategy['labeled'], unlabeled_s//self.strategy['unlabeled']) 
        labeled_s = s * self.strategy['labeled']
        unlabeled_s = s * self.strategy['unlabeled']

        self.num_samples += labeled_s * self.samples_per_gpu
        self.num_samples += unlabeled_s * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # labeled: 0, unlabeled: 1
        self.labeled_indices = np.where(self.flag == 0)[0]
        self.unlabeled_indices = np.where(self.flag == 1)[0]
        assert(len(self.labeled_indices) > 0 and len(self.unlabeled_indices) > 0)
        self.iter_flag = 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        labeled_indices = self.labeled_indices.copy()
        unlabeled_indices = self.unlabeled_indices.copy()

        labeled_indices = labeled_indices[list(
            torch.randperm(int(len(labeled_indices)), generator=g).numpy())].tolist()
        unlabeled_indices = unlabeled_indices[list(
            torch.randperm(int(len(unlabeled_indices)), generator=g).numpy())].tolist()
        
        Z = self.strategy['labeled'] + self.strategy['unlabeled']  
        offset_labeled = 0
        offset_unlabeled = 0

        indices = []
        for _ in range(self.num_replicas):
            for part_i in range(self.num_samples // self.samples_per_gpu):
                if part_i % Z < self.strategy['labeled']:
                    # 从有标签数据中采样
                    if offset_labeled + self.samples_per_gpu <= len(labeled_indices):
                        indice = labeled_indices[offset_labeled:offset_labeled+self.samples_per_gpu]
                        indices.append(indice)
                        offset_labeled += self.samples_per_gpu
                    else:
                        num_extra = offset_labeled + self.samples_per_gpu - len(labeled_indices)

                        indice = labeled_indices[offset_labeled:]
                        if len(indice) != 0:
                            indice = np.concatenate(
                                    [indice, np.random.choice(indice, num_extra)])
                        else:
                            indice = np.random.choice(unlabeled_indices, num_extra)

                        indices.append(indice)
                        offset_labeled = len(labeled_indices)
                else:
                    # 从无标签数据中采样
                    if offset_unlabeled + self.samples_per_gpu <= len(unlabeled_indices):
                        indice = unlabeled_indices[offset_unlabeled:offset_unlabeled+self.samples_per_gpu]
                        indices.append(indice)
                        offset_unlabeled += self.samples_per_gpu
                    else:
                        num_extra = offset_unlabeled + self.samples_per_gpu - len(unlabeled_indices)
                        indice = unlabeled_indices[offset_unlabeled:]
                        if len(indice) != 0:
                            indice = np.concatenate(
                                    [indice, np.random.choice(indice, num_extra)])
                        else:
                            indice = np.random.choice(unlabeled_indices, num_extra)

                        indices.append(indice)           
                        offset_unlabeled = len(unlabeled_indices)     

        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
