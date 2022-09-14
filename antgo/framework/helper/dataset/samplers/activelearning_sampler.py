# Copyright (c) OpenMMLab. All rights reserved.
import math
from os import unlink

import numpy as np
import torch
from antgo.framework.helper.runner import get_dist_info
from torch.utils.data import Sampler
import copy


class ActiveLearningSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        # flag 标记有标签和无标签
        # 0： 无标签
        # 1： 有标签
        assert hasattr(dataset, 'flag')
        
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.reset()
        self.iter_flag = 0

    def reset(self):
        self.flag = self.dataset.flag.astype(np.int64)
        self.label_and_unlabel_group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        labeled_s = int(
                math.ceil(self.label_and_unlabel_group_sizes[0] * 1.0 / self.samples_per_gpu))
        self.num_samples += labeled_s * self.samples_per_gpu
        # labeled: 0, unlabeled: 1
        self.labeled_indices = np.where(self.flag == 0)[0]
        # assert(len(self.labeled_indices) > 0)

    def __iter__(self):
        # 基于策略采集样本
        indices = []
        indice = np.where(self.flag == 0)[0]
        np.random.shuffle(indice)
        num_extra = int(np.ceil(self.label_and_unlabel_group_sizes[0] / self.samples_per_gpu)
                        ) * self.samples_per_gpu - len(indice)
        indice = np.concatenate(
            [indice, np.random.choice(indice, num_extra)])
        indices.append(indice)

        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedActiveLearningSampler(Sampler):
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
                 seed=0):
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
        
        self.reset()

    def reset(self):
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        self.num_samples += int(
            math.ceil(self.group_sizes[0] * 1.0 / self.samples_per_gpu /
                        self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        size = self.group_sizes[0]
        indice = np.where(self.flag == 0)[0]
        assert len(indice) == size
        # add .numpy() to avoid bug when selecting indice in parrots.
        # TODO: check whether torch.randperm() can be replaced by
        # numpy.random.permutation().
        indice = indice[list(
            torch.randperm(int(size), generator=g).numpy())].tolist()
        extra = int(
            math.ceil(
                size * 1.0 / self.samples_per_gpu / self.num_replicas)
        ) * self.samples_per_gpu * self.num_replicas - len(indice)
        # pad indice
        tmp = indice.copy()
        for _ in range(extra // size):
            indice.extend(tmp)
        indice.extend(tmp[:extra % size])
        indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
