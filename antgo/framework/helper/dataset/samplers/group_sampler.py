# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from antgo.framework.helper.runner import get_dist_info
from torch.utils.data import Sampler
import copy


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, strategy=None):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        # strtegy 中保存样本数据源比例 ({0: 2, 1: 1} 表示flag=0和flag=1在一个batch中的个数)
        self.strategy = strategy
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

        self.source_indices = []
        self.source_num_per_batch = []
        if self.strategy is not None:
            for k,v in self.strategy.items():
                k = int(k)
                self.source_indices.append(np.where(self.flag == int(k))[0])
                self.source_num_per_batch.append(int(v))
            
            assert(np.sum(self.source_num_per_batch) == samples_per_gpu)
        self.iter_flag = 0

    def sampling_by_normal(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
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

    def sampling_by_strategy(self):
        source_indices = copy.deepcopy(self.source_indices)
        # for index in range(len(source_indices)):
        #     np.random.shuffle(source_indices[index])

        offset_in_source = [0 for _ in range(len(source_indices))]
        selected_indices = []
        for batch_i in range(self.num_samples // self.samples_per_gpu):
            # 逐数据源挑选数据
            selected_source_indices = [None for _ in range(len(source_indices))]
            for source_i in range(len(source_indices)):
                offset_in_source_i = offset_in_source[source_i]
                if offset_in_source_i + self.source_num_per_batch[source_i] <= len(source_indices[source_i]):
                    selected_source_indices[source_i] = source_indices[source_i][offset_in_source_i:offset_in_source_i+self.source_num_per_batch[source_i]]
                    offset_in_source[source_i] += self.source_num_per_batch[source_i]
                else:
                    num_extra = offset_in_source_i + self.source_num_per_batch[source_i] - len(source_indices[source_i])
                    source_indices_extra = source_indices[source_i][offset_in_source_i:]

                    if len(source_indices_extra) != 0:
                        selected_source_indices[source_i] = np.concatenate(
                                [source_indices_extra, np.random.choice(source_indices_extra, num_extra)])
                    else:
                        selected_source_indices[source_i] = np.random.choice(source_indices[source_i], num_extra)
                    
                    offset_in_source[source_i] = len(source_indices[source_i])

            # 合并
            selected_indices_in_batch = np.concatenate(selected_source_indices)
            selected_indices_in_batch = selected_indices_in_batch.astype(np.int64).tolist()

            # 扩展
            selected_indices.extend(selected_indices_in_batch)

        return iter(selected_indices)
    
    def __iter__(self):
        if self.strategy is None:
            return self.sampling_by_normal()
        else:
            return self.sampling_by_strategy()

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
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
                 seed=0,strategy=None):
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
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
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
