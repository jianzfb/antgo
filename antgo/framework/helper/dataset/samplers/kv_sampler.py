import math

import numpy as np
import torch
from torch.utils.data import Sampler
import copy


# 将 index iterable 转化为 batch index iterable
# 如 chunk([4, 2, 3, 1], 2) ==> [[4, 2], [3, 1]]
def _chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


class KVSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu=1, shuffle=False, drop_last=False, strategy=None) -> None:
        super().__init__(dataset)
        assert hasattr(dataset, 'flag')
        self.flag = dataset.flag.astype(np.int64)
        # strtegy 中保存样本数据源比例 ({0: 2, 1: 1} 表示flag=0和flag=1在一个batch中的个数)
        self.strategy = strategy

        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        self.drop_last = drop_last
        self.dataset = dataset

        self.label_and_unlabel_group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.label_and_unlabel_group_sizes):
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
        # 每次返回一个batch_size
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.num_samples]

        return _chunk(iter(indices), self.samples_per_gpu)        

    def sampling_by_strategy(self):
        source_indices = copy.deepcopy(self.source_indices)
        if self.shuffle:
            for index in range(len(source_indices)):
                np.random.shuffle(source_indices[index])

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

        return _chunk(iter(selected_indices), self.samples_per_gpu)     

    def __iter__(self):
        if self.strategy is None:
            return self.sampling_by_normal()
        else:
            return self.sampling_by_strategy()

    def __len__(self):
        return self.num_samples // self.samples_per_gpu


class DistributedKVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, 
                 dataset, 
                 samplers_per_gpu=1, 
                 num_replicas=1,
                 rank=None, 
                 shuffle=True,
                 drop_last=False,
                 seed=0, strategy=None) -> None:
        super().__init__(dataset, num_replicas= num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last, seed=seed)        
        self.batch_size = samplers_per_gpu

    def __iter__(self):
        iterable = super(DistributedKVSampler, self).__iter__()
        return _chunk(iterable, self.batch_size)

    def __len__(self):
        # num_samples 是当前rank卡下，分配的样本总数
        return (self.num_samples+self.batch_size-1)//self.batch_size
