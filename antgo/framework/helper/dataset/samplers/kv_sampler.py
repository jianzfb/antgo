import math

import numpy as np
import torch
from torch.utils.data import Sampler


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
    def __init__(self, dataset, samples_per_gpu=1, shuffle=False, drop_last=False) -> None:
        super().__init__(dataset)
        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        self.num_samples = int(np.ceil(
                len(dataset) / self.samples_per_gpu)) * self.samples_per_gpu
        self.dataset = dataset
        self.drop_last = drop_last

    def __iter__(self):
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
                 seed=0) -> None:
        print(f'num_replicas {num_replicas}')
        super().__init__(dataset, num_replicas= num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last, seed=seed)        
        self.batch_size = samplers_per_gpu

    def __iter__(self):
        iterable = super(DistributedKVSampler, self).__iter__()
        return _chunk(iterable, self.batch_size)

    def __len__(self):
        # num_samples 是当前rank卡下，分配的样本总数
        return (self.num_samples+self.batch_size-1)//self.batch_size
