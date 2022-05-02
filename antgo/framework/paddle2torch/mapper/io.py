# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 12:28
# @File    : io.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
import torch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 feed_list=None,
                 places=None,
                 return_list=True,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 collate_fn=None,
                 num_workers=0,
                 use_buffer_reader=True,
                 use_shared_memory=True,
                 timeout=0,
                 worker_init_fn=None):
        assert(return_list)
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=None,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=False,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=None,
                         prefetch_factor=2,
                         persistent_workers=False)

