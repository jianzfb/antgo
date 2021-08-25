# -*- coding: UTF-8 -*-
# @Time    : 17-12-14
# @File    : simpleimages.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import copy
import multiprocessing
import six
import sys
from antgo.utils.shared_queue import SharedQueue
import numpy as np


class EndSignal():
    pass


class MultiprocessReader(object):
    def __init__(self, dataset, 
                        transformers=None,
                        num_workers=4,
                        buffer_size=128,
                        batch_size=0,
                        drop_last=True,
                        daemon=True):
        super().__init__()

        self.queue = SharedQueue(buffer_size, memsize=1024**3)

        self.num_workers = num_workers
        self.transformers = transformers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.daemon = daemon
        
    def _read_into_queue(self, samples, dataset, queue):
        end = EndSignal()
        try:
            for sample in samples:
                if sample is None:
                    raise ValueError("Sample has None.s")
                data = dataset.at(sample)
                if self.transformers is not None:
                    for t in self.transformers:
                        data = t.action(data)
                queue.put(data)

            queue.put(end)
        except:
            queue.put("")
            six.reraise(*sys.exc_info())

    def iterator_value(self):
        total_samples = [[] for i in range(self.num_workers)]
        for i, sample_i in enumerate(range(self.dataset.size)):
            index = i % self.num_workers
            total_samples[index].append(sample_i)

        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=self._read_into_queue, args=(total_samples[i], self.dataset, self.queue))
            if self.daemon:
                p.daemon = self.daemon
            p.start()

        finish_num = 0
        batch_data = [],[]
        while finish_num < self.num_workers:
            sample = self.queue.get()
            if isinstance(sample, EndSignal):
                finish_num += 1
            elif sample == "":
                raise ValueError("Multiprocess reader raises an exception.")
            else:
                if self.batch_size == 0:
                    yield sample
                else:
                    batch_data[0].append(sample[0])
                    batch_data[1].append(sample[1])
                    if len(batch_data[0]) == self.batch_size:
                        a = np.stack(batch_data[0], 0)
                        yield a, batch_data[1]
                        batch_data = [],[]

        if self.batch_size > 0:
            if len(batch_data[0]) != 0 and not self.drop_last:
                a = np.stack(batch_data[0],0)
                yield a, batch_data[1]
                batch_data = [],[]
