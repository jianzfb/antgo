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


class EndSignal():
    pass


class MultiprocessReader(object):
    def __init__(self, dataset,
                        num_workers=4,
                        buffer_size=128,
                        daemon=False):
        super().__init__()

        self.queue = SharedQueue(buffer_size, memsize=1024**3)
        total_samples = [[] for i in range(num_workers)]
        for i, sample_i in enumerate(range(dataset.size)):
            index = i % num_workers
            total_samples[index].append(sample_i)

        for i in range(num_workers):
            p = multiprocessing.Process(
                target=self._read_into_queue, args=(total_samples[i], dataset, self.queue))
            if daemon:
                p.daemon = daemon
            p.start()
        
        self.num_workers = num_workers

    def _read_into_queue(self, samples, dataset, queue):
        end = EndSignal()
        try:
            for sample in samples:
                if sample is None:
                    raise ValueError("Sample has None.s")
                queue.put(dataset.at(sample))

            queue.put(end)
        except:
            queue.put("")
            six.reraise(*sys.exc_info())

    def iterator_value(self):
        finish_num = 0
        while finish_num < self.num_workers:
            sample = self.queue.get()
            if isinstance(sample, EndSignal):
                finish_num += 1
            elif sample == "":
                raise ValueError("Multiprocess reader raises an exception.")
            else:
                yield sample
