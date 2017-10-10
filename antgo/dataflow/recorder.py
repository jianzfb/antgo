# encoding=utf-8
# @Time    : 17-6-13
# @File    : recorder.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.core import *
import copy
import numpy as np
import os
import json
from antgo.task.task import *
from antgo.dataflow.basic import *


class RecorderNode(Node):
    def __init__(self, inputs):
        super(RecorderNode, self).__init__(name=None, action=self.action, inputs=inputs, auto_trigger=True)
        self._dump_dir = None
        self._annotation_cache = []
        self._record_writer = None

    def close(self):
        if self._record_writer is not None:
            self._record_writer.close()
        self._record_writer = None
        
    @property
    def dump_dir(self):
        return self._dump_dir

    @dump_dir.setter
    def dump_dir(self, val):
        self._dump_dir = val
        self._record_writer = RecordWriter(self._dump_dir)

    def action(self, *args, **kwargs):
        assert(len(self._annotation_cache) == 0)
        assert(len(self._positional_inputs) == 1)

        value = copy.deepcopy(args[0])
        if type(value) != list:
            value = [value]
        
        for entry in value:
            cache = copy.deepcopy(entry)
            self._annotation_cache.append(cache)

    def record(self, val):
        if type(val) == list:
            if len(self._annotation_cache) > 0:
                assert(len(self._annotation_cache) == len(val))
            results = val
        else:
            if len(self._annotation_cache) > 0:
                assert(len(self._annotation_cache) == 1)
            results = [val]

        # proxy
        annotation_cache_proxy = self._annotation_cache if len(self._annotation_cache) > 0 else [None for _ in range(len(results))]

        # make a pair with annotation
        for anno, result in zip(annotation_cache_proxy, results):
            if result is None and anno is None:
                continue
            
            self._record_writer.write(Sample(groundtruth=anno, predict=result))

        self._annotation_cache[:] = []

    def iterator_value(self):
        pass
