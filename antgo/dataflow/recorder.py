# encoding=utf-8
# @Time    : 17-6-13
# @File    : recorder.py
# @Author  :
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
        if type(value) == dict or type(value) == np.ndarray:
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


# def load_records(dump_dir):
#     if os.path.exists(os.path.join(dump_dir, 'running-record.dat')):
#         result_records = []
#         annotation_records = []
#         type_records = []
#
#         fp = open(os.path.join(dump_dir, 'running-record.dat'),'r')
#         content = fp.readline()
#         while content:
#             json_obj = json.loads(content)
#             task_type = json_obj['task-type']
#             task_result = json_obj['task-result']
#
#             annotation = None
#             if 'annotation' in json_obj:
#                 annotation = json_obj['annotation']
#
#             result_records.append(task_result)
#             annotation_records.append(annotation)
#             type_records.append(task_type)
#
#             content = fp.readline()
#         fp.close()
#         return 'single', type_records, result_records, annotation_records
#     else:
#         multi_type_records = []
#         multi_result_records = []
#         multi_annotation_records = []
#         # traverse all subfolder
#         for ff in os.listdir(dump_dir):
#             if ff[0] == '.':
#                 continue
#
#             if os.path.isdir(os.path.join(dump_dir, ff)):
#                 if os.path.exists(os.path.join(dump_dir,'running-record.dat')):
#                     result_records = []
#                     annotation_records = []
#                     type_records = []
#
#                     fp = open(os.path.join(dump_dir, 'running-record.dat'), 'r')
#                     content = fp.readline()
#                     while content:
#                         json_obj = json.loads(content)
#                         task_type = json_obj['task-type']
#                         task_result = json_obj['task-result']
#
#                         annotation = None
#                         if 'annotation' in json_obj:
#                             annotation = json_obj['annotation']
#
#                         result_records.append(task_result)
#                         annotation_records.append(annotation)
#                         type_records.append(task_type)
#
#                         content = fp.readline()
#                     fp.close()
#
#                     multi_type_records.append(type_records)
#                     multi_result_records.append(result_records)
#                     multi_annotation_records.append(annotation_records)
#
#         return 'multi', multi_type_records, multi_result_records, multi_annotation_records
#

