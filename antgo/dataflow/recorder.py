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
try:
  import queue
except:
  import Queue as queue
import multiprocessing
from antgo.task.task import *
from antgo.dataflow.basic import *


class RecorderNode(Node):
  def __init__(self, inputs):
    super(RecorderNode, self).__init__(name=None, action=self.action, inputs=inputs, auto_trigger=True)
    self._dump_dir = None
    self._annotation_cache = queue.Queue()
    self._record_writer = None
    self._is_none = False
  
  @property
  def is_measure(self):
    if self._record_writer is None:
      return False
    
    if self._record_writer.size == 0:
      return False
    
    if self._is_none:
      return False
    
    return True

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
    
    if self._dump_dir is None:
      self.close()
      return
      
    # remove existed dump_dir
    if os.path.exists(self._dump_dir):
      shutil.rmtree(self._dump_dir)
    
    # mkdir
    os.makedirs(self._dump_dir)
    
    # set record workspace
    self._record_writer = RecordWriter(self._dump_dir)

  def action(self, *args, **kwargs):
    value = copy.deepcopy(args[0])
    if type(value) != list:
      value = [value]
    
    for entry in value:
      self._annotation_cache.put(copy.deepcopy(entry))

  def record(self, val, **kwargs):
    results = []
    results_label = []
    if type(val) == list or type(val) == tuple:
      for aa in val:
        if type(aa) == dict:
          results.append(aa['RESULT'])
          aa.pop('RESULT')
          results_label.append(aa)
        else:
          results.append(aa)
    else:
      if type(val) == dict:
        results.append(val['RESULT'])
        val.pop('RESULT')
        results_label.append(val)
      else:
        results = [val]
    
    for index, result in enumerate(results):
      gt = None
      if self._annotation_cache.qsize() > 0:
        gt = self._annotation_cache.get()
      
      if gt is None and not self._is_none:
        self._is_none = True
      
      if len(results_label) > 0:
        self._record_writer.write(Sample(groundtruth=gt, predict=result, predict_label=results_label[index]))
      else:
        self._record_writer.write(Sample(groundtruth=gt, predict=result, predict_label=[]))

  def iterator_value(self):
    pass
  
  @property
  def model_fn(self):
    return self._positional_inputs[0].model_fn


class QueueRecorderNode(Node):
  def __init__(self, inputs, output_queue):
    super(QueueRecorderNode, self).__init__(name=None, action=self.action, inputs=inputs,auto_trigger=True)

    self._annotation_cache = queue.Queue()

    self.recorder_output_queue = output_queue
    self._dump_dir = None
    self._is_none = False
    
    setattr(self,'model_fn', None)

  def record(self, val, **kwargs):
    results = []
    results_label = []
    if type(val) == list or type(val) == tuple:
      for aa in val:
        if type(aa) == dict:
          results.append(aa['RESULT'])
          aa.pop('RESULT')
          results_label.append(aa)
        else:
          results.append(aa)
    else:
      if type(val) == dict:
        results.append(val['RESULT'])
        val.pop('RESULT')
        results_label.append(val)
      else:
        results = [val]

    for index, result in enumerate(results):
      gt = None
      if self._annotation_cache.qsize() > 0:
        gt = self._annotation_cache.get()

      if gt is None and not self._is_none:
        self._is_none = True
      
      if len(results_label) > 0:
        self.recorder_output_queue.put((gt, (result, results_label[index])))
      else:
        self.recorder_output_queue.put((gt, result))

  def action(self, *args, **kwargs):
    value = copy.deepcopy(args[0])
    if type(value) != list:
      value = [value]

    for entry in value:
      self._annotation_cache.put(copy.deepcopy(entry))

  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, val):
    self._dump_dir = val

  @property
  def is_measure(self):
    if self._record_writer is None:
      return False

    if self._is_none:
      return False

    return True

  def iterator_value(self):
    pass