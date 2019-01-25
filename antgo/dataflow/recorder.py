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
import scipy.misc
import imageio
import requests
from antgo.context import *
from antgo.utils import *


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

  def _transfer_data(self, result, key):
      transfer_result = None
      transfer_result_type = None

      if result['%s_TYPE'%key] in ['SCALAR','FILE', 'STRING']:
        transfer_result = str(result[key])
        transfer_result_type = result['%s_TYPE'%key]
      elif result['%s_TYPE'%key] == 'JSON':
        transfer_result = json.dumps(result[key])
        transfer_result_type = 'STRING'
      elif result['%s_TYPE'%key] == 'IMAGE':
        data = result[key]
        assert(len(data.shape) <= 3 and len(data.shape) >= 2)

        if len(data.shape) == 2:
          if data.dtype == np.uint8:
            transfer_result = data
          else:
            data_min = np.min(data)
            data_max = np.max(data)
            transfer_result = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
          assert(data.shape[2] == 3)
          transfer_result = data.astype(np.uint8)

        # save path
        if not os.path.exists(self.dump_dir):
          os.makedirs(self.dump_dir)

        image_path = os.path.join(self.dump_dir, '%s.png'%str(uuid.uuid4()))
        scipy.misc.imsave(image_path, transfer_result)
        transfer_result = image_path
        transfer_result_type = 'IMAGE'
      elif result['%s_TYPE'%key] == 'VIDEO':
        data = result[key]
        assert(type(data) == list)

        # save path
        video_path = os.path.join(self.dump_dir, '%s.mp4'%str(uuid.uuid4()))
        writer = imageio.get_writer(video_path, fps=30)
        for im in data:
          writer.append_data(im.astype(np.uint8))
        writer.close()

        transfer_result = video_path
        transfer_result_type = 'VIDEO'
      else:
        logger.error('AUDIO not support')

      return transfer_result, transfer_result_type


  def record(self, val, **kwargs):
    results = []
    results_label = []
    if type(val) == list or type(val) == tuple:
      for aa in val:
        if 'RESULT' in aa and 'RESULT_TYPE' in aa:
          results.append({'RESULT': aa['RESULT'], 'RESULT_TYPE': aa['RESULT_TYPE']})
          aa.pop('RESULT')
          aa.pop('RESULT_TYPE')

        results_label.append(aa)
    else:
      if type(val) == dict:
        if 'RESULT' in val and 'RESULT_TYPE' in val:
          results.append({'RESULT': val['RESULT'], 'RESULT_TYPE': val['RESULT_TYPE']})
          val.pop('RESULT')
          val.pop('RESULT_TYPE')

        results_label.append(val)
      else:
        results = [{'RESULT': val, 'RESULT_TYPE': 'IMAGE'}]

    for index, result in enumerate(results):
      gt = None
      if self._annotation_cache.qsize() > 0:
        gt = self._annotation_cache.get()

      if gt is None and not self._is_none:
        self._is_none = True

      # result 中保存模型输出的唯一结果
      # results_label 中保存其他附加结果
      # 均转换成文件或字符串形式输出 [{'TYPE': 'FILE', 'PATH': ''},
      #                           {'TYPE': 'JSON', 'CONTENT': ''},
      #                           {'TYPE': 'STRING', 'CONTENT': ''},
      #                           {'TYPE': 'IMAGE', 'PATH': ''},
      #                           {'TYPE': 'VIDEO', 'PATH': ''},
      #                           {'TYPE': 'AUDIO', 'PATH': ''}]

      # 1.step for main results
      assert(result['RESULT_TYPE'] in ['FILE', 'JSON', 'SCALAR', 'STRING', 'IMAGE', 'VIDEO', 'AUDIO'])
      transfer_result = None
      transfer_result_type = None
      try:
        transfer_result, transfer_result_type = self._transfer_data(result, 'RESULT')
      except:
        transfer_result = None
        transfer_result_type = None

      # 2.step for additional results
      transfer_additional_results = []
      for k,v in results_label[index].items():
        if '_TYPE' not in k:
          if '%s_TYPE'%k in results_label[index]:
            a = None
            b = None
            try:
              a, b = self._transfer_data({k: v, '%s_TYPE'%k: results_label[index]['%s_TYPE'%k]}, k)
            except:
              a = None
              b = None

            if a is not None and b is not None:
              transfer_additional_results.append({k: a, 'TYPE': b})

      if len(results_label) > 0:
        self.recorder_output_queue.put((None, ({'DATA': transfer_result,
                                                'TYPE': transfer_result_type},transfer_additional_results)))
      else:
        self.recorder_output_queue.put((None, {'DATA': transfer_result,
                                               'TYPE': transfer_result_type}))

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


class LocalRecorderNode(Node):
  def __init__(self, inputs):
    super(LocalRecorderNode, self).__init__(name=None, action=self.action, inputs=inputs, auto_trigger=True)
    self._annotation_cache = queue.Queue()

    self._dump_dir = None
    self._is_none = False

    self._count = 0
    setattr(self, 'model_fn', None)

  def save(self, data, data_type, name, count, index):
    # only support 'FILE', 'JSON', 'STRING', 'IMAGE', 'VIDEO', 'AUDIO'
    assert(data_type in ['FILE', 'JSON', 'SCALAR', 'STRING', 'IMAGE', 'VIDEO', 'AUDIO'])

    if data_type == 'FILE':
      if not os.path.exists(data):
        return

      norm_path = os.path.normpath(data)
      shutil.copy(data, os.path.join(self.dump_dir, '%d_%d_%s_%s'%(count, index,name,norm_path.split('/')[-1])))
    elif data_type == 'JSON':
      with open(os.path.join(self.dump_dir, '%d_%d_%s.json'%(count, index,name)),'w') as fp:
        fp.write(json.dumps(data))
    elif data_type == 'STRING' or data_type == 'SCALAR':
      with open(os.path.join(self.dump_dir, '%d_%d_%s.txt'%(count, index,name)), 'w') as fp:
        fp.write(str(data))
    elif data_type == 'IMAGE':
      transfer_result = data
      if len(data.shape) == 2:
        if data.dtype == np.uint8:
          transfer_result = data
        else:
          data_min = np.min(data)
          data_max = np.max(data)
          transfer_result = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
      else:
        assert (data.shape[2] == 3)
        transfer_result = data.astype(np.uint8)

      scipy.misc.imsave(os.path.join(self.dump_dir, '%d_%d_%s.png'%(count, index, name)), transfer_result)
    elif data_type == 'VIDEO':
      video_path = os.path.join(self.dump_dir, '%d_%d_%s.mp4' % (count, index, name))
      writer = imageio.get_writer(video_path, fps=30)
      for im in data:
        writer.append_data(im.astype(np.uint8))
      writer.close()
    elif data_type == 'AUDIO':
      logger.error('now, dont support save audio')

  def action(self, *args, **kwargs):
    value = copy.deepcopy(args[0])
    if type(value) != list:
      value = [value]

    for entry in value:
      self._annotation_cache.put(copy.deepcopy(entry))

  def record(self, val, **kwargs):
    results = val
    if type(val) != list or type(val) != tuple:
      results = [val]

    for index, result in enumerate(results):
      gt = None
      if self._annotation_cache.qsize() > 0:
        gt = self._annotation_cache.get()

      if gt is None and not self._is_none:
        self._is_none = True

      # 均转换成文件或字符串形式输出 [{'TYPE': 'FILE', 'PATH': ''},
      #                           {'TYPE': 'JSON', 'CONTENT': ''},
      #                           {'TYPE': 'STRING', 'CONTENT': ''},
      #                           {'TYPE': 'IMAGE', 'PATH': ''},
      #                           {'TYPE': 'VIDEO', 'PATH': ''},
      #                           {'TYPE': 'AUDIO', 'PATH': ''}]

      for k, v in result.items():
        if not k.endswith('_TYPE'):
          if '%s_TYPE' % k in result:
            data = v
            data_type = result['%s_TYPE' % k]

            self.save(data, data_type, k, self._count, index)

    self._count += 1

  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, val):
    self._dump_dir = val


class EmptyRecorderNode(Node):
  def __init__(self):
    self._dump_dir = ''
    setattr(self, 'model_fn', None)

  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, val):
    self._dump_dir = val

  def record(self, val, **kwargs):
    pass


class EvaluationRecorderNode():
  def __init__(self, measure, proxy=None, signature=None):
    self.proxy = proxy
    self.signature = signature
    self.measure = measure
    self.record_cache = []
    self.ctx = get_global_context()
    self._dump_dir = ''

  @property
  def model_fn(self):
    return None

  @property
  def dump_dir(self):
    return self._dump_dir

  @dump_dir.setter
  def dump_dir(self, val):
    self._dump_dir = val

  def record(self, *args, **kwargs):
    # 1.step record all result and concat its groundtruth
    predict, gt = args
    self.record_cache.append((predict, gt))

  def finish(self):
    # 2.step evaluation measure value
    if self.measure is None:
      return 10000.0

    result = self.measure.eva(self.record_cache, None)
    if self.proxy is not None:
      try:
        url = 'http://%s/update/model/%s/'%(self.proxy, self.ctx.name)
        requests.post(url,
                      data={'signature': self.signature,
                            'evaluation_value': result['statistic']['value'][0]['value']})
      except:
        logger.error('couldnt push evaluation info to proxy')

    return result['statistic']['value'][0]['value']


class ActiveLearningRecorderNode(Node):
  def __init__(self, inputs):
    super(ActiveLearningRecorderNode, self).__init__(name=None,
                                                     action=self.action,
                                                     inputs=inputs,
                                                     auto_trigger=True)
    self._annotation_cache = queue.Queue()
    self._data_set = []

  @property
  def dump_dir(self):
    return None

  def action(self, *args, **kwargs):
    a = copy.deepcopy(args[0])
    if type(a) != list:
      a = [a]

    for aa in a:
      self._annotation_cache.put(aa)

  def record(self, feature):
    binded_data = {}
    binded_data['feature'] = feature

    if self._annotation_cache.qsize() > 0:
      binded_data['data'] = self._annotation_cache.get()

    self._data_set.append(binded_data)

  def iterator_value(self):
    for data in self._data_set:
      yield data
