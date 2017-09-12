# encoding=utf-8
# @Time    : 17-6-7
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import copy
import numpy as np
import sys
from antgo.dataflow.core import *
from antgo.utils import get_rng
from antgo.utils.concurrency import *
from antgo.context import *
try:
    import queue
except:
    import Queue as queue


class BatchData(Node):
  class _FetchDataThread(StoppableProcess):
    def __init__(self, host_node, buffer_size):
      super(BatchData._FetchDataThread, self).__init__(buffer_size)
      self.daemon = True
      self._host_node = host_node
      self._is_launched = False

    @property
    def is_launched(self):
      return self._is_launched
    @is_launched.setter
    def is_launched(self, val):
      self._is_launched = val

    def run(self):
      while True:
        try:
          data = self._host_node._fetch_batch_data()
          self.process_queue.put(data)
          
          # force input update
          for i in self._host_node._positional_inputs:
            i._force_inputs_dirty()
          for name, i in items(self._host_node._keyword_inputs):
            i._force_inputs_dirty()
            
        except StopIteration:
          with self.process_condition:
            self.process_queue.put(DIE)
            self.process_condition.wait()
          if self.stopped():
            break
        except:
          info = sys.exc_info()
          logger.error('%s:%s' % (info[0], info[1]))
          exit(-1)
      
  def __init__(self, inputs, batch_size, remainder=False, buffer_size=0):
    super(BatchData, self).__init__(name=None, action=None, inputs=inputs)
    self.batch_size = batch_size
    self.remainder = remainder
    self.stop_iteration = False
    self.buffer = None
    if buffer_size > 0:
      self.producer_wait = False
      self.fetch_data_thread = BatchData._FetchDataThread(self, buffer_size)
      self.producer_condition = self.fetch_data_thread.process_condition
      self.buffer = self.fetch_data_thread.process_queue

      # register at context
      get_global_context().register_stoppable_thread(self.fetch_data_thread)
      
  def _fetch_batch_data(self):
    try:
      if self.stop_iteration:
        raise StopIteration
    
      batch_list = []
      label_list = []
    
      for _ in range(self.batch_size):
        input = self._positional_inputs[0]
        data = input.get_value()
        a = None
        b = None
        if type(data) == tuple or type(data) == list:
          a = data[0]
          b = data[1] if len(data) == 2 else data[1:]
        else:
          a = data
          b = None
        
        batch_list.append(a)
        label_list.append(b)
        if _ != self.batch_size - 1:
          input._force_inputs_dirty()
    
      batch = BatchData._aggregate_batch(batch_list)
      # clear reset state
      self._iteration_reset_state = False
      
      return batch, label_list
    except StopIteration:
      if self.stop_iteration or not self.remainder or len(batch_list) == 0:
        # reset all input
        self._reset_iteration_state()
        self.stop_iteration = False
        raise StopIteration
    
      if self.remainder:
        self.stop_iteration = True
        batch = BatchData._aggregate_batch(batch_list)
        return batch, label_list
    except:
      info = sys.exc_info()
      logger.error('%s:%s' % (info[0], info[1]))
      exit(-1)
  
  def _evaluate(self):
    if self.buffer is not None:
      if not self.fetch_data_thread.is_launched:
        self.fetch_data_thread.start()
        self.fetch_data_thread.is_launched = True

      if self.producer_wait:
        with self.producer_condition:
          self.producer_condition.notify_all()
        self.producer_wait = False
      
      data = self.buffer.get()
      if data == DIE:
        # no enough data as batch
        self.producer_wait = True
        raise StopIteration
      else:
        return data
    else:
      return self._fetch_batch_data()
    
  @staticmethod
  def _aggregate_batch(batch_list):
    batch = None
    max_shape = np.array([data.shape for data in batch_list]).max(axis=0)
    if len(max_shape) == 1:
      batch = np.zeros((len(batch_list), max_shape[0]), dtype=batch_list[0].dtype)
    elif len(max_shape) == 2:
      batch = np.zeros((len(batch_list), max_shape[0], max_shape[1]), dtype=batch_list[0].dtype)
    elif len(max_shape) == 3:
      batch = np.zeros((len(batch_list), max_shape[0], max_shape[1], max_shape[2]), dtype=batch_list[0].dtype)
    elif len(max_shape) == 4:
      batch = np.zeros((len(batch_list), max_shape[0], max_shape[1], max_shape[2], max_shape[3]),
          dtype=batch_list[0].dtype)

    assert (batch is not None)

    for i in range(len(batch_list)):
      batch[i] = batch_list[i]

    return batch


class RandomChooseData(Node):
  def __init__(self, inputs, propabilitys):
    super(RandomChooseData, self).__init__(name=None, action=None, inputs=inputs)
    self.input_list = []
    self.input_list.extend([i for i in self._positional_inputs])

    self.propability_list = tuple(propabilitys)
    self.rng = get_rng(self)

  def _evaluate(self):
    try:
      itr = self.rng.choice(self.input_list, p=self.propability_list)
      # clear reset state
      self._iteration_reset_state = False
      return itr.get_value()
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0],info[1]))
      exit(-1)


class RandomMixData(Node):
  def __init__(self, inputs):
    super(RandomMixData, self).__init__(name=None, action=None, inputs=inputs)
    self.rng = get_rng(self)
    self.input_list_flag = np.ones((len(self._positional_inputs)), dtype=np.uint8)

  def _evaluate(self):
    selected_i = -1
    try:
      ok_inputs = np.where(self.input_list_flag == 1)[0]
      if len(ok_inputs) == 0:
          raise StopIteration

      selected_i = self.rng.randint(low=0, high=len(ok_inputs), size=1)[0]
      selected_i = ok_inputs[selected_i]
      # clear reset state
      self._iteration_reset_state = False
      return self._positional_inputs[selected_i].get_value()
    except StopIteration:
      if selected_i >= 0:
        self.input_list_flag[selected_i] = 0

        while True:
          # select the next valid data input
          try_selected_i = -1
          try:
            maybe_ok_inputs = np.where(self.input_list_flag == 1)[0]
            if len(maybe_ok_inputs) == 0:
                raise StopIteration

            try_selected_i = self.rng.randint(low=0, high=len(maybe_ok_inputs), size=1)[0]
            try_selected_i = maybe_ok_inputs[try_selected_i]
            return self._positional_inputs[try_selected_i].get_value()
          except StopIteration:
            if try_selected_i >= 0:
                self.input_list_flag[try_selected_i] = 0
            else:
                break
          except:
            pass

      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0],info[1]))
      exit(-1)


class JoinData(Node):
  def __init__(self, inputs):
    super(JoinData, self).__init__(name=None, action=None, inputs=inputs)
    self.index = 0

  def _evaluate(self):
    try:
      try:
        self._iteration_reset_state = False
        return self._positional_inputs[self.index].get_value()
      except StopIteration:
        self.index = (self.index + 1) % len(self._positional_inputs)
        if self.index == 0:
          raise StopIteration
        self._iteration_reset_state = False
        return self._positional_inputs[self.index].get_value()
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0],info[1]))
      exit(-1)


class SerializedData(Node):
  def __init__(self, inputs):
    super(SerializedData, self).__init__(name=None, action=None, inputs=inputs)
    self._data_generator = None
    self._buffer_is_empty = False

  def _buffer_pool(self):
    all_values = []
    for i in self._positional_inputs:
      value = i.get_value()
      if type(value) != list:
        value = [value]

      all_values.extend(value)

    for name, i in items(self._keyword_inputs):
      value = i.get_value()
      if type(value) != list:
          value = [value]

      all_values.extend(value)

    self._buffer_is_empty = False
    for index in range(len(all_values)):
      yield all_values[index]

      if index == len(all_values) - 1:
        self._buffer_is_empty = True

  def _evaluate(self):
    try:
      if self._data_generator is None:
        self._data_generator = self._buffer_pool()
        self._buffer_is_empty = False

      # clear reset state
      self._iteration_reset_state = False

      return next(self._data_generator)
    except StopIteration:
      if self._buffer_is_empty:
        # force all inputs dirty
        for i in self._positional_inputs:
          i._force_inputs_dirty()
        for name, i in items(self._keyword_inputs):
          i._force_inputs_dirty()

        self._data_generator = self._buffer_pool()
        self._buffer_is_empty = False
        return next(self._data_generator)

      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0], info[1]))
      exit(-1)

  def _force_inputs_dirty(self):
    self._value = DIRTY


class _TransparantNode(Node):
  def __init__(self, upper_node, name, active_request=True):
    super(_TransparantNode, self).__init__(name=name, inputs=upper_node)
    assert(len(self._positional_inputs) == 1)
    self.active_request = active_request
    self._buffer = queue.Queue()

  def set_value(self, new_value):
    self._buffer.put(new_value)
    self._set_value(new_value)

  def get_value(self):
    if DIRTY == self._value:
      self._evaluate()

    if self._buffer.qsize() == 0:
      return self._value

    return self._buffer.get()

  def _evaluate(self):
    try:
      self._positional_inputs[0]._evaluate()

      # clear reset state
      self._iteration_reset_state = False
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s' % (info[0], info[1]))
      exit(-1)

  def _force_inputs_dirty(self):
    if self.active_request:
      if self._buffer.qsize() > 0:
        return
      
      if DIRTY == self._value:
        return
      
      for i in self._positional_inputs:
        i._force_inputs_dirty()
      for name, i in items(self._keyword_inputs):
        i._force_inputs_dirty()
      self._value = DIRTY


class DataAnnotationBranch(Node):
  def __init__(self, inputs):
    super(DataAnnotationBranch, self).__init__(name=None, action=self.action, inputs=inputs)
    self.annotation_branch = _TransparantNode(upper_node=Node.inputs(self), name='annotation')

  def action(self, *args, **kwargs):
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    self.annotation_branch.set_value(annotation)
    return image

  def output(self, index=0):
    if index == 0:
      return self
    else:
      return self.annotation_branch


class AutoBranch(Node):
  def __init__(self, inputs, branch_func):
    super(AutoBranch, self).__init__(name=None, action=self.action, inputs=inputs)
    self.auto_branch = _TransparantNode(upper_node=Node.inputs(self), name='auto_branch_transparant',active_request=False)
    self.branch_func = branch_func
    
  def action(self, *args, **kwargs):
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    branch_data = self.branch_func(image, annotation)
    self.auto_branch.set_value(branch_data)

    return image, annotation

  def output(self, index=0):
    if index == 0:
      return self
    else:
      return self.auto_branch
    
    
class SplitMultiBranches(Node):
  def __init__(self, inputs, branches_num=2):
    super(SplitMultiBranches, self).__init__(name=None, action=None, inputs=inputs)
    assert(len(self._positional_inputs) == 0)

    self.branches = [_TransparantNode(upper_node=Node.inputs(self), name='%s_branch_%d'%(self.name, _),
                                      active_request=False) for _ in range(branches_num)]

  def _evaluate(self):
    try:
      for branch in self.branches:
        if branch.is_dirty():
          branch.set_value(self._positional_inputs[0].get_value())
          self._positional_inputs[0]._force_inputs_dirty()
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s' % (info[0], info[1]))
      exit(-1)

  def output(self, index=0):
    return self.branches[index]


class Episode(Node):
  def __init__(self, inputs, capacity=100, samples_per_class=1, classes=[]):
      super(Episode, self).__init__(name=None, action=None, inputs=inputs)
      self.samples_per_class = samples_per_class
      self.classes = classes
      self.capacity = capacity
      self.consume_num = 0
      self.episode_train = _TransparantNode(upper_node=Node.inputs(self), name='episode-train', active_request=False)
      self.is_episode_ok = False
      self.episode_classes_map = {c: i for i, c in enumerate(self.classes)}
      self.rng = get_rng(self)

  def _evaluate(self):
    try:
      input = self._positional_inputs[0]
      if not self.is_episode_ok:
        # prepare episode
        episode_train_data = {category_id: [] for category_id in self.classes}
        while not self.is_episode_ok:
          data = input.get_value()
          a, b = data

          if 'bbox' in b or 'segmentation' in b:
            # image with bbox or segmentation datasource
            pair_list = [(b_i, b_c) for b_i,b_c in enumerate(b['category_id'].tolist())]
            self.rng.shuffle(pair_list)
            for b_index, b_category_id in pair_list:
              if b_category_id in self.classes and \
                              len(episode_train_data[b_category_id]) < self.samples_per_class:

                obj_img = None
                obj_mask = None
                obj_bbox = np.zeros((1, 4), dtype=np.float32)
                if 'bbox' in b:
                    x0, y0, x1, y1 = b['bbox'][b_index, :]
                    x0 = np.maximum(int(x0), 0)
                    y0 = np.maximum(int(y0), 0)
                    x1 = np.minimum(int(x1), a.shape[1] - 1)
                    y1 = np.minimum(int(y1), a.shape[0] - 1)
                    obj_img = a[y0:y1, x0:x1, :]
                    obj_bbox[0, 0] = x0
                    obj_bbox[0, 1] = y0
                    obj_bbox[0, 2] = x1
                    obj_bbox[0, 3] = y1

                if 'segmentation' in b:
                    obj_mask_full = b['segmentation'][b_index]
                    yyxx = np.where(obj_mask_full > 0)
                    yy = yyxx[0]
                    xx = yyxx[1]
                    x0 = np.min(xx)
                    y0 = np.min(yy)
                    x1 = np.max(xx)
                    y1 = np.max(yy)
                    obj_img = a[y0:y1, x0:x1, :]
                    obj_mask = obj_mask_full[y0:y1, x0:x1, :]
                    obj_bbox[0, 0] = x0
                    obj_bbox[0, 1] = y0
                    obj_bbox[0, 2] = x1
                    obj_bbox[0, 3] = y1

                annotation = {'bbox': obj_bbox,
                              'category_id': np.array([self.episode_classes_map[b_category_id]]),
                              'category': [b['category'][b_index]],
                              'info': a.shape}
                if obj_mask is not None:
                    annotation['segmentation'] = [obj_mask]

                episode_train_data[b_category_id].append((obj_img, annotation))
                break

            self.is_episode_ok = True
            for k, v in episode_train_data.items():
                if len(v) < self.samples_per_class:
                    self.is_episode_ok = False
                    break
          else:
              # image datasource
              pass

          input._force_inputs_dirty()

          if self.is_episode_ok:
              tt = []
              for k, v in episode_train_data.items():
                  tt.extend(v)

              self.episode_train.set_value(tt)
              self.consume_num = 0

      if self.consume_num >= self.capacity:
          raise StopIteration

      while True:
        a, b = input.get_value()
        if 'bbox' in b or 'segmentation' in b:
          # image with bbox or segmentation datasource
          keep_index = [i for i, l in enumerate(b['category_id'].tolist()) if l in self.classes]

          if len(keep_index) == 0:
              input._force_inputs_dirty()
              continue

          b_cpy = copy.deepcopy(b)
          b_cpy['category'] = [b_cpy['category'][i] for i in keep_index]
          b_cpy['category_id'] = np.array([self.episode_classes_map[b_cpy['category_id'][i]] for i in keep_index])
          if 'supercategory' in b_cpy:
              b_cpy['supercategory'] = [b_cpy['supercategory'][i] for i in keep_index]
          if 'area' in b_cpy:
              b_cpy['area'] = np.array([b_cpy['area'][i] for i in keep_index])
          if 'segmentation' in b_cpy:
              b_cpy['segmentation'] = [b_cpy['segmentation'][i] for i in keep_index]
          if 'bbox' in b_cpy:
              b_cpy['bbox'] = b_cpy['bbox'][keep_index,:]
          if 'iscrowd' in b_cpy:
              b_cpy['iscrowd'] = [b_cpy['iscrowd'][i] for i in keep_index]

          self.consume_num += 1
          return (a, b_cpy)
      else:
        # image datasource
        return (None, None)

    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      self.is_episode_ok = False
      self.consume_num = 0
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0], info[1]))
      exit(-1)

  def output(self, index=0):
    if index == 0:
      return self
    else:
      return self.episode_train


class Zip(Node):
  def __init__(self, inputs):
    super(Zip, self).__init__(name=None, action=None, inputs=inputs)

  def _evaluate(self):
    try:
      input_num = len(self._positional_inputs)
      zip_data = [None for _ in range(input_num)]
      for n in range(input_num):
          zip_data[n] = self._positional_inputs[n].get_value()
      return zip_data
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s' % (info[0], info[1]))
      exit(-1)
