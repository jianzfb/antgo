# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
# Credit belongs to Xinyu Zhou
from __future__ import unicode_literals

import threading
import multiprocessing
import atexit
import bisect
from contextlib import contextmanager
import signal
import weakref
import six
from datetime import datetime
import time
import random
import os, sys
import subprocess
from six.moves import queue
import time
from . import logger

__all__ = ['StoppableThread', 'LoopThread','TimerThread', 'ensure_proc_terminate',
           'OrderedResultGatherProc', 'OrderedContainer', 'DIE',
           'mask_sigint', 'start_proc_mask_signal', 'StoppableProcess', 'GatherMultiProcs']


class StoppableThread(threading.Thread):
  """
  A thread that has a 'stop' event.
  """
  def __init__(self):
    super(StoppableThread, self).__init__()
    self._stop_evt = threading.Event()
    
    self._stop_condition = None

  def stop(self):
    """ stop the thread"""
    self._stop_evt.set()

  def stopped(self):
    """ check whether the thread is stopped or not"""
    return self._stop_evt.isSet()

  def queue_put_stoppable(self, q, obj):
    """ put obj to queue, but will give up if the thread is stopped"""
    while not self.stopped():
      try:
        q.put(obj, timeout=5)
        break
      except queue.Full:
        pass

  def queue_get_stoppable(self, q):
    """ take obj from queue, but will give up if the thread is stopped"""
    while not self.stopped():
      try:
        return q.get(timeout=5)
      except queue.Empty:
        pass
  
  @property
  def stop_condition(self):
    if self._stop_condition is None:
      self._stop_condition = threading.Condition()
    return self._stop_condition
  
  
class LoopThread(StoppableThread):
  """ A pausable thread that simply runs a loop"""
  def __init__(self, func, pausable=True):
    """
    :param func: the function to run
    """
    super(LoopThread, self).__init__()
    self._func = func
    self._pausable = pausable
    if pausable:
        self._lock = threading.Lock()
    self.daemon = True

  def run(self):
    while not self.stopped():
        if self._pausable:
            self._lock.acquire()
            self._lock.release()
        self._func()

  def pause(self):
    assert self._pausable
    self._lock.acquire()

  def resume(self):
    assert self._pausable
    self._lock.release()


class TimerThread(StoppableThread):
  ''' A timer thread that run func periodically'''
  def __init__(self,func,periodic = 10):
    '''
    :param func:
    :param periodical: unit is sec
    '''
    super(TimerThread, self).__init__()
    self._func = func
    self.daemon = True
    self.periodic = periodic

  def run(self):
    while not self.stopped():
      if type(self._func) == list:
        for func in self._func:
            func()
      else:
        self._func()
      time.sleep(self.periodic)


class DIE(object):
  """ A placeholder class indicating end of queue """
  pass


def ensure_proc_terminate(proc):
  if isinstance(proc, list):
    for p in proc:
        ensure_proc_terminate(p)
    return

  def stop_proc_by_weak_ref(ref):
    proc = ref()
    if proc is None:
      return
    if not proc.is_alive():
      return
    proc.terminate()
    proc.join()

  assert isinstance(proc, multiprocessing.Process)
  atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))


@contextmanager
def mask_sigint():
  sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  yield
  signal.signal(signal.SIGINT, sigint_handler)


def start_proc_mask_signal(proc):
  if not isinstance(proc, list):
    proc = [proc]

  with mask_sigint():
    for p in proc:
      p.start()


def subproc_call(cmd, timeout=None):
  try:
    output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
    return output
  except subprocess.TimeoutExpired as e:
    logger.warn("Command timeout!")
    logger.warn(e.output)
  except subprocess.CalledProcessError as e:
    logger.warn("Commnad failed: {}".format(e.returncode))
    logger.warn(e.output)


class OrderedContainer(object):
  """
  Like a priority queue, but will always wait for item with index (x+1) before producing (x+2).
  """
  def __init__(self, start=0):
    self.ranks = []
    self.data = []
    self.wait_for = start

  def put(self, rank, val):
    idx = bisect.bisect(self.ranks, rank)
    self.ranks.insert(idx, rank)
    self.data.insert(idx, val)

  def has_next(self):
    if len(self.ranks) == 0:
      return False
    return self.ranks[0] == self.wait_for

  def get(self):
    assert self.has_next()
    ret = self.data[0]
    rank = self.ranks[0]
    del self.ranks[0]
    del self.data[0]
    self.wait_for += 1
    return rank, ret


class OrderedResultGatherProc(multiprocessing.Process):
  """
  Gather indexed data from a data queue, and produce results with the
  original index-based order.
  """
  def __init__(self, data_queue, nr_producer, start=0):
    """
    :param data_queue: a multiprocessing.Queue to produce input dp
    :param nr_producer: number of producer processes. Will terminate after receiving this many of DIE sentinel.
    :param start: the first task index
    """
    super(OrderedResultGatherProc, self).__init__()
    self.data_queue = data_queue
    self.ordered_container = OrderedContainer(start=start)
    self.result_queue = multiprocessing.Queue()
    self.nr_producer = nr_producer

  def run(self):
    nr_end = 0
    try:
      while True:
        task_id, data = self.data_queue.get()
        if task_id == DIE:
          self.result_queue.put((task_id, data))
          nr_end += 1
          if nr_end == self.nr_producer:
            return
        else:
          self.ordered_container.put(task_id, data)
          while self.ordered_container.has_next():
            self.result_queue.put(self.ordered_container.get())
    except Exception as e:
      import traceback
      traceback.print_exc()
      raise e

  def get(self):
    return self.result_queue.get()


class StoppableProcess(multiprocessing.Process):
  def __init__(self, queue_size=-1):
    super(StoppableProcess, self).__init__()
    self._is_stop = False
    self._condition = None
    self._queue = multiprocessing.Queue(queue_size)

  def stop(self):
    """ stop the thread"""
    self._is_stop = True

  def stopped(self):
    """ check whether the thread is stopped or not"""
    return self._is_stop
  
  @property
  def process_condition(self):
    if self._condition == None:
      self._condition = multiprocessing.Condition()
    return self._condition
  
  @property
  def process_queue(self):
    return self._queue
  
  
class GatherMultiProcs(object):
  @staticmethod
  def process_func(data_flow, data_pipe, condition):
    # 1.step prepare random seed for current pid
    seed = (os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    random.seed(seed)
    
    while True:
      # 2.step put in queue
      for data in data_flow.iterator_value():
        data_pipe.put(data)
      
      # 3.step add DIE flag
      with condition:
        data_pipe.put(DIE)
        condition.wait()

  def __init__(self, data_flow, nr=2, cache=2):
    self._is_running = False
    self._nr = nr
    self._queue = multiprocessing.Queue(nr * cache)
    self._condition = multiprocessing.Condition()
    self._processes = [multiprocessing.Process(target=GatherMultiProcs.process_func,
                       args=(data_flow, self._queue, self._condition)) for _ in range(nr)]

    for p in self._processes:
      p.daemon = True
  
  def iterator_value(self):
    assert(self._queue.empty())
    
    # launch all processes
    if not self._is_running:
      for p in self._processes:
        p.start()
      self._is_running = True
    
    # notify all waiting processes
    with self._condition:
      self._condition.notify_all()

    while True:
      data = self._queue.get()
      if data == DIE:
        DIE_nr = 1
        # waiting until nr DIE
        while DIE_nr < self._nr:
          temp = self._queue.get()
          if temp == DIE:
            DIE_nr += 1
          else:
            yield temp
          
        raise StopIteration
      yield data