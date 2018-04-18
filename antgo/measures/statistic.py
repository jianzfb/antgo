# encoding=utf-8
# @Time    : 17-5-8
# @File    : statistic.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from contextlib import contextmanager
from collections import defaultdict
from ..utils.cpu import *
from ..utils.gpu import *
from ..utils.concurrency import *
from antgo.utils import logger
import time
import psutil
import os
import numpy as np
import subprocess

_RUNNING_STATISTIC = defaultdict(dict)
_RUNNING_STATISTIC_TREE = []

@contextmanager
def performance_statistic_region(running_block):
  global _RUNNING_STATISTIC
  if running_block in _RUNNING_STATISTIC:
    _RUNNING_STATISTIC.pop(running_block)

  _RUNNING_STATISTIC_TREE.append(running_block)
  statistic_tree_block = '/'.join(_RUNNING_STATISTIC_TREE)

  # current process
  pid = os.getpid()

  # now cpu status
  now_cpu_status = cpu_running_info(pid)
  running_mem_usage = [now_cpu_status['cpu_mem_usage']]
  running_cpu_util = [now_cpu_status['cpu_util']]
  running_cpu_occupy = [now_cpu_status['occupy_cpus']]
  cpu_model = [now_cpu_status['cpus']]

  def _cpu_info_func(pid):
    try:
      cpu_info = cpu_running_info(pid)
      mem_usage = cpu_info['cpu_mem_usage']
      cpu_util = cpu_info['cpu_util']
      cpu_occupy = cpu_info['occupy_cpus']
      running_mem_usage.append(mem_usage)
      running_cpu_util.append(cpu_util)
      running_cpu_occupy.append(cpu_occupy)
    except:
      logger.error('some error happen when sampling cpu state')

  # now gpu status
  now_gpu_status = gpu_running_info(pid)
  gpu_model = [now_gpu_status['gpus']] if now_gpu_status is not None else []
  gpu_driver_version = [now_gpu_status['driver-version']] if now_gpu_status is not None else []
  
  running_gpu_mem_usage = [[float(now_gpu_status['gpu_mem_usage'][int(index)]) for index in now_gpu_status['occupy_gpus']]] \
                          if now_gpu_status is not None else []
  running_gpu_util = [[float(now_gpu_status['gpu_util'][int(index)]) for index in now_gpu_status['occupy_gpus']]] \
                      if now_gpu_status is not None else []
  running_gpu_occupy = [now_gpu_status['occupy_gpus']] if now_gpu_status is not None else []

  def _gpu_info_func(pid):
    try:
      gpu_info = gpu_running_info(pid)
      if gpu_info is not None:
        assert(len(gpu_info['occupy_gpus']) <= len(gpu_info['gpu_mem_usage']))
        assert(len(gpu_info['occupy_gpus']) <= len(gpu_info['gpu_util']))
        assert(len(gpu_info['gpu_mem_usage']) == len(gpu_info['gpu_util']))
        if len(gpu_info['occupy_gpus']) == 0:
          return
        
        running_gpu_mem_usage.append([float(gpu_info['gpu_mem_usage'][int(index)]) for index in gpu_info['occupy_gpus']])
        running_gpu_util.append([float(gpu_info['gpu_util'][int(index)]) for index in gpu_info['occupy_gpus']])
        running_gpu_occupy.append(gpu_info['occupy_gpus'])
  
        if len(gpu_model) == 0:
          gpu_model.append(gpu_info['gpus'])
        
        if len(gpu_driver_version) == 0:
          gpu_driver_version.append(gpu_info['driver-version'])
    except:
      logger.error('some error happen when sampling gpu state')

  # running before
  # 1.step running time start
  start = time.time()

  # 2.step cpu and gpu statistic periodically
  timer_thread = TimerThread([lambda:_cpu_info_func(pid), lambda:_gpu_info_func(pid)], periodic=5)
  timer_thread.start()

  yield
  
  # stop thread
  timer_thread.stop()

  # running after
  elapsed_time = time.time() - start

  time_length = len(running_mem_usage)
  sta_tstart = int(time_length * 0.3)
  sta_tstop = int(time_length * 0.8)

  if sta_tstart == sta_tstop:
    sta_tstop = sta_tstart + 1
  
  # memory util status
  running_mem_usage_mean = np.mean(running_mem_usage[sta_tstart:sta_tstop])
  running_mem_usage_median = np.median(running_mem_usage[sta_tstart:sta_tstop])
  running_mem_usage_max = np.max(running_mem_usage[sta_tstart:sta_tstop])

  # cpu util status
  running_cpu_util_mean = np.mean(running_cpu_util[sta_tstart:sta_tstop])
  running_cpu_util_median = np.median(running_cpu_util[sta_tstart:sta_tstop])
  running_cpu_util_max = np.max(running_cpu_util[sta_tstart:sta_tstop])
  _RUNNING_STATISTIC[statistic_tree_block]['time'] = {}
  _RUNNING_STATISTIC[statistic_tree_block]['time']['elapsed_time'] = elapsed_time
  _RUNNING_STATISTIC[statistic_tree_block]['cpu'] = {}
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['mem_mean_usage'] = running_mem_usage_mean
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['mem_median_usage'] = running_mem_usage_median
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['mem_max_usage'] = running_mem_usage_max
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['cpu_mean_usage'] = running_cpu_util_mean
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['cpu_median_usage'] = running_cpu_util_median
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['cpu_max_usage'] = running_cpu_util_max
  _RUNNING_STATISTIC[statistic_tree_block]['cpu']['cpu_model'] = cpu_model[0][0]

  if len(gpu_model) > 0:
    # sampling gpu
    time_length = len(running_gpu_occupy)
    sta_tstart = int(time_length * 0.3)
    sta_tstop = int(time_length * 0.8)
  
    if sta_tstart == sta_tstop:
      sta_tstop = sta_tstart + 1
    
    # filter invalid gpu sampling
    occupied_gpu_num = 0
    for sampling_i in range(sta_tstart, sta_tstop, 1):
      if occupied_gpu_num < len(running_gpu_occupy[sampling_i]):
        occupied_gpu_num = len(running_gpu_occupy[sampling_i])
    
    if occupied_gpu_num > 0:
      running_gpu_mem_usage = [vv for vv in running_gpu_mem_usage if len(vv) == occupied_gpu_num]
      running_gpu_util = [vv for vv in running_gpu_util if len(vv) == occupied_gpu_num]
      
      # gpu memory util status
      running_gpu_mem_usage_mean = np.mean(np.array(running_gpu_mem_usage[sta_tstart:sta_tstop]), axis=0).tolist()
      running_gpu_mem_usage_median = np.median(running_gpu_mem_usage[sta_tstart:sta_tstop], axis=0).tolist()
      running_gpu_mem_usage_max = np.max(running_gpu_mem_usage[sta_tstart:sta_tstop], axis=0).tolist()
  
      # gpu util status
      running_gpu_util_usage_mean = np.mean(np.array(running_gpu_util[sta_tstart:sta_tstop]), axis=0).tolist()
      running_gpu_util_usage_median = np.median(running_gpu_util[sta_tstart:sta_tstop], axis=0).tolist()
      running_gpu_util_usage_max = np.max(running_gpu_util[sta_tstart:sta_tstop], axis=0).tolist()
  
      _RUNNING_STATISTIC[statistic_tree_block]['gpu'] = {}
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_mean_usage'] = running_gpu_mem_usage_mean
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_median_usage'] = running_gpu_mem_usage_median
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_max_usage'] = running_gpu_mem_usage_max
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mean_usage'] = running_gpu_util_usage_mean
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_median_usage'] = running_gpu_util_usage_median
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_max_usage'] = running_gpu_util_usage_max
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_model'] = gpu_model[0][0]
      _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_driver_version'] = gpu_driver_version[0]

  # waiting until stop
  timer_thread.join()
  _RUNNING_STATISTIC_TREE.pop(-1)


def get_performance_statistic(running_block=None):
  return _RUNNING_STATISTIC if running_block is None else _RUNNING_STATISTIC[running_block]
