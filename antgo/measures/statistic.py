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
import time
import psutil
import os
import numpy as np

_RUNNING_STATISTIC = defaultdict(dict)
_RUNNING_STATISTIC_TREE = []

@contextmanager
def running_statistic(running_block):
    global _RUNNING_STATISTIC
    if running_block in _RUNNING_STATISTIC:
        _RUNNING_STATISTIC.pop(running_block)

    _RUNNING_STATISTIC_TREE.append(running_block)

    statistic_tree_block = '/'.join(_RUNNING_STATISTIC_TREE)
    running_mem_usage = []
    running_cpu_util = []
    running_cpu_occupy = []
    cpu_model = []

    def _cpu_info_func(pid):
        cpu_info = cpu_running_info(pid)
        mem_usage = cpu_info['cpu_mem_usage']
        cpu_util = cpu_info['cpu_util']
        cpu_occupy = cpu_info['occupy_cpus']
        running_mem_usage.append(mem_usage)
        running_cpu_util.append(cpu_util)
        running_cpu_occupy.append(cpu_occupy)
        if len(cpu_model) == 0:
            cpu_model.append(cpu_info['cpus'])

    gpu_model = []
    gpu_driver_version = []
    running_gpu_mem_usage = []
    running_gpu_util = []
    running_gpu_occupy = []

    def _gpu_info_func(pid):
        gpu_info = gpu_running_info(pid)
        if gpu_info is not None:
            occupy_gpus = gpu_info['occupy_gpus']
            gpu_mem_usage = gpu_info['gpu_mem_usage']
            gpu_util = gpu_info['gpu_util']
            running_gpu_mem_usage.append([gpu_mem_usage[index] for index in occupy_gpus])
            running_gpu_util.append([gpu_util[index] for index in occupy_gpus])
            running_gpu_occupy.append(occupy_gpus)
            if len(gpu_model) == 0:
                gpu_model.append(gpu_info['gpus'])
            if len(gpu_driver_version) == 0:
                gpu_driver_version.append(gpu_info['driver-version'])

    # current process
    pid = os.getpid()

    # running before
    # 1.step running time start
    start = time.time()
    # 2.step cpu and gpu statistic periodically
    timer_thread = TimerThread([lambda:_cpu_info_func(pid), lambda:_gpu_info_func(pid)], periodic=1)
    timer_thread.start()

    yield
    # stop thread
    timer_thread.stop()

    # running after
    elapsed_time = time.time() - start

    time_length = len(running_mem_usage)
    sta_tstart = int(time_length * 0.3)
    sta_tstop = int(time_length * 0.8)

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

    # if len(gpu_model) > 0:
    #     # gpu memory util status
    #     running_gpu_mem_usage_mean = np.mean(np.array(running_gpu_mem_usage[sta_tstart:sta_tstop]),axis=0)
    #     running_gpu_mem_usage_median = np.median(running_gpu_mem_usage[sta_tstart:sta_tstop],axis=0)
    #     running_gpu_mem_usage_max = np.max(running_gpu_mem_usage[sta_tstart:sta_tstop],axis=0)
    #
    #     # gpu util status
    #     running_gpu_util_usage_mean = np.mean(np.array(running_gpu_util[sta_tstart:sta_tstop]),axis=0)
    #     running_gpu_util_usage_median = np.median(running_gpu_util[sta_tstart:sta_tstop],axis=0)
    #     running_gpu_util_usage_max = np.max(running_gpu_util[sta_tstart:sta_tstop],axis=0)
    #
    #     running_gpu_model = [gpu_model[0][index] for index in running_gpu_occupy]
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu'] = {}
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_mean_usage'] = running_gpu_mem_usage_mean
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_median_usage'] = running_gpu_mem_usage_median
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mem_max_usage'] = running_gpu_mem_usage_max
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_mean_usage'] = running_gpu_util_usage_mean
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_median_usage'] = running_gpu_util_usage_median
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_max_usage'] = running_gpu_util_usage_max
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_model'] = running_gpu_model
    #     _RUNNING_STATISTIC[statistic_tree_block]['gpu']['gpu_driver_version'] = gpu_driver_version[0]

    # waiting until stop
    timer_thread.join()
    _RUNNING_STATISTIC_TREE.pop(-1)


def get_running_statistic(running_block=None):
    return _RUNNING_STATISTIC if running_block is None else _RUNNING_STATISTIC[running_block]
