# encoding=utf-8
# @Time    : 17-5-8
# @File    : cpu.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import re
import psutil
import os
from antgo.utils import timer
import numpy.distutils.cpuinfo
unit = {'b':1, 'k':2**10, 'm':2**20, 'g':2**30}


def get_nr_cpu():
    return psutil.cpu_count()


def cpu_running_info(pid=None,interval=1):
    '''
    key:
        mem_usage is RSS memory.RSS is thetotal memory actually held in RAM for a process.
        RSS can be misleading, because it reports the total all of the shared libraries 
        that the process uses, even though a shared library is only loaded into memory 
        once regardless of how many processes use it.
    :param pid: 
    :return: 
    '''
    mem_max = psutil.virtual_memory().total / unit['m']

    cpus = []
    cpu_info = numpy.distutils.cpuinfo.cpu.info
    if type(cpu_info) == list:
        for cpu_index in range(len(cpu_info)):
            if 'model name' in cpu_info[cpu_index]:
                cpus.append(cpu_info[cpu_index]['model name'])
            else:
                cpus.append('unkown')
    else:
        if 'model name' in cpu_info:
            cpus.append(cpu_info['model name'])
        else:
            cpus.append('unkown')

    if pid is not None:
        p = psutil.Process(pid)
        mem_usage = p.memory_percent() * 0.01 * mem_max
        cpu_usage = p.cpu_percent(interval=interval)
        num_threads = p.num_threads()
        if getattr(p,'cpu_affinity',None) is not None:
            occupy_cpus = p.cpu_affinity()
        else:
            occupy_cpus = [0]

        return {'cpus': cpus,'cpu_mem_usage':mem_usage,
                'cpu_mem_max':mem_max,'cpu_util':cpu_usage,
                'occupy_cpus':occupy_cpus,'num_threads':num_threads}

    return {'cpus': cpus,'cpu_mem_max':mem_max}