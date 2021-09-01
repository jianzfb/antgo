# -*- coding: UTF-8 -*-
# @Time    : 17-11-27
# @File    : sandbox.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os, sys
from contextlib import contextmanager
from antgo.utils.concurrency import *
from antgo.utils import logger
import antvis.client.mlogger as mlogger
import multiprocessing
import time


@contextmanager
def running_sandbox(*wargs, **kwargs):
  def _sandbox_thread(**kwargs):
    if 'sandbox_time' in kwargs and kwargs['sandbox_time'] is not None:
      launch_time = kwargs['sandbox_launch_time']
      sandbox_running_time = kwargs['sandbox_time']
      # now time
      now_time = time.time()
      if now_time - launch_time > sandbox_running_time:
        # arriving custom max running time
        logger.info('max running time is arriving, exit antgo running environment')

        # experiment result save process
        if 'sandbox_dump_dir' in kwargs and \
                'sandbox_experiment' in kwargs and \
                'sandbox_user_token' in kwargs:
          # TODO 当实验运行超时时，执行固定操作
          pass

        # exit globally
        mlogger.exit()
        os._exit(0)

  try:
    # timer thread
    timer_thread = None

    sandbox_thread_kwargs = kwargs
    sandbox_thread_kwargs['sandbox_launch_time'] = time.time()
    if 'sandbox_time' in kwargs and kwargs['sandbox_time'] is not None:
      sandbox_time = kwargs['sandbox_time']
      # check time format (s,m,h,d)
      time_unit = sandbox_time[-1]
      if time_unit not in ['s', 'm', 'h', 'd']:
        logger.error('sand time isnt correct')
        exit(-1)
      time_value = sandbox_time[:-1]
      time_value = int(time_value)

      second = time_value
      if time_unit == 'm':
        second = second * 60
      elif time_unit == 'h':
        second = second * 3600
      elif time_unit == 'd':
        second = second * 3600 * 24
      running_time = second

      sandbox_thread_kwargs['sandbox_time'] = running_time

    if 'sandbox_time' in sandbox_thread_kwargs:
      timer_thread = TimerThread([lambda: _sandbox_thread(**sandbox_thread_kwargs)], periodic=60)
      # start thread
      timer_thread.start()

    yield

    # 线程退出
    if timer_thread is not None:
      # stop thread
      timer_thread.stop()
      timer_thread.join()

    logger.info('exit antgo running environment')
  except Exception as e:
    print(e)
    logger.error('error occured in antgo running environment')