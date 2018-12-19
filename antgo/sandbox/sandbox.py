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
from antgo.utils.dht import *
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
          # 1.step launch experiment save process
          logger.info('launch upload experiment record in the running end')
          process = multiprocessing.Process(target=experiment_upload_dht,
                                            args=(kwargs['sandbox_dump_dir'],
                                                  kwargs['sandbox_experiment'],
                                                  kwargs['sandbox_user_token'],
                                                  kwargs['sandbox_user_proxy'],
                                                  kwargs['sandbox_user_signature']))
          process.start()

          # 2.step wating until save process stop
          process.join()

        # exit globally
        os._exit(0)

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

  if len(sandbox_thread_kwargs) > 0:
    timer_thread = TimerThread([lambda: _sandbox_thread(**sandbox_thread_kwargs)], periodic=60)
    # start thread
    timer_thread.start()

  yield

  if timer_thread is not None:
    # stop thread
    timer_thread.stop()
    timer_thread.join()

  # launch experiment save process
  logger.info('launch upload experiment record in the running end')
  process = multiprocessing.Process(target=experiment_upload_dht,
                                    args=(kwargs['sandbox_dump_dir'],
                                          kwargs['sandbox_experiment'],
                                          kwargs['sandbox_user_token'],
                                          kwargs['sandbox_user_proxy'],
                                          kwargs['sandbox_user_signature']))

  process.start()
  process.join()
  logger.info('exit antgo running environment')