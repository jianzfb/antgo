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
import tarfile
import time
import tempfile
import ipfsapi
import shutil

def _hierarchy_folder_tar(tar, level):
  return tar

_ipfs_api_controler = None
def ipfa_api_controler():
  global _ipfs_api_controler
  if _ipfs_api_controler is None:
    _ipfs_api_controler = ipfsapi.connect('127.0.0.1', 5001)

  return _ipfs_api_controler

@contextmanager
def running_sandbox(*wargs, **kwargs):
  def _sandbox_thread(**kwargs):
    if 'sandbox_time' in kwargs:
      sandbox_start_time = kwargs['sandbox_time']['start_time']
      sandbox_running_time = kwargs['sandbox_time']['running_time']
    
      now_time = time.time()
      if now_time - sandbox_start_time > sandbox_running_time:
        logger.info('running time is arriving')
        os._exit(0)

    if 'sandbox_dump_dir' in kwargs and 'token' in kwargs:
      dump_dir = kwargs['sandbox_dump_dir']
      try:
        # 1.step tar dump dir (tempdir)
        tar_file_path = os.path.join(tempfile.tempdir, '%s.tar'%dump_dir.split('/')[-1])
        tar = tarfile.open(tar_file_path, 'w')
        tar = _hierarchy_folder_tar(tar, dump_dir)
        tar.close()

        # 2.step symmetric encryption
        encryption_tar_file_path = tar_file_path

        # 3.step add to ipfs (kv: experiment <-> HASH)
        # 3.1.step delete old record
        old_dump_hash = None

        # 3.2.step add new record
        cc = ipfa_api_controler()
        res = cc.add(tar_file_path)
        new_dump_hash = res['Hash']

        # 4.step clear data
        shutil.rmtree(tar_file_path)
        shutil.rmtree(encryption_tar_file_path)
      except:
        pass


  # timer thread
  timer_thread = None

  sandbox_thread_kwargs = {}
  if 'sandbox_time' in kwargs and kwargs['sandbox_time'] is not None:
    sandbox_time = kwargs['sandbox_time']
    # check time format (s,m,h,d)
    try:
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

      start_time = time.time()  # seconds
      sandbox_thread_kwargs['sandbox_time'] = \
        {'start_time': start_time, 'running_time': running_time}
    except:
      logger.error('sand time isnt correct')
      exit(-1)
  
  if len(sandbox_thread_kwargs) > 0:
    timer_thread = TimerThread([lambda: _sandbox_thread(**sandbox_thread_kwargs)], periodic=2)
    # start thread
    timer_thread.start()
    
  yield
  
  if timer_thread is not None:
    # stop thread
    timer_thread.stop()
