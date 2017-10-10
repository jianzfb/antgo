#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: jian(jian@mltalker.com)
from __future__ import unicode_literals

import os, sys
from six.moves import urllib
from . import logger

def mkdir_p(dirname):
  """ make a dir recursively, but do nothing if the dir exists"""
  assert dirname is not None
  if dirname == '' or os.path.isdir(dirname):
    return
  try:
    os.makedirs(dirname)
  except OSError as e:
    if e.errno != 17:
      raise e

def download(url, dir, fname=None):
  mkdir_p(dir)
  if fname is None:
    fname = url.split('/')[-1]
  fpath = os.path.join(dir, fname)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' %
                     (fname,
                         min(float(count * block_size)/ total_size,
                             1.0) * 100.0))
    sys.stdout.flush()
  try:
    fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=_progress)
    statinfo = os.stat(fpath)
    size = statinfo.st_size
  except:
    logger.error("Failed to download {}".format(url))
    raise
  assert size > 0, "Download an empty file!"
  sys.stdout.write('\n')
  print('Succesfully downloaded ' + fname + " " + str(size) + ' bytes.')
  return fpath

def maybe_here(dest_dir,target_file):
  maybe_dest_dir = dest_dir
  while maybe_dest_dir is not None:
    target_file_path = os.path.join(maybe_dest_dir, target_file)
    if not os.path.exists(target_file_path):
      is_continue = False
      for ff in os.listdir(maybe_dest_dir):
        if ff[0] == '.':
          continue
        if os.path.isdir(os.path.join(maybe_dest_dir, ff)):
          maybe_dest_dir = os.path.join(maybe_dest_dir, ff)
          is_continue = True
          break

      if not is_continue:
        maybe_dest_dir = None
      else:
        break
    else:
      break

  return maybe_dest_dir

def maybe_here_match_format(dest_dir, target_pattern):
  maybe_dest_dir = dest_dir
  while maybe_dest_dir is not None:
    files = os.listdir(maybe_dest_dir)
    is_has_files = len([f for f in files if target_pattern in f])
    if is_has_files == 0:
      is_continue = False
      for ff in os.listdir(maybe_dest_dir):
        if ff[0] == '.':
          continue
        if os.path.isdir(os.path.join(maybe_dest_dir, ff)):
          maybe_dest_dir = os.path.join(maybe_dest_dir, ff)
          is_continue = True
          break

      if not is_continue:
        maybe_dest_dir = None
      else:
        break

  return maybe_dest_dir

def maybe_here_fixed_3_hierarchy(dest_dir,target_pattern):
  maybe_dest_dir = dest_dir
  while maybe_dest_dir is not None:
    check_dirs = [os.path.join(maybe_dest_dir, family, character) \
                              for family in os.listdir(maybe_dest_dir) \
                              if os.path.isdir(os.path.join(maybe_dest_dir, family)) \
                              for character in os.listdir(os.path.join(maybe_dest_dir, family))]

    if len(check_dirs) > 0:
      files = os.listdir(check_dirs[0])
      is_has_files = len([f for f in files if target_pattern in f])
      if is_has_files > 0:
        return maybe_dest_dir

    is_continue = False
    for ff in os.listdir(maybe_dest_dir):
      if ff[0] == '.':
        continue
      if os.path.isdir(os.path.join(maybe_dest_dir, ff)):
        maybe_dest_dir = os.path.join(maybe_dest_dir, ff)
        is_continue = True
        break

    if not is_continue:
      maybe_dest_dir = None
      return maybe_dest_dir

if __name__ == '__main__':
    download('http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz', '.')
