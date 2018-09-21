# -*- coding: UTF-8 -*-
# @Time : 17/03/2018
# @File : utils.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import imghdr
import os
import shutil

def url_path_join(*pieces):
  """Join components of url into a relative url

  Use to prevent double slash when joining subpath. This will leave the
  initial and final / in place

  Copied from notebook.utils.url_path_join
  """
  initial = pieces[0].startswith('/')
  final = pieces[-1].endswith('/')
  stripped = [s.strip('/') for s in pieces]
  result = '/'.join(s for s in stripped if s)

  if initial:
    result = '/' + result
  if final:
    result = result + '/'
  if result == '//':
    result = '/'

  return result


def check_file_types(file_path, check_types=['jpg']):
  if check_types[0] in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
    # IMAGE
    file_type = imghdr.what(file_path)
    file_path = os.path.normpath(file_path)
    file_name = file_path.split('/')[-1]
    if file_type != file_name.split('.')[-1]:
      file_name = '%s.%s'%(file_name.split('.')[0], file_type)
      os.rename(file_path, os.path.join('/'.join(file_path.split('/')[0:-1]), file_name))
      file_path = os.path.join('/'.join(file_path.split('/')[0:-1]), file_name)

    if 'jpg' in check_types and 'jpeg' not in check_types:
      check_types.append('jpeg')

    if file_type in check_types:
      return True, file_path

    return False, None
  elif check_types[0] in ['mp4', 'avi']:
    # VIDEO
    file_path = os.path.normpath(file_path)
    file_name = file_path.split('/')[-1]
    if file_name.split('.')[-1] == '':
      return False, None

    return True, file_path
  elif check_types[0] in ['txt']:
    # 文本
    file_path = os.path.normpath(file_path)
    file_name = file_path.split('/')[-1]
    if file_name.split('.')[-1] == '':
      return False, None

    return True, file_path

  else:
    # 不支持
    return False, None
