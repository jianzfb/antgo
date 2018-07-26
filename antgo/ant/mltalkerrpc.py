# -*- coding: UTF-8 -*-
# @Time : 2018/7/24
# @File : mltalkerrpc.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import requests
import json

class MLTalkerRPC(object):
  def __init__(self, mltalker_ip, mltalker_port, mltalker_token):
    self._mltalker_ip = mltalker_ip
    self._mltalker_port = mltalker_port
    self._mltalker_token = mltalker_token

  def __getattr__(self, item):
    def func(**kwargs):
      request_url = ''
      if item.startswith('post_'):
        request_url = 'http://%s:%d/%s'%(self._mltalker_ip, self._mltalker_port, item.replace('post_'))
      elif item.startswith('get_'):
        request_url = 'http://%s:%d/%s'%(self._mltalker_ip, self._mltalker_port, item.replace('get_'))
      elif item.startswith('delete_'):
        request_url = 'http://%s:%d/%s' % (self._mltalker_ip, self._mltalker_port, item.replace('delete_'))
      elif item.startswith('patch_'):
        request_url = 'http://%s:%d/%s'%(self._mltalker_ip, self._mltalker_port, item.replace('patch_'))

      if request_url == '':
        return None

      try:
        response = requests.post(request_url,
                                data=kwargs,
                                headers={'Authorization': "token " + self._mltalker_token})

        if response is None:
          return None

        if response.status_code not in [200, 201]:
          return None

        return json.loads(response.content)
      except:
        return None

    return func