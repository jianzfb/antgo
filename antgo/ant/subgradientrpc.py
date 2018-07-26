# -*- coding: UTF-8 -*-
# @Time : 2018/6/22
# @File : subgradientrpc.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import requests
import json
import Crypto
from Crypto.Hash import SHA256
from Crypto.Signature import pkcs1_15
from Crypto.PublicKey import RSA
import uuid
import os

class SubgradientRPC(object):
  def __init__(self, subgradientserver_ip, subgradientserver_port):
    self._subgradientserver_ip = subgradientserver_ip
    self._subgradientserver_port = subgradientserver_port

  @property
  def private_key(self):
    return open('%s/.ssh/id_rsa'%os.environ['HOME']).read()

  @property
  def public_key(self):
    return open('%s/.ssh/id_rsa.pub'%os.environ['HOME']).read()

  @property
  def subgradientserver_ip(self):
    return self._subgradientserver_ip
  @property
  def subgradientserver_port(self):
    return self._subgradientserver_port

  def signature(self, secret_option=''):
    secret = str(uuid.uuid4())
    message = '%s.%s' % (secret, secret_option)
    key = RSA.import_key(self.private_key)
    message_sha256 = SHA256.new(message.encode('utf-8'))
    signature = pkcs1_15.new(key).sign(message_sha256)
    return secret, signature

  def make_computingpow_order(self,
                              rental_time,
                              os_platform,
                              os_version,
                              software_framework,
                              cpu_model,
                              cpu_num,
                              cpu_mem,
                              gpu_model,
                              gpu_num,
                              gpu_mem,
                              dataset,
                              max_fee):
    # 1.step 订购满足条件的算力
    result = requests.post('http://%s:%d/api/auto/make/order'%(self.subgradientserver_ip,
                                                               self.subgradientserver_port),
                           {
                            'OS_PLATFORM': os_platform,
                            'OS_VERSION': os_version,
                            'SOFTWARE_FRAMEWORK': software_framework,
                            'CPU_MODEL': cpu_model,
                            'CPU_NUM': cpu_num,
                            'CPU_MEM': cpu_mem,
                            'GPU_MODEL': gpu_model,
                            'GPU_NUM': gpu_num,
                            'GPU_MEM': gpu_mem,
                            'DATASET': dataset,
                            'RENTAL_TIME': rental_time,
                            'PUBLIC_KEY': self.public_key,
                            'MAX_FEE': max_fee})
    if result is None:
      return None

    if result.status_code not in [200, 201]:
      return None

    response = json.loads(result.content)
    if response['result'] == 'fail' and response['reason'] == 'no enough money':
      print('has no enough money to make computing pow order')
      return None
    elif response['result'] == 'fail':
      return None

    order_response = json.loads(result.content)
    order_ip_address = order_response['ip_address']
    order_rpc_port = order_response['rpc_port']
    order_ssh_port = order_response['ssh_port']
    order_id = order_response['order_id']

    return {'order_ip': order_ip_address,
            'order_rpc_port': order_rpc_port,
            'order_ssh_port': order_ssh_port,
            'order_id': order_id}

  def __getattr__(self, item):
    if item not in ['authorize', 'refresh','launch', 'status', 'ping']:
      raise NotImplementedError

    def func(order_ip, order_port, **kwargs):
      request_url = 'http://%s:%d/%s'%(order_ip, order_port, item)
      result = requests.post(request_url, kwargs)
      if result.status_code not in [200, 201]:
        return {'result': 'fail', 'reason': 'UNKOWN_ERROR'}

      response = json.loads(result.content)
      return response

    return func
