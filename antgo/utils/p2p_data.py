# -*- coding: UTF-8 -*-
# Time: 12/31/17
# File: p2p_data.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import time
import requests
import subprocess
import json
import traceback
from antgo import config
Config = config.AntConfig


def data_download_local(dataset, ipfs_url, pwd=None):
  try:
    # 1.step dataset root folder
    data_factory = getattr(Config, 'data_factory', None)

    # 2.step get dataset DHT address
    dataset_hash = ipfs_url.split('/')[-1]

    # 3.step download from DHT
    os.chdir(data_factory)
    import ipfsapi
    ipfs = ipfsapi.connect('127.0.0.1', 5001)
    ipfs.get(dataset_hash)

    # 4.step post process
    rename_shell = 'mv %s %s.tgz'%(dataset_hash, dataset)
    subprocess.call(rename_shell, shell=True, cwd=data_factory)

    untar_shell = 'tar -zxvf %s.tgz'%dataset
    subprocess.call(untar_shell, shell=True, cwd=data_factory)

    os.remove(os.path.join(data_factory, "%s.tgz"%dataset))

    return True
  except:
    traceback.print_exc()
    return False


def data_publish_dht(dataset, token, pwd=None):
  try:
    if token is None:
      return False
    
    # 1.step dataset root folder
    data_factory = getattr(Config, 'data_factory', None)

    # 2.step tar all files
    tar_shell = 'tar -zcvf %s.tgz %s'%(dataset, dataset)
    subprocess.call(tar_shell, shell=True, cwd=data_factory)

    # 3.step publish to DHT
    import ipfsapi
    ipfs = ipfsapi.connect('127.0.0.1', 5001)
    hash_res = ipfs.add(os.path.join(data_factory, '%s.tgz'%dataset))
    print(hash_res)
    os.remove(os.path.join(data_factory, '%s.tgz'%dataset))

    # 3.step notify mltalker
    dataset_hash = hash_res['Hash']
    server = getattr(Config, 'server_ip', 'www.mltalker.com')
    port = getattr(Config, 'server_port', '8999')
    user_authorization = {'Authorization': "token " + token}

    res = requests.patch('http://%s:%s/hub/api/terminal/update/dataset'%(server, port),
                         data={'dataset-name': dataset,
                               'dataset-url': 'ipfs://%s'%dataset_hash},
                         headers=user_authorization)

    if res.status_code == 200:
      return True
    return False
  except:
    traceback.print_exc()
    return False
