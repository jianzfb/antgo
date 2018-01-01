# -*- coding: UTF-8 -*-
# Time: 12/31/17
# File: p2p_experiment.py
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
import shutil
from antgo import config
Config = config.AntConfig

def experiment_download_local(dump_dir, experiment, pwd, token):
  # call in an independent process
  try:
    experiment_path = os.path.join(dump_dir, experiment)
    if not os.path.exists(experiment_path):
      os.makedirs(experiment_path)

    import ipfsapi
    # 1.step get experiment address
    user_authorization = {'Authorization': "token " + token}

    server = getattr(Config, 'server_ip', 'www.mltalker.com')
    port = getattr(Config, 'server_port', '8999')
    res = requests.get('http://%s:%s/hub/api/experiment/%s/address' % (server, port, experiment),
                       headers=user_authorization)
    experiment_info = json.loads(res.content)

    address = experiment_info['ADDRESS']
    address_update_time = experiment_info['ADDRESS_UPDATE_TIME']

    # 2.step get from dht
    os.chdir(experiment_path)
    ipfs = ipfsapi.connect('127.0.0.1', 5001)
    ipfs.get(address)

    # 3.step post process (rename and extract)
    # 3.1.step rename
    rename_shell = 'mv %s %s.tar.gz'%(address, experiment)
    subprocess.call(rename_shell, shell=True, cwd=experiment_path)
    # 3.2.step untar
    untar_shell = 'openssl enc -d -aes256 -in %s.tar.gz -k %s | tar xz -C %s'%(experiment, pwd, '.')
    subprocess.call(untar_shell, shell=True, cwd=experiment_path)
    # 3.3.step clear tar file
    os.remove(os.path.join(experiment_path, '%s.tar.gz'%experiment))
    return True
  except:
    # remove unincomplete experiment
    shutil.rmtree(os.path.join(dump_dir, experiment))
    # error info
    traceback.print_exc()
    return False

def experiment_publish_dht(dump_dir, experiment, pwd, token):
  # call in an independent process
  try:
    import ipfsapi
    # 1.step tar all files
    tar_shell = 'tar -czf - * | openssl enc -e -aes256 -out %s.tar.gz -k %s' % (experiment, pwd)
    subprocess.call(tar_shell, shell=True, cwd=os.path.join(dump_dir, experiment))

    # 2.step ipfs add to dht
    ipfs = ipfsapi.connect('127.0.0.1', 5001)
    res = ipfs.add(os.path.join(os.path.join(dump_dir, experiment), '%s.tar.gz' % experiment))
    experiment_hash = res['Hash']

    # 3.step notify mltalker
    user_authorization = {'Authorization': "token " + token}

    server = getattr(Config, 'server_ip', 'www.mltalker.com')
    port = getattr(Config, 'server_port', '8999')
    requests.patch('http://%s:%s/hub/api/experiment/%s/address' % (server, port, experiment),
                   data={'address': experiment_hash, 'time': time.time()},
                   headers=user_authorization)

    # 4.step clear
    os.remove(os.path.join(dump_dir, experiment, '%s.tar.gz' % experiment))
    return True
  except:
    traceback.print_exc()
    return False

# experiment_download_local('/Users/zhangken/PycharmProject/antgo/antgo/dump/', '20180101.015106.406021', '023f49c20c8a4f04bde72dbfefa41a3e','023f49c20c8a4f04bde72dbfefa41a3e')