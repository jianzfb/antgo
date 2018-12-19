# -*- coding: UTF-8 -*-
# Time: 12/31/17
# File: dht.py
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
import threading
import sys
from antgo.dataflow.basic import *
from antgo.utils.serialize import *
import numpy as np
import shutil
from antgo.ant.warehouse import *

from antgo import config
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import Queue as queue
elif PYTHON_VERSION == 3:
    import queue as queue

Config = config.AntConfig


class FuncThread(threading.Thread):
  def __init__(self, func, blocks, data_q):
    threading.Thread.__init__(self)
    self._func = func
    self._blocks = blocks
    self._data_q = data_q
    self.daemon = True

  def run(self):
    self._func(self._blocks, self._data_q)


def dataset_upload_dht(dataset_name, data_generators, dump_dir):
  dataset_readme = {}

  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  # parse dataset and split to blocks
  for dataset_c, dataset_content in data_generators.items():
    # data sample generator
    dataset_generator = dataset_content['generator']
    # data sample number
    dataset_num = dataset_content['num']
    # block number
    block_num = dataset_content['block']

    # rectify block number
    if block_num > dataset_num:
      block_num = 1

    dataset_index = np.arange(0, int(dataset_num)).tolist()

    # prepare block content
    blocks_content = []
    blocks_name = ['%s_%s_block_%d'%(dataset_name, dataset_c, bi) for bi in range(block_num)]
    blocks_index = {}

    block_size = int(int(dataset_num) // int(block_num))

    current_block_index = 0
    for index, content in zip(dataset_index, dataset_generator):
      data, label = content
      block_index = int(index // block_size)
      if block_index == block_num:
        block_index = block_num - 1

      if current_block_index != block_index:
        with open(os.path.join(dump_dir, blocks_name[current_block_index]), 'wb') as fp:
          fp.write(dumps(blocks_content))

        blocks_content = []
        current_block_index = block_index

      # block content
      blocks_content.append((index, data, label))

      # sample index
      if blocks_name[block_index] not in blocks_index:
        blocks_index[blocks_name[block_index]] = []

      blocks_index[blocks_name[block_index]].append(index)

    with open(os.path.join(dump_dir, blocks_name[current_block_index]), 'wb') as fp:
      fp.write(dumps(blocks_content))

    # dataset sample number
    dataset_readme['%s_count'%dataset_c] = dataset_num
    # dataset blocks
    dataset_readme['%s_block'%dataset_c] = blocks_name
    # dataset block sample index
    dataset_readme['%s_index'%dataset_c] = blocks_index

  # save readme
  with open(os.path.join(dump_dir, 'readme.yaml'), 'w') as fp:
    yaml.safe_dump(dataset_readme, fp)

  # upload to dht
  short_dir = dump_dir.split('/')[-1]
  # s = subprocess.Popen('ipfs add -r %s'%short_dir,
  #                      shell=True,
  #                      cwd='/'.join(dump_dir.split('/')[0:-1]), stdout=subprocess.PIPE)
  # result = s.communicate()[0].decode()
  #
  # dataset_hash_code = None
  # for ff in result.split('\n'):
  #   if len(ff) > 0:
  #     cmd, hash_code, file_h = ff.split(' ')
  #     if file_h == short_dir:
  #       dataset_hash_code = hash_code
  #       break
  import ipfsapi
  ipfs_host = os.environ.get('IPFS_HOST', '127.0.0.1')
  ipfs = ipfsapi.connect(ipfs_host, 5001)

  dataset_hash_code = ''
  result = ipfs.add(dump_dir, recursive=True)
  for ff in result:
    if ff['Name'] == short_dir:
      dataset_hash_code = ff['Hash']
      break

  return dataset_hash_code


def dataset_download_dht(dataset_folder, train_or_test, dht_address, data_queue, db, threads_num):
  # 1.step download dataset readme.yaml
  import ipfsapi
  ipfs_host = os.environ.get('IPFS_HOST', '127.0.0.1')
  ipfs = ipfsapi.connect(ipfs_host, 5001)

  if not os.path.exists(os.path.join(dataset_folder,'temp')):
    os.makedirs(os.path.join(dataset_folder,'temp'))

  os.chdir(os.path.join(dataset_folder,'temp'))
  ipfs.get('%s/readme.yaml' % dht_address)

  with open(os.path.join(dataset_folder, 'temp', 'readme.yaml'),'r') as fp:
    readme_config = yaml.load(fp)
    # notify dataset sample number
    data_queue.put({'count': readme_config['%s_count'%train_or_test]})

  # all data blocks and their index
  block_list = readme_config['%s_block'%train_or_test]
  block_index_list = readme_config['%s_index'%train_or_test]

  # 2.step download data block in parallel
  ok_block_list = []
  ok_index_list = []

  for b in block_list:
    if db.get(b) is not None:
      ok_block_list.append(b)
      ok_index_list.extend(block_index_list[b])

  # notify prepared block index
  if len(ok_index_list) > 0:
    data_queue.put(ok_index_list)

  # prepare waiting blocks
  block_address_list = [(block_index, '%s/%s' % (dht_address, block_name)) for block_index, block_name in
                        enumerate(block_list) if block_name not in ok_block_list]

  if len(block_address_list) == 0:
    return

  # continue download data from dht
  def _pull_from_dht(block_addresses, data_q):
    ipfs_host = os.environ.get('IPFS_HOST', '127.0.0.1')
    local_ipfs = ipfsapi.connect(ipfs_host, 5001)

    for block_index, block_address in block_addresses:
      try:
        # 1.step pull
        local_ipfs.get(block_address)

        # 3.step read content from block
        data_q.put((block_index, block_address.split('/')[-1]))
      except:
        pass

    data_q.put(None)

  waiting_block_num = len(block_address_list)
  if waiting_block_num < threads_num:
    threads_num = waiting_block_num

  blocks_in_thread = int(int(waiting_block_num) / int(threads_num))
  assign_responsability_blocks = [None for _ in range(threads_num)]
  for s in range(threads_num - 1):
    assign_responsability_blocks[s] = block_address_list[s*blocks_in_thread:(s+1)*blocks_in_thread]

  assign_responsability_blocks[threads_num-1] = block_address_list[(threads_num-1)*blocks_in_thread:]

  q = queue.Queue()
  # initialize thread
  process_threads = [FuncThread(_pull_from_dht, assign_responsability_blocks[i], q) for i in range(threads_num)]
  # launch thread
  for pt in process_threads:
    pt.start()

  none_num = threads_num
  while True:
    content = q.get()
    if content == None:
      none_num -= 1
      if none_num == 0:
        break

    if content is not None:
      block_index, block_file = content

      ok_samples = []
      with open(os.path.join(dataset_folder, 'temp', block_file), 'rb') as fp:
        # parse content
        binary_content = fp.read()
        data_list = loads(binary_content)

        for a,b,c in data_list:
          ok_samples.append((a, b, c, block_file))

      # notfiy prepared block
      if len(ok_samples) > 0:
        data_queue.put(ok_samples)

  # wating until all stop
  for pt in process_threads:
    pt.join()


def experiment_download_dht(dump_dir, experiment, token, proxy=None, signature='123',target='qiniu', address=None):
  # call in an independent process
  try:
    if experiment is None:
      kterms = dump_dir.split('/')
      dump_dir = '/'.join(kterms[:-1])
      experiment = kterms[-1]

    experiment_path = os.path.join(dump_dir, experiment)
    if not os.path.exists(experiment_path):
      os.makedirs(experiment_path)

    if token is not None:
      # 1.step get experiment address
      user_authorization = {'Authorization': "token " + token}

      server = getattr(Config, 'server_ip', 'www.mltalker.com')
      port = getattr(Config, 'server_port', '8999')
      res = requests.get('http://%s:%s/hub/api/experiment/%s/address' % (server, port, experiment),
                         headers=user_authorization)
      experiment_info = json.loads(res.content)

      address = experiment_info['ADDRESS']
      address_update_time = experiment_info['ADDRESS_UPDATE_TIME']
    elif proxy is not None:
      res = requests.get('http://%s/update/model/%s/' % (proxy, experiment),
                    data={'signature': signature})
      experiment_info = json.loads(res.content)
      if experiment_info['code'] != 'Success':
        logger.error('couldnt get experiment address')
        return False

      address = experiment_info['address']

    if address is None:
      logger.error('couldnt download experiment')
      return False

    # 2.step get from dht
    try:
      if target == 'ipfs':
        import ipfsapi
        os.chdir(experiment_path)
        ipfs_host = os.environ.get('IPFS_HOST', '127.0.0.1')
        ipfs = ipfsapi.connect(ipfs_host, 5001)
        ipfs.get(address)
      elif target == 'qiniu':
        os.chdir(experiment_path)
        result = qiniu_download(address, experiment_path)
        if result is None:
          return False
        address = result.split('/')[-1]
      else:
        logger.error('dont support target %s platform'%target)
        return False
    except:
      logger.error('couldnt get experiment address')
      return False

    # 3.step post process (rename and extract)
    # 3.1.step rename
    rename_shell = 'mv %s %s.tar.gz'%(address, experiment)
    subprocess.call(rename_shell, shell=True, cwd=experiment_path)
    # 3.2.step untar
    untar_shell = 'openssl enc -d -aes256 -in %s.tar.gz -k %s | tar xz -C %s'%(experiment, signature, '.')
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


def experiment_upload_dht(dump_dir, experiment, token, proxy=None, signature='123', bucket='experiment', target='qiniu'):
  # call in an independent process
  try:
    if token is None and proxy is None:
      return

    if experiment is None:
      kterms = dump_dir.split('/')
      dump_dir = '/'.join(kterms[:-1])
      experiment = kterms[-1]

    if not os.path.exists(os.path.join(dump_dir, experiment)):
      logger.error('experiment %s does not exist'%experiment)
      return

    # 1.step tar all files
    tar_shell = 'tar -czf - * | openssl enc -e -aes256 -out %s.tar.gz -k %s' % (experiment, signature)
    subprocess.call(tar_shell, shell=True, cwd=os.path.join(dump_dir, experiment))

    # 2.step ipfs add to dht
    experiment_hash = ''
    if target == 'ipfs':
      import ipfsapi
      ipfs_host = os.environ.get('IPFS_HOST', '127.0.0.1')
      ipfs = ipfsapi.connect(ipfs_host, 5001)
      res = ipfs.add(os.path.join(os.path.join(dump_dir, experiment), '%s.tar.gz' % experiment))
      experiment_hash = res['Hash']
    elif target == 'qiniu':
      result = qiniu_upload(os.path.join(dump_dir, experiment, '%s.tar.gz'%experiment), bucket=bucket, max_size=500)
      if result is None:
        logger.error('couldnt upload experiment %s to qiniu/%s'%(experiment, bucket))
        return

      experiment_hash = result
    else:
      # dont support now
      logger.error('dont support target %s platform'%target)
      return

    # 3.step notify mltalker
    try:
      if token is not None:
        user_authorization = {'Authorization': "token " + token}

        server = getattr(Config, 'server_ip', 'www.mltalker.com')
        port = getattr(Config, 'server_port', '8999')
        requests.patch('http://%s:%s/hub/api/experiment/%s/address' % (server, port, experiment),
                       data={'address': experiment_hash, 'time': time.time()},
                       headers=user_authorization)
      elif proxy is not None:
        requests.post('http://%s/update/model/%s/'%(proxy, experiment),
                      data={'signature': signature, 'address': experiment_hash})
    except:
      logger.error('couldnt notify server')

    # 4.step clear
    os.remove(os.path.join(dump_dir, experiment, '%s.tar.gz' % experiment))
    return True
  except:
    traceback.print_exc()
    return False


# test update experiment record
# experiment_upload_dht('/Users/jian/PycharmProjects/antgo/antgo/example/',
#                       'de03d0b6-9a2f-4704-94f0-a48cda92d8d5-20181219-143919-050970',
#                       token=None,
#                       proxy='127.0.0.1:10001',
#                       signature='aaa'
#                       )

# test download experiment record
# experiment_download_dht('/Users/Jian/Downloads/',
#                         'de03d0b6-9a2f-4704-94f0-a48cda92d8d5-20181219-143919-050970',
#                         token=None,
#                         proxy='127.0.0.1:10001',
#                         signature='aaa')