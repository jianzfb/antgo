# -*- coding: UTF-8 -*-
# @Time : 2018/7/17
# @File : warehouse.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
# import qiniu
# from qiniu import Auth, put_file, etag, urlsafe_base64_encode
from antgo.utils import logger
import os
from six.moves import urllib
import uuid

def qiniu_upload(file_path, bucket='mltalker', out_url_base = 'http://experiment.mltalker.com', max_size=10):
  return 'qiniu:%s/%s' % ('http:127.0.0.1', 'aaa')
  access_key = 'ZSC-X2p4HG5uvEtfmn5fsTZ5nqB3h54oKjHt0tU6'
  secret_key = 'Ya8qYwIDXZn6jSJDMz_ottWWOZqlbV8bDTNfCGO0'
  q = Auth(access_key, secret_key)

  if max_size is not None:
    # check file size
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    if fsize > max_size:
      logger.error('file size is larger than limit (%dMB)'%max_size)
      return None

  key = file_path.split('/')[-1]
  token = q.upload_token(bucket, key, 3600)
  ret, info = put_file(token, key, file_path)
  if ret['key'] == key and ret['hash'] == etag(file_path):
    logger.info('success to upload')
    return 'qiniu:%s/%s' % (out_url_base, key)

  return None


def qiniu_download(address, file_path):
  access_key = 'ZSC-X2p4HG5uvEtfmn5fsTZ5nqB3h54oKjHt0tU6'
  secret_key = 'Ya8qYwIDXZn6jSJDMz_ottWWOZqlbV8bDTNfCGO0'
  q = Auth(access_key, secret_key)
  if address.startswith('qiniu:'):
    address = address.replace('qiniu:','')
  private_url = q.private_download_url(address, expires=3600)

  if os.path.isdir(file_path):
    file_path = os.path.join(file_path, str(uuid.uuid4()))

  try:
    fpath, _ = urllib.request.urlretrieve(private_url, file_path)
    statinfo = os.stat(fpath)
    size = statinfo.st_size

    if size == 0:
      logger.error('couldnt download data')
      return None

    return file_path
  except:
    logger.error('couldnt download data')
    return None


# import tarfile
# import subprocess
#
# def _recursive_tar(root_path, path, tar, ignore=None):
#   if path.split('/')[-1][0] == '.':
#     return
#
#   if os.path.isdir(path):
#     for sub_path in os.listdir(path):
#       _recursive_tar(root_path, os.path.join(path, sub_path), tar)
#   else:
#     if ignore is not None:
#       if path.split('/')[-1] == ignore:
#         return
#     arcname = os.path.relpath(path, root_path)
#     tar.add(path, arcname=arcname)
#
# random_code_package_name = str(uuid.uuid4())
# code_tar_path = os.path.join('/Users/Jian/Downloads/aaa', '%s_code.tar.gz' % random_code_package_name)
# tar = tarfile.open(code_tar_path, 'w:gz')
# for sub_path in os.listdir('/Users/Jian/Downloads/aaa'):
#   _recursive_tar('/Users/Jian/Downloads/aaa',
#                       os.path.join('/Users/Jian/Downloads/aaa', sub_path),
#                       tar,
#                       ignore='%s_code.tar.gz' % random_code_package_name)
# tar.close()
#
# crypto_code = str(uuid.uuid4())
#
# crypto_shell = 'openssl enc -e -aes256 -in %s -out %s -k %s' % (
# '%s_code.tar.gz' % random_code_package_name,
# '%s_code_ssl.tar.gz' % random_code_package_name,
# crypto_code)
# subprocess.call(crypto_shell, shell=True, cwd='/Users/Jian/Downloads/aaa')
#
# # 解密
# decrypto_shell = 'openssl enc -d -aes256 -in %s -out %s -k %s'%('%s_code_ssl.tar.gz' % random_code_package_name,
#                                                                               '%s_code.tar.gz' % random_code_package_name,
#                                                                               crypto_code)
# subprocess.call(decrypto_shell, shell=True, cwd='/Users/Jian/Downloads/aaa')

# # qiniu_address = qiniu_upload(code_tar_path, bucket='mltalker', max_size=10)
# # print(qiniu_address)
#
# qiniu_download('http://pbzz7zw0y.bkt.clouddn.com/14697433-5bd9-4b3c-a392-ba72172c666e_code.tar.gz', '/Users/Jian/Downloads/zj')