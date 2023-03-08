from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant import environment
from dataloader import KVReader
import os

# step1: 配置hdfs后端存储
class ByteHDFS(environment.HDFSClient):
    def get(self, remote_path, local_path):
        pass

    def rm(self, remote_path, is_folder=False):
        pass

    def ls(self, remote_path):
        pass

    def mkdir(self, remote_path, p=False):
        pass

    def put(self, remote_path, local_path, is_exist=False):
        pass

    def mv(self, remote_src_path, remote_target_path):
        pass

    def exists(self, remote_path):
        pass


print('hello the world')
environment.hdfs_client = ByteHDFS()
print('after')

# step2: 配置KV Reader & Writer
environment.KVReader = KVReader
environment.KVWriter = None