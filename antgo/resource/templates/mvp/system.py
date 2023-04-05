###########################################################
# 定制化绑定本地后端配置，如文件HDFS系统和KV系统
###########################################################

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant import environment
import os
import subprocess


# step1: 配置hdfs后端存储
class YourHDFS(environment.HDFSClient):
    def get(self, remote_path, local_path):
        # 下载远程文件到本地
        # remote_path: 远程hdfs地址
        # local_path: 本地地址
        os.system(f'hdfs dfs -get {remote_path} {local_path}')
        return True

    def rm(self, remote_path, is_folder=False):
        # 删除远程文件
        # remote_path: 远程hdfs地址
        return True

    def ls(self, remote_path):
        # 获得远程地址下的文件列表
        result = subprocess.Popen(f'hdfs dfs -ls {remote_path}', shell=True, stdout=subprocess.PIPE).stdout
        file_list = []
        for line_info in result.read().decode('utf-8').split('\n'):
            if 'hdfs' in line_info:
                a = line_info.split('hdfs://')[-1].strip()
                file_list.append(f'hdfs://{a}')
            
        return file_list

    def mkdir(self, remote_path, p=False):
        # 在远程地址下创建目录
        if p:
            os.system(f'hdfs dfs -mkdir -p {remote_path}')
        else:
            os.system(f'hdfs dfs -mkdir {remote_path}')
        return True

    def put(self, remote_path, local_path, is_exist=False):
        # 上传本地文件到远程地址
        if is_exist:
            os.system(f'hdfs dfs -put {local_path} {remote_path}')
        else:
            os.system(f'hdfs dfs -put -f {local_path} {remote_path}')
        return True

    def mv(self, remote_src_path, remote_target_path):
        # 移动远程src地址到tgt地址
        return True

    def exists(self, remote_path):
        # 检查远程地址是否存在
        result = subprocess.Popen(f'hdfs dfs -ls {remote_path}', shell=True, stdout=subprocess.PIPE).stdout
        if result.read() == b'':
            return False
        else:
            return True

environment.hdfs_client = YourHDFS()


# step2: 配置KV Reader & Writer
class YourKVReader(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def read_many(self, keys):
        # keys: ["", ""]
        # 基于传入的keys读取
        return []


class YourKVWriter(object):
    def __init__(self, *args, **kwargs):
        pass

    def write_many(self, keys, values):
        # keys: ["", ""], values: [b'', b'']
        # 将keys,values写入
        pass

    def flush(self):
        # 刷新写缓存
        pass

# environment.KVReader = YourKVReader
# environment.KVWriter = YourKVWriter