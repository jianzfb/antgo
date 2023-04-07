from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

class HDFSClient(object):
    def get(self, remote_path, local_path):
        raise NotImplementedError

    def rm(self, remote_path, is_folder=False):
        raise NotImplementedError

    def ls(self, remote_path):
        raise NotImplementedError

    def mkdir(self, remote_path, p=False):
        raise NotImplementedError

    def put(self, remote_path, local_path, is_exist=False):
        raise NotImplementedError

    def mv(self, remote_src_path, remote_target_path):
        raise NotImplementedError

    def exists(self, remote_path):
        raise NotImplementedError


class DummyKVReader(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def read_many(self, index):
        pass


class DummyKVWriter(object):
    def __init__(self, *args, **kwargs):
        pass
    def write_many(self, keys, values):
        pass
    def flush(self):
        pass


hdfs_client = HDFSClient()
KVReader = DummyKVReader
KVWriter = DummyKVWriter
