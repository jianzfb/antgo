from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

class HDFSClient(object):
    def get(self, remote_path):
        raise NotImplementedError

    def rm(self, remote_path, is_folder=False):
        raise NotImplementedError

    def ls(self, remote_path):
        raise NotImplementedError

    def mkdir(self, remote_path):
        raise NotImplementedError

    def put(self, remote_path, local_path, is_exist=False):
        raise NotImplementedError

    def mv(self, remote_src_path, remote_target_path):
        raise NotImplementedError

hdfs_client = HDFSClient()
