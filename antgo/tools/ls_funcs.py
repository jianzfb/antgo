from antgo.framework.helper.fileio.file_client import *
import os
import logging

def ls_from_aliyun(src_path):
    ali = AliBackend()
    if not src_path.startswith("ali://"):
        src_path = f'{"ali://"}{src_path}'
    
    for file_name in ali.ls(src_path):
        print(file_name)
