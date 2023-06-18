from antgo.framework.helper.fileio.file_client import *
import os
import logging


def upload_to_aliyun(target_folder, src_path):
    if not target_folder.startswith('ali://'):
        target_folder = f'ali://{target_folder}'
    
    if not os.path.exists(src_path):
        logging.error(f"{src_path} dont exist")
        return

    ali = AliBackend()
    ali.upload(target_folder, src_path)
