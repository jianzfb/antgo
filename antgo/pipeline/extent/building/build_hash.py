import hashlib
import os


def path_hash(path):
    md5 = hashlib.md5()
    md5.update(path.encode('utf-8'))
    return md5.hexdigest()[:8]


def get_file_hash(fname):
    return str(int(os.path.getmtime(fname)))
