from ..config import config
from ..utils import makedirs
from .build_hash import path_hash
import os


def get_virtual_dirname(path):
    if config.BUILD_IN_LOCAL_PATH:
        return path
    assert os.path.isdir(path), path
    dirname, basename = os.path.split(path)
    # hash dirname
    hash_dirname = path_hash(dirname)
    new_path = os.path.normpath(os.path.join(
        config.BUILD_PATH, 'build', '{}_{}'.format(basename, hash_dirname)))
    makedirs(new_path, exist_ok=True)
    tag_fname = os.path.join(new_path, 'ORIGINAL_PATH')
    if not os.path.exists(tag_fname):
        with open(tag_fname, 'w') as fout:
            fout.write(os.path.abspath(path))
    return new_path


def change_exts(lst, rules):
    res = []
    mappings = dict(rules)
    for name in lst:
        sp = os.path.splitext(name)
        if sp[1] and sp[1][0] == '.':
            ext = sp[1][1:]
            if ext in mappings:
                new_ext = mappings[ext]
                name = sp[0] + '.' + new_ext
        res.append(name)
    return res


def add_path(path, files):
    return list(map(lambda x: os.path.join(path, x), files))
