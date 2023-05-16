import os
import subprocess

ENV_PATH = os.path.dirname(__file__)
__all__ = ['ENV_PATH', 'list_gpus', 'get_git_hash']


def list_gpus_impl():
    result = ''
    nvidia_smi = ['nvidia-smi', '/usr/bin/nvidia-smi',
                  '/usr/local/nvidia/bin/nvidia-smi']
    for cmd in nvidia_smi:
        try:
            result = subprocess.check_output(
                [cmd, "-L"], universal_newlines=True)
            break
        except Exception:
            pass
    else:
        return range(0)
    return list(range(len([i for i in result.split('\n') if 'GPU' in i])))


GPUS_LIST = list_gpus_impl()


def list_gpus():
    """Return a list of GPUs
        Adapted from [MXNet](https://github.com/apache/incubator-mxnet)

    Returns
    -------
    list of int:
        If there are n GPUs, then return a list [0,1,...,n-1]. Otherwise returns
        [].
    """
    return GPUS_LIST


def get_git_hash():
    try:
        GIT_HEAD_PATH = os.path.join(ENV_PATH, '..', '.git')
        line = open(os.path.join(GIT_HEAD_PATH, 'HEAD')
                    ).readline().strip()
        if line[:4] == 'ref:':
            ref = line[5:]
            return open(os.path.join(GIT_HEAD_PATH, ref)).readline().strip()[:7]
        return line[:7]
    except FileNotFoundError:
        return 'custom'


def with_metaclass(meta, *bases):
    class metaclass(type):
        def __new__(cls, name, _bases, attrs):
            return meta(name, bases, attrs)

        @classmethod
        def __prepare__(cls, name, _bases):
            return meta.__prepare__(name, bases)
    return type.__new__(metaclass, 'class', (), {})


def makedirs(name, mode=511, exist_ok=False):
    '''makedirs(name [, mode=0o777][, exist_ok=False])

Super-mkdir; create a leaf directory and all intermediate ones.  Works like
mkdir, except that any intermediate path segment (not just the rightmost)
will be created if it does not exist. If the target directory already
exists, raise an OSError if exist_ok is False. Otherwise no exception is
raised.  This is recursive.'''
    try:
        os.makedirs(name, mode)
    except FileExistsError:
        if not exist_ok:
            raise
