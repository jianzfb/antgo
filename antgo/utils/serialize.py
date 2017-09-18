# encoding=utf-8
# @Time     : 17-8-14
# @File     : serialize.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

__all__ = ['loads', 'dumps']


def dumps(obj):
    return msgpack.dumps(obj,use_bin_type=True)


def loads(buf):
    return msgpack.loads(buf,encoding='utf-8')
