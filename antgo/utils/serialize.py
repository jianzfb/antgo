#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py
# Author: jian<jian@mltalker.com>
from __future__ import unicode_literals

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

__all__ = ['loads', 'dumps']


def dumps(obj):
    return msgpack.dumps(obj,use_bin_type=True)


def loads(buf):
    return msgpack.loads(buf,encoding='utf-8')
