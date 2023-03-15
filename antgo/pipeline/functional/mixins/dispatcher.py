# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:31
# @File    : dispatcher.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine.factory import ops


class DispatcherMixin:
    """
    Mixin for call dispatcher for data collection

    >>> @register(name='add_1')
    ... def add_1(x):
    ...     return x+1

    >>> dc = DataCollection.range(5).stream()
    >>> dc.add_1['a','b','c']() #doctest: +ELLIPSIS
    <map object at ...>
    """

    def resolve(self, path, index, *arg, **kws):
        return getattr(ops, path)[index](*arg, **kws)
