# -*- coding: UTF-8 -*-
# @Time    : 2022/9/18 12:48
# @File    : runas_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from typing import Callable

from antgo.pipeline.engine import register

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name

@register(name='runas_op')
class runas_op:
    """
    Convert a user-defined function as an operator and execute.

    Args:
        func (`Callable`):
            The user-defined function.

    Examples:

    >>> from antgo.pipeline.functional import DataCollection
    >>> from antgo.pipeline.functional.entity import Entity
    >>> entities = [Entity(a=i, b=i) for i in range(5)]
    >>> dc = DataCollection(entities)
    >>> res = dc.runas_op['a', 'b'](func=lambda x: x - 1).to_list()
    >>> res[0].a == res[0].b + 1
    True
    """
    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kws):
        return self._func(*args, **kws)


# __test__ = {'run': run.__doc__}
