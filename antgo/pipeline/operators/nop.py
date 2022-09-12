# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 17:58
# @File    : nop.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from typing import Any, Dict, NamedTuple

from .base import PyOperator


class NOPOperator(PyOperator):
    """No-op operators. Input arguments are redefined as a `NamedTuple` and returned as
    outputs.
    """

    def __init__(self):
        #pylint: disable=useless-super-delegation
        super().__init__()

    def __call__(self, **args: Dict[str, Any]) -> NamedTuple:
        fields = [(name, type(val)) for name, val in args.items()]
        return NamedTuple('Outputs', fields)(**args)  # pylint: disable=not-callable
