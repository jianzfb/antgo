# -*- coding: UTF-8 -*-
# @Time    : 2022/9/6 23:17
# @File    : __init__.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from pathlib import Path
from .operator_registry import OperatorRegistry # pylint: disable=import-outside-toplevel


register = OperatorRegistry.register
resolve = OperatorRegistry.resolve


DEFAULT_LOCAL_CACHE_ROOT = Path.home() / '.antgo'
LOCAL_PIPELINE_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'pipelines'
LOCAL_OPERATOR_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'operators'
