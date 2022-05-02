# -*- coding: UTF-8 -*-
# @Time    : 2022/5/1 18:28
# @File    : utils.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import paddle


def is_available():
  return paddle.fluid.is_compiled_with_cuda()