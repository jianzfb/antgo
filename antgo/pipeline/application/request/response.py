# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : response.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *

class ResponseOp(object):
    def __init__(self, check_func):
        self.check_func = check_func

    @property
    def fixedOutIndex(self):
        return '__response__'

    def __call__(self, *args):
        is_ok = self.check_func(*args)
        if is_ok:
            return {
                'code': 0,
                'message': 'success',
            }
        else:
            return {
                'code': -1,
                'message': 'fail',
            }