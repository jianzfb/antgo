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
    def __init__(self, check_func, detail=''):
        self.check_func = check_func
        self.detail = detail

    def info(self):
        return ['session_id']

    @property
    def outIndex(self):
        return '__response__'

    def __call__(self, *args, session_id):
        is_ok = self.check_func(*args)
        if is_ok:
            # 填写响应标记
            return {
                'code': 0,
                'message': 'success',
            }
        else:
            # 设置请求退出标记
            set_context_exit_info(session_id, detail='', status_code=200)

            # 填写响应标记
            return {
                'code': -1,
                'message': 'fail',
                'info': self.detail
            }
