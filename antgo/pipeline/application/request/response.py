# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : response.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.utils.reserved import *


class ResponseOp(object):
    def __init__(self, check_func, detail=''):
        self.check_func = check_func
        self.detail = detail

    def info(self):
        return ['session_id']

    def __call__(self, *args, session_id):
        is_ok = self.check_func(*args)
        if is_ok:
            # 填写响应标记
            return {
                'code': 0,
                'message': 'success',
            }
        else:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': self.detail
                },
                session_id=session_id,
                status_code=200,
                message=self.detail
            )
