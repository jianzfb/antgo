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
    def __init__(self, check_func, detail='', response=None):
        self.check_func = check_func
        self.detail = detail
        self.response = response

    def info(self):
        return ['session_id']

    def __call__(self, *args, session_id):
        # 常用于条件判断，检验是否满足后续计算的约束。
        # 校验失败，不意味服务崩溃，故返回code标记默认为0
        is_ok = self.check_func(*args)
        if is_ok:
            # 默认标记
            return {
                'code': 0,
                'message': 'success',
            }
        else:
            response = {
                'code': 0,
                'message': self.detail
            }
            if self.response is not None:
                response.update(self.response)

            return ReservedRtnType(
                index = '__response__',
                data = response,
                session_id=session_id,
                status_code=200
            )
