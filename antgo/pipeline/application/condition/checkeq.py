# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : checkeq.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.utils.reserved import *
from sqlalchemy import and_, or_

class CheckeqOp(object):
    def __init__(self, target, detail=None, status_code=200):
        self.target = target
        self.detail = detail
        self.status_code = status_code

    def info(self):
        return ['session_id']

    def __call__(self, *args, session_id):
        if args[0] is None or args[0] != self.target:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "request not allow" if self.detail is None else self.detail
                },
                message = "request not allow" if self.detail is None else self.detail,
                status_code = self.status_code,
                session_id=session_id,
            )

        return None
