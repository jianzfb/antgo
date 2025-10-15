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
from antgo.pipeline.application.common.env import *
from sqlalchemy import and_, or_
from antgo.pipeline.utils.reserved import *


class CheckeqOp(object):
    def __init__(self, field, target, detail=None):
        self.field = field
        self.target = target
        self.detail = detail

    def info(self):
        return ['session_id']

    @resource_db_env
    def __call__(self, *args, session_id, **kwargs):
        check_val = None
        if '/' not in self.field:
            # 表内属性
            check_val = getattr(args[0], self.field)
        else:
            # 跨表属性
            related_obj,related_field = self.field.split('/')
            related_obj = getattr(args[0], related_obj)
            if related_obj is not None:
                check_val = getattr(related_obj, related_field)

        if check_val is None or check_val != self.target:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "request not allow" if self.detail is None else self.detail
                },
                session_id=session_id,
                status_code=401,
                message="request not allow" if self.detail is None else self.detail
            )

        return True