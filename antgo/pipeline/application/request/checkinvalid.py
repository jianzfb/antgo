# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : checkinvalid.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.utils.reserved import *

class CheckinvalidOp(object):
    def __init__(self, *args, detail='', status_code=200):
        self.invalid_vals = args
        self.detail = detail
        self.status_code = status_code

    def info(self):
        return ['session_id']

    def __call__(self, *args, session_id):
        is_invalid = False
        for check_var, invalid_val in zip(args, self.invalid_vals):
            if check_var is None:
                # None is invalid
                is_invalid = True
                break
            if invalid_val is None:
                continue

            if check_var == invalid_val:
                is_invalid = True
                break

        if is_invalid:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': 'invalid request params' if self.detail == '' else self.detail
                },
                session_id=session_id,
                status_code=self.status_code,
                message=''
            )

        return args[0] if len(args) == 1 else args