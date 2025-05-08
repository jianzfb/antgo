# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : checkeq.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *
from sqlalchemy import and_, or_

class CheckeqOp(object):
    def __init__(self, target, detail=None):
        self.target = target
        self.detail = detail

    def info(self):
        return ['session_id']

    def __call__(self, *args, session_id):
        if args[0] is None or args[0] != self.target:
            set_context_exit_info(session_id, detail="request not allow" if self.detail is None else self.detail)
            return False

        return True
