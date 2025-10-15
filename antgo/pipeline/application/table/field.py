# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : field.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.application.common.env import *
from sqlalchemy import and_, or_


class FieldOp(object):
    def __init__(self, field):
        if isinstance(self.field, str):
            self.field = [field]

    @resource_db_env
    def __call__(self, *args, **kwargs):
        # 需要考虑跨线程对象共享
        info = []
        for data_name in self.field:
            if '/' not in data_name:
                # 表内属性
                info.append(getattr(args[0], data_name))
            else:
                # 跨表属性
                related_obj,related_field = data_name.split('/')
                related_obj = getattr(args[0], related_obj)
                if related_obj is not None:
                    info.append(getattr(related_obj, related_field))

        return tuple(info)

