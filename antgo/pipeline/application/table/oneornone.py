# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : oneornone.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *
from sqlalchemy import and_, or_


class OneornoneOp(object):
    def __init__(self, table, fields, data=None, prefix='and', allow_none=False):
        self.table = table
        self.data = data
        self.fields = fields if isinstance(fields, list) else [fields]
        assert(prefix in ['and', 'or'])
        self.prefix = prefix
        self.allow_none = allow_none

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def __call__(self, *args, session_id):
        orm = get_db_orm()
        orm_table = getattr(orm, self.table.capitalize())
        prefix_op = and_ if self.prefix == 'and' else or_
        obj = None
        db = get_thread_session()
        if len(self.fields) == 1:
            obj = db.query(orm_table).filter(getattr(orm_table, self.fields[0]) == args[0]).one_or_none()
        elif len(self.fields) == 2:
            obj = db.query(orm_table).filter(
                prefix_op(getattr(orm_table, self.fields[0]) == args[0], getattr(orm_table, self.fields[1]) == args[1])
            ).one_or_none()

        if obj is None and (not self.allow_none):
            set_context_exit_info(session_id, detail="not existed in db")
            return None

        if self.data is None or obj is None:
            return obj

        obj_info = {}
        for data_name in self.data:
            if '/' not in data_name:
                # 表内属性
                obj_info[data_name] = getattr(obj, data_name)
            else:
                # 跨表属性
                related_obj,related_field = data_name.split('/')
                related_obj = getattr(obj, related_obj)
                info_dict[data_name] = None
                if related_obj is not None:
                    info_dict[data_name] = getattr(related_obj, related_field)

        return obj_info

