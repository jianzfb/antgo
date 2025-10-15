# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : oneornone.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.env import *
from antgo.pipeline.utils.reserved import *
from sqlalchemy import and_, or_


class OneornoneOp(object):
    def __init__(self, table, field, export=None, prefix='and', allow_none=False, detail=None):
        self.table = table
        self.export = export
        self.field = field if isinstance(field, list) else [field]
        assert(prefix in ['and', 'or'])
        self.prefix = prefix
        self.allow_none = allow_none
        self.detail = detail

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    @resource_db_env
    def __call__(self, *args, session_id, db):
        orm_table = getattr(get_db_orm(), self.table.capitalize())
        prefix_op = and_ if self.prefix == 'and' else or_
        obj = None
        if len(self.field) == 1:
            obj = db.query(orm_table).filter(getattr(orm_table, self.field[0]) == args[0]).one_or_none()
        elif len(self.field) == 2:
            obj = db.query(orm_table).filter(
                prefix_op(getattr(orm_table, self.field[0]) == args[0], getattr(orm_table, self.field[1]) == args[1])
            ).one_or_none()

        if obj is None and (not self.allow_none):
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "not existed in db" if self.detail is None else self.detail
                },
                session_id=session_id,
                status_code=401,
                message="not existed in db" if self.detail is None else self.detail
            )

        if self.export is None or obj is None:
            return obj

        obj_info = {}
        for data_name in self.export:
            if '/' not in data_name:
                # 表内属性
                obj_info[data_name] = getattr(obj, data_name)
            else:
                # 跨表属性
                related_obj,related_field = data_name.split('/')
                related_obj = getattr(obj, related_obj)
                obj_info[data_name] = None
                if related_obj is not None:
                    obj_info[data_name] = getattr(related_obj, related_field)

        return obj_info
