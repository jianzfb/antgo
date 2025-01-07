# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : add.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *
from sqlalchemy import and_, or_


class AddOp(object):
    def __init__(self, table, fields, data=None, keys=None):
        self.table = table
        self.fields = fields if isinstance(fields, list) else [fields]
        self.data = data
        self.key_i = [0]
        if keys is not None:
            self.key_i = []
            for key in keys:
                self.key_i.append(self.fields.index(key))

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def __call__(self, *args, session_id):
        orm_handler = get_db_orm()
        orm_table = getattr(orm_handler, self.table.capitalize())
        record = None
        with thread_session_context(get_db_session()) as db:
            # 检查是否已经存在
            if len(self.key_i) == 1:
                record = db.query(orm_table).filter(getattr(orm_table, self.fields[self.key_i[0]]) == args[self.key_i[0]]).one_or_none()
            elif len(self.key_i) == 2:
                record = db.query(orm_table).filter(and_(getattr(orm_table, self.fields[self.key_i[0]]) == args[self.key_i[0]], getattr(orm_table, self.fields[self.key_i[1]]) == args[self.key_i[1]])).one_or_none()

            if record is not None:
                set_context_exit_info(session_id, detail="existed in db")
                return None

            # 添加一条记录
            if record is None:
                field_info = {}
                for key, value in zip(self.fields, args):
                    field_info[key] = value
                record = orm_table(**field_info)
                db.add(record)
                db.commit()

        # 如果不需要提取指定字段，则返回对象
        if self.data is None:
            return record

        # 返回指定字段数据
        obj_info = {}
        for data_name in self.data:
            obj_info[data_name] = getattr(record, data_name)
        return obj_info