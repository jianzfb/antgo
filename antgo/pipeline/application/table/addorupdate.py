# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : addorupdate.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.env import *
from sqlalchemy import and_, or_


class AddorupdateOp(object):
    def __init__(self, table, field, export=None, keys=None, default=None):
        # field:   对应输入的字段名字
        # keys:     标记过滤关键字段
        self.table = table
        self.field = field if isinstance(field, list) else [field]
        self.default = default
        self.export = export
        self.key_i = [0]
        if keys is not None:
            self.key_i = []
            for key in keys:
                self.key_i.append(self.field.index(key))

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    @resource_db_env
    def __call__(self, *args, session_id, db):
        orm_handler = get_db_orm()
        orm_table = getattr(orm_handler, self.table.capitalize())
        record = None
        # 检查是否已经存在
        if len(self.key_i) == 1:
            record = db.query(orm_table).filter(getattr(orm_table, self.field[self.key_i[0]]) == args[self.key_i[0]]).one_or_none()
        elif len(self.key_i) == 2:
            record = db.query(orm_table).filter(and_(getattr(orm_table, self.field[self.key_i[0]]) == args[self.key_i[0]], getattr(orm_table, self.field[self.key_i[1]]) == args[self.key_i[1]])).one_or_none()

        is_new_record = True
        if record is not None:
            is_new_record = False

        if is_new_record:
            # 添加一条记录
            field_info = {}
            # 用户指定数据
            for key, value in zip(self.field, args):
                if value is not None:
                    # 仅对非None数据进行赋值
                    if '/' in key:
                        foreign_table_name, foreign_field_name = key.split('/')
                        foreign_orm_table = getattr(orm_handler, foreign_table_name.capitalize())
                        foreign_record = db.query(foreign_orm_table).filter(getattr(foreign_orm_table, foreign_field_name) == value).one_or_none()
                        field_info[foreign_table_name] = foreign_record
                    else:
                        field_info[key] = value

            # 默认数据
            if self.default is not None:
                for key, value in self.default.items():
                    if value is not None:
                        # 仅对非None数据进行赋值
                        if '/' in key:
                            foreign_table_name, foreign_field_name = key.split('/')
                            foreign_orm_table = getattr(orm_handler, foreign_table_name.capitalize())
                            foreign_record = db.query(foreign_orm_table).filter(getattr(foreign_orm_table, foreign_field_name) == value).one_or_none()
                            field_info[foreign_table_name] = foreign_record
                        else:
                            field_info[key] = value

            record = orm_table(**field_info)
            db.add(record)
            db.commit()
        else:
            field_info = {}
            for key, value in zip(self.field, args):
                if value is not None:
                    # 仅对非None数据进行赋值
                    setattr(record, key, value)
            db.commit()

        # 如果不需要提取指定字段，则返回对象
        if self.export is None:
            return record

        # 返回指定字段数据
        obj_info = {}
        for data_name in self.export:
            if '/' not in data_name:
                # 表内属性
                obj_info[data_name] = getattr(record, data_name)
            else:
                # 跨表属性
                related_obj,related_field = data_name.split('/')
                related_obj = getattr(record, related_obj)
                obj_info[data_name] = None
                if related_obj is not None:
                    obj_info[data_name] = getattr(related_obj, related_field)

        return obj_info