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
    def __init__(self, table, field, export=None, keys=None, default=None, detail=None, allow_increment=False):
        # field:   对应输入的字段名字
        # keys:     标记过滤关键字段
        self.table = table
        self.field = field if isinstance(field, list) else [field]
        self.default = default
        self.export = export
        self.allow_increment = allow_increment
        self.detail = None
        self.key_i = [0]
        if keys is not None:
            self.key_i = []
            for key in keys:
                self.key_i.append(self.field.index(key))

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def __call__(self, *args, session_id):
        orm_handler = get_db_orm()
        orm_table = getattr(orm_handler, self.table.capitalize())
        db = get_thread_session()
        args_cpy = list(args)

        # 检查是否已经存在
        if len(self.key_i) == 1:
            records = db.query(orm_table).filter(getattr(orm_table, self.field[self.key_i[0]]) == args_cpy[self.key_i[0]]).all()
        elif len(self.key_i) == 2:
            records = db.query(orm_table).filter(and_(getattr(orm_table, self.field[self.key_i[0]]) == args_cpy[self.key_i[0]], getattr(orm_table, self.field[self.key_i[1]]) == args[self.key_i[1]])).all()

        if len(records) != 0 and not self.allow_increment:
            set_context_exit_info(session_id, detail="existed in db" if self.detail is None else self.detail)
            return None

        record_num = len(records)
        if record_num > 0:
            # 名字+1
            args_cpy[self.key_i[0]] = f'{args_cpy[self.key_i[0]]}-{record_num}'

        # 添加一条记录
        field_info = {}
        for key, value in zip(self.field, args_cpy):
            if value is not None:
                # 仅对非None数据进行赋值
                if '/' in key:
                    foreign_table_name, foreign_field_name = key.split('/')
                    foreign_orm_table = getattr(orm_handler, foreign_table_name.capitalize())
                    foreign_record = db.query(foreign_orm_table).filter(getattr(foreign_orm_table, foreign_field_name) == value).one_or_none()
                    field_info[foreign_table_name] = foreign_record
                else:
                    field_info[key] = value
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