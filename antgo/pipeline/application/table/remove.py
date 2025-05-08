# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : remove.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *
from sqlalchemy import and_, or_


class RemoveOp(object):
    def __init__(self, table, field, keys=None, export=None):
        # field:   对应输入的字段名字
        self.table = table
        self.field = field if isinstance(field, list) else [field]
        self.export = export
        self.key_i = [0]
        if keys is not None:
            self.key_i = []
            for key in keys:
                self.key_i.append(self.field.index(key))

    def __call__(self, *args):
        orm = get_db_orm()
        orm_table = getattr(orm, self.table.capitalize())
        db = get_thread_session()
        objs = []
        if len(self.key_i) == 1:
            objs = db.query(orm_table).filter(getattr(orm_table, self.field[0]) == args[0]).all()
        elif len(self.key_i) == 2:
            objs = db.query(orm_table).filter(
                and_(getattr(orm_table, self.field[0]) == args[0], getattr(orm_table, self.field[1]) == args[1])
            ).all()

        obj_info_list = []
        for record in objs:
            if self.export is not None:
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
                obj_info_list.append(obj_info)

            db.delete(record)
        if len(objs) > 0:
            db.commit()

        return obj_info_list