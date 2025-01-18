# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : filter.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from sqlalchemy import and_, or_


class FilterOp(object):
    def __init__(self, table, fields, data=None, prefix='and'):
        self.table = table
        self.data = data
        self.fields = fields if isinstance(fields, list) else [fields]
        assert(prefix in ['and', 'or'])
        self.prefix = prefix

    def __call__(self, *args):
        orm = get_db_orm()
        orm_table = getattr(orm, self.table.capitalize())
        prefix_op = and_ if self.prefix == 'and' else or_
        objs = None
        db = get_thread_session()
        if len(self.fields) == 1:
            objs = db.query(orm_table).filter(getattr(orm_table, self.fields[0]) == args[0]).all()
        elif len(self.fields) == 2:
            objs = db.query(orm_table).filter(
                prefix_op(getattr(orm_table, self.fields[0]) == args[0], getattr(orm_table, self.fields[1]) == args[1])
            ).all()

        if self.data is None:
            return objs

        obj_infos = []
        for filter_obj in objs:
            info_dict = {}
            for data_name in self.data:
                if '/' not in data_name:
                    # 表内属性
                    info_dict[data_name] = getattr(filter_obj, data_name)
                else:
                    # 跨表属性
                    related_obj,related_field = data_name.split('/')
                    related_obj = getattr(filter_obj, related_obj)
                    info_dict[data_name] = None
                    if related_obj is not None:
                        info_dict[data_name] = getattr(related_obj, related_field)

            obj_infos.append(info_dict)
        return obj_infos
