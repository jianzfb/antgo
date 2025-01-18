# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : remove.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from sqlalchemy import and_, or_


class RemoveOp(object):
    def __init__(self, table, fields):
        # fields:   对应输入的字段名字
        self.table = table
        self.fields = fields if isinstance(fields, list) else [fields]

    def __call__(self, *args):
        orm = get_db_orm()
        orm_table = getattr(orm, self.table)
        db = get_thread_session()
        objs = []
        if len(self.key_i) == 1:
            objs = db.query(orm_table).filter(getattr(orm_table, self.fields[0]) == args[0]).all()
        elif len(self.key_i) == 2:
            objs = db.query(orm_table).filter(
                and_(getattr(orm_table, self.fields[0]) == args[0], getattr(orm_table, self.fields[1]) == args[1])
            ).all()

        for obj in objs:
            db.delete(obj)
        if len(objs) > 0:
            db.commit()

        return True