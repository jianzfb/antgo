# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : remove.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2

class RemoveOp(object):
    def __init__(self, table, fields, data=None, key=None):
        self.table = table
        self.fields = fields if isinstance(fields, list) else [fields]
        self.data = data
        self.key_i = 0 if key is None else self.fields.index(key)

    def __call__(self, *args):
        orm = get_db_orm()
        orm_table = getattr(orm, self.table)
        with thread_session_context(get_db_session()) as db:
            objs = db.query(orm_table).filter(getattr(orm_table, self.fields[self.key_i]) == args[self.key_i]).all()
            for obj in objs:
                db.delete(obj)
            if len(objs) > 0:
                db.commit()

        return True