# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : signup.py
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
import uuid


class SignupOp(object):
    def __init__(self,cookie_prefix='antgo-auth'):
        self.cookie_prefix = cookie_prefix

    def info(self):
        # db, cookie
        return ['session_id', 'username', 'password', 'headers', 'cookie']

    @resource_db_env
    def __call__(self, session_id, username, password, headers, cookie, db):
        orm = get_db_orm()

        # 创建用户
        # 1. username, password
        if username is None or password is None:       
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "username or password empty"
                },
                session_id=session_id,
                status_code=400,
                message="username or password empty"
            )

        u = db.query(orm.User).filter(orm.User.name == username).one_or_none()
        if u is not None:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "user exists"
                },
                session_id=session_id,
                status_code=400,
                message="user exists"
            )

        u = orm.User(name=username, password=password)
        db.add(u)
        db.commit()
        return True
