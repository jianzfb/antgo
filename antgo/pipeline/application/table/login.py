# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : login.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.env import *
from sqlalchemy import and_, or_
import uuid

class LoginOp(object):
    def __init__(self, cookie_prefix='antgo-auth', is_cas = False, detail='login fail'):
        self.cookie_prefix = cookie_prefix
        self.is_cas = is_cas

    def info(self):
        # db, cookie
        return ['session_id', 'username', 'password', 'headers', 'cookie']

    @resource_db_env
    def __call__(self, session_id, username, password, headers, cookie, db):
        orm = get_db_orm()

        # auth
        # 1. username, password
        # 2. cookie
        # 3. token

        # cookie 方法
        # cookie.get(), cookie.set(), cookie.clear()

        # cookie 登陆
        cookie_id = cookie.get(self.cookie_prefix, None)
        if cookie_id is not None:
            u = db.query(orm.User).filter(orm.User.cookie_id == cookie_id).one_or_none()
            if u is not None:
                return True

        # headers 方法
        # headers.get()
        # api token 登陆
        api_token = headers.get('Authorization', None)
        if api_token is not None:
            api_token = orm.APIToken.find(db, api_token, kind='user')
            if api_token is not None and api_token.user is not None:
               return True

        # username, password 登陆
        if username is None or password is None:
            return False
        u = db.query(orm.User).filter(orm.User.name == username, orm.User.password == password).one_or_none()
        if u is None:
            return False

        # 添加cookie
        cookie_id = str(uuid.uuid4()) + '/' + str(uuid.uuid4())
        cookie.set(self.cookie_prefix, cookie_id)
        u.cookie_id = cookie_id
        db.commit()

        if self.is_cas:
            # 删除已有票据
            if u.service_ticket is not None:
                for st in u.service_ticket:
                    db.delete(st)
            db.commit()

            # 创建新票据
            service_ticket = u.new_service_ticket()                

        return True