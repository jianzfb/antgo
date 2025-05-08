# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : user.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.functional.mixins.db import *
from urllib.parse import unquote_plus, quote_plus
import os
import cv2
import base64
import numpy as np
import re
from sqlalchemy import and_, or_


# pattern for the authentication token header
auth_header_pat = re.compile(r'^token\s+([^\s]+)$')


class CasOp(object):
    def __init__(self, cas_ip, cas_port, cas_proto='http', cas_prefix='antvis', server_router="/#/Login"):
        self.cas_url = f'{cas_proto}://{cas_ip}:{cas_port}/{cas_prefix}'
        self.cas_ip = cas_ip
        self.cas_port = cas_port
        self.cas_proto = cas_proto
        self.cas_prefix = cas_prefix
        self.server_router = server_router

    def info(self):
        return ['ST', 'token', 'username', 'password', 'session_id']

    def get_current_user_from_token(self, db, token):
        """get_current_user from Authorization header token"""
        if token is None:
            return None
        match = auth_header_pat.match(auth_header)
        if not match:
            return None
        token = match.group(1)
        orm_token = get_db_orm().APIToken.find(db, token)
        if orm_token is None:
            return None
        else:
            return orm_token.user

    def get_current_user_from_argments(self, db, username, password):
        if username is None or password is None:
            return None

        user = db.query(get_db_orm().User).filter(and_(get_db_orm().User.name==username, get_db_orm().User.password==password)).one_or_none()
        return user

    def get_current_user(self, db, token, username, password):
        """get current username"""
        # # 1.step 从cookie获得登录要换个户
        # user = self.get_current_user_cookie()
        # if user is not None:
        #     return user
        # 2.step 从api_token获得登录用户
        user = self.get_current_user_from_token(db, token)
        if user is not None:
            return user
        # 3.step 从用户名密码获得登录用户
        user = self.get_current_user_from_argments(db, username, password)
        if user is not None:
            return user

        return user

    def __call__(self, *args, ST=None, token=None, username=None, password=None, session_id=None):
        db = get_thread_session()
        current_user = self.get_current_user(db, token, username, password)
        if current_user is not None:
            return current_user

        if ST is None:
            # 无票据信息，需要重新登录
            re_server_router = self.server_router
            if self.server_router.startswith('/'):
                re_server_router = self.server_router[1:]

            server_url = ''
            if re_server_router.startswith('#'):
                server_url = '{}/{}'.format(self.cas_url, quote_plus(re_server_router))

            cas_url = '{}/cas/auth/?redirect={}'.format(self.cas_url, server_url)
            set_context_redirect_info(session_id, cas_url)
            set_context_exit_info(session_id, detail="login or re-auth user")
            return None

        if current_user is None or current_user.service_ticket != ST:
            # 从CAS获得登录信息
            response = HttpRpc('', self.cas_prefix, self.cas_ip, self.cas_port).cas.auth.post(ST=ST)
            if response['status'] == 'OK':
                # 当前用户登录成功
                user_name = response['content']['user']
                is_admin = response['content']['admin']
                # 记录当前用户信息，如果不存在则创建
                user = db.query(get_db_orm().User).filter(get_db_orm().User.name == user_name).one_or_none()

                if user is None:
                    # 创建用户
                    user = get_db_orm().User(name=user_name, admin=is_admin)
                    db.add(user)
                    db.commit()

                if user.admin != is_admin:
                    user.admin = is_admin

                # 设置登录状态
                user.cookie_id = str(uuid.uuid4()) + '/' + str(uuid.uuid4())
                db.commit()
                set_context_cookie_info(session_id, 'antgo-user', user.cookie_id)
                return user
            else:
                # 票据失效，需要重新登录
                re_server_router = self.server_router
                if self.server_router.startswith('/'):
                    re_server_router = self.server_router[1:]

                server_url = ''
                if re_server_router.startswith('#'):
                    server_url = '{}/{}'.format(self.cas_url, quote_plus(re_server_router))

                cas_url = '{}/cas/auth/?redirect={}'.format(self.cas_url, server_url)
                set_context_redirect_info(session_id, cas_url)
                set_context_exit_info(session_id, detail="login or re-auth user")
                return None

        return current_user
