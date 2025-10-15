# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : auth.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.env import *
from antgo.pipeline.application.common.db import *
from urllib.parse import unquote_plus, quote_plus
from antgo.pipeline.utils.reserved import *
import os
import cv2
import base64
import numpy as np
import re
from sqlalchemy import and_, or_


# pattern for the authentication token header
auth_header_pat = re.compile(r'^token\s+([^\s]+)$')


class AuthOp(object):
    def __init__(self, cas_ip=None, cas_port=None, cas_proto='http', cas_prefix='antvis', server_router="/#/Login", cookie_prefix='antgo-auth'):
        self.cas_url = f'{cas_proto}://{cas_ip}:{cas_port}/{cas_prefix}'
        self.cas_ip = cas_ip
        self.cas_port = cas_port
        self.cas_proto = cas_proto
        self.cas_prefix = cas_prefix
        self.server_router = server_router
        self.cookie_prefix = cookie_prefix

    def info(self):
        return ['ST', 'username', 'password', 'session_id', 'db', 'headers', 'cookie']

    def get_current_user(self, db, username, password, headers, cookie):
        """get current username"""
        orm = get_db_orm()
        # 1.step 从cookie获得登录要换个户
        cookie_id = cookie.get(self.cookie_prefix, None)
        if cookie_id is not None:
            user = db.query(orm.User).filter(orm.User.cookie_id == cookie_id).one_or_none()
            if user is not None:
                return user

        # 2.step 从api_token获得登录用户
        api_token = headers.get('Authorization', None)
        if api_token is not None:
            api_token = orm.APIToken.find(db, api_token, kind='user')
            if api_token is not None and api_token.user is not None:
               return api_token.user

        # 3.step 从用户名密码获得登录用户
        if username is None or password is None:
            return None
        user = db.query(orm.User).filter(orm.User.name == username, orm.User.password == password).one_or_none()
        if user is not None:
            return user

        return None

    @resource_db_env
    def __call__(self, ST, username, password, session_id, db, headers, cookie):
        current_user = self.get_current_user(db, username, password, headers, cookie)
        if current_user is not None:
            return current_user

        if self.cas_ip is None or self.cas_port is None:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "need to login"
                },
                session_id=session_id,
                status_code=401,
                message="need to login"
            )

        if ST is None:
            # 无票据信息，需要重新登录
            re_server_router = self.server_router
            if self.server_router.startswith('/'):
                re_server_router = self.server_router[1:]

            server_url = ''
            if re_server_router.startswith('#'):
                server_url = '{}/{}'.format(self.cas_url, quote_plus(re_server_router))

            cas_url = '{}/cas/auth/?redirect={}'.format(self.cas_url, server_url)
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "login or re-auth user"
                },
                session_id=session_id,
                status_code=401,
                message="login or re-auth user",
                redirect_url=cas_url
            )

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
                cookie.set(self.cookie_prefix, user.cookie_id)

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
                return ReservedRtnType(
                    index = '__response__',
                    data = {
                        'code': -1,
                        'message': 'fail',
                        'info': "login or re-auth user"
                    },
                    session_id=session_id,
                    status_code=401,
                    message="login or re-auth user",
                    redirect_url=cas_url
                )

        return current_user
