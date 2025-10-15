# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : logout.py
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


class LogoutOp(object):
    def __init__(self,cookie_prefix='antgo-auth'):
        self.cookie_prefix = cookie_prefix

    def info(self):
        return ['session_id', 'cookie']

    @resource_db_env
    def __call__(self, *args, session_id, cookie, **kwargs):
        cookie.clear(self.cookie_prefix)
        return True