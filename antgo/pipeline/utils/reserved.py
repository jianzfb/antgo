# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : reserved.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


class ReservedRtnType(object):
    def __init__(self, index=None, data=None, message='', status_code=200, session_id=None, redirect_url=None):
        self.index = index
        self.data = data
        self.message = message
        self.status_code = status_code
        self.session_id = session_id
        self.redirect_url = redirect_url
