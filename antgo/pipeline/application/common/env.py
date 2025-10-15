# -*- coding: UTF-8 -*-
# @Time    : 2025/10/10 22:42
# @File    : env.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.application.common.db import *
from antgo.pipeline.utils.reserved import *
import os
import sys
import traceback

def resource_db_env(func):
    def wrapper(*args, **kwargs):
        use_db = False
        result = None
        try:
            if kwargs.get('db', None) is None:
                kwargs['db'] = get_db_session()()
                use_db = True
            result = func(*args, **kwargs)
        except Exception as e:
            if use_db:
                kwargs['db'].rollback()

            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "pipeline inner error"
                },
                status_code=500,
                message="pipeline inner error"
            )
        finally:
            if use_db:
                kwargs['db'].close()
        return result

    return wrapper