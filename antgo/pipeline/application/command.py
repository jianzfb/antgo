# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : command.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo import config
import os
import cv2
import base64
import numpy as np

# 启动，进度
class BackgroundTask(object):
    def __init__(self):
        self.task_message = None # 记录到数据库 task_user, task_id, task_message
        pass

    def _info(self):
        return ['db', 'user']

    def _config(self):
        # 自定义db设计
        {
            'table': 'task',    # 构建新表名字
            'link': 'user',      # 关联表
            'fields': {
                'task_name': 'str',
                'task_create_time': 'date'
            }
        }

    def __call__(self, *args, db=None, user=None):
        # input
        # db, user, params

        # 执行
        # 独立线程，执行后台func(task_id, task_message, **params)


        # output
        # task_id, create_time
        pass
