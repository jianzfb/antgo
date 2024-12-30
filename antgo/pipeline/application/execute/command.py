# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : command.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.functional.mixins.db import *
from antgo import config
import threading
import os
import cv2
import base64
import numpy as np

# 配置数据库表示例
# get_table_info().update({
#     'application/execute/command': {
#             'table': 'task',        # 构建新表名字
#             'links': ['user'],         # 关联表
#             'fields': {
#                 'task_name': 'str',
#                 'task_progress': 'str',
#                 'task_create_time': 'date',
#                 'task_stop_time': 'date',
#                 'task_is_finish': 'bool'
#             }
#     }
# })



class CommandOp(object):
    def __init__(self, func):
        self.func = func

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def progress(self, user_name, task_name, task_progress, task_finish):
        orm = get_db_orm()
        with thread_session_context(get_db_session()) as db:
            task = db.query(orm.Task).filter(orm.Task.task_name == task_name).one_or_none()
            if task is None:
                # no task
                return

            task.task_progress = task_progress
            if task_finish:
                task.task_is_finish = True
                task.task_stop_time = datetime.datetime.now()

            db.commit()

    def __call__(self, *args, session_id):
        # 启动新任务或查询进度
        current_user, task_name = args[:2]
        orm = get_db_orm()
        with thread_session_context(get_db_session()) as db:
            task = db.query(orm.Task).filter(orm.Task.user_id == current_user.id).one_or_none()
            if task is not None:
                # 返回任务进度
                return {'status': 'running', 'progress': task.task_progress, 'create_time': task.task_create_time.strftime('%Y-%m-%d %H:%M:%S'), 'stop_time': task.task_stop_time.strftime('%Y-%m-%d %H:%M:%S') if task.task_is_finish else ''}

            # 创建新任务
            task = orm.Task(task_name=task_name, user=current_user)
            db.add(task)
            db.commit()

            # 启动任务
            thread = threading.Thread(target=self.func, args=(current_user.name, task_name, self.progress))
            thread.start()
            return {'status': 'start', 'progress': '', 'create_time': task.task_create_time.strftime('%Y-%m-%d %H:%M:%S')}

