# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : command.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.application.table.table import *
from antgo.pipeline.functional.mixins.db import *
from antgo import config
import threading
import logging
import os
import cv2
import base64
import numpy as np
from sqlalchemy import and_, or_
import traceback


# 配置数据库表示例
# update_table_info( {
#             'table': 'task',        # 构建新表名字
#             'links': ['user'],         # 关联表
#             'fields': {
#                 'task_name': 'str',
#                 'task_progress': 'str',
#                 'task_create_time': 'date',
#                 'task_stop_time': 'date',
#                 'task_is_finish': 'bool'
#             }
#     })


class SafeCall(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *argc, **kwargs):
        try:
            self.func(*argc, **kwargs)
        except Exception as e:  # swallow any exception
            print(e)
            traceback.print_exc()


class CommandOp(object):
    def __init__(self, func):
        self.func = SafeCall(func)

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def progress(self, user_name, task_name, task_progress, task_finish):
        orm_handler = get_db_orm()
        orm_table = orm_handler.Task
        with local_session_context() as db:
            task = db.query(orm_table).filter(orm_table.task_name == task_name).one_or_none()
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
        orm_handler = get_db_orm()
        orm_table = orm_handler.Task
        db = get_thread_session()
        current_user, task_name, task_script = args[:3]
        task = db.query(orm_table).filter(and_(orm_table.user == current_user, orm_table.task_name == task_name)).one_or_none()
        if task is not None:
            # 返回任务进度
            return {'status': 'running', 'progress': task.task_progress, 'create_time': task.task_create_time.strftime('%Y-%m-%d %H:%M:%S'), 'stop_time': task.task_stop_time.strftime('%Y-%m-%d %H:%M:%S') if task.task_is_finish else ''}

        # 创建新任务
        params = {'user': current_user}
        params['task_name'] = task_name
        params['task_script'] = task_script
        task = orm_table(**params)
        db.add(task)
        db.commit()

        # 启动任务
        thread = threading.Thread(target=self.func, args=(current_user.name, task_name, task_script, self.progress))
        thread.start()
        return {'status': 'start', 'progress': 'start', 'create_time': task.task_create_time.strftime('%Y-%m-%d %H:%M:%S')}
