# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : command.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.application.table.table import *
from antgo.pipeline.application.common.db import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.env import *
from antgo.pipeline.utils.reserved import *
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
    def __init__(self, func, bind):
        self.func = SafeCall(func)
        self.bind = bind

    def info(self):
        # 设置需要使用隐信息（数据库、session_id）
        return ['session_id']

    def progress(self, user_name, task_name, task_progress, task_finish, task_success=True):
        orm_handler = get_db_orm()
        orm_table = orm_handler.Task

        try:
            db = get_db_session()()
            task = db.query(orm_table).filter(orm_table.task_name == task_name).one_or_none()
            if task is None:
                # no task
                return

            task.task_progress = task_progress
            if task_finish:
                task.task_is_finish = True
                task.task_is_success = task_success
                task.task_stop_time = datetime.datetime.now()
            db.commit()
        except Exception as e:
            db.rollback()
        finally:
            db.close()


    @resource_db_env
    def __call__(self, *args, session_id, db):
        # 启动新任务或查询进度
        orm_handler = get_db_orm()
        orm_table = orm_handler.Task
        current_user, task_name, bind_obj, task_script = args[:4]
        task = db.query(orm_table).filter(and_(orm_table.user == current_user, orm_table.task_name == task_name)).one_or_none()
        if task is not None:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': "task has existed in db"
                },
                session_id=session_id,
                status_code=401,
                message="task has existed in db"
            )

        # 创建新任务
        params = {'user': current_user}
        params['task_name'] = task_name
        params['task_script'] = task_script
        params[self.bind] = [bind_obj]
        task = orm_table(**params)
        db.add(task)
        db.commit()

        # 启动任务
        thread = threading.Thread(target=self.func, args=(current_user.name, task_name, task_script, self.progress))
        thread.start()
        return {'status': 'start', 'progress': 'start', 'create_time': task.task_create_time.strftime('%Y-%m-%d %H:%M:%S')}
