# -*- coding: UTF-8 -*-
# @Time    : 2022/9/10 23:06
# @File    : base_execution.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import traceback
from antgo.pipeline.engine.execution.base_data import *
from antgo.pipeline.deploy.cpp_op import CppOp
from antgo.pipeline.control.ifnotnone_op import IfNotNone
from antgo.pipeline.remote.remote_api import RemoteApiOp
from antgo.pipeline.functional.common.env import *

class BaseExecution:
    """
    Execute an operators
    """
    def __apply__(self, *arg, **kws):
        # Multi inputs.
        if isinstance(self._index[0], tuple):
            args = [getattr(arg[0], x) if hasattr(arg[0], x) else None for x in self._index[0]]
        # Single input or No input.
        else:
            if isinstance(self._index, str) or len(self._index) == 1:
                # no input
                args = []
            else:
                # single input
                args = [getattr(arg[0], self._index[0]) if hasattr(arg[0], self._index[0]) else None]
        if isinstance(self._op, CppOp):
            self._op._index = self._index
        if isinstance(self._op, IfNotNone):
            self._op._index = self._index
        if isinstance(self._op, RemoteApiOp):
            self._op._index = self._index

        if hasattr(self._op, 'info'):
            info = self._op.info()
            for key in self._op.info():
                kws.update({key: getattr(arg[0], key, None)})
        return self._op(*args, **kws)

    def __is_need_exit__(self, session_id=None):
        if session_id is None:
            return None
        exit_condition = get_context_exit_info(session_id, None)
        return exit_condition

    def __call__(self, *arg, **kws):
        if self.__is_need_exit__(arg[0].__dict__.get('session_id', None)):
            # 检查退出标记
            return None

        try:
            if bool(self._index):
                res = self.__apply__(*arg, **kws)

                # Multi outputs.
                if (isinstance(self._index, tuple) or isinstance(self._index, list)) and (isinstance(self._index[1], tuple) or isinstance(self._index[1], list)):
                    if not isinstance(self._index[1],
                                    tuple) or len(self._index[1]) != len(res):
                        raise IndexError(
                            f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                        )
                    for i, j in zip(self._index[1], res):
                        setattr(arg[0], i, j)
                # Single output.
                else:
                    if isinstance(res, NoUpdate):
                        return arg[0]

                    if isinstance(self._index, str):
                        setattr(arg[0], self._index, res)
                    else:
                        setattr(arg[0], self._index[1], res)
                return arg[0]
            else:
                if isinstance(self._op, CppOp):
                    self._op._index = self._index  
                if isinstance(self._op, IfNotNone):
                    self._op._index = self._index
                if isinstance(self._op, RemoteApiOp):
                    self._op._index = self._index

                if hasattr(self._op, '__info'):
                    info = self._op.info()
                    for key in self._op.__info():
                        kws.update({key: getattr(arg[0], key, None)})
                res = self._op(*arg, **kws)
                return res
        except Exception:
            # 打印异常信息
            traceback.print_exc()
            # 设置退出标记
            set_context_exit_info(arg[0].__dict__.get('session_id', None), detail='pipeline run abnormal')