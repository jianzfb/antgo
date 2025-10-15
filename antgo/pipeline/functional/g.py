# -*- coding: UTF-8 -*-
# @Time    : 2025/10/08 22:43
# @File    : g.py
# @Author  : jian<jian@mltalker.com>


class gEnv(object):
    # 记录管线名字和对应的算子管线
    # {
    #   pipeline_name: [...]
    # }
    _g_pipeline_op_map = {}
    # 当前正在装配的管线名字
    _g_active_pipeline_name = None