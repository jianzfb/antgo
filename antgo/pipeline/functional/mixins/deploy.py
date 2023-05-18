# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 21:56
# @File    : deploy.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import os
import json
from antgo.pipeline.functional.common.config import *
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
from antgo.pipeline.extent.op.loader import *


def auto_generate_eagleeye_op(op_name, op_index, op_args, op_kwargs):
    func = extent.func(op_name)

    # 创建header文件
    eagleeye_warp_h_code_content = \
        gen_code(
            './templates/op_code.h',
            )

    # 创建cpp文件
    eagleeye_warp_cpp_code_content = \
        gen_code('./templates/op_code.cpp')
    
    source_to_so_ctx(build_path, srcs, target_name, ctx)
    pass


def __android_package_build():
    graph_config = get_graph_info()
    for graph_op_info in graph_config:
        op_name = graph_op_info['op_name']
        op_index = graph_op_info['op_index']
        op_args = graph_op_info['op_args']
        op_kwargs = graph_op_info['op_kwargs']

        if op_name.startswith('deploy'):
            # 需要独立编译
            # 1.step 生成eagleeye算子封装

            # 2.step 编译封装的算子
            pass
        else:
            # eagleey核心库中存在的算子
            pass
    pass


def __linux_package_build():
    pass


class DeployMixin:
    def build(self, platform='android', output_path='./deploy'):
        # 编译
        # 1.step 基于不同平台对CPP算子编译，并生成静态库
        if platform == 'android':
            __android_package_build()
        elif platform == 'linux':
            __linux_package_build()
        else:
            return False

        # 2.step 编译eagleeye 插件库，并关联上面的静态库

        print(graph_config)
        return True

    def run(self, platform='android'):
        # 编译
        # 运行
        pass