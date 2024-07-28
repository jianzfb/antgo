# -*- coding: UTF-8 -*-
# @Time    : 2022/9/6 23:17
# @File    : __init__.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from pathlib import Path
import sys
import os
import traceback
from .operator_registry import OperatorRegistry # pylint: disable=import-outside-toplevel
from ..control.group_op import Group
from contextlib import contextmanager
import importlib
import copy

ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)

register = OperatorRegistry.register
resolve = OperatorRegistry.resolve

DEFAULT_LOCAL_CACHE_ROOT = Path.home() / '.antgo'
LOCAL_PIPELINE_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'pipelines'
LOCAL_OPERATOR_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'operators'

GroupDefMap = dict()

class GroupDef(object):
    def __init__(self, name):
        self.name = name

        self.op_creator_map = {
            'inner': self.create_inner_op,
            'eagleeye': self.create_eagleeye_op,
            'deploy': self.create_deploy_op
        }
        self.op_name_list = []
        self.op_args_list = []
        self.op_category_list = []
        self.op_relation = []

        self.op_name_cache = ''
        self.op_prefix = ''
        self.op_offset = 0

    def create_eagleeye_op(self, op_name, op_params):
        module, category, name = op_name.split('/')
        os.makedirs(ANTGO_DEPEND_ROOT, exist_ok=True)
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', 'py')):
            if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')):
                os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

            if 'darwin' in sys.platform:
                os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash osx_build.sh BUILD_PYTHON_MODULE && mv install py')
            else:
                first_comiple = False
                if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye','py')):
                    first_comiple = True
                os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash linux_build.sh BUILD_PYTHON_MODULE && mv install py')
                if first_comiple:
                    # 增加搜索.so路径
                    cur_abs_path = os.path.abspath(os.curdir)
                    so_abs_path = os.path.join(cur_abs_path, f"{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64")
                    os.system(f'echo "{so_abs_path}" >> /etc/ld.so.conf && ldconfig')
        
        op_cls_gen = None
        if category.lower() == 'op':
            op_cls_gen = getattr(importlib.import_module('antgo.pipeline.eagleeye.core_op'), 'CoreOp', None)
        elif category.lower() == 'node':
            op_cls_gen = getattr(importlib.import_module('antgo.pipeline.eagleeye.node_op'), 'CoreNode', None)
        elif category.lower() == 'pipeline':
            op_cls_gen = getattr(importlib.import_module('antgo.pipeline.eagleeye.pipeline_op'), 'CorePipeline', None)
        else:
            op_cls_gen = getattr(importlib.import_module('antgo.pipeline.eagleeye.exe_op'), 'Exe', None)
        op_params.update({
            'func_op_name': name.replace('-', '_')
        })
        return op_cls_gen(**op_params)

    def create_deploy_op(self, op_name, op_params):
        # 加入算子
        module, fname = op_name.split('/')
        op_cls_gen = getattr(importlib.import_module('antgo.pipeline.deploy.cpp_op'), 'CppOp', None)
        op_params.update({
            'func_op_name': fname.replace('-', '_')
        })

        return op_cls_gen(**op_params)

    def create_inner_op(self, op_name, op_params):
        op_name = op_name.replace('_','-')
        op_cls = OperatorRegistry.resolve(op_name)

        if op_params is None or len(op_params) == 0:
            op = op_cls()
        else:
            op = op_cls(**op_params)        
        return op

    def __call__(self, params, relation):
        # 定义图结构
        self.op_relation = relation

        # 定义算子
        group_op_list = []
        for op_name, op_category, op_param in zip(self.op_name_list, self.op_category_list, params):
            # 复制一份
            self.op_args_list.append(copy.deepcopy(op_param))

            # 生成op
            op = self.op_creator_map[op_category](op_name, op_param)
            group_op_list.append(op)

        # 动态创建类
        group_name = self.name
        group_op_relation = relation
        group_cls = \
            type(
                group_name, 
                (Group,), 
                {
                    '__init__': lambda self: 
                        Group.__init__(self, group_op_list, group_op_relation)
                }
            )

        # 注册到管线中
        OperatorRegistry.REGISTRY[group_name] = group_cls

    def __getattr__(self, name):
        if name.startswith('__'):
            return self.__dict__[name]

        if (name.startswith('deploy') or self.op_prefix == 'deploy'):
            # deploy.xxx
            if self.op_name_cache == '':
                self.op_name_cache = name
            else:
                self.op_name_cache += '/'+name

            if self.op_prefix == '':
                self.op_prefix = 'deploy'
            
            self.op_offset += 1
            if self.op_offset != 2:
                return self

            # 已经获得完整名字
            name = self.op_name_cache
            self.op_name_cache = ''
            self.op_prefix = ''
            self.op_offset = 0

            self.op_name_list.append(name)
            self.op_category_list.append('deploy')
            return self
        elif (name.startswith('eagleeye') or self.op_prefix == 'eagleeye'):
            # eagleeye.xxx.yyy
            if self.op_name_cache == '':
                self.op_name_cache = name
            else:
                self.op_name_cache += '/'+name

            if self.op_prefix == '':
                self.op_prefix = 'eagleeye'
            
            self.op_offset += 1
            if self.op_offset != 3:
                return self

            # 已经获得完整名字
            name = self.op_name_cache
            self.op_name_cache = ''
            self.op_prefix = ''
            self.op_offset = 0

            self.op_name_list.append(name)
            self.op_category_list.append('eagleeye')
            return self

        self.op_name_list.append(name)
        self.op_category_list.append('inner')        
        return self


@contextmanager
def GroupRegister(name):
  try:
    assert(name not in GroupDefMap)
    groupdef = GroupDef(name)
    yield groupdef
    GroupDefMap[name] = groupdef
  except:
    traceback.print_exc()
    raise sys.exc_info()[0]
