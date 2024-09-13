# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 17:48
# @File    : operator_loader.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import importlib
import sys
import os
import subprocess
from pathlib import Path
from typing import Any, List, Dict, Union
import traceback
from antgo.pipeline.operators import Operator
from antgo.pipeline.operators.nop import NOPOperator
from antgo.pipeline.engine import *
from antgo.pipeline.hparam import param_scope
import pathlib
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)


class OperatorLoader:
    """Wrapper class used to load operators from either local cache or a remote
    location.

    Args:
        cache_path: (`str`)
            Local cache path to use. If not specified, it will default to
            `$HOME/.antgo/operators`.
    """
    def __init__(self, cache_path: str = None):
        if cache_path is None:
            self._cache_path = LOCAL_OPERATOR_CACHE
        else:
            self._cache_path = Path(cache_path)

    def _load_interal_op(self, op_name: str, arg: List[Any], kws: Dict[str, Any]):
        if op_name in ['_start_op', '_end_op']:
            return NOPOperator()
        elif op_name == '_concat':
            return ConcatOperator(*arg, **kws)
        else:
            # Not a interal operators
            return None

    def load_operator_from_internal(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        return self._load_interal_op(function, arg, kws)

    def load_operator_from_registry(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        op = OperatorRegistry.resolve(function)
        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_mm(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if not function.startswith('mm'):
            return None

        module, fname = function.split('/')
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        module = '.'.join(['antgo.pipeline.hub.external', '{}.{}'.format(module, fname)])

        try:
            op = getattr(importlib.import_module(module), op_cls)
        except:
            subprocess.check_call(['pip3', 'install', 'openmim'])
            subprocess.check_call(['mim', 'install', 'mmengine'])
            subprocess.check_call(["mim", "install", "mmcv>=2.0.0rc1"])
            if fname == 'detector':
                subprocess.check_call(['pip3', 'install', 'mmdet'])
            elif fname == 'segmentor':
                subprocess.check_call(['pip3', 'install', 'mmsegmentation'])
            elif fname == 'classification':
                subprocess.check_call(['pip3', 'install', 'mmcls'])
            elif fname == 'ocr':
                subprocess.check_call(['pip3', 'install', 'mmdet'])
                subprocess.check_call(['pip3', 'install', 'mmocr'])
            elif fname == 'pose':
                subprocess.check_call(['pip3', 'install', 'mmpose'])
            elif fname == 'editing':
                subprocess.check_call(['git', 'clone', 'https://github.com/open-mmlab/mmediting.git'])
                subprocess.check_call(['cd', 'mmediting'])
                subprocess.check_call(['pip3', 'install', '-e', '.'])

            op = getattr(importlib.import_module(module), op_cls)

        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_mp(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str):
        if not function.startswith('mp'):
            return None

        module, fname = function.split('/')
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        module = '.'.join(['antgo.pipeline.hub.external', '{}.{}'.format(module, fname)])

        try:
            op = getattr(importlib.import_module(module), op_cls)
        except:
            subprocess.check_call(['pip3', 'install', 'mediapipe'])
            op = getattr(importlib.import_module(module), op_cls)
        
        return self.instance_operator(op, arg, kws) if op is not None else None


    def load_operator_from_deploy(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if not function.startswith('deploy'):
            return None

        module, fname = function.split('/')
        op = getattr(importlib.import_module('antgo.pipeline.deploy.cpp_op'), 'CppOp', None)
        if op is None:
            return None        
        kws.update({
            'func_op_name': fname.replace('-', '_')
        })
        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_eagleeye(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if not function.startswith('eagleeye'):
            return None

        module, category, fname = function.split('/')
        os.makedirs(ANTGO_DEPEND_ROOT, exist_ok=True)
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', 'py')):
            if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')):
                os.system('cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

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
        op = None
        if category.lower() == 'op':
            op = getattr(importlib.import_module('antgo.pipeline.eagleeye.core_op'), 'CoreOp', None)
        elif category.lower() == 'node':
            op = getattr(importlib.import_module('antgo.pipeline.eagleeye.node_op'), 'CoreNode', None)
        elif category.lower() == 'pipeline':
            op = getattr(importlib.import_module('antgo.pipeline.eagleeye.pipeline_op'), 'CorePipeline', None)
        else:
            op = getattr(importlib.import_module('antgo.pipeline.eagleeye.exe_op'), 'Exe', None)
        if op is None:
            return None
        kws.update({
            'func_op_name': fname.replace('-', '_')
        })
        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_packages(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        try:
            if len(function.split('/')) > 2:
                return None

            module, fname = function.split('/')
            load_func = getattr(self, f'load_operator_from_{module}', None)
            if load_func is not None:
                return load_func(function, arg, kws, tag)

            fname = fname.replace('-', '_')
            op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))

            # module = '.'.join([module, fname, fname])
            module = '.'.join(['antgo.pipeline.models', '{}.{}'.format(module, fname), fname])
            op = getattr(importlib.import_module(module), op_cls)
            return self.instance_operator(op, arg, kws) if op is not None else None
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            return None

    def split_function(self, function_key_name_list):
        # eagleeye
        # mm
        # mp
        # deploy
        function_list = []
        while len(function_key_name_list) != 0:
            if function_key_name_list[0].startswith('eagleeye'):
                function_list.append('/'.join(function_key_name_list[:3]))
                function_key_name_list = function_key_name_list[3:]
            elif function_key_name_list[0].startswith('mm'):
                function_list.append('/'.join(function_key_name_list[:2]))
                function_key_name_list = function_key_name_list[2:]
            elif function_key_name_list[0].startswith('mp'):
                function_list.append('/'.join(function_key_name_list[:2]))
                function_key_name_list = function_key_name_list[2:]
            elif function_key_name_list[0].startswith('deploy'):
                function_list.append('/'.join(function_key_name_list[:2]))
                function_key_name_list = function_key_name_list[2:]
            else:
                function_list.append(function_key_name_list[0])
                function_key_name_list = function_key_name_list[1:]
        return function_list

    def load_operator_from_control(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        if not function.startswith('control'):
            return None

        info = function.split('/')
        control_op_name = info[1]
        if control_op_name == 'For':
            function_op_name_list = self.split_function(info[2:])
            function_op_name = function_op_name_list[0]
            function_op = self.load_operator(function_op_name, arg, kws.get(function_op_name.replace('-', '_'), {}), tag)
            assert(function_op is not None)
            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.for_op'), 'For', None)
            return self.instance_operator(control_op_cls, [], dict(func=function_op, parallel_num=kws.get('parallel_num', 1)))
        elif control_op_name == 'If':
            true_func_index = info.index('true-func')
            false_func_index = info.index('false-func')
            function_op_name_list = self.split_function(info[true_func_index+1:false_func_index])
            true_func_name = function_op_name_list[0]

            function_op_name_list = self.split_function(info[false_func_index+1:])
            false_func_name = function_op_name_list[0]

            true_func_op = self.load_operator(true_func_name, [], kws.get('true_func', dict()), tag)
            false_func_op = self.load_operator(false_func_name, [], kws.get('false_func', dict()), tag)

            assert(true_func_op is not None)
            assert(false_func_op is not None)

            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.if_op'), 'If', None)
            return self.instance_operator(control_op_cls, [], dict(true_func=true_func_op, false_func=false_func_op))
        elif control_op_name == 'IfNotNone':
            function_op_name_list = self.split_function(info[2:])
            function_op_name = function_op_name_list[0]
            function_op = self.load_operator(function_op_name, arg, kws, tag)
            assert(function_op is not None)

            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.ifnotnone_op'), 'IfNotNone', None)
            return self.instance_operator(control_op_cls, [], dict(func=function_op))
        elif control_op_name == 'Interval':
            function_op_name_list = self.split_function(info[2:])
            function_op_name = function_op_name_list[0]

            interval = 1
            if 'interval' in kws:
                interval = kws.pop('interval')
            function_op = self.load_operator(function_op_name, arg, kws.get(function_op_name, {}), tag)
            assert(function_op is not None)
            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.interval_op'), 'Interval', None)
            return self.instance_operator(control_op_cls, [], dict(func=function_op, interval=interval))
        elif control_op_name == 'Cache':
            function_op_name_list = self.split_function(info[2:])
            function_op_name = function_op_name_list[0]

            function_op = self.load_operator(function_op_name, arg, kws.get(function_op_name, {}), tag)
            assert(function_op is not None)
            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.cache_op'), 'Cache', None)
            cache_kwargs = dict(func=function_op)
            if function_op_name in kws:
                kws.pop(function_op_name)
            cache_kwargs.update(kws)
            return self.instance_operator(control_op_cls, [], cache_kwargs)
        elif control_op_name == 'DetectOrTracking':
            function_op_name_list = self.split_function(info[2:])
            det_function_op_name = function_op_name_list[0]
            tracking_function_op_name = None
            if len(function_op_name_list) > 1:
                tracking_function_op_name = function_op_name_list[1]

            det_function_op = self.load_operator(det_function_op_name, arg, kws.get(det_function_op_name, {}), tag)
            assert(det_function_op is not None)
            tracking_function_op = None
            if tracking_function_op_name is not None:
                tracking_function_op = self.load_operator(tracking_function_op_name, arg, kws.get(tracking_function_op_name, {}), tag)
                assert(tracking_function_op is not None)

            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.detect_or_tracking_op'), 'DetectOrTracking', None)
            detect_or_tracking_kwargs = dict(det_func=det_function_op, tracking_func=tracking_function_op)
            if det_function_op_name in kws:
                kws.pop(det_function_op_name)
            if tracking_function_op_name is not None:
                if tracking_function_op_name in kws:
                    kws.pop(tracking_function_op_name)

            detect_or_tracking_kwargs.update(kws)
            return self.instance_operator(control_op_cls, [], detect_or_tracking_kwargs)
        elif control_op_name == 'Asyn':
            function_op_name_list = self.split_function(info[2:])
            function_op_name = function_op_name_list[0]
            function_op = self.load_operator(function_op_name, arg, kws.get(function_op_name.replace('-', '_'), {}), tag)
            assert(function_op is not None)
            control_op_cls = getattr(importlib.import_module('antgo.pipeline.control.asyn_op'), 'Asyn', None)
            return self.instance_operator(control_op_cls, [], dict(func=function_op))
        return None

    def load_operator_from_remote(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:
        # 使用云端API
        return None

    def load_operator(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:
        """Attempts to load an operators from cache. If it does not exist, looks up the
        operators in a remote location and downloads it to cache instead. By standard
        convention, the operators must be called `Operator` and all associated data must
        be contained within a single directory.

        Args:
            function: (`str`)
                Origin and method/class name of the operators. Used to look up the proper
                operators in cache.
        Raises:
            FileExistsError
                Cannot find operators.
        """

        for factory in [self.load_operator_from_internal,
                        self.load_operator_from_registry,
                        self.load_operator_from_remote,
                        self.load_operator_from_packages,
                        self.load_operator_from_eagleeye,
                        self.load_operator_from_deploy,
                        self.load_operator_from_control]:
            op = factory(function, arg, kws, tag)
            if op is not None:
                return op
        return None

    def instance_operator(self, op, arg: List[Any], kws: Dict[str, Any]) -> Operator:
        with param_scope() as hp:
            return op(*arg, **kws) if kws is not None else op()
