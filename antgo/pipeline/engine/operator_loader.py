# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 17:48
# @File    : operator_loader.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import importlib
import sys
import subprocess
from pathlib import Path
from typing import Any, List, Dict, Union

# import pkg_resources
# from pkg_resources import DistributionNotFound
from antgo.pipeline.operators import Operator
from antgo.pipeline.operators.nop import NOPOperator
# from towhee.operators.concat_operator import ConcatOperator
from antgo.pipeline.engine import *
# from towhee.hub.file_manager import FileManager
# from towhee.hparam import param_scope

from antgo.pipeline.hparam import param_scope


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
        module, fname = function.split('/')
        op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
        module = '.'.join(['antgo.pipeline.hub.external', '{}.{}'.format(module, fname)])

        try:
            op = getattr(importlib.import_module(module), op_cls)
        except:
            subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'openmim'])
            subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmengine'])
            subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmcv-full'])
            if fname == 'detector':
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmdet'])
            elif fname == 'segmentor':
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmsegmentation'])
            elif fname == 'classification':
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmcls'])
            elif fname == 'ocr':
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmocr'])
            elif fname == 'pose':
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', 'mmpose'])
            elif fname == 'editing':
                subprocess.check_call([sys.executable, '-m', 'git', 'clone', 'https://github.com/open-mmlab/mmediting.git'])
                subprocess.check_call([sys.executable, '-m', 'cd', 'mmediting'])
                subprocess.check_call([sys.executable, '-m', 'pip3', 'install', '-e', '.'])

            op = getattr(importlib.import_module(module), op_cls)

        return self.instance_operator(op, arg, kws) if op is not None else None

    def load_operator_from_packages(self, function: str, arg: List[Any], kws: Dict[str, Any], tag: str) -> Operator:  # pylint: disable=unused-argument
        try:
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
                        self.load_operator_from_packages]:
            op = factory(function, arg, kws, tag)
            if op is not None:
                return op
        return None

    def instance_operator(self, op, arg: List[Any], kws: Dict[str, Any]) -> Operator:
        with param_scope() as hp:
            return op(*arg, **kws) if kws is not None else op()
