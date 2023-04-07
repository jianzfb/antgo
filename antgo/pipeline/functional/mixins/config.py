# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:49
# @File    : config.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from typing import Union, List
from antgo.pipeline.hparam import param_scope


class ConfigMixin:
    """
    Mixin to config DC, such as set the `parallel`, `chunksize`, `jit`.
    """

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_config'):
            self._config = parent._config
        else:
            self._config = None
        if parent is None or not hasattr(parent, '_num_worker'):
            self._num_worker = None
        if parent is None or not hasattr(parent, '_chunksize'):
            self._chunksize = None
        if parent is None or not hasattr(parent, '_jit'):
            self._jit = None
        if parent is None or not hasattr(parent, '_format_priority'):
            self._format_priority = None

    def config(self, parallel: int = None, chunksize: int = None, jit: Union[str, dict] = None, format_priority: List[str] = None):
        """
        Set the parameters in DC.

        Args:
            parallel (`int`):
               Set the number of parallel execution for following calls.
            chunksize (`int`):
               Set the chunk size for arrow.
            jit (`Union[str, dict]`):
               It can set to "numba", this mode will speed up the Operator's function, but it may also need to return to python mode due to JIT
               failure, which will take longer, so please set it carefully.
            format_priority (`List[str]`):
                The priority list of format.
        """
        dc = self
        if jit is not None:
            dc = dc.set_jit(compiler=jit)
        if parallel is not None:
            dc = dc.set_parallel(num_worker=parallel)
        if chunksize is not None:
            dc = dc.set_chunksize(chunksize=chunksize)
        if format_priority is not None:
            dc = dc.set_format_priority(format_priority=format_priority)
        return dc

    def get_config(self):
        """
        Return the config in DC, such as `parallel`, `chunksize`, `jit` and `format_priority`.
        """
        self._config = {}

        if hasattr(self, '_num_worker'):
            self._config['parallel'] = self._num_worker
        if hasattr(self, '_chunksize'):
            self._config['chunksize'] = self._chunksize
        if hasattr(self, '_jit'):
            self._config['jit'] = self._jit
        if hasattr(self, '_format_priority'):
            self._config['format_priority'] = self._format_priority
        return self._config

    def pipeline_config(self, parallel: int = None, chunksize: int = None, jit: Union[str, dict] = None, format_priority: List[str] = None):
        """
        Set the parameters in DC.

        Args:
            parallel (`int`):
               Set the number of parallel execution for following calls.
            chunksize (`int`):
               Set the chunk size for arrow.
            jit (`Union[str, dict]`):
               It can set to "numba", this mode will speed up the Operator's function, but it may also need to return to python mode due to JIT
               failure, which will take longer, so please set it carefully.
            format_priority (`List[str]`):
                The priority list of format.
        """
        dc = self
        if jit is not None:
            dc = dc.set_jit(compiler=jit)
        if parallel is not None:
            dc = dc.set_parallel(num_worker=parallel)
        if chunksize is not None:
            dc = dc.set_chunksize(chunksize=chunksize)
        if format_priority is not None:
            dc = dc.set_format_priority(format_priority=format_priority)
        return dc

    def get_pipeline_config(self):
        """
        Return the config in DC, such as `parallel`, `chunksize`, `jit` and `format_priority`.
        """
        self._pipeline_config = {}

        if hasattr(self, '_num_worker'):
            self._pipeline_config['parallel'] = self._num_worker
        if hasattr(self, '_chunksize'):
            self._pipeline_config['chunksize'] = self._chunksize
        if hasattr(self, '_jit'):
            self._pipeline_config['jit'] = self._jit
        if hasattr(self, '_format_priority'):
            self._pipeline_config['format_priority'] = self._format_priority
        return self._pipeline_config
