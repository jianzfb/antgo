# -*- coding: UTF-8 -*-
# @Time    : 2022/9/10 23:06
# @File    : base_execution.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import traceback

class BaseExecution:
    """
    Execute an operators
    """
    def __apply__(self, *arg, **kws):
        # Multi inputs.
        if isinstance(self._index[0], tuple):
            args = [getattr(arg[0], x) for x in self._index[0]]
        # Single input.
        else:
            args = [getattr(arg[0], self._index[0])]
        
        return self._op(*args, **kws)

    def __call__(self, *arg, **kws):
        self.__check_init__()
        try:
            if bool(self._index):
                res = self.__apply__(*arg, **kws)

                # Multi outputs.
                if isinstance(res, tuple):
                    if not isinstance(self._index[1],
                                    tuple) or len(self._index[1]) != len(res):
                        raise IndexError(
                            f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                        )
                    for i, j in zip(self._index[1], res):
                        setattr(arg[0], i, j)
                # Single output.
                else:
                    setattr(arg[0], self._index[1], res)

                return arg[0]
            else:
                res = self._op(*arg, **kws)
                return res
        except Exception:
            traceback.print_exc()