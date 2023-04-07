# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:52
# @File    : column.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from enum import Flag, auto

from antgo.pipeline.hparam.hyperparameter import param_scope

from antgo.pipeline.functional.storages import ChunkedTable, WritableTable


# pylint: disable=import-outside-toplevel
# pylint: disable=bare-except
class ColumnMixin:
    """
    Mixins to support column-based storage.
    """

    class ModeFlag(Flag):
        ROWBASEDFLAG = auto()
        COLBASEDFLAG = auto()
        CHUNKBASEDFLAG = auto()

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_chunksize'):
            self._chunksize = parent._chunksize

    def set_chunksize(self, chunksize):
        """
        Set chunk size for arrow
        """

        self._chunksize = chunksize
        chunked_table = ChunkedTable(chunksize=chunksize, stream=self.is_stream)
        chunked_table.feed(self._iterable)
        return self._factory(chunked_table, parent_stream=False, mode=self.ModeFlag.CHUNKBASEDFLAG)

    def get_chunksize(self):
        return self._chunksize
