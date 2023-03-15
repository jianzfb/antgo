# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 20:52
# @File    : stream.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from typing import Iterator

from antgo.pipeline.functional.mixins.dag import register_dag


class StreamMixin:
    """
    Stream related mixins.
    """
    @register_dag
    def stream(self):
        """
        Create a stream data collection.

        Examples:
        1. Convert a data collection to streamed version

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc.is_stream
        False

        >>> dc = dc.stream()
        >>> dc.is_stream
        True
        """
        # pylint: disable=protected-access
        iterable = iter(self._iterable) if not self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

    @register_dag
    def unstream(self):
        """
        Create a unstream data collection.

        Examples:

        1. Create a unstream data collection

        >>> dc = DataCollection(iter(range(5))).unstream()
        >>> dc.is_stream
        False

        2. Convert a streamed data collection to unstream version

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True
        >>> dc = dc.unstream()
        >>> dc.is_stream
        False
        """
        iterable = list(self._iterable) if self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

    @property
    def is_stream(self):
        """
        Check whether the data collection is stream or unstream.

        Examples:

        >>> from typing import Iterable
        >>> dc = DataCollection([0,1,2,3,4])
        >>> dc.is_stream
        False

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        False
        >>> result._iterable
        [1, 2, 3, 4, 5]

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        True
        >>> isinstance(result._iterable, Iterable)
        True

        """
        return isinstance(self._iterable, Iterator)
