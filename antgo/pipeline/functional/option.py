# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 22:44
# @File    : option.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from typing import Callable, Generic
from typing import Any, TypeVar

A = TypeVar('A', covariant=True)
B = TypeVar('B')
T = TypeVar('T')


class _Reason:
    """
    reason for `Empty` value
    """

    def __init__(self, x: Any, e: Exception) -> None:
        self._value = x
        self._exception = e

    @property
    def value(self):
        return self._value

    @property
    def exception(self):
        return self._exception


class Option(Generic[A]):
    """
    Functional-style error handling.

    Option[A] = Some(A) or Empty[A]
    1. Some(A): just a container for result;
    2. Empty[A]: result is empty, because of input error or computation error;

    Examples:

    >>> a = Some(10)
    >>> a.map(lambda x: x/2.0)
    Some(5.0)

    >>> a.map(lambda x: x/0)
    Empty()

    >>> b = Empty()
    >>> b.map(lambda x: x/2.0)
    Empty()
    """

    @staticmethod
    def of(x: T):
        """Return a boxed value

        Args:
            x (T): input value

        Returns:
            Some(T): boxed value
        """
        return Some(x)

    @staticmethod
    def empty():
        """Return an empty value

        Returns:
            Empty: empty value
        """
        return Empty()

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        """Apply boxed version of callable

        Args:
            f (Callable[[A], Option[B]]): boxed version of callable

        Returns:
            Option[B]: boxed value

        Examples:

        >>> Option.of(1).flat_map(lambda x: x+1)
        2

        >>> Option.empty().flat_map(lambda x: x+1)
        Empty()
        """
        if isinstance(self, Some):
            return f(self._value)
        else:
            return self

    def map(self, f: Callable[[A], 'B']) -> 'Option[B]':
        """Apply function to value

        Args:
            f (Callable[[A], B]): unboxed function

        Returns:
            Option[B]: boxed return value
        """

        def wrapper(x):
            try:
                return Some(f(x))
            except Exception as e:  # pylint: disable=broad-except
                return Empty(x, e)

        return self.flat_map(wrapper)

    def is_empty(self):
        """Return True if the value is empty.
        """
        return isinstance(self, Empty)

    def is_some(self):
        """Return True if the value is boxed value.
        """
        return isinstance(self, Some)

    def get_or_else(self, default):
        """Return unboxed value, or default is the value is empty.

        Examples:

        >>> Option.of(0).get_or_else(1)
        0

        >>> Option.empty().get_or_else(1)
        1
        """
        if self.is_some():
            return self.get()
        return default


class Some(Option[A]):
    """
    `Some` value for `Option`
    """

    def __init__(self, x: A) -> None:
        self._value = x

    def __repr__(self) -> str:
        return 'Some({})'.format(self._value)

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return f(self._value)

    def get(self):
        """Return unboxed value
        """
        return self._value


class Empty(Option[A]):
    """
    `Empty` value for `Option`
    """

    def __init__(self, x: Any = None, e: Exception = None) -> None:
        self._reason = _Reason(x, e)

    def __repr__(self) -> str:
        return 'Empty()'

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return self

    def get(self):
        """Return the reason of the empty value.
        """
        return self._reason


empty = Empty()
