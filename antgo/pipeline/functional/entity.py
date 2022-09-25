# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:11
# @File    : entity.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from typing import overload


class Entity:
    """
    Entity is the basic data type in DataCollection.

    Users can create an Entity with free schema, which means there is no limit on attribute name and type.
    """

    @overload
    def __init__(self):
        """
        Create an empty entity.
        """
        ...

    def __init__(self, **kwargs):
        """
        Create an Entity with given attributes.
        """
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self):
        """
        Define the representation of the Entity.

        Examples:

        >>> from antgo.pipeline import Entity
        >>> e = Entity(a = 1, b = 2)
        >>> e
        <Entity dict_keys(['a', 'b'])>
        """
        content = repr(self.__dict__.keys())
        return f'<{self.__class__.__name__} {content.strip()}>'

    def __str__(self) -> str:
        """
        Return the str representation of an Entity.

        Examples:

        >>> from antgo.pipeline import Entity
        >>> e = Entity(a = 1, b = 2)
        >>> str(e)
        "{'a': 1, 'b': 2}"
        """
        return str(self.__dict__)


class EntityView:
    """
    The view to iterate DataFrames.

    Args:
        offset (`int`):
            The offset of an Entity in the table.

    Examples:

    >>> from antgo.pipeline import Entity, DataFrame
    >>> e = [Entity(a=a, b=b) for a,b in zip(range(3), range(3))]
    >>> df = DataFrame(e)
    >>> df = df.to_column()
    >>> df.to_list()[0]
    <EntityView dict_keys(['a', 'b'])>
    >>> df.to_list()[0].a
    0
    >>> df.to_list()[0].b
    0
    """

    def __init__(self, offset: int, table):
        self._offset = offset
        self._table = table

    def __getattr__(self, name):
        value = self._table[name][self._offset]
        try:
            return value.as_py()
        # pylint: disable=bare-except
        except:
            return value

    def __setattr__(self, name, value):
        if name in ('_table', '_offset'):
            self.__dict__[name] = value
            return
        self._table.write(name, self._offset, value)

        if self._offset == len(self._table) - 1:
            self._table.seal()
        self._table = self._table.prepare()

    def __repr__(self):
        """
        Define the representation of the EntityView.

        Examples:

        >>> from antgo.pipeline import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(range(5), range(5))]
        >>> df = DataFrame(e)
        >>> df = df.to_column()
        >>> df.to_list()[0]
        <EntityView dict_keys(['a', 'b'])>

        >>> df = DataFrame(e)
        >>> df = df.set_chunksize(2)
        >>> df.to_list()[0]
        <EntityView dict_keys(['a', 'b'])>
        """
        return f'<{self.__class__.__name__} dict_keys({self._table.column_names})>'

