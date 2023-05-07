# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:10
# @File    : dataframe.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json

from typing import Dict, Any, Optional, Set, Union, List

from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.hparam import dynamic_dispatch, param_scope


class DataFrameMixin:
    """
    Mixin to help deal with Entity.

    Examples:

    1. define an operators with `register` decorator

    >>> @register
    ... def add_1(x):
    ...     return x+1

    2. apply the operators to named field of entity and save result to another named field

    >>> (
    ...     DataFrame([dict(a=1, b=2), dict(a=2, b=3)])
    ...         .as_entity()
    ...         .add_1['a', 'c']() # <-- use field `a` as input and filed `c` as output
    ...         .as_str()
    ...         .to_list()
    ... )
    ["{'a': 1, 'b': 2, 'c': 2}", "{'a': 2, 'b': 3, 'c': 3}"]

    Select the entity on the specified fields.

    Examples:

    1. Select the entity on one specified field:

    >>> df = DataFrame([Entity(a=i, b=i, c=i) for i in range(2)])
    >>> df.select['a']().to_list()
    [<Entity dict_keys(['a'])>, <Entity dict_keys(['a'])>]

    2. Select multiple fields and unpack the entity:

    >>> (
    ...     DataFrame([Entity(a=i, b=i, c=i) for i in range(5)])
    ...         .select['a', 'b']()
    ...         .as_raw()
    ...         .to_list()
    ... )
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    3. Another field selection syntax (not suggested):

    >>> (
    ...     DataFrame([Entity(a=i, b=i, c=i) for i in range(5)])
    ...         .select('a', 'b')
    ...         .as_raw()
    ...         .to_list()
    ... )
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    """

    def __init__(self):
        # pylint: disable=useless-super-delegation
        super().__init__()

    @property
    def select(self):

        @dynamic_dispatch
        def selector(*arg):
            index = param_scope()._index
            if isinstance(index, str):
                index = (index, )
            if index is None and arg is not None and len(arg) > 0:
                index = arg

            def inner(entity: Entity):
                if index is not None:
                    return Entity(
                        **{col: getattr(entity, col)
                           for col in index})

                return entity

            return self.map(inner)

        return selector

    # pylint: disable=invalid-name
    def fill_entity(self,
                    _DefaultKVs: Optional[Dict[str, Any]] = None,
                    _ReplaceNoneValue: bool = False,
                    **kws):
        """
        When DataFrame's iterable exists of Entities and some indexes missing, fill default value for those indexes.

        Args:
            _ReplaceNoneValue (`bool`):
                Whether to replace None in Entity's value.
            _DefaultKVs (`Dict[str, Any]`):
                The key-value pairs stored in a dict.

        Examples:

        >>> entities = [Entity(num=i) for i in range(3)]
        >>> df = DataFrame(entities)
        >>> df
        [<Entity dict_keys(['num'])>, <Entity dict_keys(['num'])>, <Entity dict_keys(['num'])>]

        >>> kvs = {'foo': 'bar'}
        >>> df.fill_entity(kvs).fill_entity(usage='test').to_list()
        [<Entity dict_keys(['num', 'foo', 'usage'])>, <Entity dict_keys(['num', 'foo', 'usage'])>, <Entity dict_keys(['num', 'foo', 'usage'])>]

        >>> kvs = {'FOO': None}
        >>> df.fill_entity(_ReplaceNoneValue=True, _DefaultKVs=kvs).to_list()[0].FOO
        0
        """
        if _DefaultKVs:
            kws.update(_DefaultKVs)

        def fill(entity: Entity):
            for k, v in kws.items():
                if not hasattr(entity, k):
                    setattr(entity, k, v)
                if _ReplaceNoneValue and v is None:
                    setattr(entity, k, 0)
            return entity

        return self._factory(map(fill, self._iterable))

    def as_entity(self, schema: Optional[List[str]] = None):
        """
        Convert elements into Entities.

        Args:
            schema (Optional[List[str]]):
                schema contains field names.

        Examples:
        1. convert dicts into entities:

        >>> (
        ...     DataFrame([dict(a=1, b=2), dict(a=2, b=3)])
        ...         .as_entity()
        ...         .as_str()
        ...         .to_list()
        ... )
        ["{'a': 1, 'b': 2}", "{'a': 2, 'b': 3}"]

        2. convert tuples into entities:

        >>> (
        ...     DataFrame([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .as_str()
        ...         .to_list()
        ... )
        ["{'a': 1, 'b': 2}", "{'a': 2, 'b': 3}"]

        3. convert single value into entities:

        >>> (
        ...     DataFrame([1, 2])
        ...         .as_entity(schema=['a'])
        ...         .as_str()
        ...         .to_list()
        ... )
        ["{'a': 1}", "{'a': 2}"]
        """

        if schema is None:

            def inner(x):
                return Entity(**x)
        else:

            def inner(x):
                if len(schema) == 1:
                    x = (x, )
                data = dict(zip(schema, x))
                return Entity(**data)

        return self._factory(map(inner, self._iterable))

    def parse_json(self):
        """
        Parse string to entities.

        Examples:

        >>> df = (
        ...     DataFrame(['{"x": 1}'])
        ...         .parse_json()
        ... )
        >>> df[0].x
        1
        """

        def inner(x):
            data = json.loads(x)
            return Entity(**data)

        return self.map(inner)

    def as_json(self):
        """
        Convert entities to json

        Examples:

        >>> (
        ...     DataFrame([Entity(x=1)])
        ...         .as_json()
        ... )
        ['{"x": 1}']
        """

        def inner(x):
            return json.dumps(x.__dict__)

        return self.map(inner)

    def as_raw(self):
        """
        Convert entitis into raw python values

        Examples:

        1. unpack multiple values from entities:

        >>> (
        ...     DataFrame([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .as_raw()
        ...         .to_list()
        ... )
        [(1, 2), (2, 3)]

        2. unpack single value from entities:

        >>> (
        ...     DataFrame([1, 2])
        ...         .as_entity(schema=['a'])
        ...         .as_raw()
        ...         .to_list()
        ... )
        [1, 2]
        """

        def inner(x):
            if len(x.__dict__) == 1:
                return list(x.__dict__.values())[0]
            return tuple(getattr(x, name) for name in x.__dict__)

        return self.map(inner)

    def replace(self, **kws):
        """
        Replace specific attributes with given vlues.

        Examples:


        >>> entities = [Entity(num=i) for i in range(5)]
        >>> df = DataFrame(entities)
        >>> [i.num for i in df]
        [0, 1, 2, 3, 4]

        >>> df = df.replace(num={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        >>> [i.num for i in df]
        [1, 2, 3, 4, 5]
        """

        def inner(entity: Entity):
            for index, convert_dict in kws.items():
                origin_value = getattr(entity, index)
                if origin_value in convert_dict:
                    setattr(entity, index, convert_dict[origin_value])

            return entity

        return self._factory(map(inner, self._iterable))

    def dropna(self, na: Set[str] = {'', None}) -> Union[bool, 'DataFrame']:  # pylint: disable=dangerous-default-value
        """
        Drop entities that contain some specific values.

        Args:
            na (`Set[str]`):
                Those entities contain values in na will be dropped.

        Examples:

        >>> entities = [Entity(a=i, b=i + 1) for i in range(3)]
        >>> entities.append(Entity(a=3, b=''))
        >>> df = DataFrame(entities)
        >>> df
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]

        >>> df.dropna()
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        """

        def inner(entity: Entity):
            for val in entity.__dict__.values():
                if val in na:
                    return False

            return True

        return self._factory(filter(inner, self._iterable))

    def rename(self, column: Dict[str, str]):
        """
        Rename an column in DataFrame.

        Args:
            column (`Dict[str, str]`):
                The columns to rename and their corresponding new name.

        Examples:

        >>> entities = [Entity(a=i, b=i + 1) for i in range(3)]
        >>> df = DataFrame(entities)
        >>> df
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]

        >>> df.rename(column={'a': 'A', 'b': 'B'})
        [<Entity dict_keys(['A', 'B'])>, <Entity dict_keys(['A', 'B'])>, <Entity dict_keys(['A', 'B'])>]
        """

        def inner(x):
            for key in column:
                x.__dict__[column[key]] = x.__dict__.pop(key)
            return x

        return self._factory(map(inner, self._iterable))

    @property
    def df(self):
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        if isinstance(self._iterable, pd.DataFrame):
            return self._iterable
        else:
            raise TypeError(
                'data collection is not created from pandas DataFrame.')
