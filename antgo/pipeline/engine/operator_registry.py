# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 17:22
# @File    : operator_registry.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import types
from collections import namedtuple
from typing import Any, Dict, List

from antgo.pipeline.hparam import param_scope
from antgo.pipeline.engine.uri import URI


def _get_default_namespace():
    with param_scope() as hp:
        return hp().antgo.default_namespace('anon')


class OperatorRegistry:
    """Operator Registry
    """

    REGISTRY: Dict[str, Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def resolve(name: str) -> Any:
        """
        Resolve operators by name
        """
        for n in [
                name,
                '{}/{}'.format(_get_default_namespace(), name),
                '{}/{}'.format('builtin', name),
        ]:
            if n in OperatorRegistry.REGISTRY:
                return OperatorRegistry.REGISTRY[n]
        return None

    @staticmethod
    def register(
            name: str = None,
            input_schema=None,  # TODO: parse input_schema from code @jie.hou
            output_schema=None,
            flag=None):
        """
        Register a class, function, or callable as a towhee operators.

        Examples:

        1. register a function as operators

        >>> from towhee import register
        >>> @register
        ... def foo(x, y):
        ...     return x+y

        2. register a class as operators

        >>> @register
        ... class foo_cls():
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def __call__(self, y):
        ...         return self.x + y

        By default, function/class name is used as operators name,
        which is used by the operators factory `towhee.ops` to invoke the operators.

        >>> from towhee import ops
        >>> op = ops.foo()
        >>> op(1, 2)
        3

        >>> op = ops.foo_cls(x=2)
        >>> op(3)
        5

        3. register operators with an alternative name:

        >>> @register(name='my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_foo()(1,2)
        3

        Operator URI and Namespace: The URI (unique reference identifier) of an operators has two parts: namespace and name.
        The namespace helps identify one operators and group the operators into various kinds.
        We can specific the namespace when create an operators:

        >>> ops.anon.my_foo()(1,2)
        3

        `anon` is the default namespace to which an operators is registered if no namespace is specified.
        And it's also the default searching namespace for the operators factory.

        You can also specific the fullname, including namespace when register an operators:

        >>> @register(name='my_namespace/my_foo')
        ... def foo(x, y):
        ...     return x+y
        >>> ops.my_namespace.my_foo()(1,2)
        3

        Output Schema:

        >>> @register(name='my_foo', output_schema='value')
        ... def foo(x, y):
        ...     return x+y
        >>> from towhee.hparam import param_scope
        >>> with param_scope('towhee.need_schema=1'):
        ...     ops.my_foo()(1,2)
        Output(value=3)

        Flag: Each operators type, for example: NNOperator and PyOperator, has their own default `flag`:

        >>> from towhee.operators.base import Operator, NNOperator, PyOperator
        >>> from towhee.operators.base import OperatorFlag
        >>> @register
        ... class foo(NNOperator):
        ...     pass
        >>> foo().flag
        <OperatorFlag.REUSEABLE|STATELESS: 6>

        The default flag can be override by `register(flag=someflag)`:

        >>> @register(flag=OperatorFlag.EMPTYFLAG)
        ... class foo(NNOperator):
        ...     pass
        >>> foo().flag
        <OperatorFlag.EMPTYFLAG: 1>

        Args:
            name (str, optional): operators name, will use the class/function name if None.
            input_schema(NamedTuple, optional): input schema for the operators. Defaults to None.
            output_schema(NamedTuple, optional): output schema, will convert the operators output to NamedTuple if not None.
            flag ([OperatorFlag], optional): operators flag. Defaults to OperatorFlag.EMPTYFLAG.

        Returns:
            [type]: [description]
        """
        if callable(name):
            # the decorator is called directly without any arguments,
            # relaunch the register
            cls = name
            return OperatorRegistry.register()(cls)

        if output_schema is None:  # none output schema
            output_schema = namedtuple('Output', 'col0')
        if isinstance(output_schema, str):  # string schema 'col0 col1'
            output_schema = output_schema.split()

        # list schema ['col0', 'col1']
        if isinstance(output_schema, List):
            if len(output_schema) == 0 or isinstance(output_schema[0], str):
                output_schema = namedtuple('Output', output_schema)
        # list schema [(int, (1, )), (np.float32, (-1, -1, 3))] is for triton, do nothing.

        def wrapper(cls):
            metainfo = dict(input_schema=input_schema,
                            output_schema=output_schema,
                            flag=flag)

            nonlocal name
            name = URI(cls.__name__ if name is None else name).resolve_repo(
                _get_default_namespace())

            if isinstance(cls, types.FunctionType):
                OperatorRegistry.REGISTRY[name + '_func'] = cls

            # wrap a callable to a class
            if not isinstance(cls, type) and callable(cls):
                func = cls
                cls = type(
                    cls.__name__, (object, ), {
                        '__call__': lambda _, *arg, **kws: func(*arg, **kws),
                        '__doc__': func.__doc__,
                    })

            if output_schema is not None:
                old_call = cls.__call__

                def wrapper_call(self, *args, **kws):
                    with param_scope() as hp:
                        need_schema = hp().antgo.need_schema(False)
                    if need_schema:
                        return output_schema(old_call(self, *args, **kws))
                    else:
                        return old_call(self, *args, **kws)

                cls.__call__ = wrapper_call
                cls.__abstractmethods__ = set()
            cls.metainfo = metainfo
            if flag is not None:
                cls.flag = property(lambda _: flag)
            OperatorRegistry.REGISTRY[name] = cls

            return cls

        return wrapper


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
