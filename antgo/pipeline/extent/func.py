"""A `Module` implement the `AntgoFunc` class."""
__all__ = ['AntgoFunc', 'bind']


import ctypes
import hashlib
import warnings
import numpy as np
from . import glue
from .internal.dtype import DType, CStruct, TemplateType, UnknownCType
from .building.build_utils import config


def get_func_idcode(func_name, arg_types):
    """Get Function IDCode

    Parameters
    ----------
    func_name: str
        the name of function
    arg_types: list of DType

    Returns
    -------
    idcode: str
        IDCode
    """
    arg_types_str = ','.join([e.cname for e in arg_types])
    idcode = '{func_name}:{arg_types_str}'.format(
        func_name=func_name,
        arg_types_str=arg_types_str)
    return idcode


def get_idcode_hash(idcode):
    """Get the hash string of IDCode

    Parameters
    ----------
    idcode: str
    arg_types: list of DType

    Returns
    -------
    Hash String of IDCode: str
    """
    idcode_sp = idcode.split(':')
    func_name = idcode_sp[0]
    md5 = hashlib.md5()
    md5.update(idcode[len(func_name) + 1:].encode('utf-8'))
    return '{}_{}'.format(func_name, md5.hexdigest()[:8])


class CFuncTensor:
    def __init__(self, var, ptype, glue_mod):
        self.var = var
        self.ptype = ptype
        self.glue_mod = glue_mod

    @property
    def is_const(self):
        return self.ptype.is_const


class CStructArg:
    def __init__(self, var, ptype):
        self.var = var
        self.ptype = ptype

    @property
    def is_const(self):
        return self.ptype.is_const

def _wait_to_read(var):
    if hasattr(var, 'wait_to_read'):
        var.wait_to_read()


def _wait_to_write(var):
    if hasattr(var, 'wait_to_write'):
        var.wait_to_write()


def _get_raw_pointer(arg, const_vars, mutable_vars):
    if isinstance(arg, CFuncTensor):
        p = arg.glue_mod.Tensor(arg.var).data_ptr
        if isinstance(p, (list, tuple)):
            p, v = p
            if arg.is_const:
                const_vars.append(v)
            else:
                mutable_vars.append((arg.var, v))
        return p
    if isinstance(arg, CStructArg):
        if arg.is_const:
            const_vars.append(arg.var)
            return ctypes.byref(const_vars[-1])
        else:
            mutable_vars.append((arg.var.val, arg.var))
            return ctypes.byref(mutable_vars[-1][-1])

    return arg


def _get_async_pointer(arg):
    if isinstance(arg, CFuncTensor):
        return arg.glue_mod.Tensor(arg.var).async_data_ptr
    return arg


def _get_raw_pointers(args, const_vars, mutable_vars):
    return [_get_raw_pointer(a, const_vars, mutable_vars) for a in args]


def _get_async_pointers(args):
    return list(map(_get_async_pointer, args))


def _arg_wait_to_rw(arg):
    if isinstance(arg, CFuncTensor):
        if arg.is_const:
            _wait_to_read(arg)
        else:
            _wait_to_write(arg)


def _args_wait_to_rw(args):
    for arg in args:
        _arg_wait_to_rw(arg)


class CFuncDef:
    """The definition of CFunction."""
    KERNEL = 1
    FUNC = 2
    CLASS = 3

    def __init__(self, func_name, func_kind, arg_names=None, arg_types=None, rtn_type=None,
                 template_list=None, loader=None, loader_kwargs=None):
        self.func_name = func_name
        self.func_kind = func_kind
        self.arg_names = arg_names or list()
        self.arg_types = arg_types
        self.rtn_type = rtn_type
        self.template_list = template_list or list()
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.func_c_handler = None

    def __call__(self, arg_datas, arg_types, dev_id, init_key_values=None, glue_mod=None, using_async=False):
        if dev_id is None:
            ctx = 'cpu'
            dev_id = -1
        else:
            ctx = config.GPU_BACKEND
        # function loader
        if self.func_c_handler is None:
            self.func_c_handler = self.loader(self, arg_types, ctx, **self.loader_kwargs)
        _args_wait_to_rw(arg_datas)
        const_vars = []
        mutable_vars = []
        raw_pointers = _get_raw_pointers(arg_datas, const_vars, mutable_vars)
        res = None
        if self.func_kind == CFuncDef.FUNC:
            res = self.func_c_handler(*raw_pointers)
        elif self.func_kind == CFuncDef.CLASS:
            res = self.func_c_handler(*raw_pointers, **init_key_values)
        else:
            raise TypeError(
                'Unsupported func kind: {}'.format(self.func_kind))

        out = [] if res is None else [res]
        for v_i, (target, value) in enumerate(mutable_vars):
            if isinstance(value, np.ndarray):
                # for CFuncTensor
                target[:] = value
                if res is None:
                    out.append(target)
            elif target is None:
                # value is dict
                target_dict = {}
                for info in value._fields_:
                    elem = getattr(value, info[0])
                    if isinstance(elem, ctypes.Array):
                        target_dict[info[0]] = [elem[i] for i in range(len(elem))]
                    elif isinstance(elem, ctypes._Pointer):
                        # 不知道具体指针长度，无法拆解出具体数值
                        print('Dont support pointer type, dangers for memory leak')
                        target_dict[info[0]] = elem
                    else:
                        target_dict[info[0]] = elem
                out.append(target_dict)
            else:
                # value is numpy
                target = np.ctypeslib.as_array(value.data, shape=[value.dims[i] for i in range(value.dim_size)])
                # 自动释放
                if value.data != mutable_vars[v_i][1].data:
                    # clone 
                    target = target.copy()

                    # c函数内部创建，需要销毁内存
                    self.func_c_handler.clear(ctypes.byref(value))

                if res is None:
                    out.append(target)

        return out


class AntgoFunc:
    """An encapsulation for CFunction

    Parameters:
    -----------
    name: str
        function name
    func: CFuncDef
    """

    def __init__(self, name, func):
        self.name = name
        self.func = func

        self.wait_to_read_list = []
        self.wait_to_write_list = []
        for i, ptype in enumerate(self.func.arg_types):
            if ptype.is_pointer:
                if ptype.is_const:
                    self.wait_to_read_list.append(i)
                else:
                    self.wait_to_write_list.append(i)

    def __call__(self, *args, **kwargs):
        # move kwargs into args
        args = list(args)
        if self.func.func_kind == CFuncDef.FUNC:
            # 对于函数将kwargs参数合入args
            for name in self.func.arg_names[len(args):]:
                args.append(kwargs[name])
            # 清空
            kwargs = {}

        # type check
        arg_datas = []
        dev_id = None
        arg_types = []
        template_mapping = dict()

        glue_mod = self._get_glue_mod(args)
        using_async = config.USING_ASYNC_EXEC and glue_mod is not None and hasattr(
            glue_mod, 'get_async_func')

        if not using_async:
            # Pre-process
            for i in self.wait_to_read_list:
                _wait_to_read(args[i])
            for i in self.wait_to_write_list:
                _wait_to_write(args[i])

        try:
            # parse args 
            for var, ptype in zip(args, self.func.arg_types):
                if ptype.is_pointer:
                    if hasattr(ptype, 'constructor'):
                        var_dev_id = None
                        ctype = ctypes.POINTER(ptype.cstruct)
                        try:
                            data = CStructArg(ptype.constructor(var), ptype)
                        except TypeError:
                            data = CStructArg(ptype.constructor(*var), ptype)
                    else:
                        # The type of `var` is Tensor.
                        data, var_dev_id, ctype = self._get_tensor_info(
                            var, ptype, template_mapping, using_async)
                else:
                    # The type of `var` is Scalar.
                    data, var_dev_id, ctype = self._get_scalar_info(var, ptype)

                arg_datas.append(data)
                if isinstance(ctype, UnknownCType):
                    ctype.is_const = ptype.is_const
                    arg_types.append(ctype)
                else:
                    # pointer
                    arg_types.append(DType(ctype, is_const=ptype.is_const))

                # update `dev_id`
                if var_dev_id is not None:
                    if dev_id is not None:
                        assert var_dev_id == dev_id, ValueError(
                            "Don't use multiple devices in a call :-(")
                    else:
                        dev_id = var_dev_id

            # try to know the unknown ctype
            for i, vtype in enumerate(arg_types):
                if isinstance(vtype, UnknownCType):
                    assert vtype.tname in template_mapping,\
                        Exception(
                            'Unknown template name: {}'.format(vtype.tname))
                    ctype = template_mapping[vtype.tname]._type_
                    arg_types[i] = DType(ctype, vtype.is_const)
                    arg_datas[i] = ctype(arg_datas[i])

            # kwargs (class op)
            if self.func.func_kind == CFuncDef.CLASS:
                for name, ptype in zip(kwargs.keys(), self.func.loader_kwargs['construct_arg_types']):
                    var = kwargs[name]

                    if ptype.is_pointer:
                        if hasattr(ptype, 'constructor'):
                            var_dev_id = None
                            ctype = ctypes.POINTER(ptype.cstruct)
                            try:
                                data = CStructArg(ptype.constructor(var), ptype)
                            except TypeError:
                                data = CStructArg(ptype.constructor(*var), ptype)
                        else:
                            # The type of `var` is Tensor.
                            data, var_dev_id, ctype = self._get_tensor_info(
                                var, ptype, template_mapping, using_async)
                    else:
                        # The type of `var` is Scalar.
                        data, var_dev_id, ctype = self._get_scalar_info(var, ptype)

                    kwargs[name] = data
        except TypeError:
            raise TypeError('Unmatched parameters list of the function `{}`:\n\t{}\n\t\tvs\n\t{}'.format(
                self.name, self.func.arg_types, list(map(type, args))))

        rtn = self.func(arg_datas=arg_datas,
                        arg_types=arg_types,
                        init_key_values=kwargs,
                        dev_id=dev_id,
                        glue_mod=glue_mod,
                        using_async=using_async)
        return rtn

    @staticmethod
    def _get_tensor_info(var, ptype, template_mapping, using_async=False):
        """Get tensor info

        Parameters
        ----------
        var: object
            input variable
        ptype: DType | TemplateType
            the type of argument
        template_mapping: dict
            the mapping from template name to ctype
        using_async: bool
            whether to use asynchronous execution

        Returns
        -------
        data: CFuncTensor
        dev_id: int | None
            the id of device
        ctype: ctypes.POINTER | ctypes.c_*
            the ctype of data
        """
        glue_mod = glue.backend.get_var_glue(var)
        if glue_mod is None:
            raise TypeError()
        data = CFuncTensor(var, ptype, glue_mod)
        tensor = glue_mod.Tensor(var)
        dev_id = tensor.dev_id
        ctype = ctypes.POINTER(tensor.ctype)
        if isinstance(ptype, DType):
            expected_ctype = ptype.ctype
        elif ptype.tname in template_mapping:
            expected_ctype = template_mapping[ptype.tname]
        else:
            template_mapping[ptype.tname] = expected_ctype = ctype
        assert ctype == expected_ctype,\
            TypeError('Expected Type {} instead of {}'.format(
                expected_ctype, ctype))
        return data, dev_id, ctype

    @staticmethod
    def _get_scalar_info(var, ptype):
        """Get scalar info

        Parameters
        ----------
        var: object
            input variable
        ptype: DType | TemplateType
            the type of argument

        Returns
        -------
        data: ctyoes.c_void_p
            the pointer of data
        dev_id: int | None
            the id of device
        ctype: ctypes.POINTER | ctypes.c_*
            the ctype of data
        """

        dev_id = None
        if isinstance(ptype, TemplateType):
            data = var
            ctype = type(var) if hasattr(
                var, '_type_') else UnknownCType(ptype.tname)
        else:
            data = var if isinstance(
                var, ctypes.c_void_p) else ptype.ctype(var)
            ctype = ptype.ctype
        return data, dev_id, ctype

    @staticmethod
    def _get_glue_mod(datas):
        mods = map(glue.backend.get_var_glue, datas)
        mods = list(filter(lambda x: x is not None, mods))
        if mods:
            glue_mod = mods[0]
            # all glue modules in datas are consistent
            if all(map(lambda x: x == glue_mod, mods)):
                return glue_mod
        return None

    def build(self, ctx, template_types=None):
        """Build this function

        Parameters
        ----------
        ctx: str
            context Name
        template_types: list or tuple or dict, default: []
            list: a list of template type Names
            tuple: a tuple of template type Names
            dict: a mapping from template name to type name

        Examples
        --------
        >>> mobula.func.add.build('cpu', ['float'])
        """
        arg_types = []
        par_type = self.func.arg_types
        if template_types is None:
            template_types = list()
        if isinstance(template_types, (list, tuple)):
            template_mapping = dict()  # tname -> ctype
            for vtype in par_type:
                if isinstance(vtype, TemplateType):
                    tname = vtype.tname
                    if tname in template_mapping:
                        ctype = template_mapping[tname]
                    else:
                        ctype = getattr(ctypes, 'c_{}'.format(
                            template_types.pop(0)))
                        template_mapping[tname] = ctype
                    arg_types.append(vtype(ctype))
                else:
                    arg_types.append(vtype)
            assert not template_types, Exception('redundant type')
        else:
            assert isinstance(template_types, dict), TypeError(
                'The type of template_types should be list or tuple or dict.')
            template_name = set()
            for vtype in par_type:
                if isinstance(vtype, TemplateType):
                    tname = vtype.tname
                    assert tname in template_types, KeyError(
                        'Unknown Template Type: {}'.format(tname))
                    template_name.add(tname)
                    ctype = getattr(ctypes, 'c_{}'.format(
                        template_types[tname]))
                    arg_types.append(vtype(ctype))
                else:
                    arg_types.append(vtype)
            assert len(template_name) == len(template_types), Exception(
                'Different template name: {} vs {}'.format(
                    template_name, set(template_types.keys())))
        func = self.func
        func.loader(func, arg_types, ctx, **func.loader_kwargs)


_binded_functions = dict()


def bind(functions):
    global _binded_functions
    """Bind Functions to mobula.func.<function name>

    Parameters
    ----------
    functions: dict
        name -> CFuncDef
    """
    for k, func in functions.items():
        if k in _binded_functions:
            warnings.warn('Duplicated function name {}'.format(k))
        func = AntgoFunc(k, func)
        globals()[k] = func
        _binded_functions[k] = func
