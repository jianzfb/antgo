"""Operator Loader."""
from collections import namedtuple
import os
import sys
import re
import time
import ctypes
import json
import warnings
import portalocker
from ..internal.edict import edict
from ..func import CFuncDef, bind, get_func_idcode, get_idcode_hash
from ..building.build import source_to_so_ctx, get_virtual_dirname, build_context, file_is_changed, ENV_PATH
from ..building.build_utils import *
from ..utils import get_git_hash, makedirs
from ..internal.dtype import DType, CStruct, TemplateType, CTYPENAME2CTYPE
from ..glue.common import CSTRUCT_CONSTRUCTOR, register_cstruct
from ..glue.backend import get_glue_modules
from .gen_code import get_gen_rel_code
import antgo
import numpy as np


gen_code = get_gen_rel_code(os.path.dirname(__file__))


if sys.version_info[0] >= 3:
    import importlib.util

    def load_module(name, pathname):
        """Load Module.

        Paramters
        ---------
        name: str
            the name of module.
        pathname:
            the name of path.

        Returns
        -------
        Module
        """
        spec = importlib.util.spec_from_file_location(name, pathname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
else:
    import imp

    def load_module(name, pathname):
        """Load Module.

        Paramters
        ---------
        name: str
            the name of module.
        pathname:
            the name of path.

        Returns
        -------
        Module
        """
        module = imp.load_source(name, pathname)
        return module


def _get_func_head_reg(name):
    """Get a pattern object for CFunction Head.

    Paramters
    ---------
    name: str
        Function name.

    Returns
    -------
    A pattern object
    """
    return re.compile(r'^\s*{}\s*(.*)'.format(name))


ANTGO_KERNEL_REG = _get_func_head_reg('ANTGO_(CLASS|FUNC)')
DEPEND_3RD_REG = _get_func_head_reg('#include')

ANTGO_STRUCT_REG = re.compile('^struct \s*')
STRUCT_NAME_REG = re.compile(r'struct\s*(.*?)\s*{')
VAR_REG = re.compile(r'(char\*|bool\*|unsigned int\*|int\*|float\*|double\*|void\*|unsigned char|char|int|unsigned int|uint|char_p|void_p|ubyte|byte|float|double|int|unsigned int|float_p|double_p|int_p|uint_p|bool|bool_p)\s*(.*?)\s*(.*?)\s*;')
ARRAY_REG = re.compile(r'\[(\d*[+|-|*|/]?\d*)\]')

FUNC_REG = re.compile(
    r'^\s*(.*?)\s*\((.*?)\)(?:.*?)*')
CPP_TEMPLATE_REG = re.compile(r'^\s*template\s*\<(.*?)\>\s*')

FUNC_IN_STRUCT_REG = re.compile(
    r'^\s*[\w~]*\(.*?\)'
)

CLASS_REG = re.compile(
    r'\s*class\s*(\w*)\s*'
)

RUN_FUNC_IN_CLASS_REG = re.compile(
    r'^\s*void\s*run'
)

ctype_map={
    'float': 'ctypes.c_float',
    'double': 'ctypes.c_double',
    'void*': 'ctypes.c_void_p',
    'unsigned char': 'ctypes.c_ubyte',
    'char': 'ctypes.c_byte',
    'int': 'ctypes.c_int',
    'unsigned int': 'ctypes.c_uint',
    'uint': 'ctypes.c_uint',
    'char*': 'ctypes.c_char_p',
    'char_p': 'ctypes.c_char_p',
    'void_p': 'ctypes.c_void_p',
    'ubyte': 'ctypes.c_ubyte',
    'byte': 'ctypes.c_byte',
    'float*': 'ctypes.POINTER(ctypes.c_float)',
    'double*': 'ctypes.POINTER(ctypes.c_double)',
    'int*': 'ctypes.POINTER(ctypes.c_int)',
    'unsigned int*': 'ctypes.POINTER(ctypes.c_uint)',
    'float_p': 'ctypes.POINTER(ctypes.c_float)',
    'double_p': 'ctypes.POINTER(ctypes.c_double)',
    'int_p': 'ctypes.POINTER(ctypes.c_int)',
    'uint_p': 'ctypes.POINTER(ctypes.c_uint)',
    'bool': 'ctypes.c_bool',
    'bool_p': 'ctypes.POINTER(ctypes.c_bool)',
    'bool*': 'ctypes.POINTER(ctypes.c_bool)'
}


def _get_template_decl(code):
    match = CPP_TEMPLATE_REG.search(code)
    if match is None:
        return None
    blocks = match.groups()[0].split(',')
    templates = []
    for block in blocks:
        block_sp = block.split()
        dtype, dname = block_sp
        if dtype.strip() == 'typename':
            templates.append(dname.strip())
    return templates


def parse_parameter_decl(decl):
    """Parse the code of parameter declaration

    Parameters
    ----------
    decl : str
        The C++ code of parameter declaration

    Returns
    -------
    Tuple
        (DType Instance,  variable name)
    """
    num_star = decl.count('*')
    assert num_star <= 1,\
        Exception('Only support pass-by-value or pass-by-1-level-pointer, \
            Error declaration: {}'.format(decl))
    is_pointer = num_star > 0
    if is_pointer:
        decl = decl.replace('*', '')
    decl = decl.strip()
    if decl.startswith('const '):
        is_const = True
        decl = decl[len('const '):]
    else:
        is_const = False
    decl_sp = decl.split(' ')

    # type_name and variable_name in C++ code
    type_name, var_name = decl_sp

    # void* func(...)
    if type_name == 'void':
        assert is_pointer
        return DType(ctypes.c_void_p, is_const=is_const), var_name

    # ctype func(...)
    ctype_name = 'c_{}'.format(type_name)
    if type_name == 'char':
        assert is_pointer
        ctype = getattr(ctypes, 'c_char_p')
        return DType(ctype, is_const=is_const), var_name

    if hasattr(ctypes, ctype_name):
        ctype = getattr(ctypes, ctype_name)
        if is_pointer:
            ctype = ctypes.POINTER(ctype)
        return DType(ctype, is_const=is_const), var_name

    if type_name in CSTRUCT_CONSTRUCTOR:
        info = CSTRUCT_CONSTRUCTOR[type_name]
        return CStruct(type_name, is_const=is_const, cstruct=info[0], constructor=info[1]), var_name

    # template type
    return TemplateType(tname=type_name, is_pointer=is_pointer, is_const=is_const), var_name


def parse_parameters_list(plist):
    """Parse the code of parameter declaration list

    Parameters
    ----------
    plist : str
        The code of parameter declaration list

    Returns
    -------
    rtn_type :
        The type of return value
    func_name : str
        function name
    pars_list: list
        [(DType|TemplateType, variable name), ...]
    """

    match = FUNC_REG.search(plist)
    head, plist = match.groups()
    head_split = re.split(r'\s+', head)
    plist_split = filter(lambda x: len(x) > 0, re.split(r'\s*,\s*', plist))
    func_name = head_split[-1]
    rtn_type = head_split[-2] if len(head_split) == 3 else None
    pars_list = list(map(parse_parameter_decl, plist_split))
    return rtn_type, func_name, pars_list


def parse_struct_list(plist):
    # struct_name, type_name_list, var_name_list
    match = STRUCT_NAME_REG.search(plist)
    struct_name = match.groups()[0]

    match = VAR_REG.findall(plist)
    type_name_list = []
    var_name_list = []
    for type_name, _, var_name in match:
        type_name_list.append(type_name)
        var_name_list.append(var_name)
    return struct_name, type_name_list, var_name_list


struct_def_str = '''
class {STRUCT_NAME}(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        if data is None:
            v = []
            for name, ct in self._fields_:
                v.append(ct())
            
            super().__init__(*v)
            return

        v = []
        for name, ct in self._fields_:
            if isinstance(data[name], list) or isinstance(data[name], tuple):
                # 对于数组类型，需要输入的数据为list
                v.append(ct(*data[name]))
            elif data[name] is None or isinstance(data[name], np.ndarray):
                # 对于指针类型，需要输入的数据为numpy类型
                if data[name] is None:
                    v.append(ct())
                else:
                    v.append(ctypes.cast(data[name].ctypes.data, ct))
            else:
                v.append(ct(data[name]))
        super().__init__(*v)
    
    _fields_={STRUCT_FIELD}
'''


def register_struct_info(struct_name, type_name_list, var_name_list):
    # 动态创建 ctypes.Structure
    fields = []
    for type_name, var_name in zip(type_name_list, var_name_list):
        ctype_name = ctype_map[type_name]
        if '[' in var_name:
            # 数组类型
            match = ARRAY_REG.search(var_name)
            array_num = match.groups()[0]
            var_name = var_name.replace(f'[{array_num}]', '')
            if '+' in array_num or '-' in array_num or '*' in array_num or '/' in array_num:
                array_num = eval(array_num)
            ctype_name = f'{ctype_name} * {int(array_num)}'

        fields.append((f'\"{var_name}\"', ctype_name))

    struct_cls_str = struct_def_str.format(STRUCT_NAME=struct_name, STRUCT_FIELD=fields)
    struct_cls_str = struct_cls_str.replace('\'', '')
    exec(struct_cls_str)
    struct_cls = locals()[struct_name]

    register_cstruct(struct_name, struct_cls)


# runtime
FuncInfo = namedtuple('FuncInfo', ['func', 'cpp_info', 'init'])
CTX_FUNC_MAP = dict()  # CTX_FUNC_MAP[ctx][cpp_fname] -> FuncInfo


class CPPInfo:
    """The class of the C++ file's information.

    Parameters
    ----------
    cpp_fname: str
        the filename of C++ file.
    """

    def __init__(self, cpp_fname):
        self.cpp_fname = cpp_fname
        self.function_args = dict()
        self.dll = None

    def load_dll(self, dll_fname):
        """Load Dynamic-Link Library(*.so or *.dll).

        Parameters
        ----------
        dll_fname:
            The name of Dynamic-Link Library.
        """
        # keep reference
        self.dll = ctypes.CDLL(dll_fname)


def _build_lib(cpp_fname, code_buffer, ctx, target_name, depedent_src=[]):
    # the virtual dirname of the source code
    cpp_path, cpp_basename = os.path.split(cpp_fname)
    create_time = time.strftime('%a %Y-%m-%d %H:%M:%S (%z)', time.localtime())
    git_hash = get_git_hash()
    extra_code = gen_code('./templates/wrapper.cpp')(
        cpp_fname=cpp_fname,
        git_hash=git_hash,
        create_time=create_time,
        inc_fname=os.path.abspath(cpp_fname),
        code=code_buffer)

    build_path_ctx = os.path.join(os.path.dirname(target_name),ctx)
    makedirs(build_path_ctx, exist_ok=True)

    # build so
    cpp_wrapper_fname = os.path.join(build_path_ctx,
                                     os.path.splitext(cpp_basename)[0] + '_wrapper.cpp')
    with open(cpp_wrapper_fname, 'w') as fout:
        fout.write(extra_code)

    # build lib
    srcs = [cpp_wrapper_fname]
    srcs.extend(depedent_src)
    build_path = os.path.dirname(os.path.dirname(target_name))
    source_to_so_ctx(build_path, srcs, target_name, ctx)


def _dtype_to_tvm_value_type(dtype):
    if dtype.is_pointer:
        return 'v_handle'
    if 'int' in dtype.cname:
        return 'v_int64'
    return 'v_float64'


def _get_args_inst_mx(i, t):
    s = 'args.values[%d].%s' % (i, _dtype_to_tvm_value_type(t))
    if t.is_pointer:
        return '''
          static_cast<{dtype}>(
            static_cast<DLTensor*>({tv})->data)'''.format(dtype=t.cname, tv=s)
    else:
        s = '\n          ' + s
    return s


def _generate_kernel_code(func_idcode_hash, arg_types, arg_names, func_name):
    args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])
    args_inst = ', '.join(arg_names)

    kernel_code = gen_code('./templates/kernel_code.cpp')(
        func_idcode_hash=func_idcode_hash,
        args_def=args_def,
        func_name=func_name,
        args_inst=args_inst)
    kernel_code += '\n'

    args_def_async_mx = ', '.join(['{ctype} {name}'.format(
        ctype='NDArrayHandle' if dtype.is_pointer else dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])

    using_async_mx = all(
        map(lambda dtype: 'void' not in dtype.cname, arg_types))
    if using_async_mx:
        args_inst_mx = [_get_args_inst_mx(i, t)
                        for i, t in enumerate(arg_types)]
        const_loc = []
        for i, dtype in enumerate(arg_types):
            if dtype.is_const and dtype.is_pointer:
                const_loc.append(i)
        num_const = len(const_loc)
        const_loc_code = 'nullptr' if num_const == 0 else 'std::array<int, %d>({%s}).data()' % (
            num_const, ','.join([str(u) for u in const_loc]))
        async_mx_code = gen_code('./templates/async_mx_code.cpp')(
            func_idcode_hash=func_idcode_hash,
            func_name=func_name,
            args_inst=args_inst,
            args_inst_mx=','.join(args_inst_mx),
            num_const=num_const,
            const_loc_code=const_loc_code,
            args_def_async_mx=args_def_async_mx,
        )
        async_mx_code += '\n'
        kernel_code += async_mx_code
    return kernel_code


def _generate_func_code(func_idcode_hash, rtn_type, arg_types, arg_names, func_name):
    if rtn_type is None:
        rtn_type = 'void'

    args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])
    args_inst = ', '.join(arg_names)

    code = gen_code('./templates/func_code.cpp')(
        return_value=rtn_type,
        return_statement='' if rtn_type == 'void' else 'return',
        func_idcode_hash=func_idcode_hash,
        args_def=args_def,
        func_name=func_name,
        args_inst=args_inst,
    )

    return code

def _generate_class_code(func_idcode_hash, rtn_type, arg_types, arg_names, func_name, constructor_arg_types, constructor_arg_names):
    # 需要封装构造函数和运行函数
    args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])
    args_inst = ', '.join(arg_names)

    constructor_args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(constructor_arg_types, constructor_arg_names)])
    constructor_args_inst = ', '.join(constructor_arg_names)

    code = gen_code('./templates/class_code.cpp')(
        func_idcode_hash=func_idcode_hash,
        class_name=func_name,        
        run_args_def=args_def,
        run_args_inst=args_inst,
        init_args_def=constructor_args_def,
        init_args_inst=constructor_args_inst
    )
    return code


def _update_template_inst_map(idcode, template_functions, cfunc, arg_types):
    # template function
    func_name = cfunc.func_name
    func_idcode_hash = get_idcode_hash(idcode)
    # Check Template Type Mapping
    template_mapping = dict()
    for rtype, dtype in zip(arg_types, cfunc.arg_types):
        if not isinstance(dtype, TemplateType):
            continue
        tname = dtype.tname
        rtype = str(rtype).replace(
            'const', '').replace('*', '').strip()
        if tname in template_mapping:
            assert template_mapping[tname] == rtype,\
                Exception('Excepted template type {} instead of {}'.
                          format(template_mapping[tname], rtype))
        else:
            template_mapping[tname] = rtype
    assert len(template_mapping) == len(cfunc.template_list),\
        Exception('Template List: {}, mapping: {}'.
                  format(cfunc.template_list, template_mapping))

    template_inst = [template_mapping[tname]
                     for tname in cfunc.template_list]
    template_post = '<%s>' % (', '.join(template_inst)
                              ) if template_inst else ''
    rtn_type = cfunc.rtn_type
    if rtn_type in template_mapping:
        rtn_type = template_mapping[rtn_type]

    func_kind = cfunc.func_kind
    if func_kind == CFuncDef.CLASS:
        # class op
        code = _generate_class_code(
            func_idcode_hash, rtn_type, arg_types, cfunc.arg_names, func_name+template_post, cfunc.loader_kwargs["construct_arg_types"], cfunc.loader_kwargs["construct_arg_names"])
    else:
        # func op
        code = _generate_func_code(
            func_idcode_hash, rtn_type, arg_types, cfunc.arg_names, func_name + template_post)
    template_functions[idcode] = (code, rtn_type)


def _add_function(func_map, func_idcode, rtn_type, cpp_info, dll_fname, func_kind):
    func_idcode_hash = get_idcode_hash(func_idcode)
    if func_kind == CFuncDef.CLASS:
        init = getattr(cpp_info.dll, f'{func_idcode_hash}_init', None)
        init.restype = CTYPENAME2CTYPE['void*']

        func = getattr(cpp_info.dll, f'{func_idcode_hash}_run', None)
        func.restype = None

        func_map[func_idcode] = FuncInfo(func=func, cpp_info=cpp_info, init=init)
        return

    func = getattr(cpp_info.dll, func_idcode_hash, None)
    if func is None:
        functions = [name for name in dir(
            cpp_info.dll) if not name.startswith('_')]
        raise NameError('No function `{}` in DLL {}, current functions: {}'.format(
            func_idcode, dll_fname, functions))
    func.restype = CTYPENAME2CTYPE[rtn_type]

    old_func = func_map.get(func_idcode, None)
    if old_func is not None:
        if old_func.cpp_info.cpp_fname != cpp_info.cpp_fname:
            warnings.warn('The function `{}` in `{}` will be overridden by that in `{}`'.format(
                func_idcode, old_func.cpp_info.cpp_fname, cpp_info.cpp_fname))

    func_map[func_idcode] = FuncInfo(func=func, cpp_info=cpp_info, init=None)


class OpLoader:
    '''Import Operator Loader.
    It's actual to load the operator.

    Parameters
    ----------
    cfunc: CFuncDef
        The definition of function to call.
    arg_types: list of {DType|TemplateType}
        Argument declaration list.
    ctx: str
        Building context.
    cpp_info: CPPInfo
        Related to cfunc.

    Returns
    -------
    CTX_FUNC_MAP[ctx][fname][idcode] : FuncInfo
    '''

    def __init__(self, cfunc, arg_types, ctx, cpp_info, **kwargs):
        idcode = get_func_idcode(cfunc.func_name, arg_types)
        if ctx not in CTX_FUNC_MAP:
            CTX_FUNC_MAP[ctx] = dict()
        cpp_fname = cpp_info.cpp_fname
        if cpp_fname not in CTX_FUNC_MAP[ctx]:
            CTX_FUNC_MAP[ctx][cpp_fname] = dict()
        # func_map: dict mapping idcode to CFunction
        func_map = CTX_FUNC_MAP[ctx][cpp_fname]

        if idcode not in func_map:
            '''
            *load function* when one of the following conditions is True:
            1. idcode is not loaded
            2. loading the function with same function name but different cpp filename
            '''
            cpp_path, cpp_basename = os.path.split(cpp_fname)
            cpp_path = get_virtual_dirname(cpp_path)
            build_path = os.path.join(os.curdir, '.temp', 'op', cpp_path.split('/')[-1], 'build')

            use_template = bool(cfunc.template_list)
            makedirs(build_path, exist_ok=True)
            build_info_fname = os.path.join(
                build_path, os.path.splitext(cpp_basename)[0] + '.json')
            build_info_fs = open(build_info_fname, 'a+')
            portalocker.lock(build_info_fs, portalocker.LOCK_EX)
            build_info_fs.seek(0)
            js_data = build_info_fs.read()
            if js_data:
                map_data = json.loads(js_data)
            else:
                map_data = dict(version=antgo.__version__)
            del js_data

            # try to load the instance of template function
            # map_data is a dict which records build information
            if map_data.get('version') > antgo.__version__:
                portalocker.unlock(build_info_fs)
                raise Exception(
                    """Unsupported higher version %s of wrapper file (Current AntgoOP ver: %s) :-(.Please update AntgoOP.""" % (map_data.get('version'), antgo.__version__))
            build_id = map_data.get('build_id', 0)
            is_old_version = map_data.get(
                'version') < antgo.__version__
            # load the information of template functions
            TEMPLATE_FUNCTION_NAME = 'functions'
            if is_old_version:
                template_functions = dict()
            else:
                template_functions = map_data.get(
                    TEMPLATE_FUNCTION_NAME, dict())

            so_prefix = os.path.join(build_path, os.path.splitext(cpp_basename)[0])
            # The filename of build target
            dll_fname_format = '{prefix}_{ctx}'.format(
                prefix=so_prefix, ctx=ctx) + '_{build_id}.so'
            dll_fname = dll_fname_format.format(build_id=build_id)

            file_changed = file_is_changed(cpp_fname)
            dll_existed = os.path.exists(dll_fname)
            func_existed = idcode in template_functions

            if file_changed or not dll_existed or not func_existed or is_old_version:
                # Rebuild DLL file
                try:
                    # try to remove old DLL file
                    os.remove(dll_fname)
                except:
                    pass
                if file_changed:
                    # clear template_functions since some functions may have been deleted or renamed after codefile is changed.
                    template_functions.clear()
                if file_changed or not func_existed:
                    '''
                    we increase `build_id` by 1 when one of the following conditions is True:
                    1. the cpp file has been changed
                    2. new idcode

                    When the cpp file is not changed, and idcode exists in template_functions,
                    `build_id` will be not changed.
                    '''
                    build_id += 1
                dll_fname = dll_fname_format.format(build_id=build_id)
                # build code
                if idcode not in template_functions:
                    _update_template_inst_map(
                        idcode, template_functions, cfunc, arg_types)
                # collects template instances code into code_buffer
                code_buffer = ''.join([v[0]
                                       for v in template_functions.values()])

                with build_context():
                    try:
                        _build_lib(cpp_fname, code_buffer, ctx, dll_fname, cfunc.loader_kwargs.get('depedent_src', []))
                    except:
                        # if build fail, unlock the build info file
                        portalocker.unlock(build_info_fs)
                        raise
                # update template_functions
                map_data = dict(version=antgo.__version__,
                                build_id=build_id)
                map_data[TEMPLATE_FUNCTION_NAME] = template_functions
                # clear the old context and write json data
                build_info_fs.seek(0)
                build_info_fs.truncate()
                json.dump(map_data, build_info_fs)
                build_info_fs.flush()
                os.fsync(build_info_fs.fileno())
            portalocker.unlock(build_info_fs)

            # load all functions in the dll
            cpp_info.load_dll(dll_fname)

            # import all functions
            for func_idcode in template_functions.keys():
                _add_function(func_map,
                              func_idcode, template_functions[func_idcode][1], cpp_info, dll_fname, cfunc.func_kind)

        # 算子/函数
        self.func = func_map[idcode].func
        self.init = func_map[idcode].init
        self.init_args = cfunc.loader_kwargs['construct_arg_names'] if 'construct_arg_names' in cfunc.loader_kwargs else None
        self.init_handler = None
        self.cpp_info = func_map[idcode].cpp_info
        self.idcode_hash = get_idcode_hash(idcode)

    def __call__(self, *args, **kwargs):
        if self.init is not None and self.init_handler is None:
            self.init_handler = self.init(*[kwargs[k] for k in self.init_args])

        if self.init_handler is not None:
            args = (self.init_handler,) +args

        result = self.func(*args)
        return result

    def clear(self, data):
        # 资源销毁函数
        getattr(self.cpp_info.dll, f'destroy_{str(data._obj.__class__.__name__).lower()}')(data)

    def get_async_func(self, glue_mod):
        async_name = getattr(glue_mod, 'async_name', None)
        if async_name is None:
            return None
        return glue_mod.get_async_func(self.cpp_info, self.idcode_hash)


def _get_struct_define_from_cpp(cpp_fname):
    struct_started = False
    unmatched_brackets = 0
    struct_def = ''
    function_started = False
    unmatched_function_brackets = 0
    # 忽略结构体中的函数定义
    for line in open(cpp_fname):
        if not struct_started:
            match = ANTGO_STRUCT_REG.search(line)
            if match is not None:
                struct_started = True

        if struct_started:
            if not function_started:
                function_match = FUNC_IN_STRUCT_REG.search(line)
                if function_match is not None:
                    function_started = True

            if function_started:
                unmatched_function_brackets += line.count('{') - line.count('}')
                if unmatched_function_brackets == 0:
                    function_started = False
                    continue

            if not function_started:
                unmatched_brackets += line.count('{') - line.count('}')
                struct_def += line
                if unmatched_brackets == 0:
                    # 读完结构体的所有定义
                    struct_started = False
                    struct_def = struct_def.replace('\n', '').replace('\r', '')
                    struct_name, type_name_list, var_name_list = parse_struct_list(struct_def)
                    register_struct_info(struct_name, type_name_list, var_name_list)


def _get_functions_from_cpp(cpp_fname):
    unmatched_brackets = 0
    func_def = ''
    func_kind = ''
    func_started = False
    template_list = []
    cpp_info = CPPInfo(cpp_fname=cpp_fname)
    function_args = cpp_info.function_args
    function_depedent_src = []
    base_folder = os.path.dirname(cpp_fname)
    for line in open(cpp_fname):
        # 检查第三方依赖
        match_include = DEPEND_3RD_REG.search(line)
        if match_include is not None:
            if 'opencv' in match_include.groups()[0]:
                config.USING_OPENCV = True
                continue

            if 'Eigen' in match_include.groups()[0]:
                config.USING_EIGEN = True
                continue

            if 'eagleeye' in match_include.groups()[0]:
                config.USING_EAGLEEYE = True
                continue

            check_cpp_src = match_include.groups()[0][1:-1].split('.')[0]
            check_cpp_src = f'{check_cpp_src}.cpp'
            if os.path.exists(os.path.join(base_folder, check_cpp_src)):
                function_depedent_src.append(os.path.join(base_folder, check_cpp_src))

        # 检查函数定义信息
        if not func_started:
            current_template_list = _get_template_decl(line)
            if current_template_list is not None:
                template_list = current_template_list
            match = ANTGO_KERNEL_REG.search(line)
            if match is not None:
                func_def = ''
                func_kind_str = match.groups()[0]
                if func_kind_str == 'FUNC':
                    func_kind = CFuncDef.FUNC
                    func_started = True

        # In a declaration of a function
        if func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            func_def += line
            if unmatched_brackets == 0:
                func_def = func_def.replace('\n', '').replace('\r', '')
                func_started = False
                rtn_type, kernel_name, par_list = parse_parameters_list(
                    func_def)
                # template name check
                template_set = set(template_list)
                assert len(template_set) == len(template_list),\
                    Exception('Duplicated template name in {}'.format(
                        ', '.join(template_list)))
                use_template = False
                for dtype, _ in par_list:
                    if isinstance(dtype, TemplateType):
                        assert dtype.tname in template_set,\
                            Exception(
                                "template name '{}' is not defined".format(dtype.tname))
                        use_template = True
                if not use_template:
                    template_list = []

                if func_kind == CFuncDef.FUNC:
                    func_name = kernel_name
                else:
                    raise Exception(
                        'Unknown function kind: {}'.format(func_kind))

                # Arguments
                funcdef_args = edict(func_name=func_name,
                                     func_kind=func_kind,
                                     arg_names=[t[1] for t in par_list],
                                     arg_types=[t[0] for t in par_list],
                                     rtn_type=rtn_type,
                                     template_list=template_list,
                                     loader=OpLoader,
                                     loader_kwargs=dict(
                                         cpp_info=cpp_info,
                                         depedent_src=function_depedent_src
                                     )
                                     )
                template_list = []
                function_args[func_name] = funcdef_args

    assert unmatched_brackets == 0,\
        Exception('# unmatched brackets: {}'.format(unmatched_brackets))

    if len(function_args) == 0:
        return None

    # Load dynamic file
    functions = dict(
        (name, CFuncDef(**kwargs)) for name, kwargs in function_args.items())
    # Load dynamic function for MXNet
    return functions


def _parse_class_meta(class_def):
    constructor_func_started = False
    call_func_started = False
    class_name = None
    
    constructor_func_def = ''
    run_func_def = ''

    CONSTRUCTOR_FUNC_IN_CLASS_REG = None
    unmatched_brackets = 0
    for line in class_def.split('\n'):
        if class_name is None:
            match_include = CLASS_REG.search(line)
            if match_include is not None:
                class_name = match_include.groups()[0]
        if class_name is None:
            continue

        if not constructor_func_started:
            if CONSTRUCTOR_FUNC_IN_CLASS_REG is None:
                CONSTRUCTOR_FUNC_IN_CLASS_REG = re.compile(
                    f'\s*{class_name}\s*\('
                )
            match_include =  CONSTRUCTOR_FUNC_IN_CLASS_REG.search(line)
            if match_include is not None and '~' not in line:
                constructor_func_started = True
        if constructor_func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            constructor_func_def += line
            if unmatched_brackets == 0:
                constructor_func_started = False
                # 解析constructor_func_def
                _, constructor_name, constructor_par_list = parse_parameters_list(constructor_func_def)

            continue

        if not call_func_started:
            match_include =  RUN_FUNC_IN_CLASS_REG.search(line)
            if match_include is not None:
                call_func_started = True
            pass
        if call_func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            run_func_def += line
            if unmatched_brackets == 0:
                call_func_started = False
                # 解析constructor_func_def
                _, run_name, run_par_list = parse_parameters_list(run_func_def)

            continue
    
  
    return class_name, constructor_par_list, run_par_list

def _get_class_from_cpp(cpp_fname):
    unmatched_brackets = 0
    func_def = ''
    func_kind = ''
    class_started = False
    template_list = []
    cpp_info = CPPInfo(cpp_fname=cpp_fname)

    function_depedent_src = []
    function_args = cpp_info.function_args
    base_folder = os.path.dirname(cpp_fname)
    for line in open(cpp_fname):
        # 检查第三方依赖
        match_include = DEPEND_3RD_REG.search(line)
        if match_include is not None:
            if 'opencv' in match_include.groups()[0]:
                config.USING_OPENCV = True
                continue

            if 'Eigen' in match_include.groups()[0]:
                config.USING_EIGEN = True
                continue

            if 'eagleeye' in match_include.groups()[0]:
                config.USING_EAGLEEYE = True
                continue
            
            check_cpp_src = match_include.groups()[0][1:-1].split('.')[0]
            check_cpp_src = f'{check_cpp_src}.cpp'
            if os.path.exists(os.path.join(base_folder, check_cpp_src)):
                function_depedent_src.append(os.path.join(base_folder, check_cpp_src))

        # 检查类定义信息
        if not class_started:
            current_template_list = _get_template_decl(line)
            if current_template_list is not None:
                template_list = current_template_list
            match = ANTGO_KERNEL_REG.search(line)
            if match is not None:
                func_def = ''
                func_kind_str = match.groups()[0]
                if func_kind_str == 'CLASS':
                    func_kind = CFuncDef.CLASS
                    class_started = True

        if class_started:
            unmatched_brackets += line.count('{') - line.count('}')
            func_def += line
            if unmatched_brackets == 0:
                class_started = False

                # 针对类的信息进一步解析
                class_name, constructor_par_list, run_par_list = _parse_class_meta(func_def)

                # run func Arguments
                funcdef_args = edict(
                    func_name=class_name,
                    func_kind=func_kind,
                    arg_names=[t[1] for t in run_par_list],
                    arg_types=[t[0] for t in run_par_list],
                    rtn_type=None,
                    template_list=[],
                    loader=OpLoader,
                    loader_kwargs=dict(
                        cpp_info=cpp_info, 
                        construct_arg_names=[t[1].split('=')[0].replace(' ','') for t in constructor_par_list],
                        construct_arg_types=[t[0] for t in constructor_par_list],
                        construct_rtn_type=None,
                        depedent_src=function_depedent_src
                    ))
                function_args[class_name] = funcdef_args

    # # Load dynamic file
    functions = dict(
        (name, CFuncDef(**kwargs)) for name, kwargs in function_args.items())
    # Load dynamic function for MXNet
    return functions


def load(module_name, path=''):
    """Load Operator Module

    Parameters
    ----------
    module_name: str
        The name of Operator Module
    path: str
        The path of Operator Module [default = current path]
    """
    op_name = os.path.basename(module_name)
    if not path:
        # Find Operator Module in custom directory first
        custom_path = os.path.join(os.path.dirname(__file__), '../zoo')
        if os.path.exists(os.path.join(custom_path, op_name)):
            path = custom_path
    path = os.path.join(path, module_name)

    found = False
    cpp_fname = os.path.join(path, op_name + '.hpp')
    if not os.path.exists(cpp_fname):
        cpp_fname = os.path.join(path, op_name + '.cpp')

    if os.path.exists(cpp_fname):
        found = True

        # Get struct and register
        _get_struct_define_from_cpp(cpp_fname)

        # Get functions
        functions = _get_functions_from_cpp(cpp_fname)
        if functions is not None:
            bind(functions)
            return

        # Get class
        functions = _get_class_from_cpp(cpp_fname)
        if functions is not None:
            bind(functions)
            return

