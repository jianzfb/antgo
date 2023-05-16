import ctypes

CTYPE_INTS = [ctypes.c_short, ctypes.c_int, ctypes.c_long, ctypes.c_longlong]
CTYPE_UINTS = [ctypes.c_ushort, ctypes.c_uint,
               ctypes.c_ulong, ctypes.c_ulonglong]
CTYPENAME2CTYPE = {
    'bool': ctypes.c_bool,
    'char': ctypes.c_char,
    'char*': ctypes.c_char_p,
    'double': ctypes.c_double,
    'float': ctypes.c_float,
    'int': ctypes.c_int,
    'int8_t': ctypes.c_int8,
    'int16_t': ctypes.c_int16,
    'int32_t': ctypes.c_int32,
    'int64_t': ctypes.c_int64,
    'void*': ctypes.c_void_p,
    'void': None,
    None: None,
}


def get_ctype_name(ctype):
    # ctype.__name__ = 'c_xxx'
    if ctype in CTYPE_INTS[2:]:
        return 'int{}_t'.format(ctypes.sizeof(ctype) * 8)
    if ctype in CTYPE_UINTS[2:]:
        return 'uint{}_t'.format(ctypes.sizeof(ctype) * 8)
    name = ctype.__name__
    if name.startswith('c_'):
        name = name[2:]
    return name


class DType:
    _EXTRA_ATTRS = dict()

    def __init__(self, ctype, is_const=False):
        self.ctype = ctype
        self.is_const = is_const
        self.cname, self.is_pointer = self._get_extra_attr()

    def _get_extra_attr(self):
        idcode = (self.ctype, self.is_const)
        if idcode not in DType._EXTRA_ATTRS:
            # Assignment for self.is_pointer and self.cname
            if self.ctype.__name__.startswith('LP'):
                is_pointer = True
                basic_type = self.ctype._type_
            else:
                is_pointer = False
                basic_type = self.ctype

            ctype_name = get_ctype_name(basic_type)

            if self.is_const:
                ctype_name = 'const {}'.format(ctype_name)
            if is_pointer:
                ctype_name += '*'
            cname = ctype_name
            DType._EXTRA_ATTRS[idcode] = (cname, is_pointer)
        return DType._EXTRA_ATTRS[idcode]

    def __repr__(self):
        return self.cname

    def __call__(self, value):
        return self.ctype(value)


class CStruct:
    def __init__(self, name, is_const, cstruct, constructor):
        self.cname = name + '*'
        self.cstruct = cstruct
        self.constructor = constructor
        self.is_pointer = True
        self.is_const = is_const

    def __call__(self, *args, **kwargs):
        return self.constructor(*args, **kwargs)


class UnknownCType:
    def __init__(self, tname):
        self.tname = tname
        self.is_const = False


class TemplateType:
    def __init__(self, tname, is_pointer, is_const):
        self.tname = tname
        self.is_pointer = is_pointer
        self.is_const = is_const

    def __repr__(self):
        return '<typename {const}{tname}{pointer}>'.format(
            const='const ' if self.is_const else '',
            tname=self.tname,
            pointer='*' if self.is_pointer else '')

    def __call__(self, ctype):
        if self.is_pointer:
            ctype = ctypes.POINTER(ctype)
        return DType(ctype, is_const=self.is_const)
