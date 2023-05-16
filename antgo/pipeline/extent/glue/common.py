"""Common Functions for mobula.glue"""
import sys
import pickle
import inspect
import base64
import ctypes
import functools
import warnings


def pars_encode(data):
    """Encode data to base64 string.

    Parameters
    ----------
    data: object

    Returns
    -------
    str
        The encoding base64 string.
    """
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')


def pars_decode(data):
    """Decode base64 string to object.

    Parameters
    ----------
    data: str
        The encoding base64 string.

    Returns
    -------
    object
        The decoding object.
    """
    return pickle.loads(base64.b64decode(data.encode('utf-8')))


if sys.version_info[0] >= 3:
    getargspec = inspect.getfullargspec
else:
    getargspec = inspect.getargspec


def get_varnames(func):
    """Get the name list in parameter list of a function.

    Parameters
    ----------
    func: function

    Returns
    -------
    list of str
        The name list in parameter list of a function.
    """
    return getargspec(func).args[1:]


CUSTOM_OP_LIST = dict()
OP_MODULE_GLOBALS = None
CSTRUCT_CONSTRUCTOR = dict()


def get_in_data(*args, **kwargs):
    '''
    return:
        inputs: input variances
        pars: parameters of the operator
    '''
    op = kwargs.pop('op')
    input_names = get_varnames(op.forward)
    num_inputs = len(input_names)
    defaults = getargspec(op.forward).defaults
    num_defaults = len(defaults) if defaults is not None else 0
    # define input variances in the forward function
    # And the input variances may be in args or kwargs
    if len(args) >= num_inputs:
        inputs = args[:num_inputs]
        pars = [args[num_inputs:], kwargs]
    else:
        # len(args) <= num_inputs
        inputs = [None for _ in range(num_inputs)]
        for i, arg in enumerate(args):
            assert input_names[i] not in kwargs
            inputs[i] = arg
        # the rest of parameters
        for i in range(len(args), num_inputs - num_defaults):
            name = input_names[i]
            assert name in kwargs, "Variable %s not found" % name
            inputs[i] = kwargs.pop(name)
        num_valid_inputs = num_inputs - num_defaults
        for i in range(num_inputs - num_defaults, num_inputs):
            name = input_names[i]
            if name not in kwargs:
                break
            inputs[i] = kwargs.pop(name)
            num_valid_inputs += 1
        inputs = inputs[:num_valid_inputs]
        pars = [[], kwargs]

    return inputs, pars


def get_in_shape(in_data):
    """Get shapes of input datas.

    Parameters
    ----------
    in_data: Tensor
        input datas.

    Returns
    -------
    list of shape
        The shapes of input datas.
    """
    return [d.shape for d in in_data]


def assign(_, dst, req, src):
    """Helper function for assigning into dst depending on requirements."""
    if req == 'null':
        return
    if req in ('write', 'inplace'):
        dst[:] = src
    elif req == 'add':
        dst[:] += src


backend = None  # wait for importing in __init__.py


class MobulaTensor:
    F = None

    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def data_ptr(self):
        raise NotImplementedError

    @property
    def async_data_ptr(self):
        raise NotImplementedError

    @property
    def ctype(self):
        raise NotImplementedError

    @property
    def dev_id(self):
        raise NotImplementedError


class MobulaOperator:
    """Mobula Operator.

    Parameters
    ----------
    op: Custom Operator Class
    name: str
        The name of custom operator.
    *attrs:
        The attribute of custom operator.
    """

    def __init__(self, op, name, **attrs):
        self.op = op
        self.name = name
        self.attrs = attrs

    def __call__(self, *args, **kwargs):
        glue_mod = backend.get_args_glue(*args, **kwargs)
        assert glue_mod is not None, ValueError('No explict backend')
        new_kwargs = kwargs.copy()
        new_kwargs.update(self.attrs)
        return backend.op_gen(glue_mod, op=self.op, name=self.name)(*args, **new_kwargs)

    def __getitem__(self, input_type):
        glue_mod = backend.get_var_type_glue(input_type)
        assert glue_mod is not None, ValueError(
            'The backend of {} is not found'.format(input_type))

        def wrapper(*args, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(self.attrs)
            new_kwargs['__input_type__'] = input_type
            return backend.op_gen(glue_mod, op=self.op, name=self.name)(*args, **new_kwargs)
        return wrapper


def register_cstruct(name, cstruct, constructor=None):
    if constructor is None:
        constructor = cstruct
    assert callable(
        constructor), 'constructor {} should be callable'.format(name)
    CSTRUCT_CONSTRUCTOR[name] = (cstruct, constructor)


def register(op_name=None, **attrs):
    """Regiseter a custom operator
    1. @register
       class XXX
    2. @register("OP")
       class XXX
    3. @register(a = 3)
       class XXX
    """
    def decorator(op_name, op):
        if op_name is None:
            op_name = op.__name__
        op_inst = MobulaOperator(op=op, name=op_name, **attrs)
        if op_name in CUSTOM_OP_LIST:
            warnings.warn(
                'Duplicate operator name {}, please rename it'.format(op_name))
        CUSTOM_OP_LIST[op_name] = op_inst
        OP_MODULE_GLOBALS[op_name] = op_inst
        return op_inst
    if op_name is not None and not isinstance(op_name, str):
        return decorator(None, op_name)
    return functools.partial(decorator, op_name)


INPUT_FUNCS = dict(
    X=property(lambda self: self.in_data),
    Y=property(lambda self: self.out_data),
    dX=property(lambda self: self.in_grad),
    dY=property(lambda self: self.out_grad),
    x=property(lambda self: self.in_data[0]),
    y=property(lambda self: self.out_data[0]),
    dx=property(lambda self: self.in_grad[0]),
    dy=property(lambda self: self.out_grad[0]),
)
'''
OpGen:
    in_data, out_data, in_grad, out_grad
    req[write/add/null]
    X,Y,dX,dY,x,y,dx,dy
    F
'''
try:
    import numpy as np

    def _get_numpy_type():
        name2ctype = dict()
        pairs = [
            (np.dtype('int8'), ctypes.c_int8),
            (np.dtype('int16'), ctypes.c_int16),
            (np.dtype('int32'), ctypes.c_int32),
            (np.dtype('int64'), ctypes.c_int64),  # alias: np.int
            (np.dtype('float32'), ctypes.c_float),
            (np.dtype('float64'), ctypes.c_double),  # alias: np.float
        ]
        for dtype, ctype in pairs:
            name2ctype[dtype.name] = ctype
        return name2ctype
    NP_DTYPE_NAME2CTYPE = _get_numpy_type()

    def NPDTYPE2CTYPE(dtype):
        """Convert numpy data type into ctype.

        Parameters
        ----------
        dtype: numpy.dtype

        Returns
        -------
        ctype
        """
        ctype = NP_DTYPE_NAME2CTYPE.get(np.dtype(dtype).name, None)
        assert ctype is not None, TypeError('Unknown Type: {}'.format(dtype))
        return ctype
except ImportError:
    pass
