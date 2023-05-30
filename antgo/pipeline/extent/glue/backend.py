"""Backend Manager."""
import importlib
from itertools import chain
from .common import MobulaTensor

DTYPE_TO_GLUE = dict()  # dtype -> glue_mod
GLUE_NAME_TO_GLUE = dict()  # glue_name -> glue_mod
PKG_NAME_TO_GLUE_ARGS = dict()  # package_name -> (glue_name, types_name)


def _register_glue_real(glue_name, types_name):
    global DTYPE_TO_GLUE, GLUE_NAME_TO_GLUE
    if not isinstance(types_name, list):
        types_name = [types_name]
    glue = None
    try:
        glue = importlib.import_module('.' + glue_name + '_glue', __package__)
    except ImportError:
        pass
    if glue is not None:
        for tname in types_name:
            tname_sp = tname.split('.')
            try:
                module = importlib.import_module(tname_sp[0])
                for sub_name in tname_sp[1:]:
                    module = getattr(module, sub_name, None)
                    if module is None:
                        raise ImportError
                # create generators cache
                glue.gen_cache = dict()
                DTYPE_TO_GLUE[module] = glue
            except ImportError:
                pass
            # assert hasattr(glue.OpGen, '__call__')
            # assert hasattr(glue.OpGen, 'register')
            if hasattr(glue, 'Tensor'):
                value = glue.Tensor
                assert isinstance(value, type) and issubclass(
                    value, MobulaTensor)
            else:
                # register glue tensor
                tensors = []
                for key, value in glue.__dict__.items():
                    if isinstance(value, type) and issubclass(value, MobulaTensor):
                        if key.lower() == glue_name + 'tensor':
                            tensors.append(key)
                assert len(
                    tensors) == 1, 'Only one MobulaTensor is allowed in each glue module'
                glue.Tensor = getattr(glue, tensors[0])

            GLUE_NAME_TO_GLUE[glue_name] = glue


def register_glue(glue_name, type_names):
    """Register a glue module.

    Parameters
    ----------
    glue_name: str
        The name of glue module.
    type_names: list of str
        The list of inputs' class names.
    """
    global PKG_NAME_TO_GLUE_ARGS
    assert type_names, ValueError('type_names should be not empty')
    pkg_names = [cls_name.split('.')[0] for cls_name in type_names]
    pkg_name = pkg_names[0]
    assert all(map(lambda x: x == pkg_name, pkg_names)), TypeError(
        'The name of package should be consistent in `types_name`: {}'.format(type_names))
    PKG_NAME_TO_GLUE_ARGS[pkg_name] = (glue_name, type_names)


# register glue modules.
register_glue('mxnet', ['mxnet.nd.NDArray',
                        'mxnet.sym.Symbol', 'mxnet.numpy.ndarray'])
register_glue('numpy', ['numpy.ndarray'])
register_glue('torch', ['torch.Tensor'])
register_glue('cupy', ['cupy.core.core.ndarray'])


def get_var_type_glue(vtype):
    """Get glue module from variable's type.

    Parameters
    ----------
    vtype: data type

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """
    global DTYPE_TO_GLUE, PKG_NAME_TO_GLUE_ARGS
    glue_mod = DTYPE_TO_GLUE.get(vtype, None)
    if glue_mod is not None:
        return glue_mod
    pkg_name = vtype.__module__.split('.')[0]
    if pkg_name not in PKG_NAME_TO_GLUE_ARGS:
        return None
    # try to register glue_mod
    _register_glue_real(*PKG_NAME_TO_GLUE_ARGS[pkg_name])
    return DTYPE_TO_GLUE[vtype]


def get_var_glue(var):
    """Get glue module from variable.

    Parameters
    ----------
    var: variable

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """

    return get_var_type_glue(type(var))


def get_args_glue(*args, **kwargs):
    """Get glue module from args and kwargs.

    Parameters
    ----------
    *args
    **kwargs

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """
    glue_mods = map(get_var_glue, chain(args, kwargs.values()))
    glue_mods = list(filter(lambda x: x is not None, glue_mods))
    if glue_mods:
        glue_mod = glue_mods[0]
        assert all(map(lambda x: x == glue_mod, glue_mods)),\
            TypeError(
                'Support only 1 backend in a call, now: {}'.format(glue_mods))
        return glue_mod
    return None


def op_gen(glue_mod, op, name):
    """ Get operator generator of glue module.

    Parameters
    ----------
    glue_mod: Glue Module
    op: object
        The object of custom operator.
    name: str
        The name of custom operator.

    Returns
    -------
    The operator generator of glue module.
    """
    if name not in glue_mod.gen_cache:
        glue_mod.gen_cache[name] = glue_mod.OpGen(op=op, name=name)
    return glue_mod.gen_cache[name]


def get_glue_modules():
    return GLUE_NAME_TO_GLUE.values()
