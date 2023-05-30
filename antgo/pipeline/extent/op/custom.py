"""Custom Operator."""
from .. import glue


def Custom(op_name):
    """Get custom operator whose name is `op_name`.

    Parameters
    ----------
    op_name: str
        The name of custom operator.

    Returns
    -------
    Custom Operator
    """
    assert op_name in glue.CUSTOM_OP_LIST, KeyError(
        'Operator {} not found'.format(op_name))
    return glue.CUSTOM_OP_LIST[op_name]


def CustomList():
    """Get the name list of custom operators
    Returns
    -------
    list of str
        The name list of custom operators
    """
    return glue.CUSTOM_OP_LIST.keys()
