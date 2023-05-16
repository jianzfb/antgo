"""Register for Custom Operator."""
from .. import glue


def register(*args, **kwargs):
    """The Wrapper to regiser Custom Operator

    Parameters
    ----------
    *args
    **kwargs
    """
    return glue.register(*args, **kwargs)
