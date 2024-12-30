from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np


__global_context_env_info = {
}


def get_context_env_info():
    global __global_context_env_info
    return __global_context_env_info


def update_context_env_info(session_id, **kwargs):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        __global_context_env_info[session_id] = kwargs
        return

    __global_context_env_info[session_id].update(kwargs)


def set_context_exit_info(session_id, detail=None):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        __global_context_env_info[session_id] = {}

    __global_context_env_info[session_id]['exit'] = detail


def get_context_exit_info(session_id, default=None):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        return default
    if 'exit' not in __global_context_env_info[session_id]:
        return default

    return __global_context_env_info[session_id]['exit']


def set_context_redirect_info(session_id, redirect_url):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        __global_context_env_info[session_id] = {}

    __global_context_env_info[session_id]['redirect'] = redirect_url


def get_context_redirect_info(session_id, default=None):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        return default
    if 'redirect' not in __global_context_env_info[session_id]:
        return default

    return __global_context_env_info[session_id]['redirect']


def set_context_cookie_info(session_id, key, value):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        return False
    
    if 'cookie' not in __global_context_env_info[session_id]:
        return False

    __global_context_env_info[session_id]['cookie'].set_cookie(key, value)
    return True


def update_context_cookie_info(session_id, cookie):
    global __global_context_env_info
    if session_id not in __global_context_env_info:
        __global_context_env_info[session_id] = {}

    __global_context_env_info[session_id]['cookie'] = cookie


def clear_context_env_info(session_id):
    global __global_context_env_info
    __global_context_env_info.pop(session_id)

