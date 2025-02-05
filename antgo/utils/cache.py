import os

# TODO, 引入线程安全的数据缓存
__pipeline_data_cache = {}
def get_data_in_cache(name, key, value=None):
    global __pipeline_data_cache
    if name not in __pipeline_data_cache:
        return value

    if key not in __pipeline_data_cache[name]:
        return value

    return __pipeline_data_cache[name][key]


def set_data_in_cache(name, key, value):
    global __pipeline_data_cache
    if name not in __pipeline_data_cache:
        __pipeline_data_cache[name] = {}    

    __pipeline_data_cache[name][key] = value
