from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np

__table_info = []

def get_table_info():
    global __table_info
    return __table_info


def update_table_info(table_info):
    global __table_info
    __table_info.append(table_info)
