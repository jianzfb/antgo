# encoding=utf-8
# @Time    : 17-5-24
# @File    : processbar.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys


def progress_bar(progress, callback_func=None, prefix="", bar_length=50):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    if not isinstance(progress, float):
        progress = 0
        status = 'Error: Progress Var Must be Float\r\n'
    if progress < 0:
        progress = 0
        status = "Halt ...\r\n"
        return
    if progress >= 1.0:
        progress = 1.0
        status = '\n'

    block = int(round(bar_length * progress))
    text = "\r%s[ %s] %.2f%% %s" % (prefix, "=" * block +">" + " " * int(bar_length - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

    if progress >= 1.0:
        if callback_func is not None:
            callback_func()


def iter_progress_bar(iter_at,iter_intervals,bar_prefix="",info=None,info_format=None,bar_length=50):
    iter_at = int(iter_at)
    iter_intervals = int(iter_intervals)
    progress = (iter_at % iter_intervals + 1.0) / float(iter_intervals)

    callback_func = None
    if info is not None:
        if info_format is None:
            info_format = ','.join(['%0.2f' for _ in range(len(info))])
        callback_func = lambda : print(info_format%tuple([_() for _ in info]))

    progress_bar(progress, callback_func, bar_prefix, bar_length)