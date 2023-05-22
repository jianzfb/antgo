# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 22:03
# @File    : dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from typing import Union
from pathlib import Path

from antgo.pipeline.functional.entity import Entity


class DatasetMixin:
    """
    Mixin for dealing with dataset
    """

    # pylint: disable=import-outside-toplevel
    @classmethod
    def from_glob(cls, *args):  # pragma: no cover
        """
        generate a file list with `pattern`
        """
        from glob import glob
        files = []
        for path in args:
            files.extend(glob(path))
        if len(files) == 0:
            raise FileNotFoundError(f'There is no files with {args}.')
        return cls(files)

    @classmethod
    def placeholder(cls, *args):
        signal_set = [
            'EAGLEEYE_SIGNAL_RGB_IMAGE',
            'EAGLEEYE_SIGNAL_BGR_IMAGE',
            'EAGLEEYE_SIGNAL_RGBA_IMAGE',
            'EAGLEEYE_SIGNAL_BGRA_IMAGE',
            'EAGLEEYE_SIGNAL_GRAY_IMAGE',
            'EAGLEEYE_SIGNAL_MASK',
            'EAGLEEYE_SIGNAL_DET',
            'EAGLEEYE_SIGNAL_DET_EXT',
            'EAGLEEYE_SIGNAL_TRACKING',
            'EAGLEEYE_SIGNAL_POS_2D',
            'EAGLEEYE_SIGNAL_POS_3D',
            'EAGLEEYE_SIGNAL_LANDMARK',
            'EAGLEEYE_SIGNAL_CLS',
            'EAGLEEYE_SIGNAL_STATE',
            'EAGLEEYE_SIGNAL_SWITCH',
            'EAGLEEYE_SIGNAL_RECT',
            'EAGLEEYE_SIGNAL_LINE',
            'EAGLEEYE_SIGNAL_POINT'
        ]
        for arg in args:
            if isinstance(arg, tuple):
                assert (arg[-1] in signal_set)
        return cls(list(args))

    @classmethod
    def read_json(cls, json_path: Union[str, Path], encoding: str = 'utf-8'):
        import json

        def inner():
            with open(json_path, 'r', encoding=encoding) as f:
                string = f.readline()
                while string:
                    data = json.loads(string)
                    string = f.readline()
                    yield Entity(**data)

        return cls(inner())

    def random_sample(self):
        # core API already exists
        pass

    def filter_data(self):
        # core API already exists
        pass

