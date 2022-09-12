# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 17:39
# @File    : uri.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.utils.repo_normalize import RepoNormalize


class URI:
    """_summary_

    Examples:

    >>> op = URI('resnet-image-embedding')
    >>> op.namespace
    'towhee'
    >>> op.repo
    'resnet-image-embedding'
    >>> op.module_name
    'resnet_image_embedding'
    >>> op.resolve_module('my')
    'my/resnet_image_embedding'
    >>> op.resolve_modules('my1', 'my2')
    ['my1/resnet_image_embedding', 'my2/resnet_image_embedding']
    """

    def __init__(self, uri: str) -> None:
        self._raw = uri
        result = RepoNormalize(uri).parse_uri()
        for field in result._fields:
            setattr(self, field, getattr(result, field))

    @property
    def namespace(self):
        return self.author.replace('_', '-')

    @property
    def short_uri(self):
        return self.namespace + '/' + self.norm_repo

    @property
    def full_name(self):
        return self.namespace + '/' + self.module_name

    def resolve_module(self, ns):
        if not self.has_ns:
            self.author = ns
        return self.full_name

    def resolve_modules(self, *arg):
        return [self.resolve_module(ns) for ns in arg]

    def resolve_repo(self, ns):
        if not self.has_ns:
            self.author = ns
        return self.short_uri

    def resolve_repos(self, *arg):
        return [self.resolve_repo(ns) for ns in arg]


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
