# -*- coding: UTF-8 -*-
# @Time    : 2020-05-28 22:36
# @File    : db.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session
from antgo.pipeline.application import orm


def create_db(db_url, **db_kwargs):
    session_factory = orm.new_session_factory(
                db_url,
                reset=False,
                echo=False,
                **db_kwargs
            )
    _scoped_session = scoped_session(session_factory)
    return _scoped_session()