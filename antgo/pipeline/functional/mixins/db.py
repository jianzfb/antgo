# -*- coding: UTF-8 -*-
# @Time    : 2020-05-28 22:36
# @File    : db.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session
from sqlalchemy import (
    inspect,
    Column, Integer, ForeignKey, Unicode, Boolean,
    DateTime
)
from antgo.pipeline.extent.op.loader import *
from contextlib import contextmanager
import datetime

__global_db_session = None
__global_db_orm = None

def create_db_session(db_url, **db_kwargs):
    import orm
    global __global_db_session
    if __global_db_session is not None:
        return __global_db_session

    session_factory = orm.new_session_factory(
                db_url,
                reset=False,
                echo=False,
                **db_kwargs
            )
    __global_db_session = scoped_session(session_factory)
    return __global_db_session

def get_db_session():
    global __global_db_session
    return __global_db_session


def update_db_orm(_orm):
    global __global_db_orm
    __global_db_orm = _orm


def get_db_orm():
    global __global_db_orm
    return __global_db_orm


filed_type_map = {
    'str': ('Unicode(1024)', '""'),
    'date': ('DateTime', 'datetime.datetime.now'),
    'int': ('Integer', 0),
    'bool': ('Boolean', False)
}


def create_db_orm(configs):
    custom_tables = ''
    for config_info in configs:
        if config_info['table'] in ['task']:
            # 内部已经构建
            continue

        table_cls_name = config_info['table'].capitalize()
        table_name = config_info['table'].lower()
        table_field_info = ''
        for field_name, field_type in config_info['fields'].items():
            table_field_info += f'    {field_name} = Column({filed_type_map[field_type][0]}, default={filed_type_map[field_type][1]})\n'

        table_link_info = ''
        inverse_link_info = {}
        for link_name in config_info['links']:
            # code in table
            table_link_info += f'''
    @declared_attr
    def {link_name}_id(cls):
        return Column(Integer, ForeignKey('{link_name}.id', ondelete='CASCADE'), nullable=True)
    {link_name} = relationship('{link_name.capitalize()}', back_populates="{table_name}")
            '''

            # code in ref table
            if link_name not in inverse_link_info:
                inverse_link_info[link_name] = []
            inverse_link_info[link_name].append(f'{table_name} = relationship("{table_cls_name}", back_populates="{link_name}", cascade="all,delete, delete-orphan")')

        table_template = f'''
class {table_cls_name}(Base):
    __tablename__ = '{table_name}'
    id = Column(Integer, primary_key=True)
{table_field_info}
{table_link_info}

    def __repr__(self):
        return '{table_cls_name}'/self.id
        '''

        custom_tables += f'{table_template}\n'

    custom_inverse_links_in_user = ''
    if 'user' in inverse_link_info:
        for info in inverse_link_info['user']:
            custom_inverse_links_in_user += f'''
    {info}
            '''

    orm_content = \
        gen_code('./templates/orm')(custom_tables=custom_tables, custom_inverse_links_in_user=custom_inverse_links_in_user)

    with open('./orm.py', 'w') as fp:
        fp.write(orm_content)


@contextmanager
def thread_session_context(session_db):
    sess = session_db()
    try:
        yield sess
    except Exception as e:  # swallow any exception
        sess.rollback()
    finally:
        sess.close()
