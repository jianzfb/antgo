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
from sqlalchemy.dialects.mysql import FLOAT
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
    'str': 'Unicode(1024)',
    'date': 'DateTime', 
    'int': 'Integer', 
    'bool': 'Boolean',
    'float': 'FLOAT(precision=32, scale=4)'
}


def create_db_orm(configs):
    custom_tables = ''
    # 搜集反向关联信息
    inverse_link_info = {}
    for config_info in configs:
        if config_info['table'] in ['task', 'user']:
            # 内部已经构建
            continue
        
        table_cls_name = config_info['table'].capitalize()
        table_name = config_info['table'].lower()
        for link_name in config_info['links']:
            # code in ref table
            if link_name not in inverse_link_info:
                inverse_link_info[link_name] = []
            inverse_link_info[link_name].append(f'{table_name} = relationship("{table_cls_name}", back_populates="{link_name}", cascade="all,delete, delete-orphan")')

    # 搜集表信息，并构建
    for config_info in configs:
        if config_info['table'] in ['task']:
            # 内部已经构建
            continue

        table_cls_name = config_info['table'].capitalize()
        table_name = config_info['table'].lower()
        table_field_info = ''
        for field_name, field_info in config_info['fields'].items():
            field_type = field_info['type']
            if field_type == 'date':
                field_default = 'datetime.datetime.now'
            elif field_type in ['int', 'float', 'bool']:
                field_default = field_info["default"]
            else:
                field_default = f'"{field_info["default"]}"'

            table_field_info += f'    {field_name} = Column({filed_type_map[field_type]}, default={field_default})\n'

        table_link_info = ''
        for link_name in config_info['links']:
            # code in table
            table_link_info += f'''
    @declared_attr
    def {link_name}_id(cls):
        return Column(Integer, ForeignKey('{link_name}.id', ondelete='CASCADE'), nullable=True)
    {link_name} = relationship('{link_name.capitalize()}', back_populates="{table_name}")
            '''

        table_inverse_info = ''
        if table_name in inverse_link_info:
            for inverse_code in inverse_link_info[table_name]:
                table_inverse_info += f'{inverse_code}\n'

        table_template = f'''
class {table_cls_name}(Base):
    __tablename__ = '{table_name}'
    id = Column(Integer, primary_key=True)
{table_field_info}
{table_link_info}
    {table_inverse_info}

    def __repr__(self):
        return '{table_cls_name}'
        '''

        custom_tables += f'{table_template}\n'

    custom_inverse_links_in_user = ''
    if 'user' in inverse_link_info:
        for info in inverse_link_info['user']:
            custom_inverse_links_in_user += f'''
    {info}
            '''
    custom_inverse_links_in_task = ''
    if 'task' in inverse_link_info:
        for info in inverse_link_info['task']:
            custom_inverse_links_in_task += f'''
    {info}
            '''
    orm_content = \
        gen_code('./templates/orm')(custom_tables=custom_tables, custom_inverse_links_in_user=custom_inverse_links_in_user, custom_inverse_links_in_task=custom_inverse_links_in_task)

    with open('./orm.py', 'w') as fp:
        fp.write(orm_content)


@contextmanager
def local_session_context():
    global __global_db_session
    sess = __global_db_session()
    try:
        yield sess
    except Exception as e:  # swallow any exception
        sess.rollback()
    finally:
        sess.close()


__thread_db_info = threading.local()


@contextmanager
def thread_session_context():
    global __global_db_session
    global __thread_db_info
    __thread_db_info.sess = __global_db_session()
    try:
        yield __thread_db_info.sess
    except Exception as e:  # swallow any exception
        __thread_db_info.sess.rollback()
    finally:
        __thread_db_info.sess.close()


def get_thread_session():
    global __thread_db_info
    return __thread_db_info.sess