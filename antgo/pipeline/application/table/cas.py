# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : cas.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.application.common.db import *
from urllib.parse import unquote_plus, quote_plus
from antgo.pipeline.utils.reserved import *
import os
from sqlalchemy import and_, or_


class CasOp(object):
    def __init__(self):
        pass

    def info(self):
        return ['ST', 'session_id', 'db']

    def __call__(self, ST, session_id, db):
        # 1.step 获取service ticket
        service_ticket = ST
        if service_ticket is None:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': f"missing request param ST"
                },
                session_id=session_id,
                status_code=401,
                message=f"missing request param ST"
            )

        # 检查service ticket合法性
        service_ticket_token = orm.ServiceTicket.find(db, service_ticket)
        if service_ticket_token is None or service_ticket_token.service_ticket_user is None:
            return ReservedRtnType(
                index = '__response__',
                data = {
                    'code': -1,
                    'message': 'fail',
                    'info': f"not valid ST"
                },
                session_id=session_id,
                status_code=401,
                message=f"not valid ST"
            )

        # TODO 检查service ticket时效性

        # 
        return service_ticket_token.service_ticket_user.name, service_ticket_token.service_ticket_user.admin
