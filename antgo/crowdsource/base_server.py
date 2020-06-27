# -*- coding: UTF-8 -*-
# @Time    : 2020-06-10 12:18
# @File    : base_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from tornado import web, gen
from tornado import httpclient
import tornado.web
import json


# error code
class RESPONSE_STATUS_CODE:
    SUCCESS = 200                   # [GET]
    RESOURCE_SUCCESS_CREATED = 201  # 新建、修改数据成功 [POST/PUT/PATCH]
    RESOURCE_SUCCESS_DELETE = 204   # 数据删除成功
    RESOURCE_NOT_FOUND = 404        # 未获得指定数据
    RESOURCE_GONE = 410             # 请求的资源已经被永久删除，不可得到 [GET]
    EXECUTE_ACCEPTED = 202          # 任务等待执行
    EXECUTE_UNAUTHORIZED = 401      # 未授权执行
    EXECUTE_FORBIDDEN = 403         # 拥有授权，但访问禁止
    REQUEST_NOT_ACCEPTED = 406      # 用户请求格式错误 [GET]
    REQUEST_INVALID = 400           # 用户请求数据存在问题 [POST/PUT/PATCH]
    INTERNAL_SERVER_ERROR = 500     # 服务器发生错误


class BaseHandler(tornado.web.RequestHandler):
  """Base Handler class with access to common methods and properties."""

  def set_default_headers(self):
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header('Access-Control-Allow-Headers', '*')
    self.set_header("Access-Control-Allow-Methods", "POST,GET,PUT,DELETE,PATCH,OPTIONS")
    self.set_header("Access-Control-Expose-Headers", "Content-Disposition")
    # self.set_header("Access-Control-Allow-Credentials", "true")

  @property
  def name(self):
    return self.settings.get('name', '-')

  @property
  def html_template(self):
    return self.settings['html_template']

  @property
  def keywords_template(self):
    return self.settings.get('keywords_template', {})

  @property
  def data_folder(self):
    return self.settings['data_folder']

  @property
  def db(self):
    return self.settings['db']

  @property
  def token(self):
    return self.settings['token']

  @property
  def task_name(self):
    return self.settings['task_name']

  @property
  def task_type(self):
    return self.settings['task_type']

  @property
  def upload_folder(self):
    return self.settings.get('upload', None)

  @property
  def download_folder(self):
    return self.settings.get('download', None)

  @property
  def dump_folder(self):
    return self.settings.get('dump_path', None)

  @property
  def request_queue(self):
    return self.settings.get('request_queue', None)

  @property
  def response_queue(self):
    return self.settings.get('response_queue', None)

  def response(self, status_code=RESPONSE_STATUS_CODE.SUCCESS, message='', content={}, status=None):
    self.set_status(status_code)

    if status is None:
      status = 'OK'
      if status_code not in [200, 201, 204]:
        status = 'ERROR'

    self.write(json.dumps({
      'status': status,
      'message': message,
      'content': content
    }))