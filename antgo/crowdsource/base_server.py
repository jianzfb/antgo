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
from tornado.escape import json_decode
import time


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
    self.set_header("Access-Control-Allow-Origin", "http://localhost:8080")
    self.set_header('Access-Control-Allow-Headers', '*')
    self.set_header("Access-Control-Allow-Methods", "POST,GET,PUT,DELETE,PATCH,OPTIONS")
    self.set_header("Access-Control-Expose-Headers", "Content-Disposition")
    self.set_header("Access-Control-Allow-Credentials", "true")

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

  def get_json_argument(self, name, default=None):
    args = json_decode(self.request.body)
    # name = to_unicode(name)
    if name in args:
      return args[name]
    elif default is not None:
      return default
    else:
      raise tornado.web.MissingArgumentError(name)

  def timestamp_2_str(self, now_time):
    return time.strftime("%Y-%m-%dx%H-%M-%S", time.localtime(now_time))

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

  def set_login_cookie(self, user):
    """Set login cookies for the Hub and single-user server."""

    # create and set a new cookie token for the hub
    # if not self.get_current_user_cookie():
    self._set_user_cookie(user, 'antgo')

  def _set_user_cookie(self, user, server):
    # tornado <4.2 have a bug that consider secure==True as soon as
    # 'secure' kwarg is passed to set_secure_cookie
    if self.request.protocol == 'https':
      kwargs = {'secure': True}
    else:
      kwargs = {}
    # if self.subdomain_host:
    #     kwargs['domain'] = self.domain

    # self.log.debug("Setting cookie for %s: %s, %s", user.name, server, json.dumps(kwargs))
    self.set_secure_cookie(
      server,
      user['cookie_id'],
      **kwargs
    )

  def get_current_user_cookie(self):
    """get_current_user from a cookie token"""
    return self._user_for_cookie("antgo")

  def get_current_user(self):
    """get current username"""
    # 判断是否使用白盒用户验证机制
    if self.db.get('white_users', None) is None:
      if 'default_user' not in self.settings:
        default_user = {
          'full_name': 'ANTGO',
          'short_name': 'A',
          'labeling_sample': -1,
          'start_time': -1
        }
        self.settings['default_user'] = default_user

      return self.settings['default_user']
    user = self.get_current_user_cookie()
    return user

  def _user_for_cookie(self, cookie_name, cookie_value=None):
    """Get the User for a given cookie, if there is one"""
    # cookie_id = self.get_secure_cookie(
    #     cookie_name,
    #     cookie_value,
    #     max_age_days=self.cookie_max_age_days,
    # )
    cookie_id = self.get_secure_cookie(
      cookie_name,
      cookie_value
    )

    def clear():
      self.clear_cookie(cookie_name, path='/')

    if cookie_id is None:
      if self.get_cookie(cookie_name):
        clear()
      return

    cookie_id = cookie_id.decode('utf8', 'replace')
    # u = self.db.query(orm.User).filter(orm.User.cookie_id == cookie_id).first()
    if cookie_id not in self.db['users']:
      return None

    return self.db['users'][cookie_id]
