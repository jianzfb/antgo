# -*- coding: UTF-8 -*-
# @Time    : 2022/9/15 23:19
# @File    : show.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.interactcontext import *
from antgo.pipeline.hparam import param_scope


class ShowMixin:
  def browser(self, type={}, title='antgo'):
    ctx = InteractContext()
    with ctx.Browser(title, {}, browser={'size': 0}) as hp:
      for index, info in enumerate(self):
        data = {}

        for k, v in type.items():
          data[k] = {
            'data': getattr(info, k),
            'type': v
          }

        data['id'] = index
        hp.context.recorder.record(data)

      hp.wait_until_stop()

  def label(self, label_type='RECT', label_metas={}, title='antgo'):
    ctx = InteractContext()
    with ctx.Activelearning(title, {}, activelearning={
      'label_type': label_type,
      'label_metas': label_metas,
      'stage': 'labeling'
    }, clear=False) as activelearning:
      for index, info in enumerate(self):
        data = {
          'image': None,  # 第二优先级
          'label_info': [],
          'id': index
        }
        for k in info.__dict__.keys():
          data['image'] = getattr(info, k)
          break
        activelearning.context.recorder.record(data)

      activelearning.wait_until_stop()