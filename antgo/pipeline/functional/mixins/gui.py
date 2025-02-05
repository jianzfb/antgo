# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 22:43
# @File    : gui.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


class GUIMixin:
    def loop(self):
        for env_info in self:
            # 运行至此，GUI构建完成，需要驱动进事件循环
            env_info.__page_root__.loop()
            break
