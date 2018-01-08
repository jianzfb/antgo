# -*- coding: UTF-8 -*-
# Time: 1/6/18
# File: yesno_crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.crowdsource import *


class AntYesNoCrowdsource(AntCrowdsource):
  def __init__(self, task, name):
    super(AntYesNoCrowdsource, self).__init__(task, name)

    # set _client_response_data
    self.client_response_data = {'RESPONSE': ['YES', 'NO'], 'TYPE': 'SELECT'}

  def eva(self):
    pass
