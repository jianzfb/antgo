# -*- coding: UTF-8 -*-
# Time: 1/6/18
# File: yesno_crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.crowdsource import *
from BeautifulSoup import BeautifulSoup

class AntYesNoCrowdsource(AntCrowdsource):
  def __init__(self, task, name):
    super(AntYesNoCrowdsource, self).__init__(task, name)

    # set _client_response_data
    self.client_response_data = {'RESPONSE': ['LEFT', 'RIGHT'], 'TYPE': 'SELECT'}

    # set _client_query_html
    self.client_query_html = ''

    # set _client_query_js
    self.client_query_js = ''

    # set _client_query_data
    self.client_query_data =  {'QUERY': {'PREDICT_PREDICT': 'IMAGE', 'GROUNDTRUTH_GT': 'IMAGE'},
                               'GROUNDTRUTH': {'PREDICT_PREDICT': 'IMAGE', 'GROUNDTRUTH_GT': 'IMAGE'}}

  def where_in_table(self, worksite, element_id):
    if type(element_id) != list:
      element_id = [element_id]

    for element in element_id:
      pass

    return []


  def prepare_custom_response(self, client_id, query_index, record_db):
    # CONCLUSION, WORKSITE
    user_worksite = self._client_response_record[client_id]['RESPONSE']['WORKSITE']

    # 'PREDICT','GT' position in table
    xy = self.where_in_table(user_worksite, ['PREDICT','GT'])

    user_gt_response = ''
    if int(xy[0][1]) < int(xy[1][1]):
      user_gt_response = 'RIGHT is REAL'
    else:
      user_gt_response = 'LEFT is REAL'

    # parse user_worksite (html)
    return {'CUSTOM_GT': {'DATA': user_gt_response, 'TYPE': 'TEXT'}}

  def eva(self):
    pass
