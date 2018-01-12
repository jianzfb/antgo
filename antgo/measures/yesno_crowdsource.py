# -*- coding: UTF-8 -*-
# Time: 1/6/18
# File: yesno_crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.crowdsource import *
from bs4 import BeautifulSoup

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

    output = []
    soup = BeautifulSoup(worksite)
    for element in element_id:
      element_node = soup.find(id=element)
      container_id = element_node.parent.get('id')
      xy = container_id.split('-')[-1]
      output.append(xy)

    return output

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


ss = '<td id="table-{id}-{r}{c}" style="text-align: center;"><img src="static/hello.png" id="GT"></td>'
soup = BeautifulSoup(ss)
element_node = soup.find(id='GT')
print(element_node)