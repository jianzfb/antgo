# -*- coding: UTF-8 -*-
# Time: 1/6/18
# File: yesno_crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.crowdsource import *
from bs4 import BeautifulSoup


default = {'AntYesNoCrowdsource': ('YesorNo', '')}


class AntYesNoCrowdsource(AntCrowdsource):
  def __init__(self, task):
    super(AntYesNoCrowdsource, self).__init__(task, 'YesorNo')

    # DATA_TAG
    # 控制如何从recoder中拾取数据显示在页面上
    self.left_tag = getattr(self.task, 'yes_or_no_left', 'PREDICT')
    self.right_tag = getattr(self.task, 'yes_or_no_right', 'GT')

    # set _client_query_html
    self.client_html_template = 'yesno_crowdsource.html'
    # {HTML_ELEM: DATA_TAG, HTML_ELEM: DATA_TAG, ...}
    self.client_keywords_template = {'A': self.left_tag, 'B': self.right_tag}

    # set _client_query_data
    self.client_query_data = {'QUERY': {self.left_tag: 'IMAGE',
                                        self.right_tag: 'IMAGE'}}
    self._is_support_rank = True
    self._crowdsource_title = 'YesorNo'
    self._crowdsource_type = "YN"
  
  def __where_in_table(self, worksite, element_id):
    if type(element_id) != list:
      element_id = [element_id]

    output = []
    soup = BeautifulSoup(worksite)
    for element in element_id:
      element_node = soup.find(id=element)
      container_id = element_node.parent.get('id')
      output.append(int(container_id))

    return output

  def ground_truth_response(self, client_id, query_index, record_db):
    # CONCLUSION, WORKSITE
    user_worksite = self._client_response_record[client_id]['RESPONSE'][query_index]['WORKSITE']

    # 'PREDICT','GT' position in table
    xy = self.__where_in_table(user_worksite, [self.left_tag, self.right_tag])

    user_gt_response = ''
    if int(xy[0]) < int(xy[1]):
      user_gt_response = 'RIGHT is REAL'
    else:
      user_gt_response = 'LEFT is REAL'

    # parse user_worksite (resource)
    return {'CUSTOM_GT': {'DATA': user_gt_response, 'TYPE': 'TEXT'}}

  def eva(self, data=None, label=None):
    if len(self._client_response_record) == 0:
      with open(os.path.join(self.dump_dir, 'crowdsource_record.txt')) as fp:
        content = fp.read()
        self._client_response_record = json.loads(content)

    score = None
    score_count = None
    person_count = 0
    for _, client_response in self._client_response_record.items():
      response_data = client_response['RESPONSE']
      query_index_map = client_response['ID']

      if score is None:
        score = np.zeros((len(query_index_map), len(self._client_response_record)))
        score_count = np.zeros((len(query_index_map), len(self._client_response_record)))
      
      for query_index, client_data in enumerate(response_data):
        if client_data is None:
          # some clients may have incomplete response
          continue

        real_query_index = query_index_map[query_index]
        client_worksite = client_data['WORKSITE']
        client_conclusion = client_data['CONCLUSION']
        
        xy = self.__where_in_table(client_worksite, [self.left_tag, self.right_tag])
        score_count[real_query_index, person_count] = 1.0
        if int(xy[0]) < int(xy[1]):
          if client_conclusion == 'RIGHT':
            score[real_query_index, person_count] = 1.0
          else:
            score[real_query_index, person_count] = 0.0
        else:
          if client_conclusion == 'LEFT':
            score[real_query_index, person_count] = 1.0
          else:
            score[real_query_index, person_count] = 0.0

      person_count += 1

    denominator_s = np.sum(score_count, axis=1)
    numerator_s = np.sum(score, axis=1)
    evaluation_score = np.mean(numerator_s / denominator_s)
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': float(evaluation_score), 'type': 'SCALAR', 'score_map': score_count}]}, }