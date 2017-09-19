# -*- coding: UTF-8 -*-
# @Time    : 17-9-19
# @File    : deep_analysis.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np

def discrete_multi_model_measure_analysis():
  # 95%, 52%, 42%, 13%, only best, 0%
  pass

def _greedy_mining_by_euclidean(score):
  height, width = score.shape[0:2]
  
  pass

def continuous_multi_model_measure_analysis(samples_score, data_id, data_source, filter_tag=None, random_sampling=5):
  # filter by tag
  remained_id = []
  if filter_tag is not None:
    for data_index in data_id:
      _, label = data_source.at(data_index)
      if filter_tag in label['tag']:
        remained_id.append(data_index)
  
  if len(remained_id) == 0:
    remained_id = data_id
  
  samples_score = samples_score[:, remained_id]
  
  # low, middle, high
  model_num, samples_num = samples_score.shape[0:2]
  reorganized_samples_score = np.zeros((model_num, samples_num))
  reorganized_model_id = []
  reorganized_sample_id = []
  if model_num <= 2:
    # reorder rows (model)
    model_score = np.sum(samples_score, axis=1)
    reorganized_model_id = np.argsort(model_score)
    
    reorganized_samples_score = samples_score[reorganized_model_id.tolist(), :]
    samples_score = reorganized_samples_score.copy()
    
    # reorder cols (samples)
    s = np.sum(reorganized_samples_score, axis=0)
    reorganized_sample_id = np.argsort(s)
    reorganized_samples_score = samples_score[:, reorganized_sample_id.tolist()]
  else:
    # biclusting
    pass
  
  # low region (0 ~ 1/3)
  region_start = 0
  region_end = np.minimum(region_start+samples_num / 3, samples_num)
  low_region_sampling = remained_id[region_start:region_end]
  low_region_sampling = np.random.choice(low_region_sampling, random_sampling, False)
  
  # middle region (1/3 ~ 2/3)
  region_start = np.maximum(samples_num, 0)
  region_end = np.minimum(region_start+samples_num / 3, samples_num)
  middle_region_sampling = remained_id[region_start: region_end]
  middle_region_sampling = np.random.choice(middle_region_sampling, random_sampling, False)
  
  # high region (2/3 ~ 1)
  region_start = np.maximum(samples_num / 3 * 2, 0)
  region_end = np.minimum(region_start+samples_num / 3, samples_num)
  high_region_sampling = remained_id[region_start: region_end]
  high_region_sampling = np.random.choice(high_region_sampling, random_sampling, False)

  # samples score, [], []
  return reorganized_samples_score, \
         reorganized_model_id, \
         reorganized_sample_id, \
         low_region_sampling, \
         middle_region_sampling, \
         high_region_sampling