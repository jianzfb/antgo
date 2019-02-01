# -*- coding: UTF-8 -*-
# @Time    : 2019-01-29 10:15
# @File    : evolution_check.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import json
from antgo.automl.suggestion.models import *
from antgo.automl.suggestion.searchspace.evolution import *
# import tensorflow as tf
from antgo.codebook.tf.stublayers import *


def check_evolution_initialize():
  study_configuration = {'goal': 'MAXIMIZE',
                         'current_population': [],
                         'current_population_info': [],
                         'searchSpace': {'current_population': [],
                                         'current_population_info': [],
                                         'current_population_tag': 0},

                         }
  study_configuration = json.dumps(study_configuration)
  s = Study('aa', study_configuration=study_configuration, algorithm='', search_space=None)
  Study.create(s)

  population_size = 30

  for _ in range(100):
    es = EvolutionSearchSpace(s,
                              channel_mode='UP',
                              input_size='1,128,128,3;1,512,512,3;',
                              population_size=population_size)

    ss = json.loads(s.study_configuration)

    current_population = ss['searchSpace']['current_population']
    current_population_info = ss['searchSpace']['current_population_info']
    current_population_tag = ss['searchSpace']['current_population_tag']

    for _ in range(population_size):
      suggestion = es.get_new_suggestions()
      print(suggestion)

      # a = tf.placeholder(dtype=tf.float32,shape=[1,128,128,3])
      # b = tf.placeholder(dtype=tf.float32,shape=[1,512,512,3])
      # graph = Decoder().decode(suggestion[0].structure[0])
      # graph.update_by(suggestion[0].structure[1])
      # graph.materialization(input_nodes=[a,b],layer_factory=LayerFactory(),batch_size=1)

      suggestion[0].status = 'Completed'
      suggestion[0].objective_value = random.random()

check_evolution_initialize()

def check_evolution_search_space():
  study_configuration = {'goal': 'MAXIMIZE',
                         'current_population': [],
                         'current_population_info': [],
                         'searchSpace': {'current_population': [],
                                         'current_population_info': [],
                                         'current_population_tag': 0},
                         }
  study_configuration = json.dumps(study_configuration)
  s = Study('aa', study_configuration=study_configuration, algorithm='', search_space=None)
  Study.create(s)



  # population_size = 20
  # for i in range(200):
  #   es = EvolutionSearchSpace(s, input_size='1,128,128,3;1,512,512,3;', population_size=population_size)
  #   for _ in range(population_size):
  #     suggestion_1 = es.get_new_suggestions()
  #     print(suggestion_1)
  #
  #   dd = json.loads(s.study_configuration)
  #   current_population_tag = dd['searchSpace']['current_population_tag']
  #   trials = Trial.filter(study_name='aa', tag=current_population_tag)
  #   for trail in trials:
  #     trail.objective_value = random.random()
  #     trail.status = 'Completed'

# check_evolution_search_space()