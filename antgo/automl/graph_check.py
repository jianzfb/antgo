# -*- coding: UTF-8 -*-
# @Time    : 2019/1/11 3:33 PM
# @File    : graph_check.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

# check graph transform
from antgo.automl.graph import *
from antgo.automl.basestublayers import *
from antgo.automl.suggestion.searchspace.evolution import *

def check_replace_layer():
  graph = Graph()
  graph.layer_factory = BaseLayerFactory()
  graph.add_input(shape=(1, 128, 128, 3))

  outputs = graph.get_input()
  start_node = outputs[0]

  # group 1
  layer_1 = graph.layer_factory.conv2d(3, 16, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_1')
  output_node_1 = graph.add_layer(layer_1, start_node)
  layer_2 = graph.layer_factory.bn2d(nickname='bn2d_1')
  output_node_2 = graph.add_layer(layer_2, output_node_1)
  layer_3 = graph.layer_factory.relu(nickname='relu_1')
  output_node_3 = graph.add_layer(layer_3, output_node_2)

  # group 2
  layer_4 = graph.layer_factory.conv2d(16,32, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_32_2')
  output_node_4 = graph.add_layer(layer_4, output_node_3)
  layer_5 = graph.layer_factory.conv2d(32,64, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_64_2')
  output_node_5 = graph.add_layer(layer_5, output_node_4)
  layer_6 = graph.layer_factory.conv2d(64,128, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_128_2')

  output_node_6 = graph.add_layer(layer_6, output_node_5)

  # group 3
  layer_7 = graph.layer_factory.conv2d(128, 256, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_256_3')
  output_node_7 = graph.add_layer(layer_7, output_node_6)

  graph.visualization('bb.png')

  # add skip
  layer_3_id = graph.layer_to_id[layer_3]
  layer_6_id = graph.layer_to_id[layer_6]

  graph.to_concat_skip_model(layer_3_id, layer_6_id)
  layer_8 = graph.layer_factory.conv2d(144, 128, kernel_size_h=1, kernel_size_w=1, nickname='conv2d_after_concat')
  output_node_8 = graph.add_layer(layer_8, output_node_7)

  resize_layer_9 = graph.layer_factory.bilinear_resize(height=256, width=256)
  output_node_9 = graph.add_layer(resize_layer_9, output_node_8)

  conv_layer_10 = graph.layer_factory.conv2d(128, 334, kernel_size_h=1, kernel_size_w=1, nickname='ccc')
  output_node_10 = graph.add_layer(conv_layer_10, output_node_9)

  graph.visualization('aa.png')

  layer_5_id = graph.layer_to_id[layer_5]
  new_layer = graph.layer_factory.relu()
  graph.to_replace_layer(layer_5_id, new_layer)

  graph.visualization('cc.png')

  graph.to_remove_layer(layer_5_id)
  graph.visualization('dd.png')

  layer = graph.layer_factory.conv2d(123,256, kernel_size_h=1, kernel_size_w=1, nickname='insert')
  graph.to_insert_layer(layer_6_id, layer)
  graph.visualization('ee.png')

  graph.to_concat_skip_model(graph.layer_to_id[layer_3], graph.layer_to_id[layer_7])
  graph.visualization('ff.png')

  graph.to_concat_skip_model(graph.layer_to_id[layer_4], graph.layer_to_id[layer])
  graph.visualization('gg.png')

  graph.to_remove_skip_model(graph.layer_to_id[layer_3], graph.layer_to_id[layer_7])
  graph.visualization('hh.png')

  graph.to_remove_skip_model(graph.layer_to_id[layer_4], graph.layer_to_id[layer])
  graph.visualization('ii.png')

  graph.to_remove_layer(graph.layer_to_id[layer])
  graph.visualization('mm.png')

  graph.to_add_skip_model(graph.layer_to_id[layer_4], graph.layer_to_id[layer_7])
  graph.visualization('nn.png')

  graph.to_add_skip_model(graph.layer_to_id[layer_7], graph.layer_to_id[conv_layer_10])
  graph.visualization('ww.png')

  graph.to_remove_skip_model(graph.layer_to_id[layer_7], graph.layer_to_id[conv_layer_10])
  graph.visualization('qq.png')


def check_mutate_evolution():
  graph = Graph()
  graph.layer_factory = BaseLayerFactory()
  graph.add_input(shape=(1, 128, 128, 3))
  graph.add_input(shape=(1, 256, 256, 44))
  graph.add_input(shape=(1, 512, 512, 88))
  graph.add_input(shape=(1, 1024, 1024, 199))

  # block 1
  # cell 1
  cell_1 = Cell(3, 32)
  cell_1_branch_1 = cell_1.branch('convbn_branch', 'cell_1_branch_1')
  cell_1_branch_1_node = graph.add_layer(cell_1_branch_1, 0)
  cell_1_branch_2 = cell_1.branch('spp_branch', 'cell_1_branch_2')
  cell_1_branch_2_node = graph.add_layer(cell_1_branch_2, cell_1_branch_1_node)
  cell_1_branch_3 = cell_1.branch('seperableconv_branch', 'cell_1_branch_3')
  cell_1_branch_3_node = graph.add_layer(cell_1_branch_3, cell_1_branch_2_node)

  # cell 2
  cell_2 = Cell(3, 32)
  cell_2_branch_1 = cell_2.branch('convbn_branch', 'cell_2_branch_1')
  cell_2_branch_1_node = graph.add_layer(cell_2_branch_1, cell_1_branch_3_node)
  cell_2_branch_2 = cell_2.branch('spp_branch', 'cell_2_branch_2')
  cell_2_branch_2_node = graph.add_layer(cell_2_branch_2, cell_2_branch_1_node)
  cell_2_branch_3 = cell_2.branch('seperableconv_branch', 'cell_2_branch_3')
  cell_2_branch_3_node = graph.add_layer(cell_2_branch_3, cell_2_branch_2_node)

  # block 2
  block_2_transition= graph.layer_factory.bilinear_resize(height=256, width=256)
  block_2_transition_node = graph.add_layer(block_2_transition, cell_2_branch_3_node)

  # cell 3
  cell_3 = Cell(1, 64)

  cell_3_branch_1 = cell_3.branch('spp_branch', 'cell_3_branch_1')
  cell_3_branch_1_node = graph.add_layer(cell_3_branch_1, block_2_transition_node)


  # block 3
  block_3_transition = graph.layer_factory.bilinear_resize(height=512, width=512)
  block_3_transition_node = graph.add_layer(block_3_transition, cell_3_branch_1_node)

  # cell 4
  cell_4 = Cell(3, 128)

  cell_4_branch_1 = cell_4.branch('spp_branch', 'cell_4_branch_1')
  cell_4_branch_1_node = graph.add_layer(cell_4_branch_1, block_3_transition_node)

  cell_4_branch_2 = cell_4.branch('spp_branch', 'cell_4_branch_2')
  cell_4_branch_2_node = graph.add_layer(cell_4_branch_2, cell_4_branch_1_node)

  cell_4_branch_3 = cell_4.branch('spp_branch', 'cell_4_branch_3')
  cell_4_branch_3_node = graph.add_layer(cell_4_branch_3, cell_4_branch_2_node)

  # cell 5
  cell_5 = Cell(3, 128)

  cell_5_branch_1 = cell_5.branch('spp_branch', 'cell_5_branch_1')
  cell_5_branch_1_node = graph.add_layer(cell_5_branch_1, cell_4_branch_3_node)

  cell_5_branch_2 = cell_5.branch('spp_branch', 'cell_5_branch_2')
  cell_5_branch_2_node = graph.add_layer(cell_5_branch_2, cell_5_branch_1_node)

  cell_5_branch_3 = cell_5.branch('spp_branch', 'cell_5_branch_3')
  cell_5_branch_3_node = graph.add_layer(cell_5_branch_3, cell_5_branch_2_node)

  # cell 6
  cell_6 = Cell(3, 128)

  cell_6_branch_1 = cell_6.branch('spp_branch', 'cell_6_branch_1')
  cell_6_branch_1_node = graph.add_layer(cell_6_branch_1, cell_5_branch_3_node)

  cell_6_branch_2 = cell_6.branch('spp_branch', 'cell_6_branch_2')
  cell_6_branch_2_node = graph.add_layer(cell_6_branch_2, cell_6_branch_1_node)

  cell_6_branch_3 = cell_6.branch('spp_branch', 'cell_6_branch_3')
  cell_6_branch_3_node = graph.add_layer(cell_6_branch_3, cell_6_branch_2_node)

  graph.visualization('aa.png')

  graph_info = {'blocks':[[0,1],[2],[3,4,5]], 'cells': [[graph.layer_to_id[cell_1_branch_1],
                                                         graph.layer_to_id[cell_1_branch_2],
                                                         graph.layer_to_id[cell_1_branch_3]],
                                                        [graph.layer_to_id[cell_2_branch_1],
                                                         graph.layer_to_id[cell_2_branch_2],
                                                         graph.layer_to_id[cell_2_branch_3]],
                                                        [graph.layer_to_id[cell_3_branch_1]],
                                                        [graph.layer_to_id[cell_4_branch_1],
                                                         graph.layer_to_id[cell_4_branch_2],
                                                         graph.layer_to_id[cell_4_branch_3]],
                                                        [graph.layer_to_id[cell_5_branch_1],
                                                         graph.layer_to_id[cell_5_branch_2],
                                                         graph.layer_to_id[cell_5_branch_3]],
                                                        [graph.layer_to_id[cell_6_branch_1],
                                                         graph.layer_to_id[cell_6_branch_2],
                                                         graph.layer_to_id[cell_6_branch_3]]],
                'block_info': {0: {'mode': 'u', 'size': 0, 'transition': -1, 'input_transition': []},
                               1: {'mode': 'u', 'size': 1, 'transition': graph.layer_to_id[block_2_transition], 'input_transition': []},
                               2: {'mode': 'u', 'size': 2, 'transition': graph.layer_to_id[block_3_transition], 'input_transition': []}}}


  # # 测试random skip cell
  mutate = Mutation()
  # for try_i in range(20):
  #   skip_start = -1
  #   skip_stop = -1
  #   for start_layer_id, end_layer_id in mutate._find_allowed_skip_cell(graph_info):
  #     try:
  #       mutate._mutate_for_skip_cell(graph, start_layer_id, end_layer_id)
  #       skip_start = start_layer_id
  #       skip_stop = end_layer_id
  #       break
  #     except:
  #       pass
  #
  #   graph.visualization('random_skip_cell_%d.png'%try_i)
  #   if skip_start != -1 and skip_stop != -1:
  #     if graph.has_skip(skip_start, skip_stop):
  #       graph.to_remove_skip_model(skip_start,skip_stop)
  #
  #   graph.visualization('restore_random_skip_cell_%d.png'%try_i)


  # # 测试random skip branch
  # for try_i in range(20):
  #   skip_start = -1
  #   skip_stop = -1
  #   if random.random() < mutate._mutate_rate_for_skip_branch:
  #     for start_layer_id, end_layer_id in mutate._find_allowed_skip_branch(graph_info):
  #       try:
  #         start_layer_id = 0
  #         end_layer_id = 2
  #         mutate._mutate_for_skip_branch(graph, start_layer_id, end_layer_id)
  #         skip_start = start_layer_id
  #         skip_stop = end_layer_id
  #         break
  #       except:
  #         pass
  #   graph.visualization('random_skip_branch_%d.png'%try_i)
  #   if skip_start != -1 and skip_stop != -1:
  #     if graph.has_skip(skip_start, skip_stop):
  #       graph.to_remove_skip_model(skip_start,skip_stop)
  #
  #   graph.visualization('random_skip_branch_restore_%d.png'%try_i)


  # # 测试random mutate one branch
  # for try_i in range(20):
  #   graph, graph_info = mutate._mutate_for_branch(graph, graph_info)
  #   graph.visualization('mutate_one_branch_%d.png'%try_i)


  # # 测试random mutate one cell
  # for try_i in range(20):
  #   graph, graph_info = mutate._mutate_for_cell(graph, graph_info)
  #   graph.visualization('mutate_one_cell_%d.png'%try_i)

  # # 测试random skip block
  # for try_i in range(20):
  #   if random.random() < mutate._mutate_rate_for_skip_block:
  #     for start_layer_id, end_layer_id in mutate._find_allowed_skip_block(graph_info):
  #       try:
  #         mutate._mutate_for_skip_block(graph, start_layer_id, end_layer_id)
  #         break
  #       except:
  #         pass
  #
  #   graph.visualization('random_skip_block_%d.png'%try_i)

  # # 测试 random mutate one block
  # graph, graph_info = mutate._mutate_for_block(graph, graph_info, mutate_type=['ADD'])
  # graph.visualization('random_mutate_block_nn.png')
  #
  # graph, graph_info = mutate._mutate_for_block(graph, graph_info, mutate_type=['REMOVE'])
  # graph.visualization('random_mutate_block_mm.png')

# 1.step check replace operation
# check_replace_layer()

# 2.step check mutate operation
# check_mutate_evolution()

# 3.step test initialize population
# import json
# def check_initualize_population():
#   study_configuration = {'goal': 'MIN',
#                          'current_population': [],
#                          'next_population': []}
#   study_configuration = json.dumps(study_configuration)
#   s = Study('aa', study_configuration=study_configuration,algorithm='',search_space=None)
#   evolution_s = EvolutionSearchSpace(s)
#   evolution_s._initialize_population()
#   pass
#
# check_initualize_population()


# 4.step test graph to cnn
import json
import os
# import tensorflow as tf
# from antgo.codebook.tf.stublayers import *
def check_graph_to_cnn():
  study_configuration = {'goal': 'MIN',
                         'current_population': [],
                         'current_population_info': [],
                         'next_population': [],
                         'next_population_info': [],
                         'searchSpace': {}}
  study_configuration = json.dumps(study_configuration)
  s = Study('aa', study_configuration=study_configuration, algorithm='', search_space=None)

  es = EvolutionSearchSpace(s, population_size=1, input_size='1,128,128,3;1,512,512,3')
  es._initialize_population()

  aa = json.loads(s.study_configuration)
  graph_encoder_str = aa['searchSpace']['current_population'][0]
  graph = Decoder().decode(graph_encoder_str)
  graph.layer_factory = BaseLayerFactory()
  graph.visualization('aa.png')

  graph_info = aa['searchSpace']['current_population_info'][0]
  # ss= es.dna(graph, graph_info)

  # es.mutation_operator._mutate_for_block(graph, aa['searchSpace']['current_population_info'][0], ['ADD'])
  # graph.visualization('bb.png')
  # # os.makedirs('summary')
  # train_writer = tf.summary.FileWriter('summary/', tf.get_default_graph())

  for epoch in range(200):
    for start_layer_id, end_layer_id in es.mutation_operator._find_allowed_skip_block(graph_info):
      try:
        graph = es.mutation_operator._mutate_for_skip_block(graph, start_layer_id, end_layer_id)
        for l in graph.layer_list:
          if l.layer_name == 'add':
            input_nodes = graph.layer_id_to_input_node_ids[graph.layer_to_id[l]]
            if graph.node_list[input_nodes[0]].shape[-1] != graph.node_list[input_nodes[1]].shape[-1]:
              print('asdf')
              graph.update()

        print(graph.flops)
        break
      except:
        pass

    # random mutate one cell
    graph, graph_info = es.mutation_operator._mutate_for_cell(graph, graph_info)

    # random mutate one branch
    graph, graph_info = es.mutation_operator._mutate_for_branch(graph, graph_info)

    for start_layer_id, end_layer_id in es.mutation_operator._find_allowed_skip_cell(graph_info):
      try:
        graph = es.mutation_operator._mutate_for_skip_cell(graph, start_layer_id, end_layer_id)
        for l in graph.layer_list:
          if l.layer_name == 'add':
            input_nodes = graph.layer_id_to_input_node_ids[graph.layer_to_id[l]]
            if graph.node_list[input_nodes[0]].shape[-1] != graph.node_list[input_nodes[1]].shape[-1]:
              print('asdf')
              graph.update()

        print(graph.flops)
        break
      except:
        pass

    if random.random() < es.mutation_operator._mutate_rate_for_skip_cell:
      for start_layer_id, end_layer_id in es.mutation_operator._find_allowed_skip_cell(graph_info):
        try:
          graph = es.mutation_operator._mutate_for_skip_cell(graph, start_layer_id, end_layer_id)

          for l in graph.layer_list:
            if l.layer_name == 'add':
              input_nodes = graph.layer_id_to_input_node_ids[graph.layer_to_id[l]]
              if graph.node_list[input_nodes[0]].shape[-1] != graph.node_list[input_nodes[1]].shape[-1]:
                print('asdf')
                graph.update()
          break
        except:
          pass

    for start_layer_id, end_layer_id in es.mutation_operator._find_allowed_skip_branch(graph_info):
      try:
        graph = es.mutation_operator._mutate_for_skip_branch(graph, start_layer_id, end_layer_id)
        for l in graph.layer_list:
          if l.layer_name == 'add':
            input_nodes = graph.layer_id_to_input_node_ids[graph.layer_to_id[l]]
            if graph.node_list[input_nodes[0]].shape[-1] != graph.node_list[input_nodes[1]].shape[-1]:
              print('asdf')
              graph.update()

        print(graph.flops)
        # if epoch == 4:
        #   a = tf.placeholder(dtype=tf.float32,shape=[1,128,128,3])
        #   b = tf.placeholder(dtype=tf.float32,shape=[1,512,512,3])
        #   graph.materialization(input_nodes=[a,b],layer_factory=LayerFactory())
        break
      except:
        pass



  # graph.visualization('bb.png')
  # a = tf.placeholder(dtype=tf.float32,shape=[1,128,128,3])
  # b = tf.placeholder(dtype=tf.float32,shape=[1,256,256,3])
  # graph.materialization(input_nodes=[a,b],layer_factory=LayerFactory())
  #
  # graph.visualization('cc.png')
  # ss= es.dna(graph, graph_info)
  #
  # mutate._mutate_for_cell(graph, graph_info)
  # graph.visualization('dd.png')
  #
  # for start_layer_id, end_layer_id in mutate._find_allowed_skip_cell(graph_info):
  #   try:
  #     mutate._mutate_for_skip_cell(graph, start_layer_id, end_layer_id)
  #     break
  #   except:
  #     pass
  #
  # graph.visualization('ee.png')
  # pass
  #
  # ss = es.dna(graph, graph_info)
  # mutate._mutate_for_branch(graph, graph_info)
  # graph.visualization('mm.png')
  #
  # for start_layer_id, end_layer_id in mutate._find_allowed_skip_branch(graph_info):
  #   try:
  #     mutate._mutate_for_skip_branch(graph, start_layer_id, end_layer_id)
  #     break
  #   except:
  #     pass
  #
  # graph.visualization('nn.png')
  # ss = es.dna(graph, graph_info)

check_graph_to_cnn()


# # check evolution search space
# def check_evolution_search_space():
#   study_configuration = {'goal': 'MAXIMIZE',
#                          'current_population': [],
#                          'current_population_info': [],
#                          'searchSpace': {'current_population': [],
#                                          'current_population_info': [],
#                                          'current_population_tag': 0},
#                          }
#   study_configuration = json.dumps(study_configuration)
#   s = Study('aa', study_configuration=study_configuration, algorithm='', search_space=None)
#   Study.create(s)
#
#   population_size = 20
#   for i in range(200):
#     es = EvolutionSearchSpace(s, input_size='1,128,128,3;1,512,512,3;', population_size=population_size)
#     for _ in range(population_size):
#       suggestion_1 = es.get_new_suggestions()
#       print(suggestion_1)
#
#     dd = json.loads(s.study_configuration)
#     current_population_tag = dd['searchSpace']['current_population_tag']
#     trials = Trial.filter(study_name='aa', tag=current_population_tag)
#     for trail in trials:
#       trail.objective_value = random.random()
#       trail.status = 'Completed'
#
#
#
#
#
# check_evolution_search_space()