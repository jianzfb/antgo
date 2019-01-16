# -*- coding: UTF-8 -*-
# @Time    : 2019/1/7 10:34 PM
# @File    : evolution.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.automl.graph import *
from antgo.automl.suggestion.searchspace.abstract_searchspace import *
from antgo.automl.suggestion.models import *
from antgo.automl.suggestion.searchspace.cell import *
from antgo.automl.basestublayers import *
import json
import random
import uuid
from datetime import datetime
from antgo.utils.utils import *
from antgo.automl.suggestion.bayesian import *
from antgo.automl.suggestion.metric import *
from antgo.utils import logger
import copy

class Mutation(object):
  def __init__(self, cell, max_block_num=4, min_block_num=1, max_cell_num=10, min_cell_num=1):
    self._mutate_rate_for_skip_block = 0.5
    self._mutate_rate_for_block = 0.5
    self._mutate_rate_for_skip_cell = 0.5
    self._mutate_rate_for_cell = 0.5
    self._mutate_rate_for_branch = 0.5
    self._mutate_rate_for_skip_branch = 0.5
    self._max_block_num = max_block_num
    self._min_block_num = min_block_num
    self._max_cell_num = max_cell_num
    self._min_cell_num = min_cell_num
    self._cell_wider_mutate={}

    self._cell = cell

  def mutate(self, graph, graph_info):
    # block [[cell_id, cell_id,...],[],[]]
    # cells [[branch_id, branch_id,...],[],[]]
    try:
      # random mutate one block
      if random.random() < self._mutate_rate_for_block:
        graph, graph_info = self._mutate_for_block(graph, graph_info)

      # random skip once between block
      if random.random() < self._mutate_rate_for_skip_block:
        for start_layer_id, end_layer_id in self._find_allowed_skip_block(graph_info):
          try:
            self._mutate_for_skip_block(graph, start_layer_id, end_layer_id)
            break
          except:
            pass

      # random mutate one cell
      if random.random() < self._mutate_rate_for_cell:
        graph, graph_info = self._mutate_for_cell(graph, graph_info)

      # random mutate one branch
      if random.random() < self._mutate_rate_for_branch:
        graph, graph_info = self._mutate_for_branch(graph, graph_info)

      # random skip once between cells
      if random.random() < self._mutate_rate_for_skip_cell:
        for start_layer_id, end_layer_id in self._find_allowed_skip_cell(graph_info):
          try:
            self._mutate_for_skip_cell(graph, start_layer_id, end_layer_id)
            break
          except:
            pass

      # random skip once between branches
      if random.random() < self._mutate_rate_for_skip_branch:
        for start_layer_id, end_layer_id in self._find_allowed_skip_branch(graph_info):
          try:
            self._mutate_for_skip_branch(graph, start_layer_id, end_layer_id)
            break
          except:
            pass
    except:
      pass

  def _find_allowed_skip_block(self, graph_info):
    blocks = graph_info['blocks']
    cells = graph_info['cells']

    block_num = len(blocks)
    if block_num < 1:
      raise StopIteration

    for _ in range(50):
      a, b = random.sample(list(range(block_num)), 2)
      start_n = a if a < b else b
      end_n = a if a > b else b
      if end_n - start_n < 1:
        continue

      start_branch_id = cells[blocks[start_n][-1]][-1]
      end_branch_id = cells[blocks[end_n][-1]][-1]
      yield start_branch_id, end_branch_id

    raise StopIteration

  def _find_allowed_skip_cell(self, graph_info):
    blocks = graph_info['blocks']
    cells = graph_info['cells']

    cell_num = 0
    random_block = []
    for _ in range(50):
      random_block = random.choice(blocks)
      cell_num = len(random_block)
      if cell_num <= 1:
        continue
      break

    if cell_num <= 1:
      raise StopIteration

    for _ in range(50):
      a, b = random.sample(list(range(cell_num)), 2)
      start_n = a if a < b else b
      end_n = a if a > b else b
      if end_n - start_n < 1:
        continue

      start_branch_id = cells[random_block[start_n]][-1]
      end_branch_id = cells[random_block[end_n]][-1]
      yield start_branch_id, end_branch_id

    raise StopIteration

  def _find_allowed_skip_branch(self, graph_info):
    blocks = graph_info['blocks']
    cells = graph_info['cells']

    random_cell_id = -1
    for _ in range(50):
      random_block_index = random.choice(list(range(len(blocks))))
      random_cell_id = random.choice(blocks[random_block_index])

      if len(cells[random_cell_id]) <= 1:
        continue
      break

    if random_cell_id == -1 or len(cells[random_cell_id]) <= 1:
      raise StopIteration

    for _ in range(50):
      a, b = random.sample(list(range(len(cells[random_cell_id]))), 2)
      start_n = a if a < b else b
      end_n = a if a > b else b
      if end_n - start_n < 1:
        continue

      start_branch_id = cells[random_cell_id][start_n]
      end_branch_id = cells[random_cell_id][end_n]
      yield start_branch_id, end_branch_id

    raise StopIteration

  def _mutate_for_skip_block(self, graph, start_layer_id, end_layer_id):
    if graph.has_skip(start_layer_id, end_layer_id):
      graph.to_remove_skip_model(start_layer_id, end_layer_id)
    else:
      start_layer_output_shape= graph.layer_list[start_layer_id].output_shape
      end_layer_output_shape = graph.layer_list[end_layer_id].output_shape
      skip_type = ['ADD', 'CONCAT']
      if random.choice(skip_type) == 'ADD' and start_layer_output_shape[-1] == end_layer_output_shape[-1]:
        graph.to_add_skip_model(start_layer_id, end_layer_id)
      else:
        graph.to_concat_skip_model(start_layer_id, end_layer_id)

      graph.update()

  def _mutate_for_skip_cell(self, graph, start_layer_id, end_layer_id):
    if graph.has_skip(start_layer_id, end_layer_id):
      graph.to_remove_skip_model(start_layer_id, end_layer_id)
    else:
      start_layer_output_shape= graph.layer_list[start_layer_id].output_shape
      end_layer_output_shape = graph.layer_list[end_layer_id].output_shape

      skip_type = ['ADD','CONCAT']
      if random.choice(skip_type) == 'ADD' and start_layer_output_shape[-1] == end_layer_output_shape[-1]:
        graph.to_add_skip_model(start_layer_id, end_layer_id)
      else:
        graph.to_concat_skip_model(start_layer_id, end_layer_id)

    graph.update()

  def _mutate_for_skip_branch(self, graph, start_layer_id, end_layer_id):
    if graph.has_skip(start_layer_id, end_layer_id):
      graph.to_remove_skip_model(start_layer_id, end_layer_id)
    else:
      start_layer_output_shape= graph.layer_list[start_layer_id].output_shape
      end_layer_output_shape = graph.layer_list[end_layer_id].output_shape

      skip_type = ['ADD','CONCAT']
      if random.choice(skip_type) == 'ADD' and start_layer_output_shape[-1] == end_layer_output_shape[-1]:
        graph.to_add_skip_model(start_layer_id, end_layer_id)
      else:
        graph.to_concat_skip_model(start_layer_id, end_layer_id)

    graph.update()

  def _mutate_for_block(self, graph, graph_info, mutate_type=['ADD','REMOVE']):
    blocks = graph_info['blocks']
    cells = graph_info['cells']
    block_dict = graph_info['block_info']     # {index: {'mode': 'u', 'size': 2}}
    block_dict = {int(k): v for k,v in block_dict.items()}

    block_num = len(blocks)
    random_mutate_type = random.choice(mutate_type)
    if block_num > 1 and random_mutate_type == 'REMOVE':
      random_block_id = random.choice(list(range(block_num)))
      random_block_id = 1

      # 1.step update graph
      for cell_id in blocks[random_block_id]:
        for branch_id in cells[cell_id]:
          graph.to_remove_layer(branch_id)

      if block_dict[random_block_id]['size'] != 0:
        graph.to_remove_layer(block_dict[random_block_id]['transition'])

      if len(block_dict[random_block_id]['input_transition']) > 0:
        # remove transition layer
        graph.to_remove_layer(block_dict[random_block_id]['input_transition'][0])

        # remove add layer
        graph.to_remove_layer(block_dict[random_block_id]['input_transition'][-1])

      # 2.step update blocks and cells record
      blocks.pop(random_block_id)

      # 3.step update graph info
      block_order_index = 0
      updated_block_dict = {}
      for m in range(block_num):
        if m != random_block_id:
          updated_block_dict[block_order_index] = block_dict[m]
          block_order_index += 1

      graph_info['blocks'] = blocks
      graph_info['cells'] = cells
      graph_info['block_info'] = updated_block_dict
      return graph, graph_info
    elif block_num < self._max_block_num and random_mutate_type == 'ADD':
      occupied_placeholder = [(0, rate) for rate in range(self._max_block_num)]
      for k, v in block_dict.items():
        occupied_placeholder[int(v['size'])] = (1, int(v['size']))

      random_placeholder = random.choice([m for m in occupied_placeholder if m[0] == 0])[1]

      start_block_index = -1
      for k, v in block_dict.items():
        if v['size'] < random_placeholder and v['size'] > start_block_index:
          start_block_index = k

      blocks.insert(start_block_index+1, [len(cells)])

      last_layer_id = cells[blocks[start_block_index][-1]][-1]

      if random_placeholder != 0:
        # 处理空间分辨率大小不一致
        target_h = graph.node_list[graph.get_input()[0]].shape[1] * np.power(2,random_placeholder)
        target_w = graph.node_list[graph.get_input()[0]].shape[2] * np.power(2,random_placeholder)

        resize_layer = graph.layer_factory.bilinear_resize(height=target_h, width=target_w)
        graph.to_insert_layer(last_layer_id, resize_layer)
        last_layer_id = graph.layer_to_id[resize_layer]

      cells.append(self._cell.random(graph,
                                     last_layer_id,
                                     block_name='block_%d'%random_placeholder,
                                     cell_name='cell_%d'%len(cells)))

      input_transition = []
      if random_placeholder != 0:
        target_h = graph.node_list[graph.get_input()[0]].shape[1] * np.power(2,random_placeholder)
        target_w = graph.node_list[graph.get_input()[0]].shape[2] * np.power(2,random_placeholder)

        # needed add skip add
        if len(graph.get_input()) > 1:
          for index, input_id in enumerate(graph.get_input()):
            if index == 0:
              continue

            if graph.node_list[input_id].shape[1] == target_h and \
                    graph.node_list[input_id].shape[2] == target_w:

              # 3x3 convolution layer
              transition_conv33 = graph.layer_factory.conv2d(input_channel=None,
                                                             filters=resize_layer.output_shape[-1],
                                                             kernel_size_h=3,
                                                             kernel_size_w=3)
              graph.add_layer(transition_conv33, input_id)

              # add skip model
              graph.to_add_skip_model(graph.layer_to_id[transition_conv33], graph.layer_to_id[resize_layer])

              input_transition = [graph.layer_to_id[transition_conv33],
                                  graph.adj_list[graph.layer_id_to_output_node_ids[graph.layer_to_id[transition_conv33]][0]][0][1]]

      graph_info['blocks'] = blocks
      graph_info['cells'] = cells

      updated_block_dict = {}
      old_index = 0
      for m in range(len(blocks)):
        if m != start_block_index+1:
          updated_block_dict[m] = block_dict[old_index]
          old_index += 1
        else:
          updated_block_dict[m] = {'mode': 'u',
                                   'size': random_placeholder,
                                   'transition': -1 if random_placeholder == 0 else last_layer_id,
                                   'input_transition': input_transition}

      graph_info['block_info'] = updated_block_dict
      return graph, graph_info
    else:
      return graph, graph_info

  def _mutate_for_cell(self, graph, graph_info):
    # blocks [[cell_id, cell_id,...],[],[]]
    # cells [[branch_id, branch_id,...],[],[]]
    blocks = graph_info['blocks']
    cells = graph_info['cells']

    random_block_id = random.choice(list(range(len(blocks))))
    cell_num = len(blocks[random_block_id])

    mutate_type = ['ADD', 'REMOVE']
    random_mutate_type = random.choice(mutate_type)
    if cell_num > 1 and random_mutate_type == 'REMOVE':
      cell_id = random.choice(blocks[random_block_id])
      for branch_id in cells[cell_id]:
        graph.to_remove_layer(branch_id)

      # cells.pop(cell_id)
      remove_cell_index = -1
      for v_i, v in enumerate(blocks[random_block_id]):
        if v == cell_id:
          remove_cell_index = v_i
          break

      blocks[random_block_id].pop(remove_cell_index)

      graph_info['blocks'] = blocks
      graph_info['cells'] = cells
      return graph, graph_info
    elif cell_num < self._max_cell_num and random_mutate_type == 'ADD':
      random_cell_id = random.choice(blocks[random_block_id])
      new_cell = self._cell.random(graph,
                                   cells[random_cell_id][-1],
                                   block_name='block_%d'%graph_info['block_info'][random_block_id]['size'],
                                   cell_name='cell_%d'%len(cells))
      cells.append(new_cell)

      middle_k = -1
      for ki, k in enumerate(blocks[random_block_id]):
        if k == random_cell_id:
          middle_k = ki
          break
      blocks[random_block_id].insert(middle_k+1, len(cells)-1)

      graph_info['blocks'] = blocks
      graph_info['cells'] = cells
      return graph, graph_info
    else:
      return graph, graph_info

  def _mutate_for_branch(self, graph, graph_info):
    # blocks [[cell_id, cell_id,...],[],[]]
    # cells [[branch_id, branch_id,...],[],[]]
    blocks = graph_info['blocks']
    cells = graph_info['cells']

    random_block_id = random.choice(list(range(len(blocks))))
    random_cell_id = random.choice(blocks[random_block_id])

    mutate_type = ['CHNAGE', 'NOCHANGE']
    if random.choice(mutate_type) == 'CHNAGE':
      branch_id = random.choice(cells[random_cell_id])
      new_branch_id = self._cell.change(graph, branch_id)
      ck = -1
      for v_i, v in enumerate(cells[random_cell_id]):
        if v == branch_id:
          ck = v_i
          break
      cells[random_cell_id][ck] = new_branch_id

      graph_info['blocks'] = blocks
      graph_info['cells'] = cells
      return graph, graph_info
    else:
      return graph, graph_info


class CrossOver(object):
  def __init__(self):
    pass


class EvolutionSearchSpace(AbstractSearchSpace):
  default_params={'max_block_num': 4,
                 'min_block_num': 1,
                 'max_cell_num': 1,
                 'min_cell_num': 1,
                 'branch_num': 5,
                 'branch_base_channel': 64,
                 'block_stack_mode': 'DECODER',
                 'population_size': 1,
                 'flops': 100000,
                 'input_size': ''}

  def __init__(self, study, **kwargs):
    super(EvolutionSearchSpace, self).__init__(study, **kwargs)
    self.study = study
    self.study_configuration = json.loads(study.study_configuration)
    self.study_goal = self.study_configuration['goal']

    self.max_block_num = int(kwargs.get('max_block_num', EvolutionSearchSpace.default_params['max_block_num']))
    self.min_block_num = int(kwargs.get('min_block_num', EvolutionSearchSpace.default_params['min_block_num']))

    self.max_cell_num = int(kwargs.get('max_cell_num', EvolutionSearchSpace.default_params['max_cell_num']))
    self.min_cell_num = int(kwargs.get('min_cell_num', EvolutionSearchSpace.default_params['min_cell_num']))

    self.branch_num = int(kwargs.get('branch_num', EvolutionSearchSpace.default_params['branch_num']))
    self.branch_base_channel = int(kwargs.get('branch_base_channel',
                                              EvolutionSearchSpace.default_params['branch_base_channel']))

    self.input_size = []
    if kwargs.get('input_size') != '':
      input_size = kwargs.get('input_size')
      for s in input_size.split(';'):
        if s == '':
          continue
        a,b,c,d = s.strip().split(',')
        self.input_size.append([int(a),int(b),int(c),int(d)])

    # ['ENCODER-DECODER', 'ENCODER', 'DECODER']
    self.block_stack_mode = 'DECODER'

    self.population_size = int(kwargs.get('population_size', EvolutionSearchSpace.default_params['population_size']))

    self.current_population = self.study_configuration['current_population'] if 'current_population' in self.study_configuration else []
    self.current_population_info = self.study_configuration['current_population_info'] if 'current_population_info' in self.study_configuration else []
    self.next_population = self.study_configuration['next_population'] if 'next_population' in self.study_configuration else []
    self.next_population_info = self.study_configuration['next_population_info'] if 'next_population_info' in self.study_configuration else []
    self.cell = Cell(branch_num=self.branch_num,
                     base_channel=self.branch_base_channel)
    self.mutation_operator = Mutation(self.cell,
                                      max_block_num=self.max_block_num,
                                      min_block_num=self.min_block_num,
                                      max_cell_num=self.max_cell_num,
                                      min_cell_num=self.min_cell_num)
    self.cross_over_operator = CrossOver()

    # temp structure recommand
    # initialize bayesian optimizer
    study_configuration = json.loads(study.study_configuration)
    if study_configuration['goal'] == 'MAXIMIZE':
      self.bo = BayesianOptimizer(0.0001, Accuracy, 0.1, 2.576)
    else:
      self.bo = BayesianOptimizer(0.0001, Loss, 0.1, 2.576)

    if len(Trial.filter(study_name=self.study.name)) == 0:
      # initialize search space (study)
      self._initialize_population()

    # get all completed trials
    self.trials = Trial.filter(study_name=self.study.name, status='Completed')
    x_queue = [np.array(trial.structure_encoder) for trial in self.trials]
    y_queue = [trial.objective_value for trial in self.trials]
    if len(x_queue) > 10:
      self.bo.fit(x_queue, y_queue)

  def _initialize_population(self, random_block_ini=False, random_cell_ini=False):
    # 1.step get default graph
    if self.block_stack_mode == 'DECODER':
      self.current_population = []
      for instance_index in range(self.population_size):
        # 1.step get block number
        blocks = [[]]
        block_info = {0: {'mode': 'u', 'size': 0, 'transition': -1, 'input_transition': []}}
        for index in range(self.max_block_num):
          if index == 0:
            continue

          if random.random() < 0.5 and random_block_ini:
            block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': []}
            blocks.append([])
          else:
            if len(self.input_size) > 1:
              if index == int(np.log2(self.input_size[1][1] / self.input_size[0][1])):
                block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': []}
                blocks.append([])
            else:
              block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': []}
              blocks.append([])

        block_num = len(blocks)

        # 2.step get cell number in every block
        cell_num = []
        for _ in range(block_num):
          if random_cell_ini:
            cell_num.append(random.choice(list(range(self.min_cell_num, self.max_cell_num))))
          else:
            cell_num.append(self.max_cell_num)

        # 3.step generate cells
        default_graph = Graph()
        default_graph.layer_factory = BaseLayerFactory()
        for input_s in self.input_size:
          default_graph.add_input(shape=input_s)

        last_layer_id = -1

        cells = []
        cell_id_offset = 0
        for block_id in range(block_num):
          if block_id == 0:
            for cell_id in range(cell_num[block_id]):
              cells.append(self.cell.random(default_graph,
                                            last_layer_id,
                                            cell_name='cell_%d'%(cell_id + cell_id_offset),
                                            block_name='block_%d'%block_info[block_id]['size']))

              last_layer_id = cells[-1][-1]
              blocks[block_id].append(cell_id + cell_id_offset)
          else:
            base_size = int(np.power(2, block_info[block_id]['size']))
            block_transition_layer = default_graph.layer_factory.bilinear_resize(height=self.input_size[0][1] * base_size,
                                                                                 width=self.input_size[0][2] * base_size)
            default_graph.add_layer(block_transition_layer, default_graph.layer_id_to_output_node_ids[last_layer_id][0])
            last_layer_id = default_graph.layer_to_id[block_transition_layer]
            block_info[block_id]['transition'] = last_layer_id

            for cell_id in range(cell_num[block_id]):
              cells.append(self.cell.random(default_graph,
                                            last_layer_id,
                                            cell_name='cell_%d'%(cell_id + cell_id_offset),
                                            block_name='block_%d'%block_info[block_id]['size']))

              last_layer_id = cells[-1][-1]
              blocks[block_id].append(cell_id + cell_id_offset)

            # link with compatible input
            for input_i, input_s in enumerate(self.input_size):
              if input_s[1] == self.input_size[0][1] * base_size and input_s[2] == self.input_size[0][2] * base_size:
                # 3x3 convolution layer
                transition_conv33 = default_graph.layer_factory.conv2d(input_channel=None,
                                                                       filters=default_graph.layer_list[last_layer_id].output_shape[-1],
                                                                       kernel_size_h=3,
                                                                       kernel_size_w=3)
                default_graph.add_layer(transition_conv33, default_graph.get_input()[input_i])

                # add skip model
                default_graph.to_add_skip_model(default_graph.layer_to_id[transition_conv33],block_info[block_id]['transition'])

                input_transition = [default_graph.layer_to_id[transition_conv33],
                                    default_graph.adj_list[
                                    default_graph.layer_id_to_output_node_ids[default_graph.layer_to_id[transition_conv33]][0]][0][1]]
                block_info[block_id]['input_transition'] = input_transition

          cell_id_offset += cell_num[block_id]

        # default_graph.visualization('aa.png')

        graph_encoder_str = Encoder(skipkeys=True).encode(default_graph)
        self.current_population.append(graph_encoder_str)
        self.current_population_info.append({'blocks': blocks, 'cells': cells, 'block_info': block_info})

      # write db
      study_configuration = json.loads(self.study.study_configuration)
      study_configuration['searchSpace']['current_population'] = self.current_population
      study_configuration['searchSpace']['current_population_info'] = self.current_population_info
      self.study.study_configuration = json.dumps(study_configuration)

  def dna(self, graph, graph_info):
    unary_p = self.max_block_num * self.max_cell_num * self.branch_num
    pairwise_p = self.max_block_num * self.max_block_num + \
                 self.max_block_num * self.max_cell_num * self.max_cell_num +\
                 self.max_block_num * self.max_cell_num * self.branch_num * self.branch_num

    dna_vector = np.zeros((unary_p + pairwise_p))
    blocks = graph_info['blocks']
    cells = graph_info['cells']
    block_dict = graph_info['block_info']     # {index: {'mode': 'u', 'size': 2}}
    block_dict = {int(k): v for k, v in block_dict.items()}

    # analyze cell type
    for block_index, block in enumerate(blocks):
      block_id = block_dict[block_index]['size']      # 0, 1, 2, ..., max_block_num
      dna_block_offset = block_id * self.max_cell_num * self.branch_num

      for cell_index, cell_id in enumerate(block):
        dna_cell_offset = cell_index * self.branch_num
        for branch_index, branch_id in enumerate(cells[cell_id]):
          dna_vector[dna_block_offset + dna_cell_offset + branch_index] =\
            graph.layer_list[branch_id].layer_type_encoder

    # analyze blocks links
    dna_brancn_links_offset = unary_p
    for block_index in range(len(blocks)):
      for next_block_index in range(len(blocks)):
        if next_block_index <= block_index:
          continue

        block_id = block_dict[block_index]['size']
        next_block_id = block_dict[next_block_index]['size']

        branch_id_in_block = cells[blocks[block_index][-1]][-1]
        branch_id_in_next_block = cells[blocks[next_block_index][-1]][-1]

        if graph.has_skip(branch_id_in_block, branch_id_in_next_block):
          dna_vector[dna_brancn_links_offset + block_id * self.max_block_num + next_block_id] = 1

    # analyze cells links
    dna_cell_links_offset = unary_p + self.max_block_num*self.max_block_num
    for block_index in range(len(blocks)):
      block_id = block_dict[block_index]['size']
      dna_cell_links_offset += block_id * self.max_cell_num * self.max_cell_num

      for cell_index, cell_id in enumerate(blocks[block_index]):
        for next_cell_index, next_cell_id in enumerate(blocks[block_index]):
          if next_cell_index <= cell_index:
            continue

          branch_id_in_cell = cells[cell_index][-1]
          branch_id_in_next_cell = cells[next_cell_index][-1]

          if graph.has_skip(branch_id_in_cell, branch_id_in_next_cell):
            dna_vector[dna_cell_links_offset + cell_index * self.max_cell_num + next_cell_index] = 1

    # analyze branchs links
    dna_branch_links_offset = unary_p + \
                              self.max_block_num*self.max_block_num + \
                              self.max_block_num*self.max_cell_num*self.max_cell_num
    for block_index in range(len(blocks)):
      block_id = block_dict[block_index]['size']
      for cell_index, cell_id in enumerate(blocks[block_index]):
        for branch_index, branch_id in enumerate(cells[cell_id]):
          for next_branch_index, next_branch_id in enumerate(cells[cell_id]):
            if next_branch_index <= branch_index:
              continue

            if graph.has_skip(branch_id, next_branch_id):
              dna_vector[dna_branch_links_offset +
                         block_id*self.max_cell_num*self.branch_num*self.branch_num +
                         cell_index*self.branch_num*self.branch_num +
                         branch_index * self.branch_num + next_branch_index] = 1

    return dna_vector

  def evolve_population(self):
    pass

  def random(self, count=1):
    study_configuration = json.loads(self.study.study_configuration)
    default_graph_str = study_configuration['searchSpace']['current_population'][0]
    default_graph = Decoder().decode(default_graph_str)
    default_graph.layer_factory = BaseLayerFactory()
    default_graph_info = study_configuration['searchSpace']['current_population_info'][0]

    try_count = 0
    proposed_search_space = []
    while True:
      if try_count > 50:
        logger.warn('couldnt find valid graph structure for study %s'%self.study.name)
        return [(None, None)]

      # 1.step skip branch in branch
      graph, graph_info = self.mutation_operator._mutate_for_branch(copy.deepcopy(default_graph),
                                                                    copy.deepcopy(default_graph_info))

      # 2.step mutation branch in cell
      for _ in range(self.branch_num - 1):
        for start_layer_id, end_layer_id in self.mutation_operator._find_allowed_skip_branch(graph_info):
          try:
            temp_graph = copy.deepcopy(graph)
            self.mutation_operator._mutate_for_skip_branch(temp_graph, start_layer_id, end_layer_id)
            graph = temp_graph
            break
          except:
            pass
      graph.visualization('%s.png' % (str(uuid.uuid4())))

      graph_dna = self.dna(graph, graph_info)

      trials = Trial.filter(study_name=self.study.name)
      is_not_valid = False
      for t in trials:
        if str(t.structure_encoder) == str(graph_dna.tolist()):
          is_not_valid = True
          break

      if is_not_valid:
        try_count += 1
        continue

      proposed_search_space.append((graph_dna, Encoder(skipkeys=True).encode(graph)))
      if len(proposed_search_space) == count:
        break

    return proposed_search_space

  def get_new_suggestions(self, number=1, **kwargs):
    all_trials = []
    for _ in range(number):
      x = None
      x_obj = None
      if len(self.trials) > 10:
        x, x_obj = self.bo.optimize_acq(self)
      else:
        x, x_obj = self.random()[0]

      if x is None or x_obj is None:
        break

      trail_name = '%s-%s' % (str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))
      trial = Trial.create(Trial(self.study.name, trail_name, created_time=time.time(), updated_time=time.time()))
      trial.structure = x_obj
      trial.structure_encoder = x.tolist()
      all_trials.append(trial)

    return all_trials
