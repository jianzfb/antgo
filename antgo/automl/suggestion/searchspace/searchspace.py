# -*- coding: UTF-8 -*-
# @Time    : 2019-02-01 16:06
# @File    : searchspace.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.automl.graph import *
from antgo.automl.suggestion.searchspace.abstract_searchspace import *
from antgo.automl.suggestion.searchspace.cell import *
import numpy as np
import random
from itertools import chain


class SearchSpace(AbstractSearchSpace):
  _default_params={'max_block_num': 4,
                   'min_block_num': 1,
                   'max_cell_num': 1,
                   'min_cell_num': 1,
                   'branch_num': 5,
                   'base_channel': 64,
                   'channel_mode': 'CONSTANT',
                   'block_stack_mode': 'DECODER',
                   'input_size': ''}

  def __init__(self, study, **kwargs):
    super(SearchSpace, self).__init__(study, **kwargs)

    self.max_block_num = int(kwargs.get('max_block_num', SearchSpace._default_params['max_block_num']))
    self.min_block_num = int(kwargs.get('min_block_num', SearchSpace._default_params['min_block_num']))

    self.max_cell_num = int(kwargs.get('max_cell_num', SearchSpace._default_params['max_cell_num']))
    self.min_cell_num = int(kwargs.get('min_cell_num', SearchSpace._default_params['min_cell_num']))

    self.branch_num = int(kwargs.get('branch_num', SearchSpace._default_params['branch_num']))
    self.base_channel = int(kwargs.get('base_channel',
                                       SearchSpace._default_params['base_channel']))

    # 'CONSTANT','UP','DOWN'
    self.channel_mode = kwargs.get('channel_mode',
                                   SearchSpace._default_params['channel_mode'])

    # ['ENCODER-DECODER', 'ENCODER', 'DECODER']
    self.block_stack_mode = 'DECODER'

    self.input_size = []
    if kwargs.get('input_size') != '':
      input_size = kwargs.get('input_size')
      for s in input_size.split(';'):
        if s == '':
          continue
        a,b,c,d = s.strip().split(',')
        self.input_size.append([int(a),int(b),int(c),int(d)])

    self.block_num = 0
    if len(self.input_size) > 1:
      for index in range(self.max_block_num):
        for input_index in range(len(self.input_size)):
          if index == int(np.log2(self.input_size[input_index][1] / self.input_size[0][1])):
            self.block_num += 1
            break
    else:
      self.block_num = self.max_block_num

    self.cell_num = self.max_cell_num
    self.cell = Cell(branch_num=self.branch_num,
                     base_channel=self.base_channel)

  def random(self, random_block_ini=False, random_cell_ini=False, mode='decoder'):
    # 1.step get block number
    blocks = [[]]
    block_info = {0: {'mode': 'u', 'size': 0, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}}
    for index in range(self.max_block_num):
      if index == 0:
        continue

      if random.random() < 0.5 and random_block_ini:
        block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
        blocks.append([])
      else:
        if len(self.input_size) > 1:
          for input_index in range(len(self.input_size)):
            if index == int(np.log2(self.input_size[input_index][1] / self.input_size[0][1])):
              block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
              blocks.append([])
              break
        else:
          block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
          blocks.append([])

    block_num = len(blocks)

    # 2.step get cell number in every block
    cell_num = []
    for _ in range(block_num):
      if random_cell_ini:
        cell_num.append(random.choice(list(range(self.min_cell_num, self.max_cell_num))))
      else:
        cell_num.append(self.max_cell_num)

    # 3.step generate cells and branches
    default_graph = Graph()
    default_graph.layer_factory = BaseLayerFactory()
    for input_s in self.input_size:
      default_graph.add_input(shape=input_s)

    last_layer_id = -1

    cells = []
    cell_id_offset = 0
    last_channel = self.cell.current_channel

    # reset cell channels
    self.cell.restore_channels()
    # build block stacks
    for block_id in range(block_num):
      if block_id == 0:
        for cell_id in range(cell_num[block_id]):
          cells.append(self.cell.random(default_graph,
                                        last_layer_id,
                                        cell_name='cell_%d' % (cell_id + cell_id_offset),
                                        block_name='block_%d' % block_info[block_id]['size']))

          last_layer_id = cells[-1][-1]
          blocks[block_id].append(cell_id + cell_id_offset)
      else:
        # adjust block channels
        if self.channel_mode == 'UP':
          self.cell.increase_channels(2.0)
        elif self.channel_mode == 'DOWN':
          self.cell.decrease_channels(0.5)

        block_info[block_id]['base_channel'] = self.cell.current_channel

        # 1.step channel compatibility
        if self.cell.current_channel != last_channel:
          # 3x3 convolution layer
          transition_conv33 = default_graph.layer_factory.conv2d(input_channel=None,
                                                                 filters=self.cell.current_channel,
                                                                 kernel_size_h=3,
                                                                 kernel_size_w=3)

          default_graph.add_layer(transition_conv33, default_graph.layer_id_to_output_node_ids[last_layer_id][0])
          last_layer_id = default_graph.layer_to_id[transition_conv33]
          last_channel = self.cell.current_channel

        # 2.step spatial compatibility
        base_size = int(np.power(2, block_info[block_id]['size']))
        block_transition_layer = default_graph.layer_factory.bilinear_resize(height=self.input_size[0][1] * base_size,
                                                                             width=self.input_size[0][2] * base_size)
        default_graph.add_layer(block_transition_layer, default_graph.layer_id_to_output_node_ids[last_layer_id][0])
        last_layer_id = default_graph.layer_to_id[block_transition_layer]

        block_info[block_id]['transition'] = last_layer_id

        for cell_id in range(cell_num[block_id]):
          cells.append(self.cell.random(default_graph,
                                        last_layer_id,
                                        cell_name='cell_%d' % (cell_id + cell_id_offset),
                                        block_name='block_%d' % block_info[block_id]['size']))

          last_layer_id = cells[-1][-1]
          blocks[block_id].append(cell_id + cell_id_offset)

        # 3.step link with compatible input
        for input_i, input_s in enumerate(self.input_size):
          if input_s[1] == self.input_size[0][1] * base_size and input_s[2] == self.input_size[0][2] * base_size:
            # identity layer
            identity_layer = default_graph.layer_factory.identity()
            default_graph.add_layer(identity_layer, default_graph.get_input()[input_i])

            # add skip model
            default_graph.to_add_skip_model(default_graph.layer_to_id[identity_layer],
                                            block_info[block_id]['transition'])
            default_graph.update()

      cell_id_offset += cell_num[block_id]

    # 4.step build connection information between blocks,cells, branches
    block_connection_type = random.choice([0, 0.5, 1])
    block_connections = []
    if block_connection_type == 0:
      # no connection
      block_connections = np.zeros((block_num * block_num)).tolist()
    elif block_connection_type == 1:
      # dense connection
      block_connections = (np.ones((block_num * block_num)) - np.eye(block_num).flatten()).tolist()
    elif block_connection_type == 0.5:
      # random connection
      block_connections = np.zeros((block_num * block_num)).tolist()
      for bi in range(block_num):
        for bj in range(block_num):
          if bj > bi:
            target = random.choice([0, 1])
            block_connections[bi * block_num + bj] = target
            block_connections[bj * block_num + bi] = target

    cell_connection_type = np.random.choice([0, 0.5, 1], block_num).tolist()
    cell_connections = [np.zeros(self.max_cell_num * self.max_cell_num).tolist() for _ in range(block_num)]
    for b in range(block_num):
      if cell_connection_type[b] == 0:
        # no connection
        cell_connections[b] = np.zeros(self.max_cell_num * self.max_cell_num).tolist()
      elif cell_connection_type[b] == 1:
        # dense connection
        cell_connections[b] = (np.ones(self.max_cell_num * self.max_cell_num) - np.eye(self.max_cell_num).flatten()).tolist()
      else:
        # random connection
        cell_connections[b] = np.zeros(self.max_cell_num * self.max_cell_num).tolist()
        for ci in range(self.max_cell_num):
          for cj in range(self.max_cell_num):
            if cj > ci:
              target = random.choice([0, 1])
              cell_connections[b][ci * self.max_cell_num + cj] = target
              cell_connections[b][cj * self.max_cell_num + ci] = target

    branch_connection_type = np.random.choice([0, 0.5, 1], block_num * self.max_cell_num).tolist()
    branch_connections = [np.zeros(self.branch_num * self.branch_num).tolist() for _ in
                          range(block_num * self.max_cell_num)]
    for c in range(block_num * self.max_cell_num):
      if branch_connection_type[c] == 0:
        # no connection
        branch_connections[c] = np.zeros(self.branch_num * self.branch_num).tolist()
      elif branch_connection_type[c] == 1:
        # dense connection
        branch_connections[c] = (np.ones(self.branch_num * self.branch_num)-np.eye(self.branch_num).flatten()).tolist()
      else:
        # random connection
        branch_connections[c] = np.zeros(self.branch_num * self.branch_num).tolist()
        for bi in range(self.branch_num):
          for bj in range(self.branch_num):
            if bj > bi:
              target = random.choice([0, 1])
              branch_connections[c][bi * self.branch_num + bj] = target
              branch_connections[c][bj * self.branch_num + bi] = target

    default_graph.layer_factory = None
    graph_encoder_str = Encoder(skipkeys=True).encode(default_graph)
    graph_info = {'blocks': blocks,
                  'cells': cells,
                  'block_info': block_info,
                  'connection': {'block': [block_connection_type],
                                 'block_connection': block_connections,
                                 'cell': cell_connection_type,
                                 'cell_connection': cell_connections,
                                 'branch': branch_connection_type,
                                 'branch_connection': branch_connections},
                  'ops': [default_graph.layer_list[s].layer_name for s in list(chain(*cells))]}

    return graph_encoder_str, graph_info

  def encoder(self, graph, graph_info):
    # graph only include no skip connection
    block_num = len(graph_info['blocks'])
    op_length = block_num * self.max_cell_num * self.branch_num
    connection_length = 1 + block_num * block_num + \
                        block_num + block_num * self.max_cell_num * self.max_cell_num + \
                        block_num * self.max_cell_num + \
                        block_num * self.max_cell_num * self.branch_num * self.branch_num

    encoder_vector = np.zeros((op_length+connection_length))

    blocks = graph_info['blocks']
    cells = graph_info['cells']
    # for op region
    for block_index, block in enumerate(blocks):
      dna_block_offset = block_index * self.max_cell_num * self.branch_num

      for cell_index, cell_id in enumerate(block):
        dna_cell_offset = cell_index * self.branch_num

        for branch_index, branch_id in enumerate(cells[cell_id]):
          encoder_vector[dna_block_offset + dna_cell_offset + branch_index] =\
            graph.layer_list[branch_id].layer_type_encoder

    # for connection region
    # 1.step block connection region
    offset = op_length
    encoder_vector[offset] = graph_info['connection']['block'][0]
    encoder_vector[offset+1:offset+1+block_num*block_num] = np.array(graph_info['connection']['block_connection'])

    # 2.step cell connection region
    offset += 1 + block_num * block_num
    encoder_vector[offset:offset+block_num] = np.array(graph_info['connection']['cell'])

    long_v = []
    for bb in range(block_num):
      long_v += graph_info['connection']['cell_connection'][bb]

    encoder_vector[offset+block_num:
               offset+block_num+block_num * self.max_cell_num * self.max_cell_num] = np.array(long_v)


    # 3.step branch connection region
    offset += block_num + block_num * self.max_cell_num * self.max_cell_num
    encoder_vector[offset:offset+block_num * self.max_cell_num] = np.array(graph_info['connection']['branch'])
    offset_in = offset + block_num * self.max_cell_num

    long_v = []
    for cc in range(len(cells)):
      long_v += graph_info['connection']['branch_connection'][cc]

    encoder_vector[offset_in:] = np.array(long_v)
    return encoder_vector