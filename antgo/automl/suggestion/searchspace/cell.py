# -*- coding: UTF-8 -*-
# @Time    : 2019/1/10 5:38 PM
# @File    : cell.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.suggestion.searchspace.branch import *
import random


class Cell(object):
  def __init__(self, branch_num, base_channel):
    self.branch_pool = {'convbn_branch': ConvBnBranch,
                        'seperableconv_branch': SeperableConvBranch,
                        'spp_branch': SPPBranch,
                        'focus_branch': FocusBranch,
                        'se_branch': SEBranch,
                        'regionse_branch': RegionSEBranch,
                        'res_branch': ResBranch,
                        'bottleneck_res_branch': BottleNeckResBranch}
    self.branch_num = branch_num
    self.base_channel = base_channel
    self.current_channel = base_channel

  def _branch(self, cell_name, block_name, output_channel, target_type=None):
    branch_list = [v for k,v in self.branch_pool.items()]
    if target_type is None:
      branch_class = random.choice(branch_list)
      return branch_class(output_channel=output_channel, cell_name=cell_name, block_name=block_name)
    else:
      return self.branch_pool[target_type](output_channel=output_channel, cell_name=cell_name, block_name=block_name)

  def random(self, graph, branch_start, cell_name='', block_name=''):
    cell_branches = []
    cell_branches_id = []
    last_branch = branch_start
    for index in range(self.branch_num):
      cell_branches.append(self._branch(cell_name=cell_name,
                                        block_name=block_name,
                                        output_channel=self.current_channel))
      if branch_start == -1 and index == 0:
        graph.add_layer(cell_branches[-1], 0)
        last_branch = graph.layer_to_id[cell_branches[-1]]
        cell_branches_id.append(last_branch)
      else:
        graph.to_insert_layer(last_branch, cell_branches[-1])
        last_branch = graph.layer_to_id[cell_branches[-1]]
        cell_branches_id.append(last_branch)

    return cell_branches_id

  def change(self, graph, branch_id, target_type=None):
    channels = graph.layer_list[branch_id].output_shape[-1]
    branch = self._branch(cell_name=graph.layer_list[branch_id].cell_name,
                          block_name=graph.layer_list[branch_id].block_name,
                          output_channel=channels,
                          target_type=target_type)
    graph.to_replace_layer(branch_id, branch)
    return graph.layer_to_id[branch]

  def branch(self, name, nickname='', cell_name='', block_name=''):
    return self.branch_pool[name](output_channel=self.current_channel,
                                  nickname=nickname,
                                  cell_name=cell_name,
                                  block_name=block_name)

  def increase_channels(self, ratio=None):
    if ratio is None:
      self.current_channel = int(self.current_channel * 2)
    else:
      if ratio < 1.0:
        ratio = 1.0 / ratio
      self.current_channel = int(self.current_channel * ratio)

  def decrease_channels(self, ratio=None):
    if ratio is None:
      self.current_channel = min(int(self.current_channel * 0.5), 2)
    else:
      if ratio > 1:
        ratio = 1.0 / ratio
      self.current_channel = min(int(self.current_channel * ratio), 2)

  def restore_channels(self):
    self.current_channel = self.base_channel

  def use_channels(self, val):
    self.current_channel = val