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

  def _branch(self, cell_name, block_name, output_channel):
    branch_list = [v for k,v in self.branch_pool.items()]
    branch_class = random.choice(branch_list)
    return branch_class(output_channel=output_channel, cell_name=cell_name, block_name=block_name)

  def random(self, graph, branch_start, cell_name='', block_name=''):
    cell_branches = []
    cell_branches_id = []
    last_branch = branch_start
    for index in range(self.branch_num):
      cell_branches.append(self._branch(cell_name=cell_name,
                                        block_name=block_name,
                                        output_channel=self.base_channel))
      if branch_start == -1 and index == 0:
        graph.add_layer(cell_branches[-1], 0)
        last_branch = graph.layer_to_id[cell_branches[-1]]
        cell_branches_id.append(last_branch)
      else:
        graph.to_insert_layer(last_branch, cell_branches[-1])
        last_branch = graph.layer_to_id[cell_branches[-1]]
        cell_branches_id.append(last_branch)

    return cell_branches_id

  def change(self, graph, branch_id, iswider=False):
    channels = graph.layer_list[branch_id].output_shape[-1]
    branch = self._branch(cell_name=graph.layer_list[branch_id].cell_name,
                          block_name=graph.layer_list[branch_id].block_name,
                          output_channel=channels * 2 if iswider else channels)
    graph.to_replace_layer(branch_id, branch)
    return graph.layer_to_id[branch]

  def branch(self, name, nickname='', cell_name='', block_name=''):
    return self.branch_pool[name](output_channel=self.base_channel,
                                  nickname=nickname,
                                  cell_name=cell_name,
                                  block_name=block_name)