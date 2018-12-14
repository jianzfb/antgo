# -*- coding: UTF-8 -*-
# @Time    : 2018/11/29 3:36 PM
# @File    : search_space.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.graph import *
from antgo.automl.suggestion.models import *
from antgo.utils.serialize import *
from random import randrange, sample
import random
import copy
import time
import uuid
from datetime import datetime
from antgo.utils.utils import *


class AbstractSearchSpace(object):
  def __init__(self, graph, flops=None):
    self.graph = graph

  def get_new_suggestions(self, study_name, number=1):
    raise NotImplementedError


class DenseArchitectureSearchSpace(AbstractSearchSpace):
  def __init__(self, graph, flops=None, **kwargs):
    super(DenseArchitectureSearchSpace, self).__init__(graph, flops)
    self.branches = kwargs.get('branches', 3)

  def get_new_suggestions(self, study_name, number=1):
    study = Study.get('name', study_name)
    if study is None:
      return [None]

    valid_num = 0
    all_trials = []
    while True:
      # 1.step make a structure suggestion
      clone_graph = copy.deepcopy(self.graph)
      outputs = self.graph.get_input()
      output_node_id = -1
      decoder_output_last = []
      for output_index, output_id in enumerate(outputs):
        if output_index > 0:
          if random.random() > 0.5:
            continue

        output = self.graph.node_list[output_id]

        temp = [output_id]
        for node_id in decoder_output_last:
          if self.graph.node_list[node_id].shape[1] != output.shape[1] or \
                  self.graph.node_list[node_id].shape[2] != output.shape[2]:
            output_node_id = self.graph.add_layer(self.graph.layer_factory.bilinear_resize(height=output.shape[1],
                                                                                 width=output.shape[2]),
                                             node_id)
            temp.append(output_node_id)
          else:
            temp.append(output_node_id)

        if len(temp) > 1:
          output_node_id = self.graph.add_layer(self.graph.layer_factory.concat(), temp)
          X = [output_node_id]
        else:
          output_node_id = temp[0]
          X = temp

        for branch_index in range(self.branches):
          # random select branch input from X
          X_index_list = list(range(len(X)))
          X_select_index_list = sample(X_index_list, random.randint(1, len(X_index_list)))
          X_selected = [X[i] for i in X_select_index_list]

          # concat all input
          if len(X_selected) > 1:
            output_node_id = self.graph.add_layer(self.graph.layer_factory.concat(), X_selected)

          # operator space
          r = random.randint(0, 2)
          if r == 0:
            # 1x1 convolution (conv + bn + relu)
            shape = self.graph.node_list[output_node_id].shape
            output_node_id = self.graph.add_layer(self.graph.layer_factory.conv2d(input_channel=shape[3],
                                                                        filters=256,
                                                                        kernel_size_h=1,
                                                                        kernel_size_w=1),
                                             output_node_id)
            output_node_id = self.graph.add_layer(self.graph.layer_factory.bn2d(),
                                             output_node_id)
            output_node_id = self.graph.add_layer(self.graph.layer_factory.relu(),
                                             output_node_id)
          elif r == 1:
            # 3x3 atrous separable convolution
            shape = self.graph.node_list[output_node_id].shape
            # rate 1,3,6,9,12,15,18,21
            min_hw = min(shape[1], shape[2])
            rate_list = [1, 3, 6, 9, 12, 15, 18, 21]
            rate_list = [rate_list[i] for i in range(len(rate_list)) if rate_list[i] < min_hw]

            rate_h = rate_list[random.randint(0, len(rate_list) - 1)]
            rate_w = rate_list[random.randint(0, len(rate_list) - 1)]

            output_node_id = self.graph.add_layer(self.graph.layer_factory.separable_conv2d(input_channel=shape[3],
                                                                                  filters=256,
                                                                                  kernel_size_h=3,
                                                                                  kernel_size_w=3,
                                                                                  rate_h=rate_h,
                                                                                  rate_w=rate_w),
                                             output_node_id)
            output_node_id = self.graph.add_layer(self.graph.layer_factory.bn2d(),
                                             output_node_id)
            output_node_id = self.graph.add_layer(self.graph.layer_factory.relu(),
                                             output_node_id)
          else:
            # spatial pyramid pooling
            shape = self.graph.node_list[output_node_id].shape
            min_hw = min(shape[1], shape[2])

            gh = [1, 2, 4, 8]
            gh = [n for n in gh if n < min_hw]
            grid_h = gh[random.randint(0, len(gh) - 1)]

            gw = [1, 2, 4, 8]
            gw = [n for n in gw if n < min_hw]
            grid_w = gw[random.randint(0, len(gw) - 1)]
            output_node_id = self.graph.add_layer(self.graph.layer_factory.spp(grid_h=grid_h, grid_w=grid_w),
                                             output_node_id)

          X.append(output_node_id)

        output_node_id = self.graph.add_layer(self.graph.layer_factory.concat(), X[1:])
        decoder_output_last.append(output_node_id)

      # check structure is valid
      current_graph_md5 = clone_graph.extract_descriptor()
      trails = Trial.filter(key='study_name', value=study_name)
      is_not_valid = False
      for t in trails:
        if (t.status is None and (time.time() - t.updated_time < 2*60*60)) or t.status == 'Completed':
          if t.md5 == current_graph_md5:
            is_not_valid = True
            break
      if is_not_valid:
        continue

      trail_name = '%s-%s' % (str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))
      trial = Trial.create(Trial(study.name, trail_name, created_time=time.time(), updated_time=time.time()))
      trial.parameter_values = Encoder(skipkeys=True).encode(clone_graph)
      trial.md5 = current_graph_md5
      all_trials.append(trial)

      if len(all_trials) == number:
        break

    return all_trials