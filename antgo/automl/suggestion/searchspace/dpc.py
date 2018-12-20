# -*- coding: UTF-8 -*-
# @Time    : 2018/11/29 3:36 PM
# @File    : dpc.py
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
from antgo.automl.suggestion.searchspace.abstract_searchspace import *
from antgo.automl.suggestion.bayesian import *
from antgo.automl.suggestion.metric import *
from antgo.automl.basestublayers import *
from antgo.utils import logger
import json


class DPCSearchSpace(AbstractSearchSpace):
  def __init__(self, study, flops=None, **kwargs):
    super(DPCSearchSpace, self).__init__(study, float(flops), **kwargs)
    self.branches = kwargs.get('branches', 3)
    self.study = study
    self.bo_min_samples = 10

    study_configuration = json.loads(self.study.study_configuration)
    graph_content = study_configuration['graph']
    self.graph = Decoder().decode(graph_content)
    self.graph.layer_factory = BaseLayerFactory()
    self.channels = kwargs.get('channels', 64)

    # get all completed trials
    self.trials = Trial.filter(study_name=self.study.name, status='Completed')

    # initialize bayesian optimizer
    self.bo = BayesianOptimizer(0.0001, Accuracy, 0.1, 2.576)
    x_queue = [np.array(trial.structure_encoder) for trial in self.trials]
    y_queue = [trial.objective_value for trial in self.trials]
    if len(x_queue) > self.bo_min_samples:
      self.bo.fit(x_queue, y_queue)

  def random(self, count=1):
    try_count = 0
    proposed_search_space = []
    while True:
      if try_count > 50:
        logger.warn('couldnt find valid graph structure')
        return [(None, None)]

      # 1.step make a structure suggestion
      clone_graph = copy.deepcopy(self.graph)
      outputs = clone_graph.get_input()
      output_node_id = -1
      decoder_output_last = []

      graph_encoder = []
      for output_index, output_id in enumerate(outputs):
        branch_offset = (1+8*8+4*4+self.branches)
        graph_output_encoder = np.zeros((branch_offset*self.branches), dtype=np.float32)

        if output_index > 0:
          if random.random() > 0.5:
            graph_encoder.extend(graph_output_encoder.tolist())
            continue

        output = clone_graph.node_list[output_id]

        temp = [output_id]
        for node_id in decoder_output_last:
          if clone_graph.node_list[node_id].shape[1] != output.shape[1] or \
                  clone_graph.node_list[node_id].shape[2] != output.shape[2]:
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.bilinear_resize(height=output.shape[1],
                                                                                             width=output.shape[2]),
                                                   node_id)
            temp.append(output_node_id)
          else:
            temp.append(output_node_id)

        if len(temp) > 1:
          output_node_id = clone_graph.add_layer(clone_graph.layer_factory.concat(), temp)
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
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.concat(), X_selected)
          else:
            output_node_id = X_selected[0]

          # encoder 1.step connect
          graph_output_encoder[branch_index*branch_offset:(branch_index+1)*branch_offset][X_select_index_list] = 1.0

          # operator space
          r = random.randint(0, 2)
          if r == 0:
            # 1x1 convolution (conv + bn + relu)
            shape = clone_graph.node_list[output_node_id].shape
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.conv2d(input_channel=shape[3],
                                                                                    filters=self.channels,
                                                                                    kernel_size_h=1,
                                                                                    kernel_size_w=1),
                                                   output_node_id)
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.bn2d(), output_node_id)
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.relu(), output_node_id)

            # encoder 2.step 1x1 convolution
            graph_output_encoder[branch_index * branch_offset:(branch_index + 1) * branch_offset][self.branches + 0] = 1.0
          elif r == 1:
            # 3x3 atrous separable convolution
            shape = clone_graph.node_list[output_node_id].shape
            # rate 1,3,6,9,12,15,18,21
            min_hw = min(shape[1], shape[2])
            rate_list = [1, 3, 6, 9, 12, 15, 18, 21]
            rate_list = [rate_list[i] for i in range(len(rate_list)) if rate_list[i] < min_hw]

            rate_h_index = random.randint(0, len(rate_list) - 1)
            rate_h = rate_list[rate_h_index]
            rate_w_index = random.randint(0, len(rate_list) - 1)
            rate_w = rate_list[rate_w_index]

            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.separable_conv2d(input_channel=shape[3],
                                                                                              filters=self.channels,
                                                                                              kernel_size_h=3,
                                                                                              kernel_size_w=3,
                                                                                              rate_h=rate_h,
                                                                                              rate_w=rate_w),
                                                   output_node_id)
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.bn2d(), output_node_id)
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.relu(), output_node_id)

            # encoder 3.step 3x3 atrous separable convolution
            graph_output_encoder[branch_index * branch_offset:(branch_index + 1) * branch_offset][
              self.branches + 1 + rate_h_index * len(rate_list) + rate_w_index] = 1.0
          else:
            # spatial pyramid pooling
            shape = clone_graph.node_list[output_node_id].shape
            min_hw = min(shape[1], shape[2])

            gh = [1, 2, 4, 8]
            gh = [n for n in gh if n < min_hw]
            grid_h_index = random.randint(0, len(gh) - 1)
            grid_h = gh[grid_h_index]

            gw = [1, 2, 4, 8]
            gw = [n for n in gw if n < min_hw]
            grid_w_index = random.randint(0, len(gw) - 1)
            grid_w = gw[grid_w_index]
            output_node_id = clone_graph.add_layer(clone_graph.layer_factory.spp(grid_h=grid_h, grid_w=grid_w),
                                                   output_node_id)

            # encoder 4.step spp
            graph_output_encoder[branch_index * branch_offset:(branch_index + 1) * branch_offset][
              self.branches + 1 + 8*8 + grid_h_index*4+grid_w_index] = 1.0

          X.append(output_node_id)

        output_node_id = clone_graph.add_layer(clone_graph.layer_factory.concat(), X[1:])
        decoder_output_last.append(output_node_id)

        graph_encoder.extend(graph_output_encoder.tolist())

      # check flops
      if clone_graph.flops > self.flops:
        try_count += 1
        continue

      # check structure is not been checked
      trials = Trial.filter(study_name=self.study.name)
      is_not_valid = False
      for t in trials:
        if str(t.structure_encoder) == str(graph_encoder):
          is_not_valid = True
          break

      if is_not_valid:
        try_count += 1
        continue

      proposed_search_space.append((np.array(graph_encoder), Encoder(skipkeys=True).encode(clone_graph)))
      if len(proposed_search_space) == count:
        break

    return proposed_search_space

  def get_new_suggestions(self, number=1, **kwargs):
    all_trials = []
    for _ in range(number):
      x = None
      x_obj = None
      if len(self.trials) > self.bo_min_samples:
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


if __name__ == '__main__':
  # default_graph = Graph()
  # default_graph.add_input(shape=(1, 14, 14, 32))
  # default_graph.add_input(shape=(1, 28, 28, 32))
  #
  # ss = Encoder(skipkeys=True).encode(default_graph)
  # with open('/Users/jian/Downloads/aa.json','w') as fp:
  #   fp.write(ss)

  with open('/Users/jian/Downloads/bb.json','r') as fp:
    content = fp.read()

  study_configuration = {'graph': content}
  #
  s = Study('hello', study_configuration=json.dumps(study_configuration), algorithm='dense')
  for _ in range(20):
    d = DPCSearchSpace(s, 10000000000)
    m = d.get_new_suggestions()
    m[0].objective_value = random.random()
    m[0].status = 'Completed'
    print(m)

    # m[0].status = 'Completed'
    # m[0].objective_value = random.random()
    #
    # graph = Decoder().decode(m[0].structure)
    # graph.layer_factory = LayerFactory()
    # pp = tf.placeholder(tf.float32, [1,14,14,32])
    # cc=graph.materialization(input_nodes=[pp])
    # print(cc)

  # for _ in range(100):
  #   d = DenseArchitectureSearchSpace(s)
  #   m = d.get_new_suggestions()
  #   print(m)
  #
  # print('hell')