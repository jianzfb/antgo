# -*- coding: UTF-8 -*-
# @Time    : 2018/11/22 2:11 PM
# @File    : graph.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.stublayer import *
from antgo.automl.layer_transformer import *
from antgo.automl.constant import *

try:
    import Queue as queue
except ImportError:
    import queue

from copy import deepcopy
from antgo.utils.serialize import *
from antgo.utils.netvis import *


class GraphNode(object):
  def __init__(self, shape, id=-1):
    self.shape = shape
    self.id = id


class NetworkDescriptor(object):
  CONCAT_CONNECT = 'concat'
  ADD_CONNECT = 'add'

  def __init__(self):
    self.skip_connections = []
    self.conv_widths = []
    self.dense_widths = []

  @property
  def n_dense(self):
    return len(self.dense_widths)

  @property
  def n_conv(self):
    return len(self.conv_widths)

  def add_conv_width(self, width):
    self.conv_widths.append(width)

  def add_dense_width(self, width):
    self.dense_widths.append(width)

  def add_skip_connection(self, u, v, connection_type):
    if connection_type not in [self.CONCAT_CONNECT, self.ADD_CONNECT]:
      raise ValueError('connection_type should be NetworkDescriptor.CONCAT_CONNECT '
                       'or NetworkDescriptor.ADD_CONNECT.')
    self.skip_connections.append((u, v, connection_type))

  def to_json(self):
    skip_list = []
    for u, v, connection_type in self.skip_connections:
      skip_list.append({'from': u, 'to': v, 'type': connection_type})
    return {'node_list': self.conv_widths, 'skip_list': skip_list}


class Graph(object):
  """A class representing the neural architecture graph of a Keras model.

  Graph extracts the neural architecture graph from a Keras model.
  Each node in the graph is a intermediate tensor between layers.
  Each layer is an edge in the graph.
  Notably, multiple edges may refer to the same layer.
  (e.g. Add layer is adding two tensor into one tensor. So it is related to two edges.)

  Attributes:
      weighted: A boolean of whether the weights and biases in the neural network
          should be included in the graph.
      input_shape: A tuple of integers, which does not include the batch axis.
      node_list: A list of integers. The indices of the list are the identifiers.
      layer_list: A list of stub layers. The indices of the list are the identifiers.
      node_to_id: A dict instance mapping from node integers to their identifiers.
      layer_to_id: A dict instance mapping from stub layers to their identifiers.
      layer_id_to_input_node_ids: A dict instance mapping from layer identifiers
          to their input nodes identifiers.
      adj_list: A two dimensional list. The adjacency list of the graph. The first dimension is
          identified by tensor identifiers. In each edge list, the elements are two-element tuples
          of (tensor identifier, layer identifier).
      reverse_adj_list: A reverse adjacent list in the same format as adj_list.
      operation_history: A list saving all the network morphism operations.
      vis: A dictionary of temporary storage for whether an local operation has been done
          during the network morphism.
  """

  def __init__(self, *args, **kwargs):
    self.weighted = kwargs.get('weighted', False)
    self.node_list = kwargs.get('node_list', [])            # record nodes
    self.layer_list = kwargs.get('layer_list', [])          # record layer (layer is between nodes)
    self.layer_id_to_input_node_ids = kwargs.get('layer_id_to_input_node_ids', {})      # record all input nodes id of layer

    if len(self.layer_id_to_input_node_ids) > 0:
      # replace key
      temp = {}
      for k,v in self.layer_id_to_input_node_ids.items():
        temp[int(k)] = v
      self.layer_id_to_input_node_ids = temp

    self.layer_id_to_output_node_ids = kwargs.get('layer_id_to_output_node_ids', {})    # record all output nodes id of layer
    if len(self.layer_id_to_output_node_ids) > 0:
      temp = {}
      for k,v in self.layer_id_to_output_node_ids.items():
        temp[int(k)] = v
      self.layer_id_to_output_node_ids = temp

    self.adj_list = kwargs.get('adj_list', {})                                          # node id -> child node id, layer
    if len(self.adj_list) > 0:
      temp = {}
      for k,v in self.adj_list.items():
        temp[int(k)] = v
      self.adj_list = temp

    self.reverse_adj_list = kwargs.get('reverse_adj_list', {})                          # node id -> parent node id, layer
    if len(self.reverse_adj_list) > 0:
      temp = {}
      for k,v in self.reverse_adj_list.items():
        temp[int(k)] = v
      self.reverse_adj_list = temp

    # node id start with 0
    self.node_to_id = {}                                    # node  -> id
    self.layer_to_id = {}                                   # layer -> id

    # configure node list
    if len(self.node_list) > 0:
      # use complete node list
      for n_i, n in enumerate(self.node_list):
        self.node_to_id[n] = n_i

    # configure layer list
    if len(self.layer_list) > 0:
      for l_i, l in enumerate(self.layer_list):
        self.layer_to_id[l] = l_i

        # reset input node
        input_nodes = l.get_input()
        if type(input_nodes) == list or type(input_nodes) == tuple:
          l.set_input([self.node_list[n.id] for n in input_nodes])
        else:
          l.set_input(self.node_list[input_nodes.id])

        # reset output node
        output_nodes = l.get_output()
        if type(output_nodes) == list or type(output_nodes) == tuple:
          l.set_output([self.node_list[n.id] for n in output_nodes])
        else:
          l.set_output(self.node_list[output_nodes.id])

    #####################################################
    self.operation_history = []
    self.vis = None
    self._layer_factory = None              # layer factory
    self.layer_transformer = None
    ######################################################

  @property
  def layer_factory(self):
    return self._layer_factory

  @layer_factory.setter
  def layer_factory(self, val):
    self._layer_factory = val
    self.layer_transformer = LayerTransfomer(val)

  def add_input(self, shape):
    self._add_node(GraphNode(shape=shape))

  def get_input(self):
    input_node_list = []
    for node in self.node_list:
      if len(self.reverse_adj_list[node.id]) == 0:
        input_node_list.append(node.id)

    return input_node_list

  def add_layer(self, layer, input_node_id):
    if isinstance(input_node_id, list):
      layer.input = list(map(lambda x: self.node_list[x], input_node_id))
      output_node_id = self._add_node(GraphNode(shape=layer.output_shape))
      for node_id in input_node_id:
        self._add_edge(layer, node_id, output_node_id)
    else:
      layer.input = self.node_list[input_node_id]
      output_node_id = self._add_node(GraphNode(shape=layer.output_shape))
      self._add_edge(layer, input_node_id, output_node_id)

    layer.output = self.node_list[output_node_id]
    return output_node_id

  def clear_operation_history(self):
    self.operation_history = []

  @property
  def n_nodes(self):
    """Return the number of nodes in the model."""
    return len(self.node_list)

  @property
  def n_layers(self):
    """Return the number of layers in the model."""
    return len(self.layer_list)

  @property
  def flops(self):
    total_flops = 0
    for l in self.layer_list:
      total_flops += l.flops()

    return total_flops

  def _add_node(self, node):
    """Add node to node list if it is not in node list."""
    node_id = len(self.node_list)
    self.node_to_id[node] = node_id
    node.id = node_id
    self.node_list.append(node)
    self.adj_list[node_id] = []
    self.reverse_adj_list[node_id] = []
    return node_id

  def _add_edge(self, layer, input_id, output_id):
    """Add an edge to the graph."""
    if layer in self.layer_to_id:
      layer_id = self.layer_to_id[layer]
      if input_id not in self.layer_id_to_input_node_ids[layer_id]:
        self.layer_id_to_input_node_ids[layer_id].append(input_id)
      if output_id not in self.layer_id_to_output_node_ids[layer_id]:
        self.layer_id_to_output_node_ids[layer_id].append(output_id)
    else:
      layer_id = len(self.layer_list)
      self.layer_list.append(layer)
      self.layer_to_id[layer] = layer_id
      self.layer_id_to_input_node_ids[layer_id] = [input_id]
      self.layer_id_to_output_node_ids[layer_id] = [output_id]

    self.adj_list[input_id].append((output_id, layer_id))
    self.reverse_adj_list[output_id].append((input_id, layer_id))

  def _redirect_edge(self, u_id, v_id, new_v_id):
    """Redirect the edge to a new node.
    Change the edge originally from `u_id` to `v_id` into an edge from `u_id` to `new_v_id`
    while keeping all other property of the edge the same.
    """
    layer_id = None
    for index, edge_tuple in enumerate(self.adj_list[u_id]):
      if edge_tuple[0] == v_id:
        layer_id = edge_tuple[1]
        self.adj_list[u_id][index] = (new_v_id, layer_id)
        break

    for index, edge_tuple in enumerate(self.reverse_adj_list[v_id]):
      if edge_tuple[0] == u_id:
        layer_id = edge_tuple[1]
        self.reverse_adj_list[v_id].remove(edge_tuple)
        break
    self.reverse_adj_list[new_v_id].append((u_id, layer_id))

    for index, value in enumerate(self.layer_id_to_output_node_ids[layer_id]):
      if value == v_id:
        self.layer_id_to_output_node_ids[layer_id][index] = new_v_id
        break

  def _replace_layer(self, layer_id, new_layer):
    """Replace the layer with a new layer."""
    old_layer = self.layer_list[layer_id]
    new_layer.input = old_layer.input
    new_layer.output = old_layer.output
    new_layer.output.shape = new_layer.output_shape
    self.layer_list[layer_id] = new_layer
    self.layer_to_id[new_layer] = layer_id
    self.layer_to_id.pop(old_layer)

  @property
  def topological_order(self):
    """Return the topological order of the node ids."""
    q = queue.Queue()
    in_degree = {}
    for node in self.node_list:
      i = node.id
      if i not in self.adj_list:
        continue
      in_degree[i] = 0

    for node in self.node_list:
      u = node.id
      if u not in self.adj_list:
        continue
      for v, _ in self.adj_list[u]:
        in_degree[v] += 1

    for node in self.node_list:
      i = node.id
      if i not in self.adj_list:
        continue
      if in_degree[i] == 0:
        q.put(i)

    order_list = []
    while not q.empty():
      u = q.get()
      order_list.append(u)
      for v, _ in self.adj_list[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
          q.put(v)
    return order_list

  def _get_pooling_layers(self, start_node_id, end_node_id):
    layer_list = []
    node_list = [start_node_id]
    self._depth_first_search(end_node_id, layer_list, node_list)
    return filter(lambda layer_id: self.layer_list[layer_id].layer_type == 'pool2d' or self.layer_list[layer_id] == 'global_pool2d', layer_list)

  def _depth_first_search(self, target_id, layer_id_list, node_list):
    u = node_list[-1]
    if u == target_id:
      return True

    for v, layer_id in self.adj_list[u]:
      layer_id_list.append(layer_id)
      node_list.append(v)
      if self._depth_first_search(target_id, layer_id_list, node_list):
        return True
      layer_id_list.pop()
      node_list.pop()

    return False

  def _search(self, u, start_dim, total_dim, n_add):
    """Search the graph for widening the layers.

    Args:
        u: The starting node identifier.
        start_dim: The position to insert the additional dimensions.
        total_dim: The total number of dimensions the layer has before widening.
        n_add: The number of dimensions to add.
    """
    if (u, start_dim, total_dim, n_add) in self.vis:
      return

    self.vis[(u, start_dim, total_dim, n_add)] = True
    for v, layer_id in self.adj_list[u]:
      layer = self.layer_list[layer_id]

      if layer.layer_type.startswith('conv'):
        new_layer = self.layer_transformer.wider_next_conv(layer, start_dim, total_dim, n_add, self.weighted)
        self._replace_layer(layer_id, new_layer)
      elif layer.layer_type == 'dense':
        new_layer = self.layer_transformer.wider_next_dense(layer, start_dim, total_dim, n_add, self.weighted)
        self._replace_layer(layer_id, new_layer)
      elif layer.layer_type.startswith('bn'):
        new_layer = self.layer_transformer.wider_bn(layer, start_dim, total_dim, n_add, self.weighted)
        self._replace_layer(layer_id, new_layer)
        self._search(v, start_dim, total_dim, n_add)
      elif layer.layer_type == 'concat':
        if self.layer_id_to_input_node_ids[layer_id][1] == u:
          # u is on the right of the concat
          # next_start_dim += next_total_dim - total_dim
          left_dim = self._upper_layer_width(self.layer_id_to_input_node_ids[layer_id][0])
          next_start_dim = start_dim + left_dim
          next_total_dim = total_dim + left_dim
        else:
          next_start_dim = start_dim
          next_total_dim = total_dim + self._upper_layer_width(self.layer_id_to_input_node_ids[layer_id][1])
        self._search(v, next_start_dim, next_total_dim, n_add)
      else:
        self._search(v, start_dim, total_dim, n_add)

    for v, layer_id in self.reverse_adj_list[u]:
      layer = self.layer_list[layer_id]
      if layer.layer_type.startswith('conv'):
        new_layer = self.layer_transformer.wider_pre_conv(layer, n_add, self.weighted)
        self._replace_layer(layer_id, new_layer)
      elif layer.layer_type == 'dense':
        new_layer = self.layer_transformer.wider_pre_dense(layer, n_add, self.weighted)
        self._replace_layer(layer_id, new_layer)
      elif layer.layer_type == 'concat':
        continue
      else:
        self._search(v, start_dim, total_dim, n_add)

  def _upper_layer_width(self, u):
    for v, layer_id in self.reverse_adj_list[u]:
      layer = self.layer_list[layer_id]
      if layer.layer_type.startswith('conv'):
        return layer.filters
      elif layer.layer_type == 'dense':
        return layer.units
      elif layer.layer_type == 'concat':
        a = self.layer_id_to_input_node_ids[layer_id][0]    # input_node 0 of current layer
        b = self.layer_id_to_input_node_ids[layer_id][1]    # input_node 1 of current layer
        return self._upper_layer_width(a) + self._upper_layer_width(b)  # input_node 0 + input_node 1 width
      else:
        return self._upper_layer_width(v)

    return self.node_list[0][-1]

  def to_conv_deeper_model(self, target_id, kernel_size):
    """Insert a relu-conv-bn block after the target block.

    Args:
        target_id: A convolutional layer ID. The new block should be inserted after the block.
        kernel_size: An integer. The kernel size of the new convolutional layer.
    """
    self.operation_history.append(('to_conv_deeper_model', target_id, kernel_size))
    target = self.layer_list[target_id]
    new_layers = self.layer_transformer.deeper_conv_block(target, kernel_size, self.weighted)
    output_id = self._conv_block_end_node(target_id)

    self._insert_new_layers(new_layers, output_id)

  def to_wider_model(self, pre_layer_id, n_add):
    """Widen the last dimension of the output of the pre_layer.

    Args:
        pre_layer_id: The ID of a convolutional layer or dense layer.
        n_add: The number of dimensions to add.
    """
    self.operation_history.append(('to_wider_model', pre_layer_id, n_add))
    pre_layer = self.layer_list[pre_layer_id]
    output_id = self.layer_id_to_output_node_ids[pre_layer_id][0]
    dim = None
    if pre_layer.layer_type == 'dense':
      dim = pre_layer.units
    elif pre_layer.layer_type.startswith('conv'):
      dim = pre_layer.filters

    self.vis = {}
    self._search(output_id, dim, dim, n_add)

    # update all children nodes shape
    for u in self.topological_order:
      for v, layer_id in self.adj_list[u]:
        self.node_list[v].shape = self.layer_list[layer_id].output_shape

  def to_dense_deeper_model(self, target_id):
    """Insert a dense layer after the target layer.

    Args:
        target_id: The ID of a dense layer.
    """
    self.operation_history.append(('to_dense_deeper_model', target_id))
    target = self.layer_list[target_id]
    new_layers = self.layer_transformer.dense_to_deeper_block(target, self.weighted)
    output_id = self._dense_block_end_node(target_id)

    self._insert_new_layers(new_layers, output_id)

  def _insert_new_layers(self, new_layers, output_id):
    new_node_id = self._add_node(deepcopy(self.node_list[self.adj_list[output_id][0][0]]))
    temp_output_id = new_node_id
    for layer in new_layers[:-1]:
      temp_output_id = self.add_layer(layer, temp_output_id)

    self._add_edge(new_layers[-1], temp_output_id, self.adj_list[output_id][0][0])
    new_layers[-1].input = self.node_list[temp_output_id]
    self.node_list[self.adj_list[output_id][0][0]].shape = new_layers[-1].output_shape
    new_layers[-1].output = self.node_list[self.adj_list[output_id][0][0]]
    self._redirect_edge(output_id, self.adj_list[output_id][0][0], new_node_id)

  def _block_end_node(self, layer_id, block_size):
    ret = self.layer_id_to_output_node_ids[layer_id][0]
    for i in range(block_size - 2):
      ret = self.adj_list[ret][0][0]
    return ret

  def _block_start_node(self, layer_id, block_size):
    ret = self.layer_id_to_input_node_ids[layer_id][0]
    for i in range(block_size - 2):
      ret = self.reverse_adj_list[ret][0][0]
    return ret

  def _dense_block_end_node(self, layer_id):
    return self.layer_id_to_output_node_ids[layer_id][0]

  def _conv_block_end_node(self, layer_id):
    """Get the output node ID of the last layer in the block by layer ID.
        Return the input node ID of the last layer in the convolutional block.

    Args:
        layer_id: the convolutional layer ID.
    """
    return self._block_end_node(layer_id, Constant.CONV_BLOCK_DISTANCE)

  def _conv_block_start_node(self, layer_id):
    """Get the input node ID of the last layer in the block by layer ID.
        Return the input node ID of the last layer in the convolutional block.

    Args:
        layer_id: the convolutional layer ID.
    """
    return self._block_start_node(layer_id, Constant.CONV_BLOCK_DISTANCE)

  def to_add_skip_model(self, start_id, end_id):
    """Add a weighted add skip-connection from after start node to end node.

    Args:
        start_id: The convolutional layer ID, after which to start the skip-connection.
        end_id: The convolutional layer ID, after which to end the skip-connection.
    """
    self.operation_history.append(('to_add_skip_model', start_id, end_id))
    conv_block_input_id = self._conv_block_end_node(start_id)
    block_last_layer_input_id = self._conv_block_end_node(end_id)

    skip_output_id = -1
    if self.node_list[conv_block_input_id].shape[1] > self.node_list[block_last_layer_input_id].shape[1] or \
            self.node_list[conv_block_input_id].shape[2] > self.node_list[block_last_layer_input_id].shape[2]:
      # 1.step 从高分辨率到低分辨率的跳跃(使用pooling)
      # Add the pooling layer chain.
      # layer_list = self._get_pooling_layers(conv_block_input_id, block_last_layer_input_id)
      # skip_output_id = conv_block_input_id
      # for index, layer_id in enumerate(layer_list):
      #   skip_output_id = self.add_layer(deepcopy(self.layer_list[layer_id]), skip_output_id)
      kh = self.node_list[conv_block_input_id].shape[1] / self.node_list[block_last_layer_input_id].shape[1]
      kw = self.node_list[conv_block_input_id].shape[2] / self.node_list[block_last_layer_input_id].shape[2]
      new_pool_layer = self.layer_factory.pool2d(kernel_size_h=kh, kernel_size_w=kw)
      skip_output_id = conv_block_input_id
      skip_output_id = self.add_layer(new_pool_layer, skip_output_id)
    elif self.node_list[conv_block_input_id].shape[1] < self.node_list[block_last_layer_input_id].shape[1] or \
            self.node_list[conv_block_input_id].shape[2] < self.node_list[block_last_layer_input_id].shape[2]:
      # 2.step 从低分辨率到高分辨率的跳跃(使用双线性插值)
      height = self.node_list[block_last_layer_input_id].shape[1]
      width = self.node_list[block_last_layer_input_id].shape[2]
      new_resize_layer = self.layer_factory.bilinear_resize(height=height, width=width)
      skip_output_id = conv_block_input_id
      skip_output_id = self.add_layer(new_resize_layer, skip_output_id)
    else:
      skip_output_id = conv_block_input_id

    aa = self._conv_block_start_node(end_id)

    concat_input_node_id = self._add_node(deepcopy(self.node_list[block_last_layer_input_id]))
    self._redirect_edge(aa, block_last_layer_input_id, concat_input_node_id)

    # add concatenate layer
    concat_layer = self.layer_factory.add()
    concat_layer.input = [self.node_list[concat_input_node_id], self.node_list[skip_output_id]]

    self._add_edge(concat_layer, concat_input_node_id, block_last_layer_input_id)
    self._add_edge(concat_layer, skip_output_id, block_last_layer_input_id)
    concat_layer.output = self.node_list[block_last_layer_input_id]
    self.node_list[block_last_layer_input_id].shape = concat_layer.output_shape


    # # Add the conv layer (relu + conv + bn)
    # new_relu_layer = self.layer_factory.relu()
    # skip_output_id = self.add_layer(new_relu_layer, skip_output_id)
    # new_conv_layer = self.layer_factory.conv2d(self.layer_list[start_id].filters, self.layer_list[end_id].filters, kernel_size=1)
    # skip_output_id = self.add_layer(new_conv_layer, skip_output_id)
    # new_bn_layer = self.layer_factory.bn2d(self.layer_list[end_id].filters)
    # skip_output_id = self.add_layer(new_bn_layer, skip_output_id)
    #
    # # Add the add layer.
    # block_last_layer_output_id = self.adj_list[block_last_layer_input_id][0][0]
    # add_input_node_id = self._add_node(deepcopy(self.node_list[block_last_layer_output_id]))
    # add_layer = self.layer_factory.add()
    #
    # self._redirect_edge(block_last_layer_input_id, block_last_layer_output_id, add_input_node_id)
    # self._add_edge(add_layer, add_input_node_id, block_last_layer_output_id)
    # self._add_edge(add_layer, skip_output_id, block_last_layer_output_id)
    # add_layer.input = [self.node_list[add_input_node_id], self.node_list[skip_output_id]]
    # add_layer.output = self.node_list[block_last_layer_output_id]
    # self.node_list[block_last_layer_output_id].shape = add_layer.output_shape

  def to_concat_skip_model(self, start_id, end_id):
    """Add a weighted add concatenate connection from after start node to end node.

    Args:
        start_id: The convolutional layer ID, after which to start the skip-connection.
        end_id: The convolutional layer ID, after which to end the skip-connection.
    """
    self.operation_history.append(('to_concat_skip_model', start_id, end_id))
    conv_block_input_id = self._conv_block_end_node(start_id)
    block_last_layer_input_id = self._conv_block_end_node(end_id)

    skip_output_id = -1
    if self.node_list[conv_block_input_id].shape[1] > self.node_list[block_last_layer_input_id].shape[1] or \
            self.node_list[conv_block_input_id].shape[2] > self.node_list[block_last_layer_input_id].shape[2]:
      # 1.step 从高分辨率到低分辨率的跳跃(使用pooling)
      # Add the pooling layer chain.
      # pooling_layer_list = self._get_pooling_layers(conv_block_input_id, block_last_layer_input_id)
      # skip_output_id = conv_block_input_id
      # for index, layer_id in enumerate(pooling_layer_list):
      #   skip_output_id = self.add_layer(deepcopy(self.layer_list[layer_id]), skip_output_id)
      #

      kh = self.node_list[conv_block_input_id].shape[1] / self.node_list[block_last_layer_input_id].shape[1]
      kw = self.node_list[conv_block_input_id].shape[2] / self.node_list[block_last_layer_input_id].shape[2]
      new_pool_layer = self.layer_factory.pool2d(kernel_size_h=kh, kernel_size_w=kw)
      skip_output_id = conv_block_input_id
      skip_output_id = self.add_layer(new_pool_layer, skip_output_id)
    elif self.node_list[conv_block_input_id].shape[1] < self.node_list[block_last_layer_input_id].shape[1] or \
            self.node_list[conv_block_input_id].shape[2] < self.node_list[block_last_layer_input_id].shape[2]:
      # 2.step 从低分辨率到高分辨率的跳跃(使用双线性插值)
      height = self.node_list[block_last_layer_input_id].shape[1]
      width = self.node_list[block_last_layer_input_id].shape[2]
      new_resize_layer = self.layer_factory.bilinear_resize(height=height, width=width)
      skip_output_id = conv_block_input_id
      skip_output_id = self.add_layer(new_resize_layer, skip_output_id)
    else:
      # 3.step 相同分辨率
      skip_output_id = conv_block_input_id

    aa = self._conv_block_start_node(end_id)

    concat_input_node_id = self._add_node(deepcopy(self.node_list[block_last_layer_input_id]))
    self._redirect_edge(aa, block_last_layer_input_id, concat_input_node_id)

    # add concatenate layer
    concat_layer = self.layer_factory.concat()
    concat_layer.input = [self.node_list[concat_input_node_id], self.node_list[skip_output_id]]

    self._add_edge(concat_layer, concat_input_node_id, block_last_layer_input_id)
    self._add_edge(concat_layer, skip_output_id, block_last_layer_input_id)
    concat_layer.output = self.node_list[block_last_layer_input_id]
    self.node_list[block_last_layer_input_id].shape = concat_layer.output_shape

  def to_remove_skip_model(self, start_id, end_id):
    start_layer_output_node_id = self._conv_block_end_node(start_id)
    end_layer_output_node_id = self._conv_block_end_node(end_id)

    has_skip = False
    skip_layer = None
    left_node = None
    right_node = None
    combine_node = None
    for output_node, left_layer in self.adj_list[end_layer_output_node_id]:
      for input_node, right_layer in self.reverse_adj_list[output_node]:
        if input_node == start_layer_output_node_id and left_layer == right_layer:
          has_skip = True
          skip_layer = left_layer
          left_node = start_layer_output_node_id
          right_node = end_layer_output_node_id
          combine_node = output_node
          break
        elif self.reverse_adj_list[input_node][0][0] == start_layer_output_node_id and \
                self.layer_list[self.reverse_adj_list[input_node][0][1]].layer_type in ['resize', 'pool2d'] and \
                left_layer == right_layer:
          has_skip = True
          skip_layer = left_layer
          left_node = start_layer_output_node_id
          right_node = end_layer_output_node_id
          combine_node = output_node

          self.to_remove_layer(self.reverse_adj_list[input_node][0][1])
          break

    if not has_skip:
      return

    self.to_remove_layer(skip_layer)
    self.update()

  def to_remove_layer(self, old_layer_id):
    start_node_list = []
    middle_node = None
    keep_node = None
    keey_layer = None

    for layer_id, input_nodes in self.layer_id_to_input_node_ids.items():
      if layer_id == old_layer_id:
        for node_id in input_nodes:
          for output_node, linked_layer in self.adj_list[node_id]:
            if linked_layer == old_layer_id:
              middle_node = output_node
              start_node_list.append(node_id)

              if len(self.adj_list[output_node]) > 0:
                keep_node, keey_layer = self.adj_list[output_node][0]

    if middle_node is None:
      return

    if keep_node is not None and keey_layer is not None:
      nearest_input_node = None
      nearest_length = 10000000

      for input_node in start_node_list:
        layer_list = []
        node_list = [input_node]
        self._depth_first_search(keep_node, layer_list, node_list)

        if nearest_length > len(node_list):
          nearest_length = len(node_list)
          nearest_input_node = input_node

      for input_node in start_node_list:
        if input_node != nearest_input_node:
          remove_index = -1
          for i, kv in enumerate(self.adj_list[input_node]):
            if kv[1] == old_layer_id:
              remove_index = i
              break

          if remove_index != -1:
            self.adj_list[input_node].pop(remove_index)

      for i, kv in enumerate(self.adj_list[nearest_input_node]):
        if kv[1] == old_layer_id:
          self.adj_list[nearest_input_node][i] = (keep_node, keey_layer)
          break

      for i, kv in enumerate(self.reverse_adj_list[keep_node]):
        if kv[0] == middle_node:
          self.reverse_adj_list[keep_node][i] = (nearest_input_node, keey_layer)
          break

      self.adj_list.pop(middle_node)
      self.reverse_adj_list.pop(middle_node)
      self.layer_to_id.pop(self.layer_list[old_layer_id])

      self.layer_id_to_input_node_ids.pop(old_layer_id)
      self.layer_id_to_output_node_ids.pop(old_layer_id)
      self.layer_list[keey_layer].input = self.node_list[nearest_input_node]
      for mi, m in enumerate(self.layer_id_to_input_node_ids[keey_layer]):
        if m == middle_node:
          self.layer_id_to_input_node_ids[keey_layer][mi] = nearest_input_node
          break
    else:
      for input_node in start_node_list:
        remove_index = -1
        for i, kv in enumerate(self.adj_list[input_node]):
          if kv[1] == old_layer_id:
            remove_index = i
            break

        if remove_index != -1:
          self.adj_list[input_node].pop(remove_index)

      self.adj_list.pop(middle_node)
      self.reverse_adj_list.pop(middle_node)
      self.layer_to_id.pop(self.layer_list[old_layer_id])

      self.layer_id_to_input_node_ids.pop(old_layer_id)
      self.layer_id_to_output_node_ids.pop(old_layer_id)
      # self.layer_list[keey_layer].input = self.node_list[start_node_list[0]] if len(start_node_list) == 1 else [self.node_list[start_node_list[i]] for i in start_node_list]


  def to_replace_layer(self, old_layer_id, layer):
    self._replace_layer(old_layer_id, layer)
    self.update()

  def to_insert_layer(self, layer_id, layer):
    output_id = self.layer_id_to_input_node_ids[layer_id][0]
    self._insert_new_layers([layer], output_id)
    self.update()

  def has_skip(self, start_id, end_id):
    conv_block_input_id = self._conv_block_end_node(start_id)
    # conv_block_input_id = self.adj_list[conv_block_input_id][0][0]
    block_last_layer_input_id = self._conv_block_end_node(end_id)

    existed_skip = False
    for linked_node, linked_layer in self.adj_list[conv_block_input_id]:
      if self.layer_list[linked_layer].layer_name == 'bilinear_resize':
        linked_node = self.adj_list[linked_node][0][0]

      for upper_node, skip_layer in self.reverse_adj_list[linked_node]:
        if upper_node == block_last_layer_input_id and self.layer_list[skip_layer].layer_name in ['add', 'concat']:
          existed_skip = True
          break

    return existed_skip

  def extract_descriptor(self):
    ret = NetworkDescriptor()
    topological_node_list = self.topological_order
    for u in topological_node_list:
      for v, layer_id in self.adj_list[u]:
        layer = self.layer_list[layer_id]
        if layer.layer_type.startswith('conv'):
          ret.add_conv_width(layer.layer_width)

        if layer.layer_type == 'dense':
          ret.add_dense_width(layer.layer_width)

    # The position of each node, how many Conv and Dense layers before it.
    pos = [0] * len(topological_node_list)
    for v in topological_node_list:
      layer_count = 0
      for u, layer_id in self.reverse_adj_list[v]:
        layer = self.layer_list[layer_id]
        weighted = 0
        if layer.layer_type.startswith('conv') or layer.layer_type == 'dense':
          weighted = 1
        layer_count = max(pos[u] + weighted, layer_count)
      pos[v] = layer_count

    for u in topological_node_list:
      for v, layer_id in self.adj_list[u]:
        if pos[u] == pos[v]:
          continue
        layer = self.layer_list[layer_id]
        if layer.layer_type == 'concat':
          ret.add_skip_connection(pos[u], pos[v], NetworkDescriptor.CONCAT_CONNECT)
        if layer.layer_type == 'add':
          ret.add_skip_connection(pos[u], pos[v], NetworkDescriptor.ADD_CONNECT)

    return ret

  def _layer_ids_in_order(self, layer_ids):
    node_id_to_order_index = {}
    for index, node_id in enumerate(self.topological_order):
      node_id_to_order_index[node_id] = index

    return sorted(layer_ids,
                  key=lambda layer_id:
                  node_id_to_order_index[self.layer_id_to_output_node_ids[layer_id][0]])

  def _layer_ids_by_type(self, type_str):
    return list(filter(lambda layer_id: self.layer_list[layer_id].layer_type.startswith(type_str), range(self.n_layers)))

  def _conv_layer_ids_in_order(self):
    return self._layer_ids_in_order(
      list(filter(lambda layer_id: self.layer_list[layer_id].kernel_size != 1,
                  self._layer_ids_by_type('conv'))))

  def _dense_layer_ids_in_order(self):
    return self._layer_ids_in_order(self._layer_ids_by_type('dense'))

  def deep_layer_ids(self, skip_last=True):
    if skip_last:
      return self._conv_layer_ids_in_order() + self._dense_layer_ids_in_order()[:-1]
    else:
      return self._conv_layer_ids_in_order() + self._dense_layer_ids_in_order()

  def wide_layer_ids(self, skip_last=True):
    if skip_last:
      return self._conv_layer_ids_in_order()[:-1] + self._dense_layer_ids_in_order()[:-1]
    else:
      return self._conv_layer_ids_in_order() + self._dense_layer_ids_in_order()

  def skip_connection_layer_ids(self):
    return self._conv_layer_ids_in_order()[:-1]

  def size(self):
    return sum(list(map(lambda x: x.size(), self.layer_list)))

  def materialization(self, input_nodes=None, output_nodes=None, layer_factory=None, batch_size=1):
    self.layers = []

    # Input
    topo_node_list = self.topological_order
    node_list = deepcopy(self.node_list)
    if input_nodes is None:
      input_id = topo_node_list[0]
      input_shape = self.node_list[input_id].shape
      input_shape[0] = batch_size
      if layer_factory is None:
        input_tensor = self.layer_factory.input(shape=input_shape)()
        node_list[input_id] = input_tensor
      else:
        input_tensor = layer_factory.input(shape=input_shape)()
        node_list[input_id] = input_tensor
    elif type(input_nodes[0]) == int:
      for n_i in input_nodes:
        input_shape = self.node_list[n_i].shape
        input_shape[0] = batch_size

        if layer_factory is None:
          input_tensor = self.layer_factory.input(shape=input_shape)()
          node_list[n_i] = input_tensor
        else:
          input_tensor = layer_factory.input(shape=input_shape)()
          node_list[n_i] = input_tensor
    else:
      for n_i, n_tensor in enumerate(input_nodes):
        node_list[n_i] = n_tensor

    # Output
    for v in topo_node_list:
      for u, layer_id in self.reverse_adj_list[v]:
        layer = self.layer_list[layer_id]

        if layer.layer_type == 'add' or layer.layer_type == 'concat':
          edge_input_tensor = list(map(lambda x: node_list[x],
                                       self.layer_id_to_input_node_ids[layer_id]))
        else:
          edge_input_tensor = node_list[u]

        layer.layer_factory = layer_factory if layer_factory is not None else self.layer_factory
        node_list[v] = layer(edge_input_tensor)

        # if getattr(layer_factory, layer.layer_name) != None:
        #   if layer_factory is None:
        #     node_list[v] = getattr(self.layer_factory, layer.layer_name)(layer)(edge_input_tensor)
        #   else:
        #     node_list[v] = getattr(layer_factory, layer.layer_name)(layer)(edge_input_tensor)
        # else:
        #   layer.layer_factory = layer_factory if layer_factory is not None else self.layer_factory
        #   node_list[v] = layer(edge_input_tensor)

    output_tensors = []
    if output_nodes is None:
      output_id = topo_node_list[-1]
      output_tensors.append(node_list[output_id])
    elif type(output_nodes[0]) == int:
      for n_i in output_nodes:
        output_tensors.append(node_list[n_i])

    return output_tensors

  def visualization(self, target_path):
    gv = GraphVis(name='computing graph')
    node_to_blocks = {}
    node_to_cells = {}
    for node in self.node_list:
      if node.id in self.adj_list:
        for output_node, layer_id in self.adj_list[node.id]:
          node_to_blocks[output_node] = self.layer_list[layer_id].block_name
          node_to_cells[output_node] = self.layer_list[layer_id].cell_name

    for node in self.node_list:
      tag = {}
      if node.id in node_to_blocks:
        tag.update({'block': node_to_blocks[node.id]})
      if node.id in node_to_cells:
        tag.update({'cell': node_to_cells[node.id]})

      gv.add_node(node.id, '%s-(%d,%d,%d)'%(str(node.id), node.shape[1], node.shape[2], node.shape[3]), '', tag=tag)

    for k,v in self.adj_list.items():
      for output_node, layer_id in v:
        gv.add_link(k, output_node, Link(label=self.layer_list[layer_id].nickname if self.layer_list[layer_id].nickname != '' else self.layer_list[layer_id].layer_name, args=''))

    graphviz_net_svg(gv, target_path)

  def update(self):
    # udpate graph information
    for node_id in self.topological_order:
      for output_id, layer_id in self.adj_list[node_id]:
        if self.layer_list[layer_id].layer_name == 'add':
          # 检查 add 所对应的所有输入节点是否通道数一致
          is_consistent = True
          max_channels = self.node_list[self.layer_id_to_input_node_ids[layer_id][0]].shape[-1]
          for layer_input_id in self.layer_id_to_input_node_ids[layer_id][1:]:
            if self.node_list[layer_input_id].shape[-1] != max_channels:
              is_consistent = False

            if self.node_list[layer_input_id].shape[-1] > max_channels:
              max_channels = self.node_list[layer_input_id].shape[-1]

          if not is_consistent:
            for layer_input_id in self.layer_id_to_input_node_ids[layer_id]:
              if self.node_list[layer_input_id].shape[-1] != max_channels and \
                      len(self.reverse_adj_list[layer_input_id]) == 1 and \
                      self.layer_list[self.reverse_adj_list[layer_input_id][0][-1]].layer_name == 'conv2d':
                self.layer_list[self.reverse_adj_list[layer_input_id][0][-1]].filters = max_channels
                self.node_list[layer_input_id].shape = \
                  self.layer_list[self.reverse_adj_list[layer_input_id][0][-1]].output_shape

        self.node_list[output_id].shape = self.layer_list[layer_id].output_shape

#from random import randrange, sample
#mport random
#import yaml

if __name__ == '__main__':
  default_graph = Graph()
  default_graph.add_input(shape=(1, 14, 14, 32))

  ss = Encoder(skipkeys=True).encode(default_graph)
  with open('/Users/jian/Downloads/aa.json','w') as fp:
    fp.write(ss)

  pass
  # # 1.step conv + bn + relu + conv + bn + relu +
  # graph = Graph()
  # graph.layer_factory = LayerFactory()
  # graph.add_input(shape=(1, 14, 14, 32))
  # graph.add_input(shape=(1, 28, 28, 64))
  # graph.add_input(shape=(1, 56, 56, 128))
  #
  # outputs = graph.get_input()
  #
  # output_node_id = -1
  # decoder_output_last = []
  # for output_index, output_id in enumerate(outputs):
  #   output = graph.node_list[output_id]
  #
  #   temp = [output_id]
  #   for node_id in decoder_output_last:
  #     if graph.node_list[node_id].shape[1] != output.shape[1] or\
  #             graph.node_list[node_id].shape[2] != output.shape[2]:
  #       output_node_id = graph.add_layer(graph.layer_factory.bilinear_resize(height=output.shape[1],
  #                                                                            width=output.shape[2]), node_id)
  #       temp.append(output_node_id)
  #     else:
  #       temp.append(output_node_id)
  #
  #   if len(temp) > 1:
  #     output_node_id = graph.add_layer(graph.layer_factory.concat(), temp)
  #     X = [output_node_id]
  #   else:
  #     output_node_id = temp[0]
  #     X = temp
  #
  #   for branch_index in range(3):
  #     # random select branch input from X
  #     X_index_list = list(range(len(X)))
  #     X_select_index_list = sample(X_index_list, random.randint(1, len(X_index_list)))
  #     X_selected = [X[i] for i in X_select_index_list]
  #
  #     # concat all input
  #     if len(X_selected) > 1:
  #       output_node_id = graph.add_layer(graph.layer_factory.concat(X_selected), X_selected)
  #
  #     # operator space
  #     r = random.randint(0, 2)
  #     if r == 0:
  #       # 1x1 convolution
  #       shape = graph.node_list[output_node_id].shape
  #       output_node_id = graph.add_layer(graph.layer_factory.conv2d(input_channel=shape[3],
  #                                                                   filters=shape[3],
  #                                                                   kernel_size_h=1,
  #                                                                   kernel_size_w=1),
  #                                        output_node_id)
  #       output_node_id = graph.add_layer(graph.layer_factory.relu(),
  #                                        output_node_id)
  #       output_node_id = graph.add_layer(graph.layer_factory.bn2d(),
  #                                        output_node_id)
  #     elif r == 1:
  #       # 3x3 atrous separable convolution
  #       shape = graph.node_list[output_node_id].shape
  #       # rate 1,3,6,9,12,15,18,21
  #       min_hw = min(shape[1],shape[2])
  #       rate_list = [1,3,6,9,12,15,18,21]
  #       rate_list = [rate_list[i] for i in range(len(rate_list)) if rate_list[i] < min_hw]
  #
  #       rate_h = rate_list[random.randint(0,len(rate_list)-1)]
  #       rate_w = rate_list[random.randint(0,len(rate_list)-1)]
  #
  #       output_node_id = graph.add_layer(graph.layer_factory.separable_conv2d(input_channel=shape[3],
  #                                                                             filters=shape[3],
  #                                                                             kernel_size_h=3,
  #                                                                             kernel_size_w=3,
  #                                                                             rate_h=rate_h,
  #                                                                             rate_w=rate_w),
  #                                        output_node_id)
  #       output_node_id = graph.add_layer(graph.layer_factory.relu(),
  #                                        output_node_id)
  #       output_node_id = graph.add_layer(graph.layer_factory.bn2d(),
  #                                        output_node_id)
  #     else:
  #       # spatial pyramid pooling
  #       shape = graph.node_list[output_node_id].shape
  #       min_hw = min(shape[1], shape[2])
  #
  #       gh = [1,2,4,8]
  #       gh = [n for n in gh if n < min_hw]
  #       grid_h = gh[random.randint(0, len(gh) - 1)]
  #
  #       gw = [1,2,4,8]
  #       gw = [n for n in gw if n < min_hw]
  #       grid_w = gw[random.randint(0,len(gw) - 1)]
  #       output_node_id = graph.add_layer(graph.layer_factory.spp(grid_h=grid_h, grid_w=grid_w), output_node_id)
  #
  #     print(graph.node_list[output_node_id].shape)
  #     X.append(output_node_id)
  #
  #   output_node_id = graph.add_layer(graph.layer_factory.concat(), X[1:])
  #   decoder_output_last.append(output_node_id)
  #
  # # print(auto_graph.extract_descriptor())
  # # ss = auto_graph.materialization(input_nodes=[0, 1, 2])
  #
  # # print(ss)
  #
  # ss = Encoder(skipkeys=True).encode(graph)
  # # with open('/Users/jian/Downloads/xx.txt', 'w') as fp:
  # #   fp.write(ss)
  #
  # bb = Decoder().decode(ss)
  # bb.layer_factory = LayerFactory()
  # bb.materialization()
  # print(bb)


  # output_node_id = 0
  # output_node_id = auto_graph.add_layer(auto_graph.layer_factory.conv2d(3, 64, kernel_size=3), output_node_id)
  # output_node_id = auto_graph.add_layer(auto_graph.layer_factory.bn2d(64), output_node_id)
  # output_node_id = auto_graph.add_layer(auto_graph.layer_factory.relu(), output_node_id)
  #
  # ss = Encoder(skipkeys=True).encode(auto_graph)
  # with open('/Users/jian/Downloads/xx.txt', 'w') as fp:
  #   fp.write(ss)



  # left_id = auto_graph.add_layer(auto_graph.layer_factory.conv2d(16,32,kernel_size=3), output_node_id)
  # right_id = auto_graph.add_layer(auto_graph.layer_factory.conv2d(16,32,kernel_size=3), output_node_id)
  #
  # concat_id = auto_graph.add_layer(auto_graph.layer_factory.concat(), [left_id, right_id])
  # conv_id = auto_graph.add_layer(auto_graph.layer_factory.conv2d(32,64,kernel_size=3), concat_id)
  # bn_id = auto_graph.add_layer(auto_graph.layer_factory.bn2d(64), conv_id)
  # relu_id = auto_graph.add_layer(auto_graph.layer_factory.relu(), bn_id)
  # softmax_id = auto_graph.add_layer(auto_graph.layer_factory.softmax(), relu_id)
  #
  # ss=auto_graph.materialization()
  # print(ss)



  # ss = Encoder(skipkeys=True).encode(auto_graph)
  # print(ss)
  #
  # bb = Decoder().decode(ss)
  # bb.layer_factory = LayerFactory()
  # bb.materialization()

  # print(bb)


  # aa = AA(a='1',b='2',bb='dd',cc={BB(c='cc',d='dd'):'a'})
  # ss = Encoder().encode(aa)
  # # print(ss)
  #
  # bb = Decoder().decode(ss)
  # print(bb)
  # print(bb.bb)
  # print(bb.cc)

  # # (0,1,2,3,4,5,6,7,8,9,10)
  # print(auto_graph.topological_order)

  # # 2.step deeper at 0 layer (测试通过)
  # auto_graph.to_conv_deeper_model(0,3)
  # print(auto_graph.topological_order)

  # # 3.step depper at 3 layer
  # auto_graph.to_conv_deeper_model(3,3)
  # print(auto_graph.topological_order)

  # 4.step widen at 4
  # auto_graph.to_wider_model(6, 15)
  # print(auto_graph.topological_order)

  # 5.step test to skip
  # auto_graph.to_add_skip_model(0, 6)
  # print(auto_graph.topological_order)

  # 6.step test to concat
  # auto_graph.to_concat_skip_model(0,6)
  # print(auto_graph.topological_order)
