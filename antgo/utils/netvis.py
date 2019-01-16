"""Tools for visualizing a lusmu graph

Copyright 2013 Eniram Ltd. See the LICENSE file at the top-level directory of
this distribution and at https://github.com/akaihola/lusmu/blob/master/LICENSE

"""

# pylint: disable=W0212
#         Allow access to protected members of client classes
# pylint: disable=W0142
#         Allow * and ** magic

from __future__ import print_function, unicode_literals

import re
import subprocess
from textwrap import dedent
import collections
from antgo.utils.serialize import *


class Link(collections.namedtuple('Link', ['label', 'args'])):
  """
  """


class NodeVis(object):
  def __init__(self, **kwargs):
    self.name = kwargs['name']                                      # name
    self.op_name = kwargs['op_name'].replace('/', '<br />')         # op name
    self.link_nodes = [] if 'link_nodes' not in kwargs else kwargs['link_nodes']
    self.label = '' if 'label' not in kwargs else kwargs['label']
    self.id = kwargs['id']
    self._in_degree = 0 if '_in_degree' not in kwargs else kwargs['_in_degree']
    self._out_degree = 0 if '_out_degree' not in kwargs else kwargs['_out_degree']
    self._attr = {} if '_attr' not in kwargs else kwargs['_attr']
    self.tag = {} if 'tag' not in kwargs else kwargs['tag']

  def add_link(self, node, link):
    for ln in self.link_nodes:
      if node == ln['node']:
        return False
    
    self.link_nodes.append({'node': node, 'link': link})
    return True

  @property
  def linked_node(self):
    for a in self.link_nodes:
      yield a

  @property
  def in_degree(self):
    return self._in_degree

  def in_degree_increment(self):
    self._in_degree += 1

  @property
  def out_degree(self):
    return self._out_degree

  def out_degree_increment(self):
    self._out_degree += 1


class GraphVis(object):
  def __init__(self, **kwargs):
    self._name = kwargs['name'] if 'name' in kwargs else kwargs['_name']
    self._graph_nodes = {} if '_graph_nodes' not in kwargs else kwargs['_graph_nodes']
    self._graph_nodes_indegree = {} if '_graph_nodes_indegree' not in kwargs else kwargs['_graph_nodes_indegree']
    self._graph_nodes_outdegree = {} if '_graph_nodes_outdegree' not in kwargs else kwargs['_graph_nodes_outdegree']
    self._nodes_num = 0 if '_nodes_num' not in kwargs else kwargs['_nodes_num']

  def add_node(self, name, op_name, label, tag=''):
    if name not in self._graph_nodes:
      self._graph_nodes[name] = NodeVis(name=name, op_name=op_name, id=self._nodes_num, label=label, tag=tag)
      self._graph_nodes_indegree[name] = 0
      self._graph_nodes_outdegree[name] = 0
      self._nodes_num += 1

  def add_link(self, from_node, to_node, link):
    try:
      assert(from_node in self._graph_nodes)
      assert(to_node in self._graph_nodes)
      
      flag = self._graph_nodes[from_node].add_link(self._graph_nodes[to_node], link)
      if flag:
        # add new link successfully
        self._graph_nodes[from_node].out_degree_increment()
        self._graph_nodes[to_node].in_degree_increment()
    except:
      print(from_node)
      print(to_node)

  @property
  def nodes(self):
    for k,v in self._graph_nodes.items():
      yield v

  def node(self, name):
    return self._graph_nodes[name]


def node_format(node_id, node):
  # shape = 'oval' if isinstance(node, Input) else 'box'
  shape='circle'
  # action = ('{br}{br}<FONT COLOR="#888888">{action}</FONT>'
  #           .format(br=' <BR ALIGN="LEFT"/>',
  #                   action=get_action_name(node._action))
  #           if isinstance(node, Node)
  #           else '')
  action = ''
  if node.in_degree == 0:
    yield ('  {node_id} '
           '[label=<<B>{name}</B>'
           '{action}>'
           ' shape={shape} style=filled fillcolor=forestgreen fontcolor=white penwidth=1];'
           .format(node_id=node_id,
                   name=node.op_name.replace(':', ':<BR ALIGN="CENTER"/>'),
                   action=action,
                   shape=shape))
  elif node.out_degree == 0:
    yield ('  {node_id} '
           '[label=<<B>{name}</B>'
           '{action}>'
           ' shape={shape} style=filled fillcolor=mediumpurple fontcolor=white penwidth=1];'
           .format(node_id=node_id,
                   name=node.op_name.replace(':', ':<BR ALIGN="CENTER"/>'),
                   action=action,
                   shape=shape))
  else:
    yield ('  {node_id} '
           '[label=<<B>{name}</B>'
           '{action}>'
           ' shape={shape} style=filled fillcolor=goldenrod fontcolor=white penwidth=1];'
           .format(node_id=node_id,
                   name=node.op_name.replace(':', ':<BR ALIGN="CENTER"/>'),
                   action=action,
                   shape=shape))
  yield '  edge [color=darkgrey ];'


def graphviz_net(graph, format_node, who_is_input):
  """Generate source lines for a Graphviz graph definition"""
  all_nodes = sorted(graph.nodes, key=id)
  if who_is_input is not None and len(who_is_input) > 0:
    input_nodes = [n for n in all_nodes if n.name in who_is_input]
  else:
    input_nodes = [n for n in all_nodes if n.in_degree==0 and n.out_degree > 0]

  yield 'digraph gr {'
  yield '  graph [ dpi = 12 ];'
  yield '  rankdir = TB;'

  yield '  { rank = source;'
  for node in input_nodes:
    yield '    n{};'.format(node.id)
  yield '  }'

  block_names = {'regular': []}
  cell_names = {'regular': []}
  block_cell_map = {}
  for node in all_nodes:
    if 'block' in node.tag and node.tag['block'] != '':
      if node.tag['block'] not in block_names:
        block_names[node.tag['block']] = []

      block_names[node.tag['block']].append(node)

      if 'cell' in node.tag and node.tag['cell'] != '':
        if node.tag['cell'] not in cell_names:
          cell_names[node.tag['cell']] = []
        cell_names[node.tag['cell']].append(node)

        if node.tag['block'] not in block_cell_map:
          block_cell_map[node.tag['block']] = []

        if node.tag['cell'] not in block_cell_map[node.tag['block']]:
          block_cell_map[node.tag['block']].append(node.tag['cell'])
      else:
        cell_names['regular'].append(node)
    else:
      block_names['regular'].append(node)

  for k,v in block_names.items():
    if k == 'regular':
      for node in v:
        if node.in_degree == 0 and node.out_degree == 0:
          continue

        # support block tag
        for line in format_node('n{}'.format(node.id), node):
          yield line
    else:
      yield ' subgraph cluster_%s{'%k
      yield ' label="%s";'%k
      yield ' bgcolor="mintcream";'

      if len(block_cell_map) == 0:
        for node in v:
          if node.in_degree == 0 and node.out_degree == 0:
            continue

          for line in format_node('n{}'.format(node.id), node):
            yield line
      else:
        for cell_name in block_cell_map[k]:
          yield ' subgraph cluster_%s{' % cell_name
          yield ' label="%s";' % cell_name
          yield ' bgcolor="cornflowerblue";'
          for node in cell_names[cell_name]:
            if node.in_degree == 0 and node.out_degree == 0:
              continue

            for line in format_node('n{}'.format(node.id), node):
              yield line
          yield ' }'
        pass
      yield ' }'

  for node in all_nodes:
    # if node.in_degree == 0 and node.out_degree == 0:
    #   continue
    #
    # # support block tag
    # for line in format_node('n{}'.format(node.id), node):
    #   yield line

    for other in node.linked_node:
      if other['link'][0] != '':
        yield ('  n{node} -> n{other}[label={label},fontcolor=darkgreen,penwidth=2,arrowsize=0.5];'
             .format(node=node.id, other=other['node'].id, label=other['link'][0]))
      else:
        yield ('  n{node} -> n{other}[fontcolor=darkgreen,penwidth=2,arrowsize=0.5];'
             .format(node=node.id, other=other['node'].id))
  yield '}'


def graphviz_net_svg(graph, file_path, who_is_input=[]):
  # 1.step extract export format
  image_format = file_path.split('.')[-1].lower()

  # 2.step dot
  graphviz = subprocess.Popen(['dot',
                                 '-T{}'.format(image_format),
                                 '-Gdpi=50',
                                 '-o', file_path],
                                stdin=subprocess.PIPE)


  # 3.step generate dot code
  source = '\n'.join(graphviz_net(graph, node_format, who_is_input))

  if image_format == 'svg':
    source = re.sub(r'(<svg\s[^>]*>)',
                 dedent(r'''
      \1
      <style type="text/css"><![CDATA[
          g.node:hover {
              stroke-width: 2;
          }
      ]]></style>
      '''),
                 source,
                 re.S)
  
  # 4.step save to file
  graphviz.communicate(source.encode('utf-8'))

  # 5.step svg content
  if image_format == 'svg':
    fp = open(file_path, 'r')
    content = fp.read()
    fp.close()
    return content

  return None

# test_graph = Graph(name='testnet')
# test_graph.add_node(name='conv', label='hello')
# test_graph.add_node(name='pool', label='world')
# test_graph.add_node(name='input', label='www')
# test_graph.add_node(name='ssd', label='hhh')
# test_graph.add_link('conv','pool',Link('ab',''))
# test_graph.add_link('input','conv',Link('bd',''))
# test_graph.add_link('input','ssd',Link('ac',''))
#
# ss = Encoder().encode(test_graph)
# print(ss)
# dd = Decoder().decode(ss)
# print(ss)

# graph_content = graph_net_visualization(test_graph, '/Users/zhangken/Downloads/testnet.svg')
# print(graph_content)
  # net_visualization(graph, '/Users/zhangken/Downloads/11.svg',net_node_format)