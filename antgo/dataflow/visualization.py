d"""Tools for visualizing a lusmu graph

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
from antgo.dataflow.core import Input, Node


def collect_nodes(collected_nodes, *args):
    """Collect all nodes belonging to the same graph

    Walks dependent Nodes and inputs recursively.

    """
    if not args:
        return
    node = args[0]
    rest = args[1:]
    collect_nodes(collected_nodes, *rest)
    if node in collected_nodes:
        return
    collected_nodes.add(node)
    collect_nodes(collected_nodes, *node._dependents)
    if isinstance(node, Node):
        collect_nodes(collected_nodes, *node._iterate_inputs())


def get_action_name(action):
    """Try to return a good representation of the name of an action callable"""
    if hasattr(action, 'name'):
        return action.name
    if hasattr(action, '__name__'):
        return action.__name__
    if hasattr(action, 'func_name'):
        return action.func_name
    return action.__class__.__name__


def format_node_default(node_id, node):
    shape = 'oval' if isinstance(node, Input) else 'box'
    action = ('{br}{br}<FONT COLOR="#888888">{action}</FONT>'
              .format(br=' <BR ALIGN="LEFT"/>',
                      action=get_action_name(node._action))
              if isinstance(node, Node)
              else '')
    yield ('  {node_id} '
           '[label=<<B>{name}</B>'
           '{action}>'
           ' shape={shape}];'
           .format(node_id=node_id,
                   name=node.name.replace(':', ':<BR ALIGN="LEFT"/>'),
                   action=action,
                   shape=shape))
    yield '  edge [color=blue];'


def graphviz_lines(nodes, node_filter, format_node):
    """Generate source lines for a Graphviz graph definition"""
    all_nodes = set()
    collect_nodes(all_nodes, *nodes)
    if node_filter:
        all_nodes = [n for n in all_nodes if node_filter(n)]
    all_nodes = sorted(all_nodes, key=id)
    input_nodes = [n for n in all_nodes if isinstance(n, Input)]

    yield 'digraph gr {'
    yield '  graph [ dpi = 48 ];'
    yield '  rankdir = LR;'
    yield '  { rank = source;'
    for node in input_nodes:
        yield '    n{};'.format(id(node))
    yield '  }'
    for node in all_nodes:
        for line in format_node('n{}'.format(id(node)), node):
            yield line
        for other in node._dependents:
            if other in all_nodes:
                yield ('  n{node} -> n{other};'
                       .format(node=id(node), other=id(other)))
    yield '}'


def visualize_graph(nodes, filename,
                    node_filter=lambda node: True,
                    format_node=format_node_default):
    """Saves a visualization of given nodes in an image file"""
    image_format = filename.split('.')[-1].lower()
    graphviz = subprocess.Popen(['dot',
                                 '-T{}'.format(image_format),
                                 '-o', filename],
                                stdin=subprocess.PIPE)
    source = '\n'.join(graphviz_lines(nodes,
                                      node_filter,
                                      format_node))
    graphviz.communicate(source.encode('utf-8'))

    # Add some CSS to SVG images
    if image_format == 'svg':
        with open(filename) as svg_file:
            svg = svg_file.read()
        svg = re.sub(r'(<svg\s[^>]*>)',
                     dedent(r'''
                     \1
                     <style type="text/css"><![CDATA[
                         g.node:hover {
                             stroke-width: 2;
                         }
                     ]]></style>
                     '''),
                     svg,
                     re.S)
        with open(filename, 'w') as svg_file:
            svg_file.write(svg)

    return source