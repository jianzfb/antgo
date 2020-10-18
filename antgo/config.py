# encoding=utf-8
# @Time    : 17-8-10
# @File    : config.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET
import os

class _Config(object):
  def __init__(self):
    self._attribs = {}

  def parse_xml(self, config_xml):
    tree = ET.ElementTree(file=config_xml)
    root = tree.getroot()

    for child in root:
      if child.tag == 'subgradientserver':
        subgradientserver_config = {}
        for subgradientserver_child in child:
          val = subgradientserver_child.text.strip() if subgradientserver_child.text is not None else None
          if subgradientserver_child.tag == 'subgradientserver_port':
            val = int(val)
          subgradientserver_config[subgradientserver_child.tag] = val

        setattr(self, child.tag, subgradientserver_config)
        continue

      val = child.text.strip() if child.text is not None else None
      setattr(self, child.tag, val)

      if child.tag == 'factory' and val is not None and val != '':
        setattr(self, 'data_factory', os.path.join(val, 'dataset'))
        setattr(self, 'task_factory', os.path.join(val, 'task'))

      self._attribs[child.tag] = val

  def write_xml(self, config_xml, attribs={}):
    all_attribs = attribs
    all_attribs.update(self._attribs)

    tree = ET.ElementTree(file=config_xml)
    root = tree.getroot()
    for child in root:
      if child.tag in all_attribs:
        child.text = all_attribs[child.tag]

    tree.write(config_xml, encoding="utf-8",xml_declaration=True)


AntConfig = _Config()