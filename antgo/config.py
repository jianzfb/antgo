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


class _Config(object):
  def __init__(self):
    pass
    
  def parse_xml(self, config_xml):
    tree = ET.ElementTree(file=config_xml)
    root = tree.getroot()

    for child in root:
      setattr(self, child.tag, child.text.strip())


AntConfig = _Config()