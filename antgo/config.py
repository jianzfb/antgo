# encoding=utf-8
# @Time    : 17-8-10
# @File    : config.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Config(object):
  def __init__(self, config_xml):
    self._parse_xml(config_xml)

  def _parse_xml(self, config_xml):
    tree = ET.ElementTree(file=config_xml)
    root = tree.getroot()

    for child in root:
      setattr(self, child.tag, child.text.strip())
