# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:02
# @File    : __init__.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .dag import DagMixin
from .config import ConfigMixin
from .column import ColumnMixin
from .dispatcher import DispatcherMixin
from .stream import StreamMixin
from .data_processing import DataProcessingMixin
from .list import ListMixin
from .dataset import DatasetMixin
from .computer_vision import ComputerVisionMixin
from .dataframe import DataFrameMixin
from .show import ShowMixin
from .serve import ServeMixin
from .demo import DemoMixin
from .deploy import DeployMixin

class DCMixins(
  DagMixin,
  ConfigMixin,
  ColumnMixin,
  DispatcherMixin,
  StreamMixin,
  DataProcessingMixin,
  ListMixin,
  DatasetMixin,
  ComputerVisionMixin,
  DataFrameMixin,
  ShowMixin,
  ServeMixin,
  DemoMixin,
  DeployMixin):

  def __init__(self) -> None:  # pylint: disable=useless-super-delegation
    super().__init__()
