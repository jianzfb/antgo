from .data_collection import DataCollection, DataFrame
from .entity import Entity
from .image import *
from .common import *
from antgo.pipeline.hparam import HyperParameter as State

from antgo.pipeline.hparam import param_scope
from antgo.pipeline.hparam import dynamic_dispatch
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.ui.data import DataS
import numpy as np
import json
import os
import tkinter as tk  


@dynamic_dispatch
def tk_env(*args, **kwargs):
    index = param_scope()._index

    class UIConfig(object):
        def __init__(self, title, height, width, x, y):
            self.root = tk.Tk()
            self.root.title(title)
            self.root.geometry(kwargs.get('shape', f'{width}x{height}+{x}+{y}'))

        def loop(self):
            self.root.mainloop()

        @property
        def element(self):
           return self.root

    def inner():
        ui_config = UIConfig(
           title=kwargs.get('title', 'antgo'),
           height=kwargs.get('height', 300),
           width=kwargs.get('width', 300),
           x=kwargs.get('x', 100),
           y=kwargs.get('y', 100)
        )
        env_entity = Entity()(**{index: ui_config, '__page_root__': ui_config})
        yield env_entity

    return DataFrame(inner())


class _DataWrap(object):
  def __init__(self):
     self.name = None

  def __getattr__(self, name):
    self.name = name
    return self

  def get(self):
     return DataS(name=self.name).get()

  def set(self, value):
     return DataS(name=self.name).set(value)

  def __call__(self, *args, **kwds):
     return DataS(name=self.name, default=args[0])


class _gui(object):
  def __getattr__(self, name):
    if name not in ['tk', 'data']:
      return None

    if name == 'data':
       return _DataWrap()

    return globals()[f'{name}_env']

gui = _gui()

