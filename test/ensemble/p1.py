# -*- coding: UTF-8 -*-
# @Time    : 2022/4/21 11:59
# @File    : m1.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.interactcontext import *
import numpy as np

def main():
  ctx = InteractContext()

  with ctx.Challenge('A', {
    'system': {
      'ip': '127.0.0.1',
      'port': 8901},
    }, dataset='image_M') as challenge:

    print('lala')
    for data, annotation in challenge.running_dataset.iterator_value():
      print("sdf")
      data = np.random.randint(0, 255, (255, 255, 3), dtype=np.uint8)

      print(annotation)

    challenge.evaluate()

if __name__ == '__main__':
  main()

