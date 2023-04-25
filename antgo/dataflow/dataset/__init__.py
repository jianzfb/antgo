#  -*- coding: UTF-8 -*-
# @Time    : 18-3-26
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from .dataset import Dataset
from .cifar import Cifar10, Cifar100
from .pascal_voc import Pascal2007, Pascal2012
from .coco2017 import COCO2017
from .interhand26M import InterHand26M
from .lsp import LSP
from .visalso import VisalSO
from .imagenet import ImageNet
from .vggface import VGGFace
from .mnist import Mnist
from .flic import FLIC
from .lip import LIP
from .ade20k import ADE20K
from .lfw import LFW
from .cityscape import Cityscape
from .mpii import MPII

__all__ = [
  'Mnist', 'Cifar10', 'Cifar100', 'ImageNet', 'Pascal2007', 'Pascal2012', 'COCO2017', 'InterHand26M', 'LSP', 'VisalSO', 'VGGFace', 'FLIC', 'LIP', 'ADE20K', 'LFW', 'Cityscape', 'MPII', 'Dataset'
]
