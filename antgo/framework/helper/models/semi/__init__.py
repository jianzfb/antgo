# from .soft_teacher import SoftTeacher
# from .distillation_teacher import DistillationTeacher
from .dense import DenseTeacher
from .mpl import MPL
from .detmpl import DETMPL
from .hook import *

__all__ = [
    'DenseTeacher', 'MPL', 'DETMPL'
]