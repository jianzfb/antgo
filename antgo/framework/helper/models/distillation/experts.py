from sys import maxsize
from numpy.lib.function_base import diff
import torch
from torch._C import device
from antgo.framework.helper.models.builder import DETECTORS, build_model
import numpy as np
import cv2
import os
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence


class Experts(torch.nn.Module):
    def __init__(self, teachers) -> None:
        super().__init__()
        assert(isinstance(teachers, torch.nn.ModuleList))
        self.teachers = teachers

    def __len__(self):
        return len(self.teachers)

    def mean(self, outs):
        mean_outs = None
        for teacher_i, teacher in enumerate(outs):
            if mean_outs is None:
                mean_outs = list(outs[teacher_i])
            else:
                for t_i in range(len(mean_outs)):
                    mean_outs[t_i][0] += outs[teacher_i][t_i][0]

        for t_i in range(len(mean_outs)):
            mean_outs[t_i][0] /= len(outs)
        return mean_outs

    def __getattr__(self, name: str):
        def __func(*args, **kwargs):
            outs = []
            for teacher_i, teacher in enumerate(self.teachers):
                teacher_out = getattr(teacher, name)(*args, **kwargs)
                outs.append(teacher_out)

        return __func