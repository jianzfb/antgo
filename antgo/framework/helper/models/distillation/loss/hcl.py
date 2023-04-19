from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class HCL(nn.Module):
    def __init__(self):
        super(HCL, self).__init__()

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            n,c,h,w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                if l >=h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all
