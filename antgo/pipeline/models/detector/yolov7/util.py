# -*- coding: UTF-8 -*-
# @Time    : 2022/9/13 21:05
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.models.utils.common import *


class Detect(nn.Module):
  stride = None  # strides computed during build
  def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
    super(Detect, self).__init__()
    self.nc = nc  # number of classes
    self.no = nc + 5  # number of outputs per anchor
    self.nl = len(anchors)  # number of detection layers
    self.na = len(anchors[0]) // 2  # number of anchors
    self.grid = [torch.zeros(1)] * self.nl  # init grid
    a = torch.tensor(anchors).float().view(self.nl, -1, 2)
    self.register_buffer('anchors', a)  # shape(nl,na,2)
    self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

  def forward(self, x):
    # x = x.copy()  # for profiling
    z = []  # inference output
    for i in range(self.nl):
      x[i] = self.m[i](x[i])  # conv
      bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
      x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

      if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
      y = x[i].sigmoid()

      y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
      y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
      z.append(y.view(bs, -1, self.no))

    out = (torch.cat(z, 1), x)
    return out

  @staticmethod
  def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

  def convert(self, z):
    z = torch.cat(z, 1)
    box = z[:, :, :4]
    conf = z[:, :, 4:5]
    score = z[:, :, 5:]
    score *= conf
    convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                  dtype=torch.float32,
                                  device=z.device)
    box @= convert_matrix
    return (box, score)


def yolo_parse_model(d, ch):  # model_dict, input_channels(3)
  # logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
  anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
  na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
  no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

  layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
  for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
    m = eval(m) if isinstance(m, str) else m  # eval strings
    for j, a in enumerate(args):
      try:
        args[j] = eval(a) if isinstance(a, str) else a  # eval strings
      except:
        pass

    n = max(round(n * gd), 1) if n > 1 else n  # depth gain
    if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
             SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
             Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
             RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
             Res, ResCSPA, ResCSPB, ResCSPC,
             RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
             ResX, ResXCSPA, ResXCSPB, ResXCSPC,
             RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
             Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
             SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
             SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
      c1, c2 = ch[f], args[0]
      if c2 != no:  # if not output
        c2 = make_divisible(c2 * gw, 8)

      args = [c1, c2, *args[1:]]
      if m in [DownC, SPPCSPC, GhostSPPCSPC,
               BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
               RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
               ResCSPA, ResCSPB, ResCSPC,
               RepResCSPA, RepResCSPB, RepResCSPC,
               ResXCSPA, ResXCSPB, ResXCSPC,
               RepResXCSPA, RepResXCSPB, RepResXCSPC,
               GhostCSPA, GhostCSPB, GhostCSPC,
               STCSPA, STCSPB, STCSPC,
               ST2CSPA, ST2CSPB, ST2CSPC]:
        args.insert(2, n)  # number of repeats
        n = 1
    elif m is nn.BatchNorm2d:
      args = [ch[f]]
    elif m is Concat:
      c2 = sum([ch[x] for x in f])
    elif m is Chuncat:
      c2 = sum([ch[x] for x in f])
    elif m is Shortcut:
      c2 = ch[f[0]]
    elif m is Foldcut:
      c2 = ch[f] // 2
    elif m in [Detect]:
      args.append([ch[x] for x in f])
      if isinstance(args[1], int):  # number of anchors
        args[1] = [list(range(args[1] * 2))] * len(f)
    elif m is ReOrg:
      c2 = ch[f] * 4
    elif m is Contract:
      c2 = ch[f] * args[0] ** 2
    elif m is Expand:
      c2 = ch[f] // args[0] ** 2
    else:
      c2 = ch[f]

    m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
    t = str(m)[8:-2].replace('__main__.', '')  # module type
    np = sum([x.numel() for x in m_.parameters()])  # number params
    m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
    # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
    save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    layers.append(m_)
    if i == 0:
      ch = []
    ch.append(c2)
  return nn.Sequential(*layers), sorted(save)

