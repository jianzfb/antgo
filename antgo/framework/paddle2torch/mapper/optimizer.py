# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 14:55
# @File    : optimizer.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch


class SGD(torch.optim.SGD):
  def __init__(self,
               learning_rate=0.001,
               parameters=None,
               weight_decay=None,
               grad_clip=None,
               name=None):
    super().__init__(
      parameters,
      lr=learning_rate.learning_rate if isinstance(learning_rate, LRSchedulerProxy) else learning_rate,
      momentum=0,
      dampening=0,
      weight_decay=0.0 if weight_decay is None else weight_decay,
      nesterov=False
    )

    if isinstance(learning_rate, LRSchedulerProxy):
      learning_rate.set_optimizer(self)

  def clear_grad(self):
    self.zero_grad()

  def clear_gradients(self):
    self.zero_grad()

class Adam(torch.optim.Adam):
  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-08,
               parameters=None,
               weight_decay=None,
               grad_clip=None,
               name=None,
               lazy_mode=False):
    super().__init__(parameters,
                     lr=learning_rate.learning_rate if isinstance(learning_rate, LRSchedulerProxy) else learning_rate,
                     betas=(beta1, beta2),
                     eps=epsilon,
                     weight_decay=weight_decay, amsgrad=False)

    if isinstance(learning_rate, LRSchedulerProxy):
      learning_rate.set_optimizer(self)

  def clear_grad(self):
    self.zero_grad()

  def clear_gradients(self):
    self.zero_grad()


class RMSProp(torch.optim.RMSprop):
  def __init__(self,
               learning_rate,
               rho=0.95,
               epsilon=1e-06,
               momentum=0.0,
               centered=False,
               parameters=None,
               weight_decay=None,
               grad_clip=None, name=None):
    super().__init__(parameters,
                     lr=learning_rate.learning_rate if isinstance(learning_rate, LRSchedulerProxy) else learning_rate,
                     alpha=rho,
                     eps=epsilon,
                     weight_decay=0 if weight_decay is None else weight_decay,
                     momentum=momentum,
                     centered=centered)

    if isinstance(learning_rate, LRSchedulerProxy):
      learning_rate.set_optimizer(self)

  def clear_grad(self):
    self.zero_grad()

  def clear_gradients(self):
    self.zero_grad()


class LR(object):
  pass


lr = LR()


class LRSchedulerProxy(object):
  def __init__(self, lr_scheduler_func):
    self.create_lr_scheduler_func = lr_scheduler_func
    self.lr_scheduler = None
    self.learning_rate = 0.0

  def set_optimizer(self, optim):
    self.lr_scheduler = self.create_lr_scheduler_func(optim)

  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    return getattr(self.lr_scheduler, key)


def add_optimizer_lr_function(func):
  setattr(lr, func.__name__, func)


class _StepDecay(LRSchedulerProxy):
  def __init__(self, learning_rate, step_size, gamma=0.1, last_epoch=- 1, verbose=False):
    super(_StepDecay, self).__init__(
      lambda x: torch.optim.lr_scheduler.StepLR(x, step_size, gamma=gamma, last_epoch=last_epoch, verbose=verbose)
    )
    self.learning_rate = learning_rate

@add_optimizer_lr_function
def StepDecay(learning_rate, step_size, gamma=0.1, last_epoch=- 1, verbose=False):
    return _StepDecay(learning_rate, step_size, gamma, last_epoch, verbose)


class _CosineAnnealingDecay(LRSchedulerProxy):
  def __init__(self, learning_rate, T_max, eta_min=0, last_epoch=- 1, verbose=False):
    super(_CosineAnnealingDecay, self).__init__(
      lambda x: torch.optim.lr_scheduler.CosineAnnealingLR(x, T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
    )
    self.learning_rate = learning_rate

@add_optimizer_lr_function
def CosineAnnealingDecay(learning_rate, T_max, eta_min=0, last_epoch=- 1, verbose=False):
    return _CosineAnnealingDecay(learning_rate, T_max, eta_min, last_epoch, verbose)


class _MultiStepDecay(LRSchedulerProxy):
  def __init__(self,learning_rate, milestones, gamma=0.1, last_epoch=- 1, verbose=False):
    super(_MultiStepDecay, self).__init__(
      lambda x: torch.optim.lr_scheduler.MultiStepLR(x, milestones, gamma=gamma, last_epoch=last_epoch, verbose=verbose)
    )
    self.learning_rate = learning_rate

@add_optimizer_lr_function
def MultiStepDecay(learning_rate, milestones, gamma=0.1, last_epoch=- 1, verbose=False):
    return _MultiStepDecay(learning_rate, milestones, gamma, last_epoch, verbose)



