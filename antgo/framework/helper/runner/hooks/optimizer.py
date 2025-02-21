
import copy
import logging
from collections import defaultdict
from itertools import chain
import torch
from torch.nn.utils import clip_grad

from antgo.framework.helper.utils import TORCH_VERSION, _BatchNorm, digit_version
from ..dist_utils import allreduce_grads
from ..fp16_utils import LossScaler, wrap_fp16_model
from .hook import HOOKS, Hook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self, grad_clip=None, detect_anomalous_params=False, amp=True):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params
        self.use_amp = amp
        self.grad_scaler = None
        if self.use_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        # clear grads
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        loss = runner.outputs['loss']
        # TODO，临时取消
        # if True:
        #     self.detect_anomalous_parameters(loss, runner)

        if not self.use_amp:
            loss.backward()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(runner.optimizer)  # NOTE: grads are unscaled before "before_optimize" callbacks

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])

        if not self.use_amp:
            runner.optimizer.step()
        else:
            self.grad_scaler.step(runner.optimizer)
            self.grad_scaler.update()

    def detect_anomalous_parameters(self, loss, runner):
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')


@HOOKS.register_module()
class GradientCumulativeOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters=1, **kwargs):
        super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter

        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters

        # 获得模型损失
        loss = runner.outputs['loss']
        loss = loss / loss_factor

        # 反向传播
        if not self.use_amp:
            loss.backward()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(runner.optimizer)  # NOTE: grads are unscaled before "before_optimize" callbacks

        # 优化器参数优化
        if (self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner)):
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])

            if not self.use_amp:
                runner.optimizer.step()
            else:
                self.grad_scaler.step(runner.optimizer)
                self.grad_scaler.update()

            # clear grads
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
