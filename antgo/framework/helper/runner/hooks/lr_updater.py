import numbers
from math import cos, pi
from .hook import HOOKS, Hook
from antgo.framework.helper.utils import build_from_cfg


class LrUpdaterHook(Hook):
    """LR Scheduler.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 begin=None, end=None, **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = self.by_epoch    # warmup_by_epoch 和 by_epoch 一致
        self.begin = begin
        self.end = end

        # warmup_epochs 在 warmup_by_epoch时使用
        # warmup_iters 在 warmup_by_iters时使用
        if self.warmup_by_epoch:
            # 如果warmup采用by_epoch，则将warmup_iters赋值给warmup_epochs
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []       # initial lr for all param groups
        self.regular_lr = []    # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            warmup_iters = self.warmup_iters
            if self.by_epoch:
                warmup_iters = self.warmup_epochs

            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            cur_iter = runner.iter
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                # 进入正常学习率调整
                self._set_lr(runner, self.regular_lr)
            else:
                # 进入warmup学习率调整
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            cur_iter = runner.epoch
            if self.warmup is None or cur_iter > self.warmup_epochs:
                return
            elif cur_iter == self.warmup_epochs:
                self._set_lr(runner, self.regular_lr)
            else:
                # 进入warmup学习率调整
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module()
class WarmupLrUpdaterHook(LrUpdaterHook):
    def __init__(self, **kwargs):
        super(WarmupLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


@HOOKS.register_module()
class FixedLrUpdaterHook(LrUpdaterHook):
    def __init__(self, factor=1, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)
        self.factor = factor

    def get_lr(self, runner, base_lr):
        return base_lr * self.factor


@HOOKS.register_module()
class StepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):
        if isinstance(step, list):
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            # 如此设置，意味着每隔step步，调整一次学习率
            # 此时需要考虑self.begin, self.end确定整体区间
            if self.begin is not None and self.end is not None:
                if self.by_epoch:
                    progress = runner.epoch - self.begin
                else:
                    progress = runner.iter - self.begin

            exp = progress // self.step
        else:
            # 如此设置，意味着在不同区间，调整一次学习率
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


@HOOKS.register_module()
class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        if self.begin is not None and self.end is not None:
            if self.by_epoch:
                progress = runner.epoch - self.begin
            else:
                progress = runner.iter - self.begin

        return base_lr * self.gamma**progress


@HOOKS.register_module()
class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=0.9, min_lr=1e-6, **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress, max_progress = 0, 0
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
            if self.end is not None and self.end is not None:
                progress = progress - self.begin
                max_progress = self.end - self.begin
        else:
            progress = runner.iter
            max_progress = runner.max_iters
            if self.end is not None and self.end is not None:
                progress = progress - self.begin
                max_progress = self.end - self.begin

        if progress < 0:
            return base_lr
        if progress > max_progress:
            return self.min_lr

        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module()
class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        if self.begin is not None and self.end is not None:
            if self.by_epoch:
                progress = runner.epoch - self.begin
            else:
                progress = runner.iter - self.begin

        return base_lr * (1 + self.gamma * progress)**(-self.power)


@HOOKS.register_module()
class CosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress, max_progress = 0, 0
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
            if self.begin is not None and self.end is not None:
                progress = runner.epoch - self.begin
                max_progress = self.end - self.begin
        else:
            progress = runner.iter
            max_progress = runner.max_iters
            if self.begin is not None and self.end is not None:
                progress = runner.iter - self.begin
                max_progress = self.end - self.begin

        if progress < 0:
            return base_lr

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress > max_progress:
            return target_lr

        return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class FlatCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """Flat + Cosine lr schedule.

    Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501

    Args:
        start_percent (float): When to start annealing the learning rate
            after the percentage of the total training steps.
            The value should be in range [0, 1).
            Default: 0.75
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 start_percent=0.75,
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        if start_percent < 0 or start_percent > 1 or not isinstance(
                start_percent, float):
            raise ValueError(
                'expected float between 0 and 1 start_percent, but '
                f'got {start_percent}')
        self.start_percent = start_percent
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(FlatCosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            max_epochs = runner.max_epochs
            progress = runner.epoch
            if self.begin is not None and self.end is not None:
                max_epochs = self.end - self.begin
                progress = progress - self.begin

            start = round(max_epochs * self.start_percent)
            progress = progress - start
            max_progress = max_epochs - start
        else:
            max_iters = runner.max_iters
            progress = runner.iter
            if self.begin is not None and self.end is not None:
                max_iters = self.end - self.begin
                progress = progress - self.begin

            start = round(max_iters * self.start_percent)
            progress = progress - start
            max_progress = max_iters - start

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress < 0:
            return base_lr
        else:
            return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class CosineRestartLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing with restarts learning rate scheme.

    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super(CosineRestartLrUpdaterHook, self).__init__(**kwargs)

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        if self.begin is not None and self.end is not None:
            if self.by_epoch:
                progress = runner.epoch - self.begin
            else:
                progress = runner.iter - self.begin

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)


def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


@HOOKS.register_module()
class CyclicLrUpdaterHook(LrUpdaterHook):
    """Cyclic LR Scheduler.

    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf

    Different from the original paper, we use cosine annealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.

    Args:
        by_epoch (bool, optional): Whether to update LR by epoch.
        target_ratio (tuple[float], optional): Relative ratio of the highest LR
            and the lowest LR to the initial LR.
        cyclic_times (int, optional): Number of cycles during training
        step_ratio_up (float, optional): The ratio of the increasing process of
            LR in the total cycle.
        anneal_strategy (str, optional): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
        gamma (float, optional): Cycle decay ratio. Default: 1.
            It takes values in the range (0, 1]. The difference between the
            maximum learning rate and the minimum learning rate decreases
            periodically when it is less than 1. `New in version 1.4.4.`
    """

    def __init__(self,
                 by_epoch=False,
                 target_ratio=(10, 1e-4),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 anneal_strategy='cos',
                 gamma=1,
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'
        assert 0 < gamma <= 1, \
            '"gamma" must be in range (0, 1]'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.gamma = gamma
        self.max_iter_per_phase = None
        self.lr_phases = []  # init lr_phases
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear

        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super(CyclicLrUpdaterHook, self).__init__(by_epoch, **kwargs)

    def before_run(self, runner):
        super(CyclicLrUpdaterHook, self).before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        self.max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * self.max_iter_per_phase)
        self.lr_phases.append([0, iter_up_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, self.max_iter_per_phase, self.target_ratio[0],
            self.target_ratio[1]
        ])

    def get_lr(self, runner, base_lr):
        curr_iter = runner.iter % self.max_iter_per_phase
        curr_cycle = runner.iter // self.max_iter_per_phase
        # Update weight decay
        scale = self.gamma**curr_cycle

        for (start_iter, end_iter, start_ratio, end_ratio) in self.lr_phases:
            if start_iter <= curr_iter < end_iter:
                # Apply cycle scaling to gradually reduce the difference
                # between max_lr and base lr. The target end_ratio can be
                # expressed as:
                # end_ratio = (base_lr + scale * (max_lr - base_lr)) / base_lr
                # iteration: 0-iter_up_phase:
                if start_iter == 0:
                    end_ratio = 1 - scale + end_ratio * scale
                # iteration: iter_up_phase-self.max_iter_per_phase
                else:
                    start_ratio = 1 - scale + start_ratio * scale
                progress = curr_iter - start_iter
                return self.anneal_func(base_lr * start_ratio,
                                        base_lr * end_ratio,
                                        progress / (end_iter - start_iter))


@HOOKS.register_module()
class OneCycleLrUpdaterHook(LrUpdaterHook):
    """One Cycle LR Scheduler.

    The 1cycle learning rate policy changes the learning rate after every
    batch. The one cycle learning rate policy is described in
    https://arxiv.org/pdf/1708.07120.pdf

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int, optional): The total number of steps in the cycle.
            Note that if a value is not provided here, it will be the max_iter
            of runner. Default: None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """

    def __init__(self,
                 max_lr,
                 total_steps=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 div_factor=25,
                 final_div_factor=1e4,
                 three_phase=False,
                 **kwargs):
        # validate by_epoch, currently only support by_epoch = False
        if 'by_epoch' not in kwargs:
            kwargs['by_epoch'] = False
        else:
            assert not kwargs['by_epoch'], \
                'currently only support "by_epoch" = False'
        if not isinstance(max_lr, (numbers.Number, list, dict)):
            raise ValueError('the type of max_lr must be the one of list or '
                             f'dict, but got {type(max_lr)}')
        self._max_lr = max_lr
        if total_steps is not None:
            if not isinstance(total_steps, int):
                raise ValueError('the type of total_steps must be int, but'
                                 f'got {type(total_steps)}')
            self.total_steps = total_steps
        # validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError('expected float between 0 and 1 pct_start, but '
                             f'got {pct_start}')
        self.pct_start = pct_start
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.lr_phases = []  # init lr_phases
        super(OneCycleLrUpdaterHook, self).__init__(**kwargs)

    def before_run(self, runner):
        if hasattr(self, 'total_steps'):
            total_steps = self.total_steps
        else:
            total_steps = runner.max_iters
        if total_steps < runner.max_iters:
            raise ValueError(
                'The total steps must be greater than or equal to max '
                f'iterations {runner.max_iters} of runner, but total steps '
                f'is {total_steps}.')

        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                _max_lr = format_param(k, optim, self._max_lr)
                self.base_lr[k] = [lr / self.div_factor for lr in _max_lr]
                for group, lr in zip(optim.param_groups, self.base_lr[k]):
                    group.setdefault('initial_lr', lr)
        else:
            k = type(runner.optimizer).__name__
            _max_lr = format_param(k, runner.optimizer, self._max_lr)
            self.base_lr = [lr / self.div_factor for lr in _max_lr]
            for group, lr in zip(runner.optimizer.param_groups, self.base_lr):
                group.setdefault('initial_lr', lr)

        if self.three_phase:
            self.lr_phases.append(
                [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
            self.lr_phases.append([
                float(2 * self.pct_start * total_steps) - 2, self.div_factor, 1
            ])
            self.lr_phases.append(
                [total_steps - 1, 1, 1 / self.final_div_factor])
        else:
            self.lr_phases.append(
                [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
            self.lr_phases.append(
                [total_steps - 1, self.div_factor, 1 / self.final_div_factor])

    def get_lr(self, runner, base_lr):
        curr_iter = runner.iter
        start_iter = 0
        for i, (end_iter, start_lr, end_lr) in enumerate(self.lr_phases):
            if curr_iter <= end_iter:
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                lr = self.anneal_func(base_lr * start_lr, base_lr * end_lr,
                                      pct)
                break
            start_iter = end_iter
        return lr


@HOOKS.register_module()
class LinearAnnealingLrUpdaterHook(LrUpdaterHook):
    """Linear annealing LR Scheduler decays the learning rate of each parameter
    group linearly.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self, min_lr=1e-6, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(LinearAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress, max_progress = 0, 0
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
            if self.begin is not None and self.end is not None:
                progress = runner.epoch - self.begin
                max_progress = self.end - self.begin
        else:
            progress = runner.iter
            max_progress = runner.max_iters
            if self.begin is not None and self.end is not None:
                progress = runner.iter - self.begin
                max_progress = self.end - self.begin

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress < 0:
            return base_lr
        if progress > max_progress:
            return self.min_lr
        return annealing_linear(base_lr, target_lr, progress / max_progress)


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start, end, factor):
    """Calculate annealing linear learning rate.

    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor


def format_param(name, optim, param):
    if isinstance(param, numbers.Number):
        return [param] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f'expected {len(optim.param_groups)} '
                             f'values for {name}, got {len(param)}')
        return param
    else:  # multi optimizers
        if name not in param:
            raise KeyError(f'{name} is not found in {param.keys()}')
        return param[name]


@HOOKS.register_module()
class ComposerLrUpdaterHook(LrUpdaterHook):
    def __init__(self, lr_updater_list, **kwargs):
        # no warmup
        kwargs['warmup'] = None
        super(ComposerLrUpdaterHook, self).__init__(**kwargs)
        self.lr_updater_list = []
        self.lr_updater_cfg = []
        begin = 0
        for lr_config in lr_updater_list:
            policy_type = lr_config.pop('policy')
            if policy_type == 'ComposerLr':
                continue

            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = build_from_cfg(lr_config, HOOKS)
            assert('begin' in lr_config and 'end' in lr_config)
            assert(begin == lr_config['begin'])
            begin = lr_config['end'] + 1
            self.lr_updater_cfg.append(lr_config)
            self.lr_updater_list.append(hook)

        self.by_epoch = self.lr_updater_list[0].by_epoch
        self.cur_policy_i = 0

    def get_lr(self, runner, base_lr):
        return 0

    def before_run(self, runner):
        super(ComposerLrUpdaterHook, self).before_run(runner)
        self.lr_updater_list[self.cur_policy_i].base_lr = self.base_lr
        self.lr_updater_list[self.cur_policy_i].regular_lr = self.base_lr

        # 记录原始epoch, iter
        cur_epoch = runner.epoch
        cur_iter = runner.iter
        if cur_epoch != 0 or cur_iter != 0:
            iter_num_in_one_epoch = cur_iter // cur_epoch
            # 模型训练，从checkpoint恢复，发现正确学习率
            runner.epoch = 0
            runner.iter = 0
            for policy_i in range(len(self.lr_updater_list)-1):
                if cur_epoch >= self.lr_updater_cfg[policy_i]['begin'] and cur_epoch <= self.lr_updater_cfg[policy_i]['end']:
                    self.cur_policy_i = policy_i
                    break

                runner.epoch = self.lr_updater_cfg[policy_i]['end']
                runner.iter = runner.epoch * iter_num_in_one_epoch
                self.lr_updater_list[policy_i].before_train_epoch(runner)
                self.lr_updater_list[policy_i].before_train_iter(runner)
                self.lr_updater_list[policy_i+1].base_lr = self.lr_updater_list[policy_i].regular_lr

        # 恢复原始epoch, iter
        runner.epoch = cur_epoch
        runner.iter = cur_iter

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        for policy_i in range(len(self.lr_updater_list)):
            if runner.epoch >= self.lr_updater_cfg[policy_i]['begin'] and runner.epoch <= self.lr_updater_cfg[policy_i]['end']:
                if self.cur_policy_i != policy_i:
                    self.lr_updater_list[policy_i].base_lr = self.lr_updater_list[self.cur_policy_i].regular_lr
                    self.cur_policy_i = policy_i
                break

        self.lr_updater_list[self.cur_policy_i].before_train_epoch(runner)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            for policy_i in range(len(self.lr_updater_list)):
                if runner.iter >= self.lr_updater_cfg[policy_i]['begin'] and runner.iter <= self.lr_updater_cfg[policy_i]['end']:
                    if self.cur_policy_i != policy_i:
                        self.lr_updater_list[policy_i].base_lr = self.lr_updater_list[self.cur_policy_i].regular_lr
                        self.cur_policy_i = policy_i
                    break
        else:
            for policy_i in range(len(self.lr_updater_list)):
                if runner.epoch >= self.lr_updater_cfg[policy_i]['begin'] and runner.epoch <= self.lr_updater_cfg[policy_i]['end']:
                    if self.cur_policy_i != policy_i:
                        self.lr_updater_list[policy_i].base_lr = self.lr_updater_list[self.cur_policy_i].regular_lr
                        self.cur_policy_i = policy_i
                    break

        self.lr_updater_list[self.cur_policy_i].before_train_iter(runner)
