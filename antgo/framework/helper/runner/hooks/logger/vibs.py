
import datetime
import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist

from antgo.framework.helper.fileio.file_client import FileClient
from antgo.framework.helper.utils import is_tuple_of, scandir
import antvis.client.mlogger as mlogger
from antgo import config
from ..hook import HOOKS
from .base import LoggerHook
from ...dist_utils import master_only
import json


@HOOKS.register_module()
class VibSLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.

        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`

    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False, record_keys=None):
        super(VibSLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                             by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.is_ready = False
        self.canvas = None
        self.record_keys = record_keys
        if self.record_keys is None:
            self.record_keys = []

        if 'lr' not in self.record_keys:
            self.record_keys.append('lr')
        if 'epoch' not in self.record_keys:
            self.record_keys.append('epoch')

    @master_only
    def before_run(self, runner):
        super(VibSLoggerHook, self).before_run(runner)

        self.elements_in_canvas = []
        if mlogger.is_ready():
            # 日志平台是否具备条件
            self.is_ready = True
            # 创建图表容器
            self.canvas = mlogger.Container()

        self.start_iter = runner.iter

    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                           f'data_time: {log_dict["data_time"]:.3f}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        runner.logger.info(log_str)

    @master_only
    def log(self, runner):
        if not self.is_ready:
            return

        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        log_dict = dict(log_dict, **runner.log_buffer.output)

        # to 控制台
        self._log_info(log_dict, runner)

        # to 实验平台
        for log_key, log_value in log_dict.items():
            if isinstance(log_value, str):
                continue

            if self.record_keys is not None:
                if log_key not in self.record_keys:
                    continue

            if log_key not in self.elements_in_canvas:
                setattr(self.canvas, log_key, mlogger.complex.Line(plot_title=log_key, is_series=True))
                self.elements_in_canvas.append(log_key)
            getattr(self.canvas, log_key).update(log_value)

        if 'basic' not in self.elements_in_canvas:
            setattr(self.canvas, 'basic', mlogger.complex.Table("basic"))
            self.elements_in_canvas.append('basic')

        eta_str = ''
        if log_dict['mode'] == 'train' and 'time' in log_dict.keys():
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        getattr(self.canvas, 'basic').r0c0 = 'eta'
        getattr(self.canvas, 'basic').r1c0 = eta_str

        getattr(self.canvas, 'basic').r0c1 = 'data_time'
        getattr(self.canvas, 'basic').r1c1 = "%.2f" % log_dict['data_time']

        getattr(self.canvas, 'basic').r0c2 = 'time'
        getattr(self.canvas, 'basic').r1c2 = log_dict['time']

        getattr(self.canvas, 'basic').update()
        mlogger.update()

    @master_only
    def after_run(self, runner):
        pass
