
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
                 reset_flag=False,
                 project='', 
                 experiment=''):
        super(VibSLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                             by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0

        self.project = project
        self.experiment = experiment
        self.is_ready = False
        self.canvas = None

    @master_only
    def before_run(self, runner):
        super(VibSLoggerHook, self).before_run(runner)
        # step 1: 检测当前路径下收否有token缓存
        token = None
        if os.path.exists('./.token'):
            with open('./.token', 'r') as fp:
                token = fp.readline()

        # step 2: 检查antgo配置目录下的配置文件中是否有token
        if token is None or token == '':
            config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
            config.AntConfig.parse_xml(config_xml)
            token = getattr(config.AntConfig, 'server_user_token', '')
        if token == '' or token is None:
            self.is_ready = False
            print('not valid token, directly return')
            return
        mlogger.config(self.project, self.experiment, token=token, server="BASELINE")
        print(f'Show Experiment Dashboard http://experiment.vibstring.com/ExperimentDashboard?token={token}')
        self.canvas = mlogger.Container()
        self.elements_in_canvas = []
        self.is_ready = True

    def _get_max_memory(self, runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

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

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)
        for log_key, log_value in log_dict.items():
            if isinstance(log_value, str):
                continue

            if log_key not in self.elements_in_canvas:
                setattr(self.canvas, log_key, mlogger.complex.Line(plot_title=log_key, is_series=True))
                self.elements_in_canvas.append(log_key)

            getattr(self.canvas, log_key).update(log_value)
        mlogger.update()

    @master_only
    def after_run(self, runner):
        pass