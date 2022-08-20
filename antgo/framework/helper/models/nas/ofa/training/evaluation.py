# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from math import inf

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, dataloader

from antgo.framework.helper.fileio import FileClient
from antgo.framework.helper.utils import is_seq_of
from antgo.framework.helper.parallel import is_module_wrapper
from antgo.framework.helper.models.nas.elastic_nn.modules.dynamic_op import *
from antgo.framework.helper.models.nas.elastic_nn.modules.dynamic_layers import *
from antgo.framework.helper.models.nas.utils.common_tools import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.runner.hooks.evaluation import EvalHook
from antgo.framework.helper.runner.hooks.hook import HOOKS

@HOOKS.register_module()
class OfaEvalHook(EvalHook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader, and return the test results. If ``None``, the default
            test function ``mmcv.engine.single_gpu_test`` will be used.
            (default: ``None``)
        greater_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. (default: ``None``)
        less_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. (default: ``None``)
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, `runner.work_dir` will be used by default. If specified,
            the `out_dir` will be the concatenation of `out_dir` and the last
            level directory of `runner.work_dir`.
            `New in version 1.3.16.`
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
            `New in version 1.3.16.`
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.

    Note:
        If new arguments are added for EvalHook, tools/test.py,
        tools/eval_metric.py may be affected.
    """

    # Since the key for determine greater or less is related to the downstream
    # tasks, downstream repos may need to overwrite the following inner
    # variable accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 out_dir=None,
                 file_client_args=None,
                 metric=None,
                 image_size_list=None,
                 ks_list=None,
                 expand_ratio_list=None,
                 depth_list=None,
                 width_mult_list=None,
                 **eval_kwargs):
        super().__init__(                 
            dataloader, 
            start, 
            interval, 
            by_epoch, 
            save_best, 
            rule, 
            test_fn, 
            greater_keys, 
            less_keys, 
            out_dir, 
            file_client_args, 
            metric, 
            **eval_kwargs
        )
        self.image_size_list = image_size_list
        self.ks_list = ks_list
        self.expand_ratio_list = expand_ratio_list
        self.depth_list = depth_list
        self.width_mult_list = width_mult_list

    def reset_running_statistics(self, model):
        bn_mean = {}
        bn_var = {}

        forward_model = copy.deepcopy(model)
        forward_model.eval()
        for name, m in forward_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

                def new_forward(bn, mean_est, var_est):
                    def lambda_forward(x):
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )

                        batch_mean = torch.squeeze(batch_mean)
                        batch_var = torch.squeeze(batch_var)

                        mean_est.update(batch_mean.data, x.size(0))
                        var_est.update(batch_var.data, x.size(0))

                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0)
                        return F.batch_norm(
                            x,
                            batch_mean,
                            batch_var,
                            bn.weight[:_feature_dim],
                            bn.bias[:_feature_dim],
                            False,
                            0.0,
                            bn.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m, bn_mean[name], bn_var[name])

        if len(bn_mean) == 0:
            # skip if there is no batch normalization layers in the network
            return

        with torch.no_grad():
            DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
            for data in self.dataloader:
                if type(data) == list or type(data) == tuple:
                    forward_model(*data, return_loss=False)
                else:
                    data.update({
                        'return_loss': False
                    })
                    forward_model(**data)
            DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        dynamic_net = runner.model
        if is_module_wrapper(dynamic_net):
            dynamic_net = dynamic_net.module
        # 
        image_size_list = self.image_size_list
        if image_size_list is None:
            image_size_list = val2list(self.dataloader.dataset.get_active_target_size(), 1)
        ks_list = self.ks_list
        if ks_list is None:
            ks_list = dynamic_net.ks_list
        expand_ratio_list = self.expand_ratio_list
        if expand_ratio_list is None:
            expand_ratio_list = dynamic_net.expand_ratio_list
        
        depth_list = self.depth_list
        if depth_list is None:
            depth_list = dynamic_net.depth_list
        
        width_mult_list = self.width_mult_list
        if width_mult_list is None:
            if "width_mult_list" in dynamic_net.__dict__:
                width_mult_list = list(range(len(dynamic_net.width_mult_list)))
            else:
                width_mult_list = [0]

        subnet_settings = []
        for d in depth_list:
            for e in expand_ratio_list:
                for k in ks_list:
                    for w in width_mult_list:
                        for img_size in image_size_list:
                            subnet_settings.append(
                                [
                                    {
                                        "image_size": img_size,
                                        "d": d,
                                        "e": e,
                                        "ks": k,
                                        "w": w,
                                    },
                                    "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                                ]
                            )

        for setting, name in subnet_settings:
            self.dataloader.dataset.set_active_target_size(
                setting.pop("image_size")
            )
            # 基于配置，设置当前激活
            dynamic_net.set_active_subnet(**setting)

            # 重新统计当前激活子网络BN
            self.reset_running_statistics(runner.model)

            # 计算网络输出结果
            results = self.test_fn(runner.model, self.dataloader)
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            # 评估指标
            key_score = self.evaluate(runner, results)

            # 保存模型
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class OfaDistEvalHook(OfaEvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader in a multi-gpu manner, and return the test results. If
            ``None``, the default test function ``mmcv.engine.multi_gpu_test``
            will be used. (default: ``None``)
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        broadcast_bn_buffer (bool): Whether to broadcast the
            buffer(running_mean and running_var) of rank 0 to other rank
            before evaluation. Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, `runner.work_dir` will be used by default. If specified,
            the `out_dir` will be the concatenation of `out_dir` and the last
            level directory of `runner.work_dir`.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 out_dir=None,
                 file_client_args=None,
                 metric=None,
                 image_size_list=None,
                 ks_list=None,
                 expand_ratio_list=None,
                 depth_list=None,
                 width_mult_list=None,                 
                 **eval_kwargs):
        super().__init__(
                 dataloader, 
                 start, 
                 interval, 
                 by_epoch, 
                 save_best,
                 rule, 
                 test_fn, 
                 greater_keys, 
                 less_keys, 
                 broadcast_bn_buffer, 
                 tmpdir, 
                 gpu_collect, 
                 out_dir, 
                 file_client_args, 
                 metric, 
                 image_size_list,
                 ks_list,
                 expand_ratio_list,
                 depth_list,
                 width_mult_list,
                 **eval_kwargs            
        )

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def reset_running_statistics(self, model):
        bn_mean = {}
        bn_var = {}

        forward_model = copy.deepcopy(model)
        for name, m in forward_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                def new_forward(bn):
                    def lambda_forward(x):
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )

                        batch_mean = torch.squeeze(batch_mean)
                        batch_var = torch.squeeze(batch_var)

                        # 需要所有节点同步平均
                        dist.all_reduce(batch_mean, op=dist.reduce_op.AVG, group=0)
                        dist.all_reduce(batch_var, op=dist.reduce_op.AVG, group=0)

                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0)
                        return F.batch_norm(
                            x,
                            batch_mean,
                            batch_var,
                            bn.weight[:_feature_dim],
                            bn.bias[:_feature_dim],
                            False,
                            0.0,
                            bn.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m)

        if len(bn_mean) == 0:
            # skip if there is no batch normalization layers in the network
            return

        with torch.no_grad():
            DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
            for images, labels in self.dataloader:
                images = images.to(forward_model.parameters().__next__().device)
                forward_model(images)
            DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        """perform evaluation and save ckpt."""
        dynamic_net = runner.model
        if is_module_wrapper(dynamic_net):
            dynamic_net = dynamic_net.module
        # 
        image_size_list = self.image_size_list
        if image_size_list is None:
            image_size_list = val2list(self.dataloader.dataset.get_active_target_size(), 1)
        ks_list = self.ks_list
        if ks_list is None:
            ks_list = dynamic_net.ks_list
        expand_ratio_list = self.expand_ratio_list
        if expand_ratio_list is None:
            expand_ratio_list = dynamic_net.expand_ratio_list
        
        depth_list = self.depth_list
        if depth_list is None:
            depth_list = dynamic_net.depth_list
        
        width_mult_list = self.width_mult_list
        if width_mult_list is None:
            if "width_mult_list" in dynamic_net.__dict__:
                width_mult_list = list(range(len(dynamic_net.width_mult_list)))
            else:
                width_mult_list = [0]

        subnet_settings = []
        for d in depth_list:
            for e in expand_ratio_list:
                for k in ks_list:
                    for w in width_mult_list:
                        for img_size in image_size_list:
                            subnet_settings.append(
                                [
                                    {
                                        "image_size": img_size,
                                        "d": d,
                                        "e": e,
                                        "ks": k,
                                        "w": w,
                                    },
                                    "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                                ]
                            )

        for setting, name in subnet_settings:
            self.dataloader.dataset.set_active_target_size(
                setting.pop("image_size")
            )
            # 基于配置，设置当前激活
            dynamic_net.set_active_subnet(**setting)

            # 重新统计当前激活子网络BN
            self.reset_running_statistics(runner.model)

            # 计算网络输出结果
            results = self.test_fn(runner.model, self.dataloader)
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            # 计算网络输出结果
            tmpdir = self.tmpdir
            if tmpdir is None:
                tmpdir = osp.join(runner.work_dir, '.eval_hook')

            # 所有节点都需要运行
            results = self.test_fn(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=False)

            if runner.rank == 0:
                # 仅在master节点进行统计模型分数
                print('\n')
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                key_score = self.evaluate(runner, results)
                # the key_score may be `None` so it needs to skip the action to
                # save the best checkpoint
                if self.save_best and key_score:
                    self._save_ckpt(runner, key_score)
