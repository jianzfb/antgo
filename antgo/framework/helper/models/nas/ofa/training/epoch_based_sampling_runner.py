# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import platform
import shutil
import time
import warnings
import random
import torch
import numpy as np
from antgo.framework.helper.runner.epoch_based_runner import EpochBasedRunner
from antgo.framework.helper.runner.builder import RUNNERS
from antgo.framework.helper.runner.checkpoint import save_checkpoint
from antgo.framework.helper.runner.utils import get_host_info


@RUNNERS.register_module()
class EpochBasedSamplingRunner(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def __init__(self,
                model,
                batch_processor=None,
                optimizer=None,
                work_dir=None,
                logger=None,
                meta=None,
                max_iters=None,
                max_epochs=None,
                dynamic_batch_size=1,
                debug=False):
        super().__init__( 
            model,
            batch_processor=batch_processor,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta,
            max_iters=max_iters,
            max_epochs=max_epochs)
        
        self.dynamic_batch_size = dynamic_batch_size
        self.debug = debug

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
                            
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')

        nBatch = len(self.data_loader)
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # for i, data_batch in enumerate(self.data_loader):
        #     self._inner_iter = i

        #     for _ in range(self.dynamic_batch_size):
        #         # 1.step 迭代前处理
        #         self.call_hook('before_train_iter')

        #         # 2.step 结构采样
        #         # set random seed before sampling
        #         subnet_seed = int("%d%.3d%.3d" % (self._epoch * nBatch + i, _, 0))
        #         random.seed(subnet_seed)
        #         subnet_settings = self.model.module.sample_active_subnet()
        #         if self.debug:
        #             subnet_str = ''
        #             subnet_str += (
        #                 "%d: " % _
        #                 + ",".join(
        #                     [
        #                         "%s_%s"
        #                         % (
        #                             key,
        #                             "%.1f" % np.mean(val)
        #                             if isinstance(val, list)
        #                             else val,
        #                         )
        #                         for key, val in subnet_settings.items()
        #                     ]
        #                 )
        #                 + " || "
        #             )
        #             print(subnet_str)

        #         # 3.step 训练
        #         self.run_iter(data_batch, train_mode=True, **kwargs)

        #         # 4.step 迭代后处理
        #         self.call_hook('after_train_iter')

        #     self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        # self.model.eval()
        # self.mode = 'val'
        # self.data_loader = data_loader
        # self.call_hook('before_val_epoch')
        # time.sleep(2)  # Prevent possible deadlock during epoch transition
        # for i, data_batch in enumerate(self.data_loader):
        #     self._inner_iter = i
        #     self.call_hook('before_val_iter')
        #     self.run_iter(data_batch, train_mode=False, **kwargs)
        #     self.call_hook('after_val_iter')

        # self.call_hook('after_val_epoch')
        raise NotImplementedError

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        # assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                os.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
