
import os.path as osp
import os
import platform
import shutil
import time
import warnings

import torch

from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()"'
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

        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            kwargs.update({'epoch': self._epoch, 'iter': self._iter})
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

        # 更新data_loader中dataset记录的epoch计数
        # TODO, 无法有效设置Dataset epoch属性，需要查找解决方案
        self.data_loader.dataset.epoch = self._epoch    # 仅在iterator dataset数据集有效

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False, **kwargs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
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
        assert len(data_loaders) == len(workflow)
        self._max_epochs = max_epochs

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow, self._max_epochs)
        self.call_hook('before_run')

        init_workflow_i = 0
        offset = 0
        for i, (mode, epochs) in enumerate(workflow):
            if mode == 'train':
                if self.epoch >= offset and self.epoch < offset + epochs:
                    init_workflow_i = i
                    break
                offset += epochs

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                if i < init_workflow_i:
                    continue

                mode, epochs = flow
                self.logger.info(f"current workflow {i} ({mode} {epochs})")
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

                if isinstance(data_loaders[i], dict):
                    # 清理之前数据集加载对象，释放资源
                    if i > 0:
                        obj = data_loaders[i-1]
                        data_loaders[i-1] = None
                        del obj

                    # 基于规则，动态创建数据集加载对象
                    from antgo.framework.helper.dataset import (build_dataset, build_dataloader, build_kv_dataloader, build_iter_dataloader)

                    dataset = build_dataset(data_loaders[i]['dataset'])
                    train_loader_cfg = data_loaders[i]['dataloader']

                    if getattr(dataset, 'is_kv', False):
                        data_loaders[i] = build_kv_dataloader(dataset, **train_loader_cfg)
                    elif isinstance(dataset, torch.utils.data.IterableDataset):
                        data_loaders[i] = build_iter_dataloader(dataset, **train_loader_cfg)
                    else:
                        data_loaders[i] = build_dataloader(dataset, **train_loader_cfg)

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
        status = save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
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

        return status