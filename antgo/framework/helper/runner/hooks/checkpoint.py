import os.path as osp
import warnings
from antgo.framework.helper.fileio import FileClient
from ..dist_utils import allreduce_params, master_only
from .hook import HOOKS, Hook
import antvis.client.mlogger as mlogger


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool, optional): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool, optional): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`

    .. warning::
        Before v1.3.16, the ``out_dir`` argument indicates the path where the
        checkpoint is stored. However, since v1.3.16, ``out_dir`` indicates the
        root directory and the final path to save checkpoint is the
        concatenation of ``out_dir`` and the last level directory of
        ``runner.work_dir``. Suppose the value of ``out_dir`` is "/path/of/A"
        and the value of ``runner.work_dir`` is "/path/of/B", then the final
        path will be "/path/of/A/B".
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args
        self.is_ready = False

    @master_only
    def before_run(self, runner):
        if not self.out_dir:
            # 在out_dir没有被设置时，继承runner.work_dir的工作目录
            self.out_dir = runner.work_dir

        self.file_client = \
            FileClient.infer_client(
                self.file_client_args, self.out_dir)

        # 添加固定路径格式
        self.out_dir = osp.join(self.out_dir, 'output', 'checkpoint')

        # 记录地址（对disk后端有效）
        self.disk_address = None

        # 设置文件日志存储根目录
        if mlogger.is_ready():
            self.is_ready = True
            backend = 'disk'
            if self.out_dir.startswith('ali'):
                backend = 'aliyun'
            elif self.out_dir.startswith('qiniu'):
                backend = 'qiniu'
            elif self.out_dir.startswith('htfs'):
                backend = 'hdfs'
            mlogger.FileLogger.root_folder = self.out_dir
            self.file_logger = mlogger.Container()
            self.file_logger.file = mlogger.FileLogger('file', backend, True)

        runner.logger.info(
            (f'Checkpoints will be saved to {self.out_dir} by '
                            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        self.args['create_symlink'] = False

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            runner.logger.info("after allreduce_params")
            self._save_checkpoint(runner)
            runner.logger.info("after save checkpoint")

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        info = runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        if self.is_ready:
            # 文件日志
            kwargs = {}
            if self.file_logger.backend == 'disk':
                # 需要记录当前机器地址 user@ip
                if self.disk_address is None:
                    self.disk_address = 'root@127.0.0.1'
                    if os.path.exists('./address'):
                        with open('./address', 'r') as fp:
                            self.disk_address = fp.read()

                kwargs.update({
                    'address': f'{self.disk_address}'
                })
            self.file_logger.file.update(info, **kwargs)

        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
