
from torch import optim
from antgo.framework.helper.runner.iter_based_runner import IterBasedRunner
from antgo.framework.helper.runner.builder import RUNNERS

@RUNNERS.register_module()
class AdaIterBasedRunner(IterBasedRunner):
    """Iteration-based Runner. Specified for AdaTask.

    This runner train models iteration by iteration.
    """
    def __init__(self,
                 model,
                 d_optimizer=None,
                 m_optimizer=None,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        if optimizer is None:
            optimizer = m_optimizer
        super(AdaIterBasedRunner, self).__init__(
                model,
                 batch_processor,
                 optimizer=optimizer,
                 work_dir=work_dir,
                 logger=logger,
                 meta=meta,
                 max_iters=max_iters,
                 max_epochs=max_epochs
        )

        self.d_optimizer = d_optimizer
        self.m_optimizer = m_optimizer

    def d_train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'

        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_d_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1


    def m_train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'

        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_m_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

