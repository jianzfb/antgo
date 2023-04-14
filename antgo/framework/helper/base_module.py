import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler
import torch.distributed as dist
import torch.nn as nn
from collections import OrderedDict
import torch
from antgo.framework.helper.utils.logging import logger_initialized, print_log


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """
    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'antgo'

        from antgo.framework.helper.cnn import initialize
        from antgo.framework.helper.cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            for sub_module in self.modules():
                del sub_module._params_init_info

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # loss = sum(_value for _key, _value in log_vars.items())
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        num_samples = 1
        losses = None
        if type(data) == list or type(data) == tuple:
            num_samples = len(data[0]['image'])
            losses = self(*data, **kwargs)
        else:
            num_samples = len(data['image'])
            losses = self(**data, **kwargs)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        num_samples = 1
        losses = None
        if type(data) == list or type(data) == tuple:
            num_samples = len(data[0]['image'])
            losses = self(*data, **kwargs)
        else:
            num_samples = len(data['image'])
            losses = self(**data, **kwargs)

        return_loss = kwargs.get('return_loss', False)

        log_vars = {}
        for loss_name, loss_value in losses.items():
            if return_loss and 'loss' in loss_name:
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

        loss = 0
        if return_loss:
            loss = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)
            log_vars['loss'] = loss

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            # TODO, 需要进行多节点损失的reduce?
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def forward(self, image, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(image, **kwargs)
        else:
            return self.forward_test(image, **kwargs)

    def forward_train(self, image, **kwargs):
        raise NotImplementedError

    def forward_test(self, image, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict in openmmlab.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
