from antgo.framework.helper.base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .base_runner import BaseRunner
from .builder import RUNNERS, build_runner
from .checkpoint import (CheckpointLoader, _load_checkpoint,
                         _load_checkpoint_with_prefix, load_checkpoint,
                         load_state_dict, save_checkpoint, weights_to_cpu)
from .default_constructor import DefaultRunnerConstructor
from .dist_utils import (allreduce_grads, allreduce_params, get_dist_info,
                         init_dist, master_only)
from .epoch_based_runner import EpochBasedRunner
from .fp16_utils import LossScaler, auto_fp16, force_fp32, wrap_fp16_model
from .hooks import (HOOKS, CheckpointHook, ClosureHook,
                    DistEvalHook, DistSamplerSeedHook, 
                    EMAHook, EvalHook, Fp16OptimizerHook,
                    SubModulesDistEvalHook, SubModulesEvalHook,
                    GradientCumulativeFp16OptimizerHook,
                    GradientCumulativeOptimizerHook, Hook, IterTimerHook,
                    LoggerHook, NeptuneLoggerHook,
                    OptimizerHook, SyncBuffersHook, TextLoggerHook)
from .hooks.lr_updater import StepLrUpdaterHook  # noqa
from .hooks.lr_updater import (CosineAnnealingLrUpdaterHook,
                               CosineRestartLrUpdaterHook, CyclicLrUpdaterHook,
                               ExpLrUpdaterHook, FixedLrUpdaterHook,
                               FlatCosineAnnealingLrUpdaterHook,
                               InvLrUpdaterHook, LinearAnnealingLrUpdaterHook,
                               LrUpdaterHook, OneCycleLrUpdaterHook,
                               PolyLrUpdaterHook)
from .hooks.momentum_updater import (CosineAnnealingMomentumUpdaterHook,
                                     CyclicMomentumUpdaterHook,
                                     LinearAnnealingMomentumUpdaterHook,
                                     MomentumUpdaterHook,
                                     OneCycleMomentumUpdaterHook,
                                     StepMomentumUpdaterHook)
from .iter_based_runner import IterBasedRunner, IterLoader
from .log_buffer import LogBuffer
from .optimizer import (OPTIMIZER_BUILDERS, OPTIMIZERS,
                        DefaultOptimizerConstructor, build_optimizer,
                        build_optimizer_constructor)
from .priority import Priority, get_priority
from .utils import get_host_info, get_time_str, obj_from_dict, set_random_seed

# initialize ipu to registor ipu runner to RUNNERS
# from mmcv.device import ipu  # isort:skip  # noqa

__all__ = [
    'BaseRunner', 'EpochBasedRunner', 'IterBasedRunner', 'LogBuffer',
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'FixedLrUpdaterHook', 'StepLrUpdaterHook', 'ExpLrUpdaterHook',
    'PolyLrUpdaterHook', 'InvLrUpdaterHook', 'CosineAnnealingLrUpdaterHook',
    'FlatCosineAnnealingLrUpdaterHook', 'CosineRestartLrUpdaterHook',
    'CyclicLrUpdaterHook', 'OneCycleLrUpdaterHook', 'MomentumUpdaterHook',
    'StepMomentumUpdaterHook', 'CosineAnnealingMomentumUpdaterHook',
    'CyclicMomentumUpdaterHook', 'OneCycleMomentumUpdaterHook',
    'OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook', 'LoggerHook',
    'TextLoggerHook', 'NeptuneLoggerHook', 'Priority',
    'get_priority', 'get_host_info', 'get_time_str', 'obj_from_dict',
    'init_dist', 'get_dist_info', 'master_only', 'OPTIMIZER_BUILDERS',
    'OPTIMIZERS', 'DefaultOptimizerConstructor', 'build_optimizer',
    'build_optimizer_constructor', 'IterLoader', 'set_random_seed',
    'auto_fp16', 'force_fp32', 'wrap_fp16_model', 'Fp16OptimizerHook',
    'SyncBuffersHook', 'EMAHook', 'build_runner', 'RUNNERS', 'allreduce_grads',
    'allreduce_params', 'LossScaler', 'BaseModule', 'EvalHook', 'DistEvalHook', 'Sequential',
    'ModuleDict', 'ModuleList', 'GradientCumulativeOptimizerHook',
    'GradientCumulativeFp16OptimizerHook', 'DefaultRunnerConstructor',
    'LinearAnnealingMomentumUpdaterHook', 'LinearAnnealingLrUpdaterHook',
    'CheckpointLoader', '_load_checkpoint', '_load_checkpoint_with_prefix', 
    'load_checkpoint', 'load_state_dict', 'save_checkpoint', 'weights_to_cpu',
    'SubModulesEvalHook', 'SubModulesDistEvalHook'
]
