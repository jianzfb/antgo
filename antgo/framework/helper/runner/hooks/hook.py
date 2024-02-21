from antgo.framework.helper.utils import Registry, is_method_overridden
import functools
HOOKS = Registry('hook')


class Hook:
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner):
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner):
        return runner.iter + 1 == runner._max_iters

    def get_triggered_stages(self):
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)

        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            'before_epoch': ['before_train_epoch', 'before_val_epoch'],
            'after_epoch': ['after_train_epoch', 'after_val_epoch'],
            'before_iter': ['before_train_iter', 'before_val_iter'],
            'after_iter': ['after_train_iter', 'after_val_iter'],
        }

        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        return [stage for stage in Hook.stages if stage in trigger_stages]


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rsetattr(obj, attr, val):
    """Set the value of a nested attribute of an object.

    This function splits the attribute path and sets the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute until it reaches the last one and sets
    its value.

    Args:
        obj (object): The object whose attribute needs to be set.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        val (any): The value to set at the specified attribute path.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively get a nested attribute of an object.

    This function splits the attribute path and retrieves the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute. If an attribute in the path does not
    exist, it returns the value specified as the third argument.

    Args:
        obj (object): The object whose attribute needs to be retrieved.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        *args (any): Optional default value to return if the attribute
            does not exist.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))