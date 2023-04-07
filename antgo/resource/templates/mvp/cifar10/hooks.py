from antgo.framework.helper.runner.hooks.hook import HOOKS, Hook

# @HOOKS.register_module()
# class YourHook(Hook):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
    
#     def before_run(self, runner):
#         # 在全局迭代前，运行一次
#         pass
#     def after_run(self, runner):
#         # 在全局推出前，运行一次
#         pass
    
#     def before_train_epoch(self, runner):
#         # 在每个epoch前，运行一次
#         pass
#     def after_train_epoch(self, runner):
#         # 在每个epoch后，运行一次
#         pass
    
#     def before_train_iter(self, runner):
#         # 在每次迭代前，运行一次
#         pass
#     def after_train_iter(self, runner):
#         # 在每次迭代后，运行一次
#         pass