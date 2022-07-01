from abc import ABCMeta, abstractmethod

import torch
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.models.builder import *

class ProxyModule(BaseModule):
    def __init__(self, feat_module, head_cfg, train_cfg=None, test_cfg=None, init_cfg=None) -> None:
        super().__init__(init_cfg)
        if train_cfg is not None:
            head_cfg.update({'train_cfg':train_cfg})
        if test_cfg is not None:
            head_cfg.update({'test_cfg':test_cfg})

        head_init_cfg = dict()
        if 'init_cfg' in head_cfg and head_cfg['init_cfg'] is not None:
            head_init_cfg = head_cfg['init_cfg']
        head_init_cfg['compute_feature'] = False
        head_cfg.update({
            'init_cfg': head_init_cfg
        })

        self.head_module = build_head(head_cfg)
        self.feat_module = feat_module

    def forward(self, *data, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return_loss = True
        if 'return_loss' in kwargs:
            return_loss = kwargs['return_loss']

        if return_loss:
            return self.forward_train(*data, **kwargs)
        else:
            return self.forward_test(*data, **kwargs)

    def forward_test(self, *data, **kwargs):
        feat = self.feat_module(kwargs['image'])
        kwargs.update({
            'feature': feat
        })
 
        results_list = self.head_module.simple_test(
            None, **kwargs)

        results = results_list
        if self.init_cfg.get('task', None) != None:
            if self.init_cfg.get('task') == 'det':
                results = {
                    'box': torch.stack([a for a, _ in results_list], dim=0),
                    'label': torch.stack([b for _, b in results_list], dim=0)
                }
        return results

    def forward_train(self, *data, **kwargs):
        feat = self.feat_module(kwargs['image'])
        kwargs.update({
            'feature': feat
        })

        return self.head_module.forward_train(None, **kwargs)
