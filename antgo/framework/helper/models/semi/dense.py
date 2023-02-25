import torch
from antgo.framework.helper.models.builder import DETECTORS, build_model
from antgo.framework.helper.models.utils.structure_utils import dict_stack, weighted_loss
from antgo.framework.helper.multi_stream_module import MultiSteamModule
from antgo.framework.helper.models.utils.box_utils import Transform2D, filter_invalid,random_ignore
from torchvision.ops import batched_nms 
import numpy as np
import cv2
import os
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence
from losses.quality_focal_loss import QualityFocalLoss


@DETECTORS.register_module()
class DenseTeacher(MultiSteamModule):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        # 默认使用teacher模型作为最佳模型
        test_cfg.update(
            {
                'inference_on': 'teacher'
            }
        )
        super(DenseTeacher, self).__init__(
            dict(
                teacher=build_model(model), 
                student=build_model(model)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        self.focus_key = ''
        self.focus_i = 0
        self.use_sigmoid = True

        self.semi_ratio = 0.5
        self.heatmap_n_thr = 0.25
        self.semi_loss_w = 1

    def _get_unsup_dense_loss(self, student_heatmap_, teacher_heatmap_):
        # student_heatmap_: N,C,H,W
        # teacher_heatmap_: N,C,H,W
        student_heatmap = student_heatmap_.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        teacher_heatmap = teacher_heatmap_.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        if self.use_sigmoid:
            student_heatmap = student_heatmap.sigmoid()
            teacher_heatmap = teacher_heatmap.sigmoid()

        with torch.no_grad():
            max_vals = torch.max(teacher_heatmap, 1)[0]
            count_num = int(teacher_heatmap.size(0) * self.semi_ratio)
            sorted_vals, sorted_inds = torch.topk(max_vals, teacher_heatmap.size(0))

            for sorted_n in range(count_num):
                if sorted_vals[sorted_n] < self.heatmap_n_thr:
                    count_num = max(1, sorted_n)
                    break

            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.0
            fg_num = sorted_vals[:count_num].sum()

        loss_heatmap = (
            QualityFocalLoss(
                student_heatmap,
                teacher_heatmap,
                weight=mask,
                reduction="mean",
            )
        )

        return {"loss_heatmap": loss_heatmap}

    def forward_train(self, images, image_metas, index, **kwargs):
        super().forward_train(images, image_metas, **kwargs)

        # 需要保证可以同时传入有标签数据和无标签数据
        # index 记录样本源
        


        # label_images, label_metas
        # unlabel_strong_images, unlabel_strong_metas
        # unlabel_weak_images, unlabel_weak_metas
        label_images = None
        label_metas = None

        unlabel_strong_images = None
        unlabel_strong_metas = None
        unlabel_weak_images = None
        unlabel_weak_metas = None

        losse = {}
        # 有监督损失
        output_dict = self.student.forward_train(label_images, label_metas)
        assert(isinstance(output_dict, dict))
        output_dict = {"labeled_" + k: v.cpu().item() for k, v in output_dict.items()}
        losses.update(**output_dict)

        # 无监督损失
        # 应该返回 heatmap
        output_unsup_strong = self.student.forward_train(unlabel_strong_images, unlabel_strong_metas)
        strong_heatmap = None
        if isinstance(output_unsup_strong, dict):
            strong_heatmap = output_unsup_strong[self.focus_key]
        else:
            strong_heatmap = output_unsup_strong[self.focus_i]

        # 应该返回 heatmap
        output_unsup_weak = self.teacher.forward_train(unlabel_weak_images, unlabel_weak_metas)
        weak_heatmap = None
        if isinstance(output_unsup_weak, dict):
            weak_heatmap = output_unsup_weak[self.focus_key]
        else:
            weak_heatmap = output_unsup_weak[self.focus_i]

        unsup_loss = self._get_unsup_dense_loss(
            student_heatmap,
            teacher_heatmap
        )
        unsup_loss_sum = (
            sum([metrics_value for metrics_value in unsup_loss.values() if metrics_value.requires_grad])
            * self.semi_loss_w
        )
        unsup_loss = {"unlabeled_" + k: v.item() for k, v in unsup_loss.items()}
        losses.update(**unsup_loss)

        return losses
    
    def simple_test(self, images, image_metas, rescale=True, **kwargs):
        return self.teacher(images, image_metas, rescale, **kwargs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs):
        # 同时兼容从base模型和dense模型加载
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )