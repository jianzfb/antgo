import torch.nn as nn
import torch.nn.functional as F
import torch
# from mmdet.models.detectors.base import BaseDetector
# from mmdet.models import build_detector
from antgo.framework.helper.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from antgo.framework.helper.base_module import *
from antgo.framework.helper.models.distillation.loss import *
from ..builder import DISTILLER, build_distill_loss, build_model

from collections import OrderedDict
import copy


@DISTILLER.register_module()
class DetectionDistiller(BaseModule):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False):

        super(DetectionDistiller, self).__init__()
        self.teacher = build_model(teacher_cfg.model)
        if teacher_pretrained is not None:
            self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student = build_model(student_cfg.model)
        self.student.init_weights()
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
            
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg   

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                    self.register_buffer(teacher_module,output)
            def hook_student_forward(module, input, output):
                    self.register_buffer(student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')


    def forward_train(self, 
                      img, 
                      img_metas, 
                      **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        student_loss = self.student.forward_train(img, img_metas, **kwargs)

        with torch.no_grad():
            # 此处提出的是不包含head的feature
            # 如果需要包含head则需要修改
            fea_t = self.teacher.extract_feat(img)

        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat)

        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)