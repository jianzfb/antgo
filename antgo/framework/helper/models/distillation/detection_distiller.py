# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from antgo.framework.helper.runner import  load_checkpoint, _load_checkpoint, load_state_dict
# from antgo.framework.helper.base_module import *
# from antgo.framework.helper.models.detectors.multi_stream_detector import *
# from antgo.framework.helper.models.distillation.loss import *
# from ..builder import DISTILLER, build_distill_loss, build_model

# from collections import OrderedDict
# import copy


# @DISTILLER.register_module()
# class DetectionDistiller(MultiSteamDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """
    def __init__(self, student, teacher, train_cfg=None, test_cfg=None):
        super(DetectionDistiller, self).__init__(
            model=dict(
                student=build_model(student.model),
                teacher=build_model(teacher.model)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        load_checkpoint(self.teacher, teacher.pretrained, map_location='cpu')
        self.freeze("teacher")

        self.student.init_weights()
        if student.pretrained:
            load_checkpoint(self.student, student.pretrained, map_location='cpu')
        self.distill_losses = nn.ModuleDict()

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                    self.register_buffer(teacher_module,output)
            def hook_student_forward(module, input, output):
                    self.register_buffer(student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in train_cfg.distill:
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward, hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

    def init_weights(self):
        # do nothing
        pass

    def forward_train(self, 
                      image, 
                      image_meta, 
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
        super().forward_train(image, image_meta, **kwargs)
        student_loss = self.student.forward_train(image, image_meta, **kwargs)

        with torch.no_grad():
            # 此处提出的是不包含head的feature
            # 如果需要包含head则需要修改
            self.teacher.extract_feat(image)

        buffer_dict = dict(self.named_buffers())
        for item_loc in self.train_cfg.distill:
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat)

        return student_loss