from antgo.framework.helper.models.distillation.experts import Experts
from sys import maxsize
from numpy.lib.function_base import diff
import torch
from torch._C import device
from antgo.framework.helper.models.builder import DETECTORS, build_model
from antgo.framework.helper.models.utils.structure_utils import dict_stack, weighted_loss
from antgo.framework.helper.models.detectors.multi_stream_detector import MultiSteamDetector
from antgo.framework.helper.models.utils.box_utils import Transform2D, filter_invalid,random_ignore
from antgo.framework.helper.models.distillation.loss import *
from antgo.framework.helper.models.detectors.losses import iou_loss
from torchvision.ops import batched_nms 
import numpy as np
import cv2
import os
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence


@DETECTORS.register_module()
class DistillationTeacher(MultiSteamDetector):
    def __init__(self, student_model, expert_model, train_cfg=None, test_cfg=None):
        if not isinstance(expert_model, list):
            expert_model = [expert_model]
        expert_model = Experts(
            torch.nn.ModuleList([build_model(cfg) for cfg in expert_model])
        )

        super(DistillationTeacher, self).__init__(
            dict(
                teacher=build_model(student_model), 
                student=build_model(student_model),
                expert=expert_model
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        # 不允许teacher和student存在不一致的旋转
        # 冻结teacher权重 （teacher权重仅有student进行外部更新）
        self.freeze('teacher')
        self.is_expert_update = self.train_cfg.get('is_expert_update', False)
        if len(expert_model) == 1:
            self.is_expert_update = False
        if not self.is_expert_update:
            # 冻结专家权重
            self.freeze('expert')

        self.unsup_weight = self.train_cfg.get('unsup_weight', 0.0)
        self.debug = self.train_cfg.get('debug', False)
        self.target_size = self.train_cfg.get('target_size', (64, 80))
        self.background_class_in_cls = self.train_cfg.get('background_class_in_cls', False)
        self.distiller_ratio = self.train_cfg.get('distiller_ratio', 0.01)

    def forward_train(self, image, image_metas, **kwargs):
        super().forward_train(image, image_metas, **kwargs)
        
        data_groups = {}
        if 'labeled' in kwargs:
            data_groups['labeled'] = dict_stack(kwargs['labeled'], image_metas[0]['fields'])
        if 'unlabeled' in kwargs:
            data_groups['unlabeled'] = {}
            temp = dict_stack(kwargs['unlabeled'], ['teacher', 'student'])
            data_groups['unlabeled']['teacher'] = []
            teacher_num = len(temp['teacher']) // len(temp['student'])
            for teacher_i in range(teacher_num):
                data_groups['unlabeled']['teacher'].append(dict_stack(temp['teacher'][teacher_i::teacher_num], image_metas[0]['fields']))
            data_groups['unlabeled']['student'] = dict_stack(temp['student'], image_metas[0]['fields'])

        loss = {}

        if "labeled" in data_groups:
            # 使用标注数据，计算损失
            sup_loss = self.student.forward_train(**data_groups["labeled"])
            sup_loss = {"labeled_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unlabeled" in data_groups:
            # 使用无标住数据，计算损失
            teacher_data = data_groups['unlabeled']["teacher"]
            student_data = data_groups['unlabeled']["student"]
            tnames = [meta["image_file"] for meta in teacher_data[0]["image_metas"]]
            snames = [meta["image_file"] for meta in student_data["image_metas"]]
            tidx = [tnames.index(name) for name in snames]

            # compute teacher info
            context_grad = torch.enable_grad if self.is_expert_update else torch.no_grad
            with context_grad():
                teacher_info = self.extract_export_info(
                        [teacher_data[teacher_i]["image"][torch.Tensor(tidx).to(teacher_data[0]["image"].device).long()] for teacher_i in range(len(teacher_data))],
                        [[teacher_data[teacher_i]["image_metas"][idx] for idx in tidx] for teacher_i in range(len(teacher_data))]
                    )      
            # compute studentinfo
            student_info = self.extract_student_info(**student_data)

            unsup_loss = weighted_loss(
                self.compute_dense_loss(student_info, teacher_info),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unlabeled_" + k: v for k, v in unsup_loss.items()}
            loss.update(unsup_loss)

        return loss

    def QFLv2(self,
            pred_sigmoid,               # (n, 80)
            teacher_sigmoid,            # (n) 0, 1-80: 0 is neg, 1-80 is positive
            weight=None,
            beta=2.0,
            reduction='mean'):
        # all goes to 0
        pt = pred_sigmoid
        zerolabel = pt.new_zeros(pt.shape)
        loss = F.binary_cross_entropy(
            pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
        pos = weight > 0

        # positive goes to bbox quality
        pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
        loss[pos] = F.binary_cross_entropy(
            pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

        valid = weight >= 0
        if reduction == "mean":
            loss = loss[valid].mean()
        elif reduction == "sum":
            loss = loss[valid].sum()
        return loss

    def _coords(self, h,w, stride=1, device=None):
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=device)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])

        shift_x = shift_x + stride // 2
        shift_y = shift_y + stride // 2
        return shift_x, shift_y

    def compute_dense_loss(self, student_info, teacher_info):
        # head_feature is tuple
        # 0: NCHW表示类别预测的heatmap 
        # 1：N4HW表示ltrb delta_x0,delta_y0,delta_x1,delta_y1
        teacher_heatmap, teacher_delta = teacher_info['head_feature']
        teacher_heatmap, teacher_delta = teacher_heatmap[0], teacher_delta[0]
        student_heatmap, student_delta = student_info['head_feature']
        student_heatmap, student_delta = student_heatmap[0], student_delta[0]

        class_num = teacher_heatmap.shape[1]
        teacher_heatmap = torch.permute(teacher_heatmap, (0,2,3,1))
        teacher_heatmap = torch.reshape(teacher_heatmap, (-1, class_num))

        student_heatmap = torch.permute(student_heatmap, (0,2,3,1))
        student_heatmap = torch.reshape(student_heatmap, (-1, class_num))
        
        B,_,H,W = teacher_delta.shape
        teacher_delta = torch.permute(teacher_delta, (0,2,3,1))
        teacher_delta = torch.reshape(teacher_delta, (B, -1, 4))         # N, HW, 4

        student_delta = torch.permute(student_delta, (0,2,3,1))
        student_delta = torch.reshape(student_delta, (B, -1, 4))         # N, HW, 4

        coords_x, coords_y = self._coords(H,W, device=teacher_delta.device)                 # 1, HW, 1; 1, HW, 1
        coords_x = torch.reshape(coords_x, (1, -1, 1))
        coords_y = torch.reshape(coords_y, (1, -1, 1))
        teacher_tl_x = coords_x - teacher_delta[:,:,0:1]
        teacher_tl_y = coords_y - teacher_delta[:,:,1:2]
        teacher_br_x = coords_x + teacher_delta[:,:,2:3]
        teacher_br_y = coords_y + teacher_delta[:,:,3:4]
        teacher_reg_bbox = torch.concat([teacher_tl_x, teacher_tl_y, teacher_br_x, teacher_br_y], -1)
        teacher_reg_bbox = torch.reshape(teacher_reg_bbox, (-1, 4))

        student_tl_x = coords_x - student_delta[:,:,0:1]
        student_tl_y = coords_y - student_delta[:,:,1:2]
        student_br_x = coords_x + student_delta[:,:,2:3]
        student_br_y = coords_y + student_delta[:,:,3:4]
        student_reg_bbox = torch.concat([student_tl_x, student_tl_y, student_br_x, student_br_y], -1)
        student_reg_bbox = torch.reshape(student_reg_bbox, (-1, 4))

        with torch.no_grad():
            # Region Selection
            # TODO, 从每张图中挑选
            count_num = int(teacher_heatmap.size(0) * self.distiller_ratio)
            max_vals = torch.max(teacher_heatmap, 1)[0]
            sorted_vals, sorted_inds = torch.topk(max_vals, teacher_heatmap.size(0))
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask=mask>0.

        loss_logits = self.QFLv2(
            student_heatmap,
            teacher_heatmap,
            weight=mask,
            reduction="sum",
        ) / fg_num

        loss_deltas = iou_loss(
            student_reg_bbox[b_mask],
            teacher_reg_bbox[b_mask]
        )

        loss = {
            "distill_loss_logits": loss_logits,
            "distill_loss_regs": loss_deltas,
        }

        if self.is_expert_update:
            expert_loss_logits = []
            for expert_i in range(self.expert.size):
                expert_heatmap, expert_delta = teacher_info['expert_head_feature'][expert_i]
                expert_heatmap, expert_delta = expert_heatmap[0], expert_delta[0]

                expert_loss_logits.append(self.QFLv2(
                    expert_heatmap,             # 专家预测结果
                    teacher_heatmap,            # 专家聚合结果
                    weight=mask,
                    reduction="sum",
                ) / fg_num)

            expert_loss_logits = torch.mean(expert_loss_logits)
            loss.update(
                {
                    'expert_distill_loss_logits': expert_loss_logits
                }
            )
        
        return loss

    def extract_student_info(self, image, image_metas, **kwargs):
        # 注意这里输入的是无标签数据
        student_info = {}
        student_info["image"] = image
        backbone_feature = self.student.extract_feat(image)
        head_feature = self.student.bbox_head(backbone_feature)
        student_info["backbone_feature"] = backbone_feature
        student_info["head_feature"] = head_feature
        student_info["image_metas"] = image_metas
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(backbone_feature[0][0].device)
            for meta in image_metas
        ]
        return student_info

    def extract_export_info(self, image, image_metas, **kwargs):
        # 注意这里输入的是无标签数据
        teacher_info = {}
        if not isinstance(image, list):
            image = [image]
            image_metas = [image_metas]
        
        ensemble_outs = None                                # 多模型多输入聚合输出（multi input and multi models）
        expert_outs = [[] for _ in range(len(self.expert))] # 每个专家的输出（多数据聚合输出）

        for img, img_metas in zip(image, image_metas):
            # 假定输入数据不存在crop,rotation,translation等空间变换操作
            expert_outs_list = self.export.forward_dummy(img)
            outs = self.expert.mean(expert_outs_list)

            for expert_i in range(len(self.expert)):
                expert_outs[expert_i].append(expert_outs_list[expert_i])
            
            if ensemble_outs is None:
                ensemble_outs = list(outs)
            else:
                for t_i in range(len(outs)):
                    ensemble_outs[t_i][0] += outs[t_i][0]

        for t_i in range(len(ensemble_outs)):
            ensemble_outs[t_i][0] /= len(image)
        
        for expert_i in range(len(self.expert)):
            expert_outs[expert_i] = self.expert.mean(expert_outs[expert_i])

        teacher_info["image"] = image
        teacher_info["head_feature"] = ensemble_outs                      # 关键featuremap组合（可使用其对student进行监督）
        teacher_info["expert_head_feature"] = expert_outs
        return teacher_info

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        for teacher_i in range(len(self.teacher)):
            pass
        
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

'''
        if self.debug:
            image_metas = image_metas[0]
            results_list = \
                self.teacher.bbox_head.get_bboxes(
                    *aug_outs, image_metas, rescale=False, with_nms=False)

            bbox_results = {
                'box': torch.stack([a for a, _, _, _ in results_list], dim=0),
                'label': torch.stack([b for _, b, _, _ in results_list], dim=0),
            }
            # clip by position
            image_h = image_metas[0]['image_shape'][0]
            image_w = image_metas[0]['image_shape'][1]
            image_wh = torch.from_numpy(np.array([[[image_w-1, image_h-1]]])).to(feat[0].device)
            bbox_results['box'][:,:,0] = torch.maximum(bbox_results['box'][:,:,0], torch.zeros_like(bbox_results['box'][:,:,0]))
            bbox_results['box'][:,:,1] = torch.maximum(bbox_results['box'][:,:,1], torch.zeros_like(bbox_results['box'][:,:,1]))
            bbox_results['box'][:,:,2] = torch.minimum(bbox_results['box'][:,:,2], image_wh[:,:,0])
            bbox_results['box'][:,:,3] = torch.minimum(bbox_results['box'][:,:,3], image_wh[:,:,1])

            # 
            pseudo_bbox_candidates = bbox_results['box']          # BxNx5 (x0,y0,x1,y1,score)
            paeudo_label_candidates = bbox_results['label']       # BxN

            pseudo_bbox_filter_list = []
            pseudo_label_filter_list = []
            for img_id in range(len(image_metas)):
                keep = batched_nms(
                    pseudo_bbox_candidates[img_id, :, :4], 
                    pseudo_bbox_candidates[img_id, :, -1].contiguous(),
                    paeudo_label_candidates[img_id], 
                    0.8)        
                
                bboxes = pseudo_bbox_candidates[img_id, keep]
                labels = paeudo_label_candidates[img_id, keep]
                pseudo_bbox_filter_list.append(bboxes)
                pseudo_label_filter_list.append(labels)

            pseudo_bbox_list, pseudo_label_list, _, _, _ = list(
                zip(
                    *[
                        filter_invalid(
                            pseudo_bbox,
                            pseudo_label,
                            pseudo_bbox[:, -1],
                            around_bbox=None,
                            index=None,
                            thr=0.5,
                            box_min_size=None,
                            aspect_ratio=None,
                            class_constraint=None
                        )
                        for pseudo_bbox, pseudo_label in zip(
                            pseudo_bbox_filter_list, pseudo_label_filter_list
                        )
                    ]
                )
            )      

            image_metas = image_metas[0]
            teacher_info["pseudo_bboxes"] = pseudo_bbox_list
            teacher_info["pseudo_labels"] = pseudo_label_list 
            teacher_info["image_metas"] = image_metas




        if self.debug:
            # 检查teacher中获得的检测狂，是否可以通过M转换到 student的图像中
            if not os.path.exists('./debug'):
                os.makedirs('./debug')
            for i in range(len(student_info['image_metas'])):
                image = student_info['image'][i] * 255
                image = image.detach().cpu().numpy()
                image = image[0].astype(np.uint8)
                bboxs = teacher_info["pseudo_bboxes"][i]
                bboxs = bboxs.detach().cpu().numpy()
                bboxs_label = teacher_info["pseudo_labels"][i].cpu().numpy()
                image = np.stack([image,image,image], -1)
                for bbox_i, bbox in enumerate(bboxs):
                    x0,y0,x1,y1,score = bbox
                    label = bboxs_label[bbox_i]
                    color = (255,0,0)
                    if label == 1:
                        color = (0,0,255)
                    cv2.rectangle(image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), color, 2)
                    cv2.putText(image, f'score {score}', ((int)(x0),(int)(y0-5)), cv2.FONT_HERSHEY_PLAIN, 1, color , 1)

                teacher_image = teacher_info['image'][0][i] * 255
                teacher_image = teacher_image.detach().cpu().numpy()
                teacher_image = teacher_image[0].astype(np.uint8)    
                teacher_image = np.stack([teacher_image,teacher_image,teacher_image], -1)
                teacher_bboxs = teacher_info["pseudo_bboxes"][i]
                teacher_bboxs = teacher_bboxs.detach().cpu().numpy()
                for bbox_i, bbox in enumerate(teacher_bboxs):
                    x0,y0,x1,y1,score = bbox
                    label = bboxs_label[bbox_i]
                    color = (255,0,0)
                    if label == 1:
                        color = (0,0,255)
                    cv2.rectangle(teacher_image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), color, 2)
                    cv2.putText(teacher_image, f'score {score}', ((int)(x0),(int)(y0-5)), cv2.FONT_HERSHEY_PLAIN, 1, color , 1)

                student_teacher_image = np.concatenate([image, teacher_image], 0)
                cv2.imwrite(f'./debug/student_teacher_{i}.png', student_teacher_image)

                i_teacher_heatmap = torch.reshape(teacher_heatmap, (B, -1, class_num))[i]
                max_vals = torch.max(i_teacher_heatmap, 1)[0]
                sorted_vals, sorted_inds = torch.topk(max_vals, i_teacher_heatmap.size(0))
                i_mask = torch.zeros_like(max_vals)
                i_mask[sorted_inds[:count_num]] = 1.
                i_b_mask=i_mask>0.

                i_teacher_reg_bbox = torch.reshape(teacher_reg_bbox, (B,-1, 4))[i]
                pseudo_teacher_reg_bbox = i_teacher_reg_bbox[i_b_mask].detach().cpu().numpy()
                teacher_image = teacher_info['image'][0][i] * 255
                teacher_image = teacher_image.detach().cpu().numpy()
                teacher_image = teacher_image[0].astype(np.uint8)    
                teacher_image = np.stack([teacher_image,teacher_image,teacher_image], -1)                
                teacher_image_h, teacher_image_w = teacher_image.shape[:2]
                for bbox in pseudo_teacher_reg_bbox:
                    x0,y0,x1,y1 = bbox
                    x0 = x0 * (teacher_image_w/self.target_size[1])
                    y0 = y0 * (teacher_image_h/self.target_size[0])
                    x1 = x1 * (teacher_image_w/self.target_size[1])
                    y1 = y1 * (teacher_image_h/self.target_size[0])
                    color = (255,0,0)
                    cv2.rectangle(teacher_image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), color, 2)

                i_student_reg_bbox = torch.reshape(student_reg_bbox, (B,-1, 4))[i]
                pseudo_student_reg_bbox = i_student_reg_bbox[i_b_mask].detach().cpu().numpy()
                student_image = student_info['image'][i] * 255
                student_image = student_image.detach().cpu().numpy()
                student_image = student_image[0].astype(np.uint8)    
                student_image = np.stack([student_image,student_image,student_image], -1)                
                for bbox_i, bbox in enumerate(pseudo_student_reg_bbox):
                    x0,y0,x1,y1 = bbox
                    x0 = x0 * (teacher_image_w/self.target_size[1])
                    y0 = y0 * (teacher_image_h/self.target_size[0])
                    x1 = x1 * (teacher_image_w/self.target_size[1])
                    y1 = y1 * (teacher_image_h/self.target_size[0])                    
                    color = (255,0,0)
                    cv2.rectangle(student_image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), color, 2)

                student_teacher_image = np.concatenate([student_image, teacher_image], 0)
                cv2.imwrite(f'./debug/student_teacher_v2_{i}.png', student_teacher_image)

'''