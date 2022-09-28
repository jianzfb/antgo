from sys import maxsize
from numpy.lib.function_base import diff
import torch
from torch._C import device
from antgo.framework.helper.models.builder import DETECTORS, build_model
from antgo.framework.helper.models.utils.structure_utils import dict_stack, weighted_loss
from antgo.framework.helper.models.multi_stream_detector import MultiSteamDetector
from antgo.framework.helper.models.utils.box_utils import Transform2D, filter_invalid,random_ignore
from torchvision.ops import batched_nms 
import numpy as np
import cv2
import os


@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacher, self).__init__(
            dict(
                teacher=build_model(model), 
                student=build_model(model)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        self.unsup_weight = self.train_cfg.unsup_weight
        self.debug = self.train_cfg.get('debug', False)
        self.target_size = self.train_cfg.get('target_size', (64, 80))
        self.enable_random_ignore = self.train_cfg.get('random_ignore', False)

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
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups['unlabeled']["teacher"], data_groups['unlabeled']["student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unlabeled_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["image_file"] for meta in teacher_data[0]["image_metas"]]
        snames = [meta["image_file"] for meta in student_data["image_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                [teacher_data[teacher_i]["image"][torch.Tensor(tidx).to(teacher_data[0]["image"].device).long()] for teacher_i in range(len(teacher_data))],
                [[teacher_data[teacher_i]["image_metas"][idx] for idx in tidx] for teacher_i in range(len(teacher_data))]
            )      
        student_info = self.extract_student_info(**student_data)
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        #  student,teacher 面对同一张图，但是可能使用了不同的几何旋转
        # 获得从teacher到student的每个样本的变换矩阵
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        # 将从teacher中获得的伪框，转换到student的图上
        # 已经检查通过，经过M变换，可以将teacher获得的框转换到student上
        pseudo_bboxes = self._transform_bbox(
            teacher_info["pseudo_bboxes"],
            M,
            [meta["image_shape"] for meta in student_info["image_metas"]],
        )
        
        image_h = student_info['image_metas'][0]['image_shape'][0]
        image_w = student_info['image_metas'][0]['image_shape'][1]
        image_wh = torch.from_numpy(np.array([[image_w-1, image_h-1]])).to(pseudo_bboxes[0].device)
        for bi in range(len(pseudo_bboxes)):
            pseudo_bboxes[bi][:,0] = torch.maximum(pseudo_bboxes[bi][:,0], torch.zeros_like(pseudo_bboxes[bi][:,0]))
            pseudo_bboxes[bi][:,1] = torch.maximum(pseudo_bboxes[bi][:,1], torch.zeros_like(pseudo_bboxes[bi][:,1]))
            pseudo_bboxes[bi][:,2] = torch.minimum(pseudo_bboxes[bi][:,2], image_wh[:,0])
            pseudo_bboxes[bi][:,3] = torch.minimum(pseudo_bboxes[bi][:,3], image_wh[:,1])

            # filter
            valid_1 = pseudo_bboxes[bi][:,2] > pseudo_bboxes[bi][:,0]
            valid_2 = pseudo_bboxes[bi][:,3] > pseudo_bboxes[bi][:,1]
            valid = valid_1 & valid_2
            pseudo_bboxes[bi] = pseudo_bboxes[bi].index_select(0,torch.nonzero(valid)[:,0])

        pseudo_labels = list(teacher_info["pseudo_labels"])
        for bi in range(len(pseudo_bboxes)):
            if pseudo_labels[bi].shape[0] > 0:
                if 'flipped' in student_info['image_metas'][bi] and student_info['image_metas'][bi]['flipped']:
                    pseudo_labels[bi] = 1-pseudo_labels[bi]

        if self.debug:
            # 检查teacher中获得的检测狂，是否可以通过M转换到 student的图像中
            if not os.path.exists('./debug'):
                os.makedirs('./debug')
            for i in range(len(student_info['image_metas'])):
                image = student_info['image'][i] * 255
                image = image.detach().cpu().numpy()
                image = image[0].astype(np.uint8)
                bboxs = pseudo_bboxes[i]
                bboxs = bboxs.detach().cpu().numpy()
                bboxs_label = pseudo_labels[i].cpu().numpy()
                image = np.stack([image,image,image], -1)
                for bbox_i, bbox in enumerate(bboxs):
                    x0,y0,x1,y1,score,_ = bbox
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
                    x0,y0,x1,y1,score,_ = bbox
                    label = bboxs_label[bbox_i]
                    if 'flipped' in student_info['image_metas'][i] and student_info['image_metas'][i]['flipped']:
                        label = 1-label
                    color = (255,0,0)
                    if label == 1:
                        color = (0,0,255)
                    cv2.rectangle(teacher_image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), color, 2)
                    cv2.putText(teacher_image, f'score {score}', ((int)(x0),(int)(y0-5)), cv2.FONT_HERSHEY_PLAIN, 1, color , 1)

                student_teacher_image = np.concatenate([image, teacher_image], 0)
                cv2.imwrite(f'./debug/student_teacher_{i}.png', student_teacher_image)

        loss = {}
        # 使用伪标签，构建student分类损失
        loss.update(
            self.unsup_loss(
                student_info, pseudo_bboxes, pseudo_labels
            )
        )
        return loss

    def unsup_loss(self, student_info, pseudo_bboxes, pseudo_labels):
        # pseudo_bboxes: x0,y0,x1,y1,score,reg_unc
        # pseudo_bboxes: [Nx5,...]
        # pseudo_labels: [N,...]
        image_metas = student_info['image_metas']
        backbone_feature = student_info['backbone_feature']
        outs = self.student.bbox_head(backbone_feature)
        loss_inputs = outs + (pseudo_bboxes, pseudo_labels, image_metas)
        loss = self.student.bbox_head.loss(*loss_inputs)
        return loss

    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, image, image_metas, **kwargs):
        # 注意这里输入的是无标签数据
        student_info = {}
        student_info["image"] = image
        feat = self.student.extract_feat(image)
        student_info["backbone_feature"] = feat
        student_info["image_metas"] = image_metas
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in image_metas
        ]
        return student_info

    def extract_teacher_info(self, image, image_metas, **kwargs):
        # 注意这里输入的是无标签数据
        teacher_info = {}
        if not isinstance(image, list):
            image = [image]
            image_metas = [image_metas]
        
        aug_outs = None
        for img, img_metas in zip(image, image_metas):
            # 假定输入数据不存在crop,rotation,translation等空间变换操作
            feat = self.teacher.extract_feat(img)
            outs = self.teacher.bbox_head(feat)

            if self.target_size is not None:
                for t_i in range(len(outs)):
                    heat_h, heat_w = outs[t_i][0].shape[2:]
                    if heat_h != self.target_size[0] or heat_w != self.target_size[1]:
                        outs[t_i][0] = torch.nn.functional.interpolate(outs[t_i][0], size=(self.target_size[0], self.target_size[1]), mode='bilinear')

            if aug_outs is None:
                aug_outs = list(outs)
            else:
                for t_i in range(len(outs)):
                    aug_outs[t_i][0] += outs[t_i][0]

        for t_i in range(len(aug_outs)):
            aug_outs[t_i][0] /= len(image)
            
        image_metas = image_metas[0]
        results_list = self.teacher.bbox_head.get_bboxes(*aug_outs, image_metas, rescale=False, with_nms=False)

        bbox_results = {
            'box': torch.stack([a for a, _, _, _ in results_list], dim=0),
            'label': torch.stack([b for _, b, _, _ in results_list], dim=0),
            'around_box': torch.stack([c for _, _, c, _ in results_list], dim=0),
            'mask': [d for _, _, _, d in results_list],
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
        pseudo_around_bbox_candidates = bbox_results['around_box']  # BxNxMx4
        paeudo_label_candidates = bbox_results['label']       # BxN
        pseudo_index_candidates = torch.arange(pseudo_bbox_candidates.shape[1])

        pseudo_bbox_filter_list = []
        pseudo_label_filter_list = []
        pseudo_index_filter_list = []
        pseudo_around_bbox_filter_list = []
        # 过滤1：使用nms
        for img_id in range(len(image_metas)):
            keep = batched_nms(
                pseudo_bbox_candidates[img_id, :, :4], 
                pseudo_bbox_candidates[img_id, :, -1].contiguous(),
                paeudo_label_candidates[img_id], 
                0.8)        
            
            bboxes = pseudo_bbox_candidates[img_id, keep]
            labels = paeudo_label_candidates[img_id, keep]
            around_bboxes = pseudo_around_bbox_candidates[img_id, keep]
            keep_index = pseudo_index_candidates[keep]

            pseudo_bbox_filter_list.append(bboxes)
            pseudo_label_filter_list.append(labels)
            pseudo_around_bbox_filter_list.append(around_bboxes)
            pseudo_index_filter_list.append(keep_index)

        # 过滤2：基于自定义规则过滤
        # 分数，面积，长宽比
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        
        aspect_ratio = self.train_cfg.get('pseduo_box_aspect_ratio', None)  # 宽高比
        box_min_size = self.train_cfg.get('pseduo_box_min_size', None)      # 最小尺寸
        class_constraint = self.train_cfg.get('class_constraint', None)     # 类别个数限制
        pseudo_bbox_list, pseudo_label_list, _, pseudo_index_list, pseudo_around_bbox_list = list(
            zip(
                *[
                    filter_invalid(
                        pseudo_bbox,
                        pseudo_label,
                        pseudo_bbox[:, -1],
                        around_bbox=pseudo_around_bbox,
                        index=pseudo_index,
                        thr=thr,
                        box_min_size=box_min_size,
                        aspect_ratio=aspect_ratio,
                        class_constraint=class_constraint
                    )
                    for pseudo_bbox, pseudo_label, pseudo_index, pseudo_around_bbox in zip(
                        pseudo_bbox_filter_list, pseudo_label_filter_list, pseudo_index_filter_list, pseudo_around_bbox_filter_list
                    )
                ]
            )
        )        
        
        # 过滤3：随机忽略目标框（防止，错误框参与训练后，无法剔除）
        if self.enable_random_ignore:
            pseudo_bbox_list, pseudo_label_list, _, pseudo_index_list, pseudo_around_bbox_list = list(
                zip(
                    *[
                        random_ignore(
                            pseudo_bbox,
                            pseudo_label,
                            pseudo_bbox[:, -1],
                            around_bbox=pseudo_around_bbox,
                            index=pseudo_index,
                        )
                        for pseudo_bbox, pseudo_label, pseudo_index, pseudo_around_bbox in zip(
                            pseudo_bbox_list, pseudo_label_list, pseudo_index_list, pseudo_around_bbox_list
                        )
                    ]
                )
            )        

        # 过滤4：控制存在目标区域
        background_constraint = self.train_cfg.get('background_constraint', 0.01)   # 低于此值认为是纯背景
        background_weight = self.train_cfg.get('background_weight', 1.0)            # 纯背景区域权重
        for bi in range(len(results_list['mask'])):
            # MASK 说明：越接近0 背景概率越大；越接近1 前景概率越大
            mask_weight = results_list['mask'][bi] < background_constraint
            mask_weight = mask_weight.float()
            if background_weight >= 0.0:
                mask_weight = mask_weight * background_weight
            else:
                mask_weight = mask_weight * (1.0-results_list['mask'][bi])
            
            mask_weight = mask_weight.view(*mask_weight.shape[2:])
            
            # 将mask写入meta中
            image_metas[bi]['background_weight'] = mask_weight

        # 计算统计框的不稳定性        
        reg_unc = \
            self.compute_location_uncertainty(
                pseudo_bbox_list, 
                pseudo_index_list, 
                pseudo_around_bbox_list)
        det_bboxes = pseudo_bbox_list

        # x0,y0,x1,y1,score (分数), uncertaity (稳定性)
        det_bboxes = [
            torch.cat([bbox, torch.unsqueeze(unc,-1)], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = pseudo_label_list
        teacher_info["pseudo_bboxes"] = det_bboxes
        teacher_info["pseudo_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in image_metas
        ]
        teacher_info["image_metas"] = image_metas
        teacher_info["image"] = image
        return teacher_info

    def compute_location_uncertainty(self, pseudo_bbox_list, pseudo_index_list, pseudo_around_bbox_list):
        # 在中心点附近的 目标框回归的所有框，计算不确定度
        pseudo_bbox_uncertainty_list = []
        for img_id in range(len(pseudo_bbox_list)):
            # pseudo_bbox, pseudo_index 经过NMS后的挑选出来的框和索引
            # 挑选出来得伪框
            pseudo_bbox = pseudo_bbox_list[img_id]          # [[x0,y0,x1,y1]]
            pseudo_index = pseudo_index_list[img_id]            
            
            pseudo_around_bbox = pseudo_around_bbox_list[img_id]        # [[x0,y0,x1,y1],[x0,y0,x1,y1],[x0,y0,x1,y1],...]
            if pseudo_bbox.size(0) == 0:
                pseudo_bbox_uncertainty_list.append(torch.empty((0)).to(pseudo_bbox.device))              # 无限不确定古
                continue

            # 计算回归框的方差
            # N,8,4
            diff = pseudo_bbox[:,:4].unsqueeze(1) - pseudo_around_bbox[:,:,:4]
            # N,4
            diff = torch.std(diff, 1)

            # Z
            pseudo_wh_denominator = (pseudo_bbox[:,3]-pseudo_bbox[:,1]) + (pseudo_bbox[:,2]-pseudo_bbox[:,0])
            pseudo_wh_denominator = 0.5 * pseudo_wh_denominator
            pseudo_wh_denominator = pseudo_wh_denominator.view(-1,1)
            diff = diff / (pseudo_wh_denominator + 1e-6)
            diff = torch.mean(diff)

            pseudo_bbox_uncertainty_list.append(diff)
        
        return pseudo_bbox_uncertainty_list

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

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )