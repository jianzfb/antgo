import torch
import torch.nn as nn
from antgo.framework.helper.cnn import bias_init_with_prob, normal_init
from antgo.framework.helper.models.builder import HEADS, build_loss
from antgo.framework.helper.models.detectors.core.utils import multi_apply
from torchvision.ops import batched_nms 
from ..utils.gaussian_target import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
import torch.nn.functional as F


@HEADS.register_module()
class FcosHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>
    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 rescale=1.0,
                 score_thresh=0.15,
                 loss_ch=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(FcosHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.rescale_x, self.rescale_y = rescale if type(rescale) == list or type(rescale) == tuple else (rescale, rescale)
        self.score_thresh = score_thresh
        self.in_channel = in_channel  
        self.feat_channel = feat_channel
        self._build_head()
        self.loss_center_heatmap=build_loss(loss_ch)
        self.loss_reg = build_loss(loss_wh)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _build_head(self):
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.in_channel, self.feat_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channel, self.num_classes, kernel_size=1, bias=True))

        self.reg_head = nn.Sequential(
            nn.Conv2d(self.in_channel,self.feat_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channel, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True))

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feats[0])
        reg_pred = self.reg_head(feats[0])
        return center_heatmap_pred, reg_pred

    def loss(self,
             center_heatmap_pred,
             reg_pred,
             bboxes,
             labels,
             image_meta,
             gt_bboxes_ignore=None):
        target_result, avg_factor = \
            self.get_targets(bboxes, labels, center_heatmap_pred.shape, image_meta, center_heatmap_pred.device)

        center_heatmap_target = target_result['center_heatmap_target']
        reg_targets = target_result['reg_targets']
        reg_weights = target_result['reg_weights']

        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred.sigmoid(), center_heatmap_target, avg_factor=avg_factor)
        loss_reg_v = self.loss_reg(
            reg_pred,
            reg_targets,
            reg_weights,
            avg_factor=avg_factor*2)

        # 
        total_loss = dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_reg=loss_reg_v)

        return total_loss

    def coords_fmap(self, h, w):
        stride = 1
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        coords = torch.stack([shift_x, shift_y], -1) + stride // 2
        return coords

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, image_meta, device):
        """Compute regression and classification targets in multiple images.
        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_shape (list[int]): image shape in [h, w] format.
        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h = image_meta[0]['image_shape'][0]
        img_w = image_meta[0]['image_shape'][1]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = torch.zeros([bs, self.num_classes, feat_h, feat_w], dtype=torch.float32, device=device)
        reg_target_list = []
        reg_weight_list = []
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]       # gt_bbox shape: Nx4
            if gt_bbox.shape[0] == 0:
                # 防止无目标样本
                reg_target_list.append(torch.zeros([4, feat_h, feat_w], dtype=torch.float32, device=device))
                reg_weight_list.append(torch.zeros([1, feat_h, feat_w], dtype=torch.float32, device=device))
                continue
            
            coords = self.coords_fmap(h=feat_h, w=feat_w).to(device=center_heatmap_target.device) #[h*w,2]
            x = coords[:, 0]
            y = coords[:, 1]
            l_off = x[:,None] - gt_bbox[...,0][None,:]*width_ratio    #[h*w,1]-[1,m]-->[h*w,m]
            t_off = y[:,None] - gt_bbox[...,1][None,:]*height_ratio
            r_off = gt_bbox[...,2][None,:]*width_ratio - x[:,None]
            b_off = gt_bbox[...,3][None,:]*height_ratio - y[:,None]
            ltrb_off = torch.stack([l_off,t_off,r_off,b_off],dim=-1)                        #[h*w,m,4]
            areas = (ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])     #[h*w,m]

            off_min = torch.min(ltrb_off,dim=-1)[0]     #[h*w,m]
            off_max = torch.max(ltrb_off,dim=-1)[0]     #[h*w,m]

            mask_in_gtboxes = off_min>0
            mask_in_level = (off_max>0)&(off_max<=80)   # 设置最大尺寸和最小尺寸约束

            radiu = 8 * 1.5
            gt_center_x = (gt_bbox[...,0]+gt_bbox[...,2])*width_ratio/2
            gt_center_y = (gt_bbox[...,1]+gt_bbox[...,3])*height_ratio/2
            c_l_off = x[:,None] - gt_center_x[None,:]#[h*w,1]-[1,m]-->[h*w,m]
            c_t_off = y[:,None] - gt_center_y[None,:]
            c_r_off = gt_center_x[None,:] - x[:,None]
            c_b_off = gt_center_y[None,:] - y[:,None]
            c_ltrb_off = torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[h*w,m,4]
            c_off_max = torch.max(c_ltrb_off,dim=-1)[0]
            mask_center = c_off_max<radiu

            mask_pos=mask_in_gtboxes&mask_in_level&mask_center      #[h*w,m]
            areas[~mask_pos] = 99999999
            areas_min_ind=torch.min(areas,dim=-1)[1]    #[h*w]

            reg_target = ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),1)]#[h*w,4]
            reg_target = torch.reshape(reg_target,(-1,4))       #[h*w,4]
            mask_pos_2 = mask_pos.long().sum(dim=-1)            #[h*w]
            mask_pos_2 = mask_pos_2 >= 1
            
            reg_target[~(mask_pos_2.to(torch.bool))] = 0

            # reshape
            reg_target = reg_target.reshape(feat_h, feat_w, 4).permute(2,0,1)
            mask_pos_2 = mask_pos_2.to(torch.float32).reshape(1, feat_h,feat_w)

            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(int(1), int(radius))

                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

            reg_target_list.append(reg_target)
            reg_weight_list.append(mask_pos_2)

        reg_targets = torch.stack(reg_target_list, 0)           #Bx4xHxW
        reg_weights = torch.stack(reg_weight_list, 0)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            reg_targets=reg_targets,
            reg_weights=reg_weights)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   reg_pred,
                   image_meta,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.
        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(reg_pred)
        result_list = []
        for img_id in range(len(image_meta)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[img_id:img_id + 1, ...],
                    reg_pred[img_id:img_id + 1, ...],
                    image_meta[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           reg_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.
        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        center_heatmap_pred = torch.sigmoid(center_heatmap_pred)
        # center_heatmap_pred = center_heatmap_pred[:,1:,:,:]
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            reg_pred,
            img_meta['image_shape'],            # batch_input_shape->input_shape
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        if rescale:
            det_bboxes[..., :4] *= det_bboxes.new_tensor((self.rescale_x, self.rescale_y, self.rescale_x, self.rescale_y))

        if 'nms' in self.test_cfg or with_nms:
            det_bboxes, det_labels = \
                self._bboxes_nms(det_bboxes, det_labels, self.test_cfg)
        
        # only for debug
        # image = cv2.imread(img_meta['image_file'])
        # det_bboxes_numpy = det_bboxes.detach().cpu().numpy()
        # for i in range(det_bboxes_numpy.shape[0]):
        #     x0,y0,x1,y1,_ = det_bboxes_numpy[i]
        #     cv2.rectangle(image, ((int)(x0),(int)(y0)), ((int)(x1),(int)(y1)), (255,0,0), 2)
        # cv2.imwrite('./aa.png', image)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       reg_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.
        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.
        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        batch = center_heatmap_pred.shape[0]
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        ltrb_off = transpose_and_gather_feat(reg_pred, batch_index)     # BxHxWx4
        tl_x = (topk_xs - ltrb_off[..., 0]) * (inp_w / width)
        tl_y = (topk_ys - ltrb_off[..., 1]) * (inp_h / height)
        br_x = (topk_xs + ltrb_off[..., 2]) * (inp_w / width)
        br_y = (topk_ys + ltrb_off[..., 3]) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            keep = batched_nms(
                bboxes[:, :4], 
                bboxes[:, -1].contiguous(),
                labels, 
                cfg.nms)
            if max_num > 0:
                bboxes = bboxes[keep][:max_num]
                labels = labels[keep][:max_num]

            remained_index = []
            for i in range(bboxes.shape[0]):
                if bboxes[i,4] > self.score_thresh:
                    remained_index.append(i)
            
            bboxes = bboxes[remained_index]
            labels = labels[remained_index]
        return bboxes, labels
