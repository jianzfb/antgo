import torch
import torch.nn as nn

from antgo.framework.helper.models.builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class L1LossMask(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, is_bbox=True):
        super(L1LossMask, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.is_bbox = is_bbox

    def forward(self,
                pred,
                target,
                indices,
                mask):
        b, c, _, _ = pred.shape
        k = target.shape[-1]  # k, the num of obj_max
        pred = pred.reshape(b, c, -1).permute(0, 2, 1)  # (b,2,h*w) -> (b,h*w,2)
        indices = indices.long()
        mid_y_pred = torch.zeros([b, k] + list(pred.size()[2:])).to(target.device)  # (b,2,2)
        for i in range(mid_y_pred.size(0)):
            mid_y_pred[i] = pred[i, indices[i], :]
        mid_y_pred = mid_y_pred.permute(0, 2, 1)  # (b, 2, 2)
        mask = mask.reshape((b, 1, -1)).repeat(1, c, 1)
        if self.is_bbox:
            mask = mask.permute(0, 2, 1)
            target = torch.squeeze(target)
        total_loss = torch.sum(torch.abs(target * mask - mid_y_pred * mask))
        if self.reduction in ['mean', 'none', None]:
            loss = total_loss / (torch.sum(mask) + 1e-5)
        else:
            loss = total_loss
        return loss * self.loss_weight
