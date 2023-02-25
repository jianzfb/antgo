import torch.nn.functional as F


def QualityFocalLoss(
    pred_sigmoid,
    teacher_sigmoid,
    weight=None,
    beta=2.0,
    reduction="mean",
):
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction="none") * pt.pow(beta)
    pos = weight > 0

    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(pred_sigmoid[pos], teacher_sigmoid[pos], reduction="none") * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss
