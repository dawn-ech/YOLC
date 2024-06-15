import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def gwdloss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.    [B, N, 4]
        target (torch.Tensor): The learning target of the prediction.   [B, N, 4]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()

    distance = torch.square(pred - target).sum(dim=-1).sqrt() # shape is [B, N]
    t = 1
    normalize = False

    if normalize:
        scale = 2 * (t.sqrt().sqrt()).clamp(1e-7)
        distance = distance / scale

    fun = "log1p"
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    elif fun == 'none':
        pass
    
    tau = 1.0
    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance



@LOSSES.register_module()
class GWDLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(GWDLoss, self).__init__()
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
        loss_bbox = self.loss_weight * gwdloss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        return loss_bbox


