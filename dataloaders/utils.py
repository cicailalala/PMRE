import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from collections import defaultdict
from torch.nn import functional as F


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, size=1):
        metric = self.metrics[metric_name]

        metric["val"] += val*size
        metric["count"] += size
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, true):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        B = logits.size(0)
        num_classes = logits.size(1)
        logits = logits.view(B, num_classes, -1)
        true = true.view(B, 1, -1)
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1).long()]

            true_1_hot = true_1_hot.permute(0, 2, 1).float()

            #true_1_hot = true_1_hot.view(B, num_classes, -1)
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (2)
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps))
        if self.reduction == 'mean':
            return (1 - dice_loss.mean())
        else:
            return (1 - dice_loss)


class BCE_2D(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCE_2D, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        B = logits.size(0)
        C = logits.size(1)
        logits = logits.view(B, C, -1)
        target = target.view(B, -1)

            
        ce_loss = F.cross_entropy(
            logits.float(),
            target.long(),
            weight=self.weight,
            reduction=self.reduction,
        )
        return ce_loss


def get_iou(pred, gt):
    pred = np.asarray(pred.detach().cpu()).flatten()
    pred = np.around(pred)
    gt = np.asarray(gt.detach().cpu()).flatten()
    iou = jaccard_score(y_pred=pred, y_true=gt)
    return iou

def get_dice(pred, gt):
    pred = np.asarray(pred.detach().cpu()).flatten()
    pred = np.around(pred)
    gt = np.asarray(gt.detach().cpu()).flatten()
    dice = f1_score(y_pred=pred, y_true=gt)
    return dice


def get_sensitivity_specificity(pred, gt):
    pred = np.asarray(pred.detach().cpu()).flatten()
    pred = np.around(pred)
    gt = np.asarray(gt.detach().cpu()).flatten()
    sensitivity, specificity, support = sensitivity_specificity_support(y_pred=pred, y_true=gt, average='binary')
    return sensitivity, specificity

