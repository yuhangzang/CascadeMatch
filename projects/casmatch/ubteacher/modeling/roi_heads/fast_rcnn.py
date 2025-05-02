import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.utils.events import get_event_storage


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn", score_topk=100):
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    num_accurate = (pred_classes == gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero(as_tuple=False).numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]
    num_fg_accurate = (fg_pred_classes == fg_gt_classes).nonzero(as_tuple=False).numel()
    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero(as_tuple=False).numel()

    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", num_fg_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

    pred_probs = pred_logits.detach()
    pred_probs = F.softmax(pred_probs, -1)
    pred_probs, _ = pred_probs.max(dim=1)
    for score_thr in [0.6, 0.7, 0.8]:
        mask = pred_probs > score_thr
        mask_inds = mask * fg_inds
        num_mask = mask_inds.nonzero(as_tuple=False).numel()
        if num_mask > 0:
            fg_gt_classes = gt_classes[mask_inds]
            fg_pred_classes = pred_classes[mask_inds]
            num_mask_accurate = (fg_pred_classes == fg_gt_classes).nonzero(as_tuple=False).numel()
            storage.put_scalar(f"{prefix}/prec_{score_thr}", num_mask_accurate / num_mask)
            storage.put_scalar(f"{prefix}/recall_{score_thr}", num_mask_accurate / num_fg)

    score_topk = min(score_topk, len(pred_probs[fg_inds]))
    score, _ = torch.topk(pred_probs[fg_inds], score_topk)
    # score = pred_probs[fg_inds]
    return score.mean().item()


# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )
        self.score_history = []
        self.update_iter_interval = 100
        self.score_topk = 100
        self.score_thr = 0.5

    def losses(self, predictions, proposals, branch="supervised"):
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        if branch == "supervised":
            score = _log_classification_stats(scores, gt_classes, score_topk=self.score_topk)
            self.score_history.append(score)
            if len(self.score_history) % self.update_iter_interval == 0:
                self.score_history = [i for i in self.score_history if not math.isnan(i)]
                self.score_thr = np.mean(self.score_history)
                storage = get_event_storage()
                storage.put_scalar("fast_rcnn/score_thr", self.score_thr)
                self.score_history = []

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": self.comput_focal_loss(scores, gt_classes),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def comput_focal_loss(self, pred_class_logits, gt_classes):
        if gt_classes.numel() == 0:
            return 0.0 * pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=pred_class_logits, target=gt_classes)
            total_loss = total_loss / gt_classes.shape[0]
            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
