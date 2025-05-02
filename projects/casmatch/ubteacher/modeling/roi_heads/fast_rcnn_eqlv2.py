import logging
import math
from functools import partial
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    _log_classification_stats,
    fast_rcnn_inference,
)
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

# from ubteacher.modeling.utils import collect_tensor_from_dist


class EQLv2FastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        num_classes: int,
        prior_prob: float,
        gamma: float = 12,
        mu: float = 0.8,
        alpha: float = 4.0,
        queue_size: int = 100,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
        self._neg_grad = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
        self.pos_neg = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))

        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

        logger = logging.getLogger("detectron2")
        logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

        self.queue_size = queue_size
        self.queue_logits = nn.Parameter(
            torch.zeros(self.num_classes, self.queue_size), requires_grad=False
        )
        self.queue_ptrs = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
        self.counter_soft = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)

    def losses_student(self, predictions, roih_soft_labels, proposals, alpha=0.0):
        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        # prediction
        pred_logits, _ = predictions
        gt_logits = roih_soft_labels
        self.n_i, self.n_c = pred_logits.size()

        # gt label
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # target = expand_label(pred_logits, gt_classes)

        # pseudo label
        gt_logits = gt_logits.sigmoid()
        bg_score = gt_logits[:, -1].view(self.n_i, 1)
        gt_logits[:, :-1] *= 1 - bg_score
        max_v, max_ind = gt_logits.max(1)
        pos_ind = torch.unique(max_ind)
        pos_ind_fg = pos_ind[:-1]
        if len(pos_ind_fg) > 0:
            for c in pos_ind_fg:
                mean_score = self.get_topk(c, default=0.7, alpha=alpha)
                cond = (max_ind == c) & (max_v < mean_score)
                max_ind[cond] = self.num_classes
        target = pred_logits.new_zeros(self.n_i, self.n_c)
        target[torch.arange(self.n_i), max_ind] = 1

        # counter
        if len(pos_ind_fg) > 0:
            for c in pos_ind_fg:
                self.counter_soft[c] += len(max_ind[max_ind == c])

        # EQL v2
        pos_w, neg_w = self.get_weight(pred_logits)
        weight = pos_w * target + neg_w * (1 - target)
        cls_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        cls_loss = torch.sum(cls_loss * weight) / self.n_i
        self.collect_grad(pred_logits.detach(), target.detach(), weight.detach())
        losses = {"loss_cls": cls_loss}

        # PR curve
        storage = get_event_storage()
        gt_inds = torch.where(gt_classes != self.num_classes)[0]
        pred_r = max_ind[gt_inds]
        gt_r = gt_classes[gt_inds]

        precision = len(max_ind[max_ind == gt_classes]) / self.n_i
        recall = len(gt_r[pred_r == gt_r]) / (len(gt_r) + 1e-3)
        storage.put_scalar("precision", precision, smoothing_hint=False)
        storage.put_scalar("recall", recall, smoothing_hint=False)
        return losses

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        #
        probs = torch.sigmoid(scores)
        pos_inds = torch.where(gt_classes != self.num_classes)[0]
        gt_classes_pos = gt_classes[pos_inds]
        probs = probs[pos_inds, gt_classes_pos]

        # if torch.distributed.is_initialized():
        #    probs, gt_classes_pos = collect_tensor_from_dist([probs, gt_classes_pos], [0, -1])

        with torch.no_grad():
            ind = gt_classes_pos != -1
            probs = probs[ind]
            gt_classes_pos = gt_classes_pos[ind]

            gt_labels_uniq = torch.unique(gt_classes_pos)
            for label in gt_labels_uniq:
                cls_score = probs[gt_classes_pos == label][: self.queue_size].cpu()
                queue_ptr = int(self.queue_ptrs[label])
                score_size = len(cls_score)
                score_ptr = 0
                if queue_ptr + score_size > self.queue_size:
                    score_ptr = self.queue_size - queue_ptr
                    self.queue_logits[label, queue_ptr:] = cls_score[:score_ptr]
                    score_size = score_size - score_ptr
                    queue_ptr = 0
                self.queue_logits[label, queue_ptr : queue_ptr + score_size] = cls_score[score_ptr:]
                self.queue_ptrs[label] = queue_ptr + score_size

        losses = {
            "loss_cls": self.eql_v2_loss(scores, gt_classes),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def eql_v2_loss(self, cls_score, label):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction="none")
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        if dist.is_initialized():
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        pos_neg = self._pos_grad / (self._neg_grad + 1e-10)
        self.pos_neg = nn.Parameter(pos_neg, requires_grad=False)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self.pos_neg.sum() == 0:
            neg_w = cls_score.new_ones((self.n_i, self.n_c))
            pos_w = cls_score.new_ones((self.n_i, self.n_c))
        else:
            neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

    def predict_probs(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ):
        num_inst_per_image = [len(p) for p in proposals]
        scores, _ = predictions

        probs = torch.sigmoid(scores)
        n_i, n_c = probs.size()
        bg_score = probs[:, -1].view(n_i, 1)
        probs[:, :-1] *= 1 - bg_score

        return probs.split(num_inst_per_image, dim=0)

    def inference(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        proposals: List[Instances],
        score_thr: float = 0.0,
    ):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        if score_thr == 0.0:
            score_thr = self.test_score_thresh
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thr,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def get_topk(self, c, default=0.7, alpha=0.5):
        ind = torch.nonzero(self.queue_logits[c], as_tuple=False).squeeze(1)
        cached_logit = self.queue_logits[c][ind]
        if len(cached_logit) == 0:
            return default
        else:
            index = int(len(cached_logit) * alpha)
            mean, _ = torch.topk(cached_logit, index)
            if len(mean) > 0:
                mean = mean[-1]
            else:
                mean = default
            return float(mean)
