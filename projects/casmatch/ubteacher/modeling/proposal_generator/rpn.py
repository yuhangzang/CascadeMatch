from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from detectron2.layers import batched_nms, cat
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import _is_tracing
from detectron2.structures import Boxes, ImageList, Instances


@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def __init__(
        self,
        cfg,
        input_shape,
        **kwargs,
    ):
        super().__init__(
            cfg=cfg,
            input_shape=input_shape,
            **kwargs,
        )
        self.batch_size_per_image_roih = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if (self.training and compute_loss) or compute_val_loss:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}

        if branch != "unsup_teacher":
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

            return proposals, losses
        else:
            # gt_labels, _ = self.label_and_sample_anchors(anchors, gt_instances)
            gt_labels = None
            proposals = self.predict_soft_proposals(
                anchors,
                pred_objectness_logits,
                pred_anchor_deltas,
                images.image_sizes,
                gt_labels,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )
            cat_pred_objectness_logits = cat(pred_objectness_logits, dim=1)
            return proposals, cat_pred_objectness_logits

    def predict_soft_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        gt_labels,
        nms_thresh: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        min_box_size: float,
        training: bool,
    ):
        with torch.no_grad():
            proposals = self._decode_proposals(anchors, pred_anchor_deltas)

            num_images = len(image_sizes)
            device = proposals[0].device

            # 1. Select top-k anchor for every level and every image
            topk_scores = []  # #lvl Tensor, each of shape N x topk
            topk_proposals = []
            level_ids = []  # #lvl Tensor, each of shape (topk,)
            batch_idx = torch.arange(num_images, device=device)

            for level_id, (proposals_i, logits_i) in enumerate(
                zip(proposals, pred_objectness_logits)
            ):
                Hi_Wi_A = logits_i.shape[1]
                if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
                    num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
                else:
                    num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

                # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
                # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
                logits_i, idx = logits_i.sort(descending=True, dim=1)
                topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
                topk_idx = idx.narrow(1, 0, num_proposals_i)

                # each is N x topk
                topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

                topk_proposals.append(topk_proposals_i)
                topk_scores.append(topk_scores_i)

                level_ids.append(
                    torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
                )

            # 2. Concat all levels together
            topk_scores = cat(topk_scores, dim=1)
            topk_proposals = cat(topk_proposals, dim=1)
            level_ids = cat(level_ids, dim=0)

            # 3. For each image, run a per-level NMS, and choose topk results.
            results: List[Instances] = []
            for n, image_size in enumerate(image_sizes):
                boxes = Boxes(topk_proposals[n])
                scores_per_img = topk_scores[n]
                lvl = level_ids

                valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(
                    scores_per_img
                )
                if not valid_mask.all():
                    if training:
                        raise FloatingPointError(
                            "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                        )
                    boxes = boxes[valid_mask]
                    scores_per_img = scores_per_img[valid_mask]
                    lvl = lvl[valid_mask]
                boxes.clip(image_size)

                # filter empty boxes
                keep = boxes.nonempty(threshold=min_box_size)
                if _is_tracing() or keep.sum().item() != len(boxes):
                    boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

                keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
                keep = keep[: self.batch_size_per_image_roih]

                res = Instances(image_size)
                res.proposal_boxes = boxes[keep]
                res.objectness_logits = scores_per_img[keep]
                results.append(res)

            return results

    def forward_student(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        rpn_soft_labels: torch.Tensor = None,
        gt_instances=None,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        gt_labels, _ = self.label_and_sample_anchors(anchors, gt_instances)
        gt_labels = torch.stack(gt_labels)

        pred_logits, _ = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_logits
        ]

        valid_mask = gt_labels >= 0
        pred_logits = cat(pred_logits, dim=1)

        loss_cls = F.binary_cross_entropy_with_logits(
            pred_logits[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        num_images = len(gt_labels)
        normalizer = self.batch_size_per_image * num_images

        losses = {"loss_rpn_cls_pseudo": loss_cls / normalizer}
        return losses
