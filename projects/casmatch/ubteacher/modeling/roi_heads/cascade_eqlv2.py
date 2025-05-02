from typing import List

import numpy as np
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .fast_rcnn_eqlv2 import EQLv2FastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CascadeEQLv2ROIHeads(CascadeROIHeads):
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
        self.alpha_list = cfg.MODEL.ROI_HEADS.Alpha_list

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                EQLv2FastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                    prior_prob=cfg.MODEL.ROI_HEADS.PRIOR_PROB,
                    gamma=cfg.MODEL.ROI_HEADS.EQLv2_GAMMA,
                    mu=cfg.MODEL.ROI_HEADS.EQLv2_MU,
                    alpha=cfg.MODEL.ROI_HEADS.EQLv2_ALPHA,
                    queue_size=cfg.MODEL.ROI_HEADS.Queue_size,
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ):
        del images
        if self.training and compute_loss:  # apply if training loss
            proposals = self.label_and_sample_proposals(proposals, targets, branch=branch)
        elif compute_val_loss:
            raise NotImplementedError

        if (self.training and compute_loss) or compute_val_loss:
            # Need targets to box head
            losses = self._forward_box(
                features, proposals, targets, compute_loss, branch, compute_val_loss
            )
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features,
                proposals,
                targets=None,
                compute_loss=compute_loss,
                branch=branch,
                compute_val_loss=compute_val_loss,
            )
            if not self.training:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_teacher(
        self,
        images,
        features,
        proposals,
    ):
        del images

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)

        proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)

        head_outputs = self._run_all_stage(features, proposals)
        pred_cls = sum([i[0] for i in head_outputs]) / len(head_outputs)
        return (proposals, pred_cls)

    def _run_all_stage(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        predictions_list = []
        for stage in range(self.num_cascade_stages):
            box_features_new = self.box_head[stage](box_features)
            predictions_list.append(self.box_predictor[stage](box_features_new))
        return predictions_list

    def forward_student(self, images, features, roih_soft_labels=None, gt_instances=None):
        del images
        features = [features[f] for f in self.box_in_features]
        losses = {}
        storage = get_event_storage()

        proposals, soft_labels = roih_soft_labels
        proposals = self.label_proposals(proposals, gt_instances)

        for stage in range(self.num_cascade_stages):
            predictions = self._run_stage(features, proposals, stage)

            with storage.name_scope("stage{}".format(stage)):
                stage_losses = self.box_predictor[stage].losses_student(
                    predictions, soft_labels, proposals, self.alpha_list[stage]
                )
            losses.update({k + "_stage{}_pseudo".format(stage): v for k, v in stage_losses.items()})
        return losses

    def _forward_box(
        self,
        features,
        proposals,
        targets=None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if (self.training and compute_loss) or compute_val_loss:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if (self.training and compute_loss) or compute_val_loss:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            batch_list = [len(j) for i in scores_per_stage for j in i]
            if len(np.unique(batch_list)) > 1:
                scores = scores_per_stage[-1]
            else:
                scores = [
                    sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                    for scores_per_image in zip(*scores_per_stage)
                ]

            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples))

        return proposals_with_gt

    @torch.no_grad()
    def label_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        proposals_with_gt = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            if has_gt:
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

                gt_classes = targets_per_image.gt_classes[matched_idxs]
                gt_classes[matched_labels == 0] = self.num_classes
                gt_classes[matched_labels == -1] = -1
            else:
                gt_classes = torch.zeros(len(proposals_per_image.proposal_boxes)) + self.num_classes
                gt_classes = gt_classes.to(proposals_per_image.proposal_boxes.device)
                gt_classes = gt_classes.long()

            proposals_per_image.gt_classes = gt_classes

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt
