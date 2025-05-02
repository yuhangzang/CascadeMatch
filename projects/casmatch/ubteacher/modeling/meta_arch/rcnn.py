from typing import Dict, List, Optional, Tuple

import torch

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures.instances import Instances
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self,
        batched_inputs,
        branch="supervised",
        given_proposals=None,
        val_mode=False,
        rpn_soft_labels=None,
        roih_soft_labels=None,
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs, branch=branch)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images,
                features,
                gt_instances,
                compute_loss=True,
                compute_val_loss=False,
                branch=branch,
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                compute_loss=True,
                branch=branch,
                compute_val_loss=False,
            )
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        elif branch == "unsup_teacher":
            # Region proposal network
            proposals_rpn, rpn_soft_labels = self.proposal_generator(
                images,
                features,
                gt_instances=gt_instances,
                compute_loss=False,
                compute_val_loss=False,
                branch=branch,
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference !
            roih_soft_labels = self.roi_heads.forward_teacher(images, features, proposals_rpn)
            return {}, {}, rpn_soft_labels, roih_soft_labels
        elif branch == "unsup_student":
            # proposal_losses = self.proposal_generator.forward_student(
            #     images, features, rpn_soft_labels, gt_instances
            # )
            detector_losses = self.roi_heads.forward_student(
                images, features, roih_soft_labels, gt_instances
            )

            losses = {}
            losses.update(detector_losses)
            # losses.update(proposal_losses)
            return losses, [], [], None
        else:
            raise NotImplementedError

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        branch: str = "",
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                if branch == "unsup_EMA":
                    proposals, _ = self.proposal_generator(
                        images,
                        features,
                        None,
                        compute_loss=False,
                        branch=branch,
                        compute_val_loss=False,
                    )
                else:
                    proposals, _ = self.proposal_generator(
                        images, features, None, compute_loss=False, compute_val_loss=False
                    )
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(
                images,
                features,
                proposals,
                None,
                compute_loss=False,
                branch=branch,
                compute_val_loss=False,
            )
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
