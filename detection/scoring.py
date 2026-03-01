"""Detection scoring and similarity functions.

Vendored from repositories/cnos/src/model/detector.py (lines 21-249) and
repositories/cnos/src/model/loss.py (lines 19-62) to avoid runtime dependency
on the cnos repository. Pure tensor math — no cnos imports.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from data_structures.template_bank import TemplateBank
from utils.detection_utils import average_patch_similarity


# ---------------------------------------------------------------------------
# Similarity primitives (from cnos src/model/loss.py)
# ---------------------------------------------------------------------------

def compute_csls_terms(proposal_descriptors: torch.Tensor,
                       template_descriptors: Dict[Any, torch.Tensor],
                       k: int = 10) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
    objs = sorted(template_descriptors.keys())

    splits = [0]
    for o in objs:
        splits.append(splits[-1] + template_descriptors[o].size(0))

    template = torch.cat([template_descriptors[o] for o in objs], dim=0)  # [Nt, D]
    prop = proposal_descriptors  # [Nq, D]

    prop = F.normalize(prop, dim=1)
    template = F.normalize(template, dim=1)

    S = prop @ template.T  # [Nq, Nt]

    kx = min(k, max(1, template.size(0) - 1))
    rx = torch.topk(S, k=kx, dim=1).values.mean(dim=1)  # [Nq]

    T = template @ template.T  # [Nt, Nt]
    T.fill_diagonal_(-float('inf'))  # exclude self
    kt = min(k, max(1, template.size(0) - 1))
    rt = torch.topk(T, k=kt, dim=1).values.mean(dim=1)  # [Nt]

    return rx, rt, splits


def cosine_similarity(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    query_norm = F.normalize(query, dim=1)
    reference_norm = F.normalize(reference, dim=1)
    S = query_norm @ reference_norm.T
    return S.clamp(min=-1.0, max=1.0)


def csls_score(query: torch.Tensor, reference: torch.Tensor,
               rx: torch.Tensor, rt: torch.Tensor) -> torch.Tensor:
    query_norm = F.normalize(query, dim=1)
    reference_norm = F.normalize(reference, dim=1)
    S = cosine_similarity(query_norm, reference_norm)
    S_csls = 2 * S - rx[:, None] - rt[None, :]
    return S_csls


# ---------------------------------------------------------------------------
# Proposal filtering and scoring (from cnos src/model/detector.py)
# ---------------------------------------------------------------------------

def filter_similarities_dict(similarities: Dict[Any, torch.Tensor],
                             idx_selected_proposals: torch.Tensor) -> None:
    for obj_id in similarities.keys():
        similarities[obj_id] = similarities[obj_id][idx_selected_proposals]


def filter_proposals(proposals_assigned_templates_ids: torch.Tensor,
                     proposals_assigned_object_ids: torch.Tensor,
                     cosine_similarity_per_proposal: torch.Tensor,
                     sorted_obj_keys: list[Any],
                     ood_detection_method: str,
                     similarities: Dict[Any, torch.Tensor],
                     db_descriptors: Dict[Any, torch.Tensor],
                     template_data: TemplateBank = None,
                     global_similarity_threshold: float = None,
                     lowe_ratio_threshold: float = None) -> torch.Tensor:
    device = cosine_similarity_per_proposal.device
    idx_proposals = torch.arange(len(cosine_similarity_per_proposal), device=device)

    if ood_detection_method == 'cosine_similarity_quantiles':
        num_proposals = proposals_assigned_object_ids.shape[0]
        assigned_template_id = \
            proposals_assigned_templates_ids[torch.arange(num_proposals, device=device), proposals_assigned_object_ids]

        template_thresholds = template_data.template_thresholds
        thresholds_for_selected_objs = []
        for det_id, obj_id in enumerate(proposals_assigned_object_ids):
            obj_name = sorted_obj_keys[obj_id]
            if template_thresholds[obj_name] is not None:
                threshold = assigned_template_id[det_id]
            else:
                threshold = torch.tensor(global_similarity_threshold, device=device)

            thresholds_for_selected_objs.append(threshold)

        thresholds_for_selected_objs = torch.stack(thresholds_for_selected_objs)

        idx_selected_proposals = idx_proposals[cosine_similarity_per_proposal > thresholds_for_selected_objs]
    elif ood_detection_method == 'global_threshold':
        assert global_similarity_threshold is not None
        idx_selected_proposals = idx_proposals[cosine_similarity_per_proposal > global_similarity_threshold]
    elif ood_detection_method == 'lowe_test':

        all_similarities = torch.cat([similarities[obj_name] for obj_name in sorted_obj_keys], dim=1)

        topk_sims, topk_indices = torch.topk(all_similarities, 2, dim=1)
        s1, s2 = topk_sims[:, 0], topk_sims[:, 1]
        lowe_ratio = s1 / s2

        idx_selected_proposals = idx_proposals[lowe_ratio > lowe_ratio_threshold]

    elif ood_detection_method == 'mahalanobis_ood_detection':
        assigned_best_descriptor = []
        mu_cs = []
        mahalanobis_taus = []
        for i, obj_id in enumerate(proposals_assigned_object_ids):
            obj_name = sorted_obj_keys[obj_id.item()]
            obj_descriptors = db_descriptors[obj_name]
            template_id = proposals_assigned_templates_ids[i, obj_id]
            best_descriptor = obj_descriptors[template_id]
            assigned_best_descriptor.append(best_descriptor)

            mu_c = template_data.class_means[obj_name]
            mu_cs.append(mu_c)
            mahalanobis_taus.append(template_data.maha_thresh_per_class[obj_name])

        assigned_best_descriptor = torch.stack(assigned_best_descriptor)
        mu_cs = torch.stack(mu_cs)
        mahalanobis_taus = torch.stack(mahalanobis_taus)
        sigma_inv = template_data.sigma_inv

        diff = assigned_best_descriptor - mu_cs
        mahalanobis_dist = (diff.unsqueeze(1) @ sigma_inv.unsqueeze(0) @ diff.unsqueeze(2)).squeeze()

        idx_selected_proposals = idx_proposals[mahalanobis_dist <= mahalanobis_taus]

    elif ood_detection_method == 'none':
        idx_selected_proposals = idx_proposals  # Keep them as they are
    else:
        raise ValueError(f'Unknown OOD detection method {ood_detection_method}')

    return idx_selected_proposals


def compute_templates_similarity_scores(
    template_data: TemplateBank,
    proposal_cls_descriptors: torch.Tensor,
    proposal_patch_descriptors: torch.Tensor,
    proposal_masks: torch.Tensor,
    similarity_metric: str,
    aggregation_function: str,
    matching_max_num_instances: int,
    global_similarity_threshold: float,
    lowe_ratio_threshold: float,
    ood_detection_method: Optional[str] = None,
    patch_descriptors_filtering: bool = False,
    min_avg_patch_cosine_similarity: float = 0.,
    descriptor_mask_detections: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:

    db_descriptors = template_data.cls_desc
    sorted_obj_keys = sorted(db_descriptors.keys())

    cosine_similarities = {}
    csls_scores = {}

    rx, rt, splits = compute_csls_terms(proposal_cls_descriptors, db_descriptors)
    for i, obj_id in enumerate(sorted_obj_keys):
        obj_descriptor = db_descriptors[obj_id]

        similarity = cosine_similarity(proposal_cls_descriptors, obj_descriptor)
        cosine_similarities[obj_id] = similarity

        rt_obj_id = rt[splits[i]:splits[i + 1]]
        csls = csls_score(proposal_cls_descriptors, obj_descriptor, rx, rt_obj_id)
        csls_scores[obj_id] = csls

    aggregated_cosine = {}
    aggregated_csls = {}
    proposals_assigned_templates_ids_cosine = {}
    proposals_assigned_templates_ids_csls = {}

    for obj_id in sorted_obj_keys:
        if aggregation_function == "mean":
            cosine_score = torch.sum(cosine_similarities[obj_id], dim=-1) / cosine_similarities[obj_id].shape[-1]
            csls_score_agg = torch.sum(csls_scores[obj_id], dim=-1) / csls_scores[obj_id].shape[-1]
            cosine_indices = torch.arange(cosine_similarities[obj_id].shape[1],
                                          device=cosine_similarities[obj_id].device)
            csls_indices = torch.arange(csls_scores[obj_id].shape[1], device=csls_scores[obj_id].device)
        elif aggregation_function == "median":
            cosine_score, cosine_indices = torch.median(cosine_similarities[obj_id], dim=-1)
            csls_score_agg, csls_indices = torch.median(csls_scores[obj_id], dim=-1)
        elif aggregation_function == "max":
            cosine_score, cosine_indices = torch.max(cosine_similarities[obj_id], dim=-1)
            csls_score_agg, csls_indices = torch.max(csls_scores[obj_id], dim=-1)
        elif aggregation_function == "avg_5":
            k = min(cosine_similarities[obj_id].shape[-1], 5)
            cosine_score, cosine_indices = torch.topk(cosine_similarities[obj_id], k=k, dim=-1)
            cosine_score = torch.mean(cosine_score, dim=-1)
            csls_score_agg, csls_indices = torch.topk(csls_scores[obj_id], k=k, dim=-1)
            csls_score_agg = torch.mean(csls_score_agg, dim=-1)
        else:
            raise ValueError("Unknown aggregation function")

        aggregated_cosine[obj_id] = cosine_score
        aggregated_csls[obj_id] = csls_score_agg
        proposals_assigned_templates_ids_cosine[obj_id] = cosine_indices
        proposals_assigned_templates_ids_csls[obj_id] = csls_indices

    cosine_per_proposal_and_object = torch.stack([aggregated_cosine[k] for k in sorted_obj_keys], dim=-1)
    csls_per_proposal_and_object = torch.stack([aggregated_csls[k] for k in sorted_obj_keys], dim=-1)
    proposals_assigned_templates_ids_cosine = torch.stack(
        [proposals_assigned_templates_ids_cosine[k] for k in sorted_obj_keys], dim=-1)
    proposals_assigned_templates_ids_csls = torch.stack(
        [proposals_assigned_templates_ids_csls[k] for k in sorted_obj_keys], dim=-1)

    # assign each proposal to the object with the highest scores
    cosine_score_per_proposal, cosine_proposals_assigned_object_ids = torch.max(cosine_per_proposal_and_object, dim=-1)
    csls_score_per_proposal, csls_proposals_assigned_object_ids = torch.max(csls_per_proposal_and_object, dim=-1)

    if similarity_metric == 'cosine':
        proposals_assigned_object_ids = cosine_proposals_assigned_object_ids
        proposals_assigned_templates_ids = proposals_assigned_templates_ids_cosine
        score_per_proposal = cosine_score_per_proposal
        sim_per_proposal_and_object = cosine_per_proposal_and_object
        similarities = cosine_similarities
    elif similarity_metric == 'csls':
        proposals_assigned_object_ids = csls_proposals_assigned_object_ids
        proposals_assigned_templates_ids = proposals_assigned_templates_ids_csls
        score_per_proposal = csls_score_per_proposal
        sim_per_proposal_and_object = csls_per_proposal_and_object
        similarities = csls_scores
    else:
        raise ValueError(f'Unknown similarity_metric value {similarity_metric}')

    selected_proposals_indices = filter_proposals(proposals_assigned_templates_ids, proposals_assigned_object_ids,
                                                  cosine_score_per_proposal, sorted_obj_keys, ood_detection_method,
                                                  similarities, db_descriptors, template_data,
                                                  global_similarity_threshold, lowe_ratio_threshold)

    if patch_descriptors_filtering and len(selected_proposals_indices) > 0:

        assigned_object_ids = proposals_assigned_object_ids[selected_proposals_indices]
        selected_object_templates = proposals_assigned_templates_ids[selected_proposals_indices]

        detections_patch_similarities = []
        for detection_id in range(len(assigned_object_ids)):

            assigned_obj_id = assigned_object_ids[detection_id]
            assigned_obj_key = sorted_obj_keys[assigned_obj_id]
            object_descriptors = template_data.patch_desc[assigned_obj_key]
            assigned_template_id = selected_object_templates[detection_id, assigned_obj_id]
            patch_descriptor = object_descriptors[assigned_template_id][None]

            assigned_mask = template_data.masks[assigned_obj_key][assigned_template_id][None]

            proposal_patch_descriptor = proposal_patch_descriptors[[detection_id]]
            proposal_mask = proposal_masks[[detection_id]]

            patch_similarity = average_patch_similarity(
                proposal_patch_descriptor,
                patch_descriptor,
                proposal_mask,
                assigned_mask,
                descriptor_mask_detections,
            )

            detections_patch_similarities.append(patch_similarity.squeeze())

        detections_patch_similarities = torch.stack(detections_patch_similarities)
        detections_similar_patches_mask = detections_patch_similarities >= min_avg_patch_cosine_similarity
        selected_proposals_indices = selected_proposals_indices[detections_similar_patches_mask]

    # for bop challenge, we only keep top 100 instances
    if len(selected_proposals_indices) > matching_max_num_instances:
        logging.info(f"Selecting top {matching_max_num_instances} instances ...")
        _, idx = torch.topk(
            score_per_proposal[selected_proposals_indices], k=matching_max_num_instances
        )
        selected_proposals_indices = selected_proposals_indices[idx]

    # Sort detections by score
    selected_proposals_indices = selected_proposals_indices[
        torch.argsort(score_per_proposal[selected_proposals_indices], descending=True)]

    pred_idx_objects = proposals_assigned_object_ids[selected_proposals_indices]
    pred_scores = score_per_proposal[selected_proposals_indices]
    pred_score_distribution = sim_per_proposal_and_object[selected_proposals_indices]

    filter_similarities_dict(similarities, selected_proposals_indices)

    sorted_db_keys_tensor = torch.tensor(sorted_obj_keys).to(pred_idx_objects.device)
    selected_objects = sorted_db_keys_tensor[pred_idx_objects]

    return selected_proposals_indices, selected_objects, pred_scores, pred_score_distribution, similarities
