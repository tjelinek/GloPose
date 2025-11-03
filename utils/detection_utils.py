import torch
import torch.nn.functional as F


def average_patch_similarity(
        patch_descriptors1: torch.Tensor,
        patch_descriptors2: torch.Tensor,
        segmentation_mask1: torch.Tensor = None,
        segmentation_mask2: torch.Tensor = None,
        use_segmentation: bool = False,
        required_share_nonzero: float = 0.25,
) -> torch.Tensor:
    B1, P, _, D = patch_descriptors1.shape
    B2, P, _, D = patch_descriptors2.shape

    desc1 = patch_descriptors1.reshape(B1, P * P, D)
    desc2 = patch_descriptors2.reshape(B2, P * P, D)

    desc1_norm = F.normalize(desc1, p=2, dim=-1)
    desc2_norm = F.normalize(desc2, p=2, dim=-1)

    if use_segmentation:
        mask1 = _create_patch_mask(segmentation_mask1, P, required_share_nonzero)
        mask2 = _create_patch_mask(segmentation_mask2, P, required_share_nonzero)
    else:
        mask1 = torch.ones(B1, P * P, dtype=torch.bool, device=desc1.device)
        mask2 = torch.ones(B2, P * P, dtype=torch.bool, device=desc2.device)

    sim_1to2 = _compute_nearest_neighbor_similarity(desc1_norm, desc2_norm, mask1, mask2)
    # sim_2to1 = _compute_nearest_neighbor_similarity(desc2_norm, desc1_norm, mask2, mask1)
    # avg_sim = (sim_1to2 + sim_2to1) / 2

    return sim_1to2


def _create_patch_mask(segmentation_mask: torch.Tensor, P: int, required_share_nonzero: float) -> torch.Tensor:
    B, H, W = segmentation_mask.shape
    patch_h = H // P
    patch_w = W // P

    mask_patches = segmentation_mask[:, :P * patch_h, :P * patch_w].reshape(B, P, patch_h, P, patch_w)
    mask_patches = mask_patches.permute(0, 1, 3, 2, 4).reshape(B, P * P, patch_h * patch_w)

    nonzero_share = (mask_patches != 0).float().mean(dim=-1)
    patch_mask = nonzero_share >= required_share_nonzero

    return patch_mask


def _compute_nearest_neighbor_similarity(desc_src: torch.Tensor, desc_tgt: torch.Tensor,
                                         mask_src: torch.Tensor, mask_tgt: torch.Tensor) -> torch.Tensor:
    B_src = desc_src.shape[0]
    B_tgt = desc_tgt.shape[0]

    similarities = torch.zeros(B_src, B_tgt, device=desc_src.device)

    for b_src in range(B_src):
        for b_tgt in range(B_tgt):
            src_valid = desc_src[b_src][mask_src[b_src]]
            tgt_valid = desc_tgt[b_tgt][mask_tgt[b_tgt]]

            if src_valid.shape[0] == 0 or tgt_valid.shape[0] == 0:
                continue

            cosine_sim = torch.mm(src_valid, tgt_valid.t())
            max_sim = cosine_sim.max(dim=1)[0]
            avg_max_sim = max_sim.mean()

            similarities[b_src, b_tgt] = avg_max_sim

    return similarities
