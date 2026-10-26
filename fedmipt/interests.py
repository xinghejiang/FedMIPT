"""Multi-interest construction (paper Sec. "Multi-Interest Construction").

An interest package for B users with up to K interests is a dict

    direction  [B, D]     normalized history-level semantic direction (UserCL)
    prototype  [B, K, D]  normalized interest prototypes p_{u,k}; zero if invalid
    mask       [B, K]     m_{u,k}: True for non-empty clusters
    mass       [B, K]     a_{u,k}: share of history items in each cluster; sums to 1

Only these four tensors leave a domain.  Item IDs, interaction order and item
content are never part of the package.
"""

from typing import Dict

import torch
import torch.nn.functional as F

InterestPackage = Dict[str, torch.Tensor]


def build_interest_package(history_semantics: torch.Tensor, history_mask: torch.Tensor,
                           num_interests: int, iterations: int = 3) -> InterestPackage:
    """Spherical k-means over each user's projected history.

    Args:
        history_semantics: [B, H, D] normalized item semantics e_i.
        history_mask: [B, H] bool, valid history positions.
        num_interests: K, the maximum number of prototypes.
        iterations: number of assignment/update steps.

    Initialization is deterministic: the first center is the history item
    closest to the mean direction, each further center is the item farthest
    from the current centers.  Users with fewer than K items get fewer active
    slots, and clusters that become empty are deactivated.  A user whose
    history is empty (possible for target packages after the training
    positive is removed) gets no active slot and a zero direction.  Callers
    run this under ``torch.no_grad``; packages carry no gradient.
    """
    lengths = history_mask.sum(dim=1)
    mask = history_mask.to(history_semantics.dtype)
    mean = torch.einsum("uh,uhd->ud", mask, history_semantics)
    mean = mean / lengths.clamp_min(1).to(history_semantics.dtype).unsqueeze(1)
    direction = F.normalize(mean, dim=1)

    normalized = F.normalize(history_semantics, dim=-1)
    rows = torch.arange(history_semantics.shape[0], device=history_semantics.device)
    closest = torch.einsum("uhd,ud->uh", normalized, direction)
    first = closest.masked_fill(~history_mask, float("-inf")).argmax(dim=1)
    centers = [normalized[rows, first]]
    for _ in range(1, num_interests):
        stacked = torch.stack(centers, dim=1)
        nearest = torch.einsum("uhd,ukd->uhk", normalized, stacked).max(dim=2).values
        farthest = nearest.masked_fill(~history_mask, float("inf")).argmin(dim=1)
        centers.append(normalized[rows, farthest])
    centers = torch.stack(centers, dim=1)

    slots = torch.arange(num_interests, device=history_semantics.device).unsqueeze(0)
    active = slots < lengths.unsqueeze(1)
    counts = history_semantics.new_zeros(history_semantics.shape[0], num_interests)
    for _ in range(iterations):
        similarity = torch.einsum("uhd,ukd->uhk", normalized, centers)
        similarity = similarity.masked_fill(~history_mask.unsqueeze(2), float("-inf"))
        similarity = similarity.masked_fill(~active.unsqueeze(1), float("-inf"))
        assignment = F.one_hot(similarity.argmax(dim=2), num_classes=num_interests)
        assignment = assignment.to(history_semantics.dtype) * history_mask.unsqueeze(2)
        counts = assignment.sum(dim=1)
        # Prototypes are means of the (unnormalized) semantics, then normalized.
        updated = torch.einsum("uhk,uhd->ukd", assignment, history_semantics)
        updated = F.normalize(updated / counts.clamp_min(1.0).unsqueeze(2), dim=-1)
        has_items = counts > 0
        centers = torch.where(has_items.unsqueeze(2), updated, centers)
        active = active & has_items
    return {
        "direction": direction,
        "prototype": centers * active.unsqueeze(2),
        "mask": active,
        "mass": counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0),
    }
