"""Null-aware entropic optimal transport between interest distributions.

Paper Sec. "User-Wise Cross-Domain Interest Transport".  For each user, the
source distribution sum_k a_k delta(S_k) is transported towards the target
distribution sum_l b_l delta(T_l):

* cost        C_kl = 1 - cos(S_k, T_l)                      (in [0, 2])
* Null state  one extra row and column with unit mass; a real->Null or
              Null->real edge costs null_cost / 2, Null->Null costs 0, so
              rejecting one unit of real mass costs ``null_cost`` in total
* marginals   rows [a, 1], columns [b, 1]; a and b are renormalized over
              valid slots
* solver      log-domain Sinkhorn with regularization epsilon and a fixed
              number of iterations (no tolerance-based stopping)
* payload     each target slot receives the plan-weighted mean of the source
              prototypes, S~_l = sum_k P_kl S_k / sum_k P_kl (then normalized)

The trainer calls this under ``torch.no_grad``: the plan is not
differentiated, and gradients reach the model only through the
candidate-conditioned readout of the transported payload.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from fedmipt.interests import InterestPackage

# Large negative log-kernel for structurally invalid edges (masked slots).
_INVALID = -1e9


def null_aware_transport(source: InterestPackage, target: InterestPackage, epsilon: float,
                         null_cost: float, iterations: int,
                         use_null: bool = True) -> Dict[str, torch.Tensor]:
    """Align source interests to the target user's interests.

    Args:
        source, target: interest packages with prototypes [B, K, D].
        epsilon: entropic regularization (> 0).
        null_cost: total cost of leaving one unit of mass unmatched (>= 0).
        iterations: Sinkhorn iterations (> 0).
        use_null: if False, solve balanced OT without Null states (ablation);
            all valid mass must then be matched.

    Returns:
        A package in the target slot layout, where ``prototype`` holds the
        transported source interests S~_l, ``mask`` the target slots that
        received mass, and ``mass`` the received real mass per slot.  It also
        contains per-user diagnostics: ``matched_mass`` m_u (real-to-real mass,
        an interest-mass statistic), ``rejected_mass``, ``target_unmatched_mass``,
        ``transport_cost`` and normalized ``transport_entropy``.
    """
    if epsilon <= 0 or iterations <= 0 or null_cost < 0:
        raise ValueError(
            f"invalid OT settings: epsilon={epsilon}, null_cost={null_cost}, iterations={iterations}"
        )
    source_prototype = F.normalize(source["prototype"], dim=-1)
    target_prototype = F.normalize(target["prototype"], dim=-1)
    if source_prototype.shape != target_prototype.shape:
        raise ValueError(
            f"source/target prototype shapes differ: {tuple(source_prototype.shape)} "
            f"vs {tuple(target_prototype.shape)}"
        )
    source_mask = source["mask"].bool()
    target_mask = target["mask"].bool()
    source_mass = source["mass"] * source_mask.to(source["mass"].dtype)
    target_mass = target["mass"] * target_mask.to(target["mass"].dtype)
    source_mass = source_mass / source_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    target_mass = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)

    cost = 1.0 - torch.einsum("bkd,bld->bkl", source_prototype, target_prototype)
    batch, k, _ = cost.shape
    real_valid = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)
    if use_null:
        augmented_cost = cost.new_full((batch, k + 1, k + 1), 0.5 * float(null_cost))
        augmented_cost[:, :k, :k] = cost
        augmented_cost[:, k, k] = 0.0
        valid = torch.ones_like(augmented_cost, dtype=torch.bool)
        valid[:, :k, :k] = real_valid
        valid[:, :k, k] = source_mask
        valid[:, k, :k] = target_mask
        null_mass = source_mass.new_ones(batch, 1)
        row_mass = torch.cat([source_mass, null_mass], dim=1)
        col_mass = torch.cat([target_mass, null_mass], dim=1)
    else:
        augmented_cost, valid = cost, real_valid
        row_mass, col_mass = source_mass, target_mass

    plan = _sinkhorn(augmented_cost, valid, row_mass, col_mass, epsilon, iterations)
    real_plan = plan[:, :k, :k] if use_null else plan

    received = real_plan.sum(dim=1)
    matched_mass = real_plan.sum(dim=(1, 2)).clamp(0.0, 1.0)
    if use_null:
        rejected_mass = plan[:, :k, k].sum(dim=1).clamp(0.0, 1.0)
        target_unmatched_mass = plan[:, k, :k].sum(dim=1).clamp(0.0, 1.0)
    else:
        rejected_mass = matched_mass.new_zeros(batch)
        target_unmatched_mass = matched_mass.new_zeros(batch)

    aligned = torch.einsum("bkl,bkd->bld", real_plan, source_prototype)
    aligned = F.normalize(aligned / received.unsqueeze(-1).clamp_min(1e-12), dim=-1)
    aligned_mask = target_mask & (received > 1e-8)
    aligned = aligned * aligned_mask.unsqueeze(-1)

    transport_cost = (real_plan * cost).sum(dim=(1, 2)) / matched_mass.clamp_min(1e-12)
    normalized_plan = real_plan / matched_mass.view(-1, 1, 1).clamp_min(1e-12)
    entropy = -(normalized_plan * normalized_plan.clamp_min(1e-12).log()).sum(dim=(1, 2))
    entropy = entropy / torch.tensor(
        float(max(2, k * k)), device=entropy.device, dtype=entropy.dtype
    ).log()
    return {
        "direction": source["direction"],
        "prototype": aligned,
        "mask": aligned_mask,
        "mass": received,
        "matched_mass": matched_mass,
        "rejected_mass": rejected_mass,
        "target_unmatched_mass": target_unmatched_mass,
        "transport_cost": transport_cost,
        "transport_entropy": entropy,
    }


def _sinkhorn(cost: torch.Tensor, valid: torch.Tensor, row_mass: torch.Tensor,
              col_mass: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """Log-domain Sinkhorn scaling.  cost/valid: [B, R, C]; marginals [B, R], [B, C]."""
    log_row = row_mass.clamp_min(1e-30).log()
    log_col = col_mass.clamp_min(1e-30).log()
    log_kernel = (-cost / float(epsilon)).masked_fill(~valid, _INVALID)
    log_u = torch.zeros_like(log_row)
    log_v = torch.zeros_like(log_col)
    for _ in range(int(iterations)):
        log_u = log_row - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_col - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
    return torch.exp(log_kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)) * valid
