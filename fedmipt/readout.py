"""Candidate-conditioned cross-domain feature (paper Sec. "Cross-Domain Feature Construction")."""

import torch
import torch.nn.functional as F

from fedmipt.interests import InterestPackage


def cross_domain_feature(item_semantic: torch.Tensor, package: InterestPackage,
                         temperature: float, mass_gamma: float) -> torch.Tensor:
    """X^cross_{u,j} = m_u^gamma * Norm(r_{u,j}) ⊙ e_j.

    Args:
        item_semantic: [N, D] normalized candidate semantics e_j.
        package: received (OT-aligned or raw) interests with prototype [N, K, D],
            mask [N, K] and mass [N, K]; ``matched_mass`` [N] is present only
            after OT.
        temperature: softmax temperature of the candidate relevance.
        mass_gamma: exponent gamma of the historical matched-mass scaling. It is
            applied only when the package carries ``matched_mass``; gamma = 0
            gives g_u = 1.

    The relevance alpha_l = softmax_l(cos-like score / temperature + log mass_l)
    is restricted to valid slots.  Returns [N, D]; rows without any valid slot
    are zero.
    """
    prototype = package["prototype"]
    mask = package["mask"]
    score = torch.einsum("bkd,bd->bk", prototype, item_semantic) / temperature
    score = score.masked_fill(~mask, -1e9)
    score = score + package["mass"].clamp_min(1e-12).log()
    weight = torch.softmax(score, dim=-1) * mask
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    selected = torch.einsum("bk,bkd->bd", weight, prototype)
    cross = F.normalize(selected, dim=-1) * item_semantic
    if "matched_mass" in package:
        confidence = package["matched_mass"].clamp_min(0.0).pow(mass_gamma)
        cross = cross * confidence.unsqueeze(-1)
    return cross
