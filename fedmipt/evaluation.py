"""Ranking evaluation on the fixed Validation or Test candidates."""

from typing import Optional

import torch

from fedmipt.config import Config
from fedmipt.data import DomainData
from fedmipt.interests import InterestPackage
from fedmipt.metrics import GROUPS, average_metrics, ranking_metrics
from fedmipt.models import DomainModel
from fedmipt.transfer import align_received, select_users
from fedmipt.utils import finite_or_raise

OT_DIAGNOSTICS = (
    "matched_mass", "rejected_mass", "target_unmatched_mass", "transport_cost", "transport_entropy",
)


def _expand_to_candidates(package: InterestPackage, num_candidates: int) -> InterestPackage:
    """Repeat each user's package for every candidate: [B, ...] -> [B * C, ...]."""
    expanded = {}
    for key, value in package.items():
        shape = value.shape[1:]
        value = value.unsqueeze(1).expand(-1, num_candidates, *shape)
        expanded[key] = value.reshape(-1, *shape)
    return expanded


@torch.no_grad()
def evaluate_domain(model: DomainModel, domain: DomainData, received: Optional[InterestPackage],
                    split: str, config: Config, device: torch.device) -> dict:
    """Score all users of one domain on ``split`` ("vali" or "test").

    Returns ``{group: {"count", "metrics"}}`` for overall/warm/cold, plus mean
    OT ``diagnostics`` when OT is used.
    """
    model.eval()
    history = domain.history(split)
    candidate_items, candidate_labels = domain.evaluation_candidates(split, config.eval_negatives)
    cold_positive = domain.positive_is_cold(split)
    num_candidates = candidate_items.shape[1]
    grouped = {group: [] for group in GROUPS}
    diagnostics = []
    for start in range(0, domain.num_users, config.eval_batch_users):
        stop = min(start + config.eval_batch_users, domain.num_users)
        users = torch.arange(start, stop, device=device)
        history_ids = history.item_ids[start:stop].to(device)
        history_mask = history.mask[start:stop].to(device)
        user = model.user_vector(users, history_ids, history_mask)
        user = user.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(-1, 1, user.shape[-1])
        item_ids = torch.as_tensor(candidate_items[start:stop], device=device).reshape(-1)

        package = None
        if received is not None:
            aligned = align_received(
                model, select_users(received, slice(start, stop)), history_ids, history_mask, config,
            )
            if config.ot_enable:
                diagnostics.append({key: aligned[key].cpu() for key in OT_DIAGNOSTICS})
            package = _expand_to_candidates(aligned, num_candidates)

        scores = model.predict(user, item_ids, package).reshape(stop - start, num_candidates).cpu()
        finite_or_raise(f"{domain.name} {split} scores", scores)
        scores = scores.numpy()
        for offset in range(stop - start):
            row = start + offset
            metrics = ranking_metrics(scores[offset], candidate_labels[row], (5, 10),
                                      candidate_items[row])
            grouped["overall"].append(metrics)
            grouped["cold" if cold_positive[row] else "warm"].append(metrics)

    result = {group: {"count": len(rows), "metrics": average_metrics(rows)}
              for group, rows in grouped.items()}
    if diagnostics:
        result["diagnostics"] = {
            key: float(torch.cat([row[key] for row in diagnostics]).mean()) for key in OT_DIAGNOSTICS
        }
    return result
