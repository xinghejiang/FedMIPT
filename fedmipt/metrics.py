"""Leave-one-out ranking metrics and their aggregation."""

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

GROUPS = ("overall", "warm", "cold")
METRIC_KEYS = ("hr@5", "hr@10", "ndcg@5", "ndcg@10")


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, top_ks: Iterable[int],
                    item_ids: np.ndarray) -> Dict[str, float]:
    """HR@k and NDCG@k for one candidate list with exactly one positive.

    Ties are broken by ascending item ID, so the result never depends on the
    candidate order in the file.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    item_ids = np.asarray(item_ids)
    positive = np.flatnonzero(labels > 0)
    if len(positive) != 1:
        raise ValueError(f"each candidate set needs exactly one positive, found {len(positive)}")
    if item_ids.shape != scores.shape:
        raise ValueError("item_ids must have the same shape as scores")
    order = np.lexsort((item_ids, -scores))
    rank = int(np.flatnonzero(order == positive[0])[0]) + 1
    result = {}
    for k in top_ks:
        result[f"hr@{k}"] = 1.0 if rank <= k else 0.0
        result[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1.0) if rank <= k else 0.0
    return result


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {key: None for key in METRIC_KEYS}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def merge_domains(a: dict, b: dict) -> dict:
    """User-count-weighted mean of two domain results (the model-selection target)."""
    merged = {}
    for group in GROUPS:
        count_a, count_b = a[group]["count"], b[group]["count"]
        metrics = {}
        for key in a[group]["metrics"]:
            values, weights = [], []
            for result, count in ((a, count_a), (b, count_b)):
                value = result[group]["metrics"][key]
                if value is not None and count:
                    values.append(value)
                    weights.append(count)
            metrics[key] = float(np.average(values, weights=weights)) if values else None
        merged[group] = {"count": count_a + count_b, "metrics": metrics}
    if "diagnostics" in a and "diagnostics" in b:
        merged["diagnostics"] = {
            key: 0.5 * (a["diagnostics"][key] + b["diagnostics"][key]) for key in a["diagnostics"]
        }
    return merged
