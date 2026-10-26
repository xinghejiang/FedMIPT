"""Experiment configuration.

A run is fully described by one flat :class:`Config`.  Values come from a JSON
file under ``configs/`` and optional ``--set key=value`` overrides; the resolved
configuration is written to every run directory.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Config:
    # Task and data.  ``data_dir`` holds one subdirectory per domain.
    task: str = "task1_phone_sports"
    data_dir: str = "data/task1_i10_u5/splits/seed2026"
    domain_a_name: str = "Phone"
    domain_a_subdir: str = "phone"
    domain_b_name: str = "Sports"
    domain_b_subdir: str = "sport"
    clip_dim: int = 512
    use_text: bool = True
    use_image: bool = True
    # Fraction of items that keep their content features (robustness study).
    content_keep_ratio: float = 1.0
    content_mask_seed: int = 2026

    # Local model.
    embedding_dim: int = 64
    num_interests: int = 3
    kmeans_iterations: int = 3
    # Probability of hiding a warm item's collaborative embedding in training.
    warm_item_dropout: float = 0.2
    contrastive_temperature: float = 0.2
    readout_temperature: float = 0.2

    # Cross-domain transfer.  ``cross_transfer=False`` is the strict
    # no-transfer ablation: no interest packages and no projector FedAvg.
    cross_transfer: bool = True
    ot_enable: bool = True
    ot_use_null: bool = True
    ot_epsilon: float = 0.07
    ot_null_cost: float = 0.7
    ot_iterations: int = 50
    # Exponent of the matched-mass confidence g_u = m_u ** gamma.
    ot_mass_gamma: float = 0.5
    # Laplace perturbation of exchanged vectors (a heuristic, not formal DP).
    noise_scale: float = 0.0
    noise_clip_norm: float = 1.0
    noise_seed: Optional[int] = None

    # Optimization.
    item_cl_weight: float = 0.04
    user_cl_weight: float = 0.01
    pretrain_rounds: int = 2
    transfer_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 512
    num_negatives: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0

    # Model selection on Validation Overall NDCG@10 (mean of both domains).
    patience: int = 5
    min_improvement: float = 1e-4
    eval_negatives: int = 99
    eval_batch_users: int = 64

    # Training seed (initialization, negative sampling, shuffling).  It never
    # changes the data split or the fixed evaluation candidates.
    seed: Optional[int] = None

    def validate(self) -> None:
        """Reject inconsistent or out-of-range settings before any work starts."""
        errors: List[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        for name in ("domain_a_subdir", "domain_b_subdir"):
            value = getattr(self, name)
            require(
                bool(value) and len(Path(value).parts) == 1 and not Path(value).is_absolute(),
                f"{name} must be a single relative directory name, got {value!r}",
            )
        require(self.domain_a_subdir != self.domain_b_subdir, "domains must use different subdirs")
        for name in (
            "clip_dim", "embedding_dim", "kmeans_iterations", "ot_iterations",
            "transfer_rounds", "local_epochs", "batch_size", "num_negatives",
            "patience", "eval_negatives", "eval_batch_users",
        ):
            require(getattr(self, name) > 0, f"{name} must be positive")
        require(self.pretrain_rounds >= 0, "pretrain_rounds must be non-negative")
        require(self.num_interests >= 0, "num_interests must be non-negative")
        require(0.0 <= self.content_keep_ratio <= 1.0, "content_keep_ratio must be in [0, 1]")
        require(0.0 <= self.warm_item_dropout < 1.0, "warm_item_dropout must be in [0, 1)")
        for name in ("contrastive_temperature", "readout_temperature", "ot_epsilon",
                     "noise_clip_norm", "lr", "grad_clip_norm"):
            require(getattr(self, name) > 0, f"{name} must be positive")
        for name in ("ot_null_cost", "ot_mass_gamma", "noise_scale", "item_cl_weight",
                     "user_cl_weight", "weight_decay", "min_improvement"):
            require(getattr(self, name) >= 0, f"{name} must be non-negative")
        if self.cross_transfer:
            require(self.num_interests > 0, "cross_transfer requires num_interests > 0")
            require(self.use_text or self.use_image,
                    "cross_transfer requires at least one content modality")
        require(self.seed is not None, "seed must be set (use --seed)")
        require(self.task == "task1_phone_sports", "release3 currently packages Task1 only")
        require(self.noise_scale == 0.0, "noise perturbation is outside the recovered Task1 contract")
        if errors:
            raise ValueError("invalid configuration:\n  " + "\n  ".join(errors))

    @property
    def transfer_enabled(self) -> bool:
        return self.cross_transfer and self.num_interests > 0 and (self.use_text or self.use_image)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_FIELDS = {field.name: field for field in dataclasses.fields(Config)}


def _coerce(name: str, value: Any) -> Any:
    """Check a JSON value against the declared field type."""
    expected = _FIELDS[name].type
    if expected in ("Optional[int]", Optional[int]):
        if value is None:
            return None
        expected = int
    if expected in (bool, "bool"):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true/false, got {value!r}")
        return value
    if expected in (int, "int"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return value
    if expected in (float, "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        return float(value)
    if expected in (str, "str"):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {value!r}")
        return value
    raise TypeError(f"unsupported config field type for {name}: {expected}")


def config_from_dict(values: Dict[str, Any]) -> Config:
    unknown = sorted(set(values) - set(_FIELDS))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Config(**{name: _coerce(name, value) for name, value in values.items()})


def parse_override(text: str) -> Dict[str, Any]:
    """Parse ``key=value``; the value is read as JSON, falling back to a string."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if key not in _FIELDS:
        raise ValueError(f"unknown config key in override: {key!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return {key: value}


def load_config(path: Path, overrides: Optional[List[str]] = None,
                seed: Optional[int] = None) -> Config:
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a JSON object")
    values.pop("_comment", None)
    for text in overrides or []:
        values.update(parse_override(text))
    if seed is not None:
        values["seed"] = seed
    config = config_from_dict(values)
    config.validate()
    return config
