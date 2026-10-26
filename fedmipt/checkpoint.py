"""Best-Validation checkpoints."""

from pathlib import Path
from typing import Any, Dict

import torch

from fedmipt.models import DomainModel


def save_checkpoint(path: Path, model_a: DomainModel, model_b: DomainModel,
                    metadata: Dict[str, Any]) -> None:
    """Store both domain models with the round, Validation result, config and data fingerprint."""
    torch.save({"model_a": model_a.state_dict(), "model_b": model_b.state_dict(), **metadata}, path)


def load_checkpoint(path: Path, model_a: DomainModel, model_b: DomainModel,
                    device: torch.device) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    try:
        model_a.load_state_dict(checkpoint["model_a"])
        model_b.load_state_dict(checkpoint["model_b"])
    except RuntimeError as error:
        raise RuntimeError(f"{path} does not match the configured model: {error}") from error
    return checkpoint
