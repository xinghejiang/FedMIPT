"""Exchange of interest packages between the two domains.

Per round, each domain builds packages from its own histories with its
current model, optionally perturbs them, and the server relays them to the
other domain.  The receiving domain aligns them to its own user's interests
with Null-aware OT before the candidate-conditioned readout.  Everything here
runs without gradients.
"""

from typing import Dict

import torch

from fedmipt.config import Config
from fedmipt.data import DomainData
from fedmipt.interests import InterestPackage, build_interest_package
from fedmipt.models import DomainModel
from fedmipt.protection import perturb_package
from fedmipt.transport import null_aware_transport

# Disjoint offsets keep the Laplace noise streams of splits, rounds and
# domains apart for a given base seed.
_NOISE_SPLIT_OFFSET = {"train": 0, "vali": 10_000_000, "test": 20_000_000}
_NOISE_ROUND_STRIDE = 10_007
_NOISE_DOMAIN_OFFSET = {"a": 1_000_003, "b": 2_000_003}


@torch.no_grad()
def domain_packages(model: DomainModel, domain: DomainData, split: str, config: Config,
                    device: torch.device) -> InterestPackage:
    """Interest packages of all users of one domain, built from the split's history."""
    model.eval()
    history = domain.history(split)
    item_ids = torch.arange(domain.num_items, device=device)
    semantics = model.item_semantic(item_ids)
    padded = torch.cat([semantics, semantics.new_zeros(1, config.embedding_dim)], dim=0)
    return build_interest_package(
        padded[history.item_ids.to(device)], history.mask.to(device),
        config.num_interests, config.kmeans_iterations,
    )


@torch.no_grad()
def exchange_packages(model_a: DomainModel, model_b: DomainModel, data, split: str,
                      config: Config, device: torch.device,
                      round_index: int = 0) -> Dict[str, InterestPackage]:
    """Build, perturb and swap packages.  Returns ``{"for_a": ..., "for_b": ...}``."""
    package_a = domain_packages(model_a, data.domain_a, split, config, device)
    package_b = domain_packages(model_b, data.domain_b, split, config, device)
    if config.noise_scale > 0:
        base = config.seed if config.noise_seed is None else config.noise_seed
        base = base + _NOISE_SPLIT_OFFSET[split] + int(round_index) * _NOISE_ROUND_STRIDE
        package_a = perturb_package(package_a, config.noise_scale,
                                    base + _NOISE_DOMAIN_OFFSET["a"], config.noise_clip_norm)
        package_b = perturb_package(package_b, config.noise_scale,
                                    base + _NOISE_DOMAIN_OFFSET["b"], config.noise_clip_norm)
    return {"for_a": package_b, "for_b": package_a}


def select_users(package: InterestPackage, users) -> InterestPackage:
    return {key: value[users] for key, value in package.items()}


@torch.no_grad()
def align_received(model: DomainModel, received: InterestPackage, history_ids: torch.Tensor,
                   history_mask: torch.Tensor, config: Config) -> InterestPackage:
    """Align received source interests to the target user's local interests.

    The target package is built from the same history the scorer sees (with
    the training positive removed during training).  Without OT the received
    interests are read out directly, which is the "w/o OT" control.
    """
    if not config.ot_enable:
        return received
    target = build_interest_package(
        model.item_semantic(history_ids), history_mask,
        config.num_interests, config.kmeans_iterations,
    )
    return null_aware_transport(
        received, target, epsilon=config.ot_epsilon, null_cost=config.ot_null_cost,
        iterations=config.ot_iterations, use_null=config.ot_use_null,
    )
