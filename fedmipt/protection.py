"""Clipping and Laplace perturbation of exchanged interest packages.

This is the robustness mechanism studied in the paper (RQ4).  The noise scale
``b`` is *not* calibrated to a privacy budget, so it provides no formal DP or
LDP guarantee.  Masses and validity masks are exchanged unperturbed.
"""

import numpy as np
import torch
import torch.nn.functional as F

from fedmipt.interests import InterestPackage


def perturb_package(package: InterestPackage, scale: float, seed: int,
                    clip_norm: float = 1.0) -> InterestPackage:
    """Clip each vector to L2 norm ``clip_norm``, add i.i.d. Laplace(0, scale), renormalize.

    Applied to ``direction`` [B, D] and to valid ``prototype`` rows [B, K, D];
    invalid prototype rows stay zero.  Noise is drawn from NumPy with the given
    seed (direction first, then prototypes), so it is reproducible across
    devices.  ``scale == 0`` returns the package unchanged.
    """
    if scale < 0:
        raise ValueError(f"Laplace noise scale must be non-negative, got {scale}")
    if clip_norm <= 0:
        raise ValueError(f"clip norm must be positive, got {clip_norm}")
    if scale == 0:
        return package
    rng = np.random.default_rng(int(seed))

    def perturb(value: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        clipped = value / torch.maximum(torch.ones_like(norm), norm / float(clip_norm))
        noise = rng.laplace(0.0, float(scale), size=tuple(value.shape)).astype(np.float32)
        noise = torch.from_numpy(noise).to(device=value.device, dtype=value.dtype)
        return F.normalize(clipped + noise, dim=-1)

    output = dict(package)
    output["direction"] = perturb(package["direction"])
    valid = package["mask"].bool().unsqueeze(-1)
    output["prototype"] = perturb(package["prototype"]) * valid.to(package["prototype"].dtype)
    return output
