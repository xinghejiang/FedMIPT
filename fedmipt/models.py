"""Local multimodal recommendation model of one domain.

Federated boundary: only ``projector`` (the shared semantic projector) is
averaged across domains.  User/item embeddings, the history query, the scale
coefficients and the scorer stay inside the domain.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmipt.interests import InterestPackage
from fedmipt.readout import cross_domain_feature

# The scorer pools over user slots with 0.2 * logsumexp(score / 0.2).  The
# released model has a single user slot, so this equals the slot score up to
# floating-point rounding; it is kept so results match the experiments bit for bit.
_SLOT_POOL_TEMPERATURE = 0.2


class SemanticProjector(nn.Module):
    """Shared linear map from CLIP space to the normalized semantic space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(features), dim=-1)


class DomainModel(nn.Module):
    def __init__(self, domain, dim: int = 64, warm_item_dropout: float = 0.2,
                 readout_temperature: float = 0.2, contrastive_temperature: float = 0.2,
                 mass_gamma: float = 0.5):
        super().__init__()
        self.num_items = domain.num_items
        self.dim = dim
        self.warm_item_dropout = float(warm_item_dropout)
        self.readout_temperature = float(readout_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.mass_gamma = float(mass_gamma)

        # Row ``num_items`` is the history padding item with no modality.
        padding = torch.zeros(1, domain.text_features.shape[1])
        self.register_buffer("text", torch.cat([domain.text_features, padding]))
        self.register_buffer("image", torch.cat([domain.image_features, padding]))
        self.register_buffer(
            "modality_mask",
            torch.cat([domain.modality_mask, torch.zeros(1, 2, dtype=torch.bool)]),
        )
        self.register_buffer("warm_item", domain.item_seen_in_train_tensor.bool())

        # Registration order fixes the parameter-initialization RNG stream.
        self.encoder = SemanticProjector(domain.text_features.shape[1], dim)
        self.user_emb = nn.Embedding(domain.num_users, dim)
        self.item_emb = nn.Embedding(domain.num_items + 1, dim, padding_idx=domain.num_items)
        self.query = nn.Parameter(torch.empty(1, dim))
        self.semantic_scale = nn.Parameter(torch.ones(()))
        self.interest_scale = nn.Parameter(torch.ones(()))
        self.mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 1))
        self._reset_parameters()

    @property
    def projector(self) -> SemanticProjector:
        """The only federated module; keep historical checkpoint keys encoder.*."""
        return self.encoder

    def _reset_parameters(self) -> None:
        # The query is drawn from a snapshot of the RNG state, so it does not
        # shift the stream used by the embeddings below.
        state = torch.random.get_rng_state()
        nn.init.xavier_uniform_(self.query)
        torch.random.set_rng_state(state)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight[:-1], std=0.02)
        with torch.no_grad():
            self.item_emb.weight[-1].zero_()
        # A zero output layer makes every initial score 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def item_semantic(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Masked-mean fusion of projected text/image features: [...] -> [..., D].

        Items without any available modality map to the zero vector.
        """
        text = self.encoder(self.text[item_ids])
        image = self.encoder(self.image[item_ids])
        weight = self.modality_mask[item_ids].to(text.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return F.normalize(weight[..., :1] * text + weight[..., 1:] * image, dim=-1)

    def user_interest(self, history_ids: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """History summary e~_u by attention with the domain query: [B, H] -> [B, 1, D]."""
        history = self.item_semantic(history_ids)
        query = F.normalize(self.query, dim=-1)
        score = torch.einsum("bhd,kd->bkh", history, query) / math.sqrt(self.dim)
        mask = history_mask.unsqueeze(1)
        weight = torch.softmax(score.masked_fill(~mask, -1e9), dim=-1) * mask
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return F.normalize(torch.einsum("bkh,bhd->bkd", weight, history), dim=-1)

    def user_vector(self, user_ids: torch.Tensor, history_ids: torch.Tensor,
                    history_mask: torch.Tensor) -> torch.Tensor:
        """U_u = E_u + gamma_u * e~_u: [B, 1, D]."""
        interest = self.user_interest(history_ids, history_mask)
        return self.user_emb(user_ids).unsqueeze(1) + self.interest_scale * interest

    def item_vector(self, item_ids: torch.Tensor, drop_warm: bool = False) -> torch.Tensor:
        """V_j = gamma_v * e_j (+ E_j for warm items): [N, D].

        In training, ``drop_warm`` hides each warm item's collaborative
        embedding with probability ``warm_item_dropout`` so the model also
        learns to score items from semantics alone, as for cold items.
        """
        semantic = self.semantic_scale * self.item_semantic(item_ids)
        collaborative = self.item_emb(item_ids)
        enabled = self.warm_item[item_ids]
        if drop_warm and self.training:
            keep = torch.rand_like(enabled, dtype=torch.float32) >= self.warm_item_dropout
            enabled = enabled & keep
        return semantic + collaborative * enabled.unsqueeze(-1)

    def predict(self, user: torch.Tensor, item_ids: torch.Tensor,
                package: Optional[InterestPackage] = None, drop_warm: bool = False) -> torch.Tensor:
        """Score f([U_u || V_j || X^cross_{u,j}]).  user: [N, 1, D]; item_ids: [N] -> [N]."""
        semantic = self.item_semantic(item_ids)
        item = self.item_vector(item_ids, drop_warm).unsqueeze(1).expand_as(user)
        if package is None:
            cross = torch.zeros_like(user)
        else:
            cross = cross_domain_feature(
                semantic, package, self.readout_temperature, self.mass_gamma,
            ).unsqueeze(1).expand_as(user)
        score = self.mlp(torch.cat([user, item, cross], dim=-1)).squeeze(-1)
        return _SLOT_POOL_TEMPERATURE * torch.logsumexp(score / _SLOT_POOL_TEMPERATURE, dim=1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, history_ids: torch.Tensor,
                history_mask: torch.Tensor, package: Optional[InterestPackage] = None,
                drop_warm: bool = False) -> torch.Tensor:
        user = self.user_vector(user_ids, history_ids, history_mask)
        return self.predict(user, item_ids, package, drop_warm)

    def item_cl_loss(self, item_ids: torch.Tensor) -> torch.Tensor:
        """ItemCL: InfoNCE between gamma_v * e_j and the detached E_j of warm batch items."""
        item_ids = torch.unique(item_ids)
        item_ids = item_ids[self.warm_item[item_ids]]
        if item_ids.numel() < 2:
            return self.user_emb.weight.sum() * 0.0
        semantic = F.normalize(self.semantic_scale * self.item_semantic(item_ids), dim=-1)
        teacher = F.normalize(self.item_emb(item_ids).detach(), dim=-1)
        labels = torch.arange(item_ids.numel(), device=item_ids.device)
        return F.cross_entropy(semantic @ teacher.T / self.contrastive_temperature, labels)
