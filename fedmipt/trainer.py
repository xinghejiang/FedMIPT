"""Federated training, Validation model selection, and Test evaluation of a fixed checkpoint."""

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmipt.checkpoint import load_checkpoint, save_checkpoint
from fedmipt.config import Config
from fedmipt.data import DomainData, DualDomainData
from fedmipt.evaluation import evaluate_domain
from fedmipt.federated import fedavg_projector
from fedmipt.interests import InterestPackage
from fedmipt.metrics import GROUPS, merge_domains
from fedmipt.models import DomainModel
from fedmipt.transfer import align_received, exchange_packages, select_users
from fedmipt.utils import finite_or_raise

# Per-round training seeds are derived from the run seed with these offsets.
_PRETRAIN_SEED_OFFSET = {"a": 0, "b": 100}
_TRANSFER_SEED_OFFSET = {"a": 1000, "b": 2000}


def user_cl_loss(local: torch.Tensor, received: torch.Tensor, temperature: float) -> torch.Tensor:
    """UserCL: InfoNCE between e~_u and the received source direction r^s_u.

    Both sides are centered over the batch and normalized; other users in the
    batch are negatives.  local, received: [B, D].
    """
    if local.shape[0] < 2:
        return local.sum() * 0.0
    local = F.normalize(local - local.mean(dim=0, keepdim=True), dim=-1)
    received = F.normalize(received - received.mean(dim=0, keepdim=True), dim=-1)
    labels = torch.arange(local.shape[0], device=local.device)
    return F.cross_entropy(local @ received.T / temperature, labels)


class Trainer:
    def __init__(self, config: Config, data: DualDomainData, device: torch.device):
        self.config = config
        self.data = data
        self.device = device
        self.model_a = self._build_model(data.domain_a)
        self.model_b = self._build_model(data.domain_b)
        # Both domains start from the same projector.
        self.model_b.projector.load_state_dict(copy.deepcopy(self.model_a.projector.state_dict()))
        self.optimizer_a = self._build_optimizer(self.model_a)
        self.optimizer_b = self._build_optimizer(self.model_b)
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self, domain: DomainData) -> DomainModel:
        return DomainModel(
            domain, dim=self.config.embedding_dim,
            warm_item_dropout=self.config.warm_item_dropout,
            readout_temperature=self.config.readout_temperature,
            contrastive_temperature=self.config.contrastive_temperature,
            mass_gamma=self.config.ot_mass_gamma,
        ).to(self.device)

    def _build_optimizer(self, model: DomainModel) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)

    # ------------------------------------------------------------------ rounds

    def _aggregate(self) -> None:
        # The strict no-transfer ablation also skips projector aggregation.
        if self.config.transfer_enabled:
            fedavg_projector(self.model_a, self.model_b,
                             len(self.data.domain_a.train), len(self.data.domain_b.train))

    def _exchange(self, split: str, round_index: int = 0) -> Optional[Dict[str, InterestPackage]]:
        if not self.config.transfer_enabled:
            return None
        return exchange_packages(self.model_a, self.model_b, self.data, split,
                                 self.config, self.device, round_index)

    def _train_round(self, round_index: int, offsets: Dict[str, int],
                     packages: Optional[Dict[str, InterestPackage]]) -> Dict[str, Dict[str, float]]:
        seed = self.config.seed + round_index
        stats_a = self._train_domain(self.model_a, self.optimizer_a, self.data.domain_a,
                                     packages["for_a"] if packages else None, seed + offsets["a"])
        stats_b = self._train_domain(self.model_b, self.optimizer_b, self.data.domain_b,
                                     packages["for_b"] if packages else None, seed + offsets["b"])
        self._aggregate()
        return {self.data.domain_a.name: stats_a, self.data.domain_b.name: stats_b}

    def _train_domain(self, model: DomainModel, optimizer: torch.optim.Optimizer,
                      domain: DomainData, received: Optional[InterestPackage],
                      seed: int) -> Dict[str, float]:
        """One local epoch (or ``local_epochs``) of BCE + ItemCL + UserCL."""
        config = self.config
        history = domain.train_history
        model.train()
        totals = {"loss": 0.0, "rec": 0.0, "item_cl": 0.0, "user_cl": 0.0}
        examples = 0
        for epoch in range(config.local_epochs):
            loader = domain.make_train_loader(config.num_negatives, config.batch_size, seed + epoch)
            for user_cpu, item_ids, labels in loader:
                history_ids = history.item_ids[user_cpu].to(self.device)
                history_mask = history.mask[user_cpu].to(self.device)
                user_ids = user_cpu.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)
                # Hide the training positive from its own history input.
                is_positive = (labels > 0.5).unsqueeze(1)
                history_mask = history_mask & ~((history_ids == item_ids.unsqueeze(1)) & is_positive)

                package = None
                if received is not None:
                    package = align_received(model, select_users(received, user_ids),
                                             history_ids, history_mask, config)

                optimizer.zero_grad()
                logits = model(user_ids, item_ids, history_ids, history_mask, package, drop_warm=True)
                rec = self.criterion(logits, labels)
                item_cl = (model.item_cl_loss(item_ids) if (config.use_text or config.use_image)
                           else rec.new_zeros(()))
                user_cl = rec.new_zeros(())
                if received is not None:
                    unique_users = torch.unique(user_ids)
                    if unique_users.numel() >= 2:
                        unique_cpu = unique_users.cpu()
                        # e~_u uses the user's full training history.
                        local = model.user_interest(history.item_ids[unique_cpu].to(self.device),
                                                    history.mask[unique_cpu].to(self.device))[:, 0]
                        user_cl = user_cl_loss(local, received["direction"][unique_users],
                                               config.contrastive_temperature)
                loss = rec + config.item_cl_weight * item_cl + config.user_cl_weight * user_cl
                finite_or_raise(f"{domain.name} training loss", loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

                count = len(labels)
                examples += count
                for key, value in (("loss", loss), ("rec", rec), ("item_cl", item_cl),
                                   ("user_cl", user_cl)):
                    totals[key] += float(value.detach()) * count
        return {key: value / max(1, examples) for key, value in totals.items()}

    # -------------------------------------------------------------- evaluation

    def evaluate(self, split: str) -> dict:
        """Evaluate both domains on ``split``; ``mean`` is the user-weighted average."""
        packages = self._exchange(split)
        result_a = evaluate_domain(self.model_a, self.data.domain_a,
                                   packages["for_a"] if packages else None,
                                   split, self.config, self.device)
        result_b = evaluate_domain(self.model_b, self.data.domain_b,
                                   packages["for_b"] if packages else None,
                                   split, self.config, self.device)
        result = {self.data.domain_a.name: result_a, self.data.domain_b.name: result_b,
                  "mean": merge_domains(result_a, result_b)}
        for name, values in result.items():
            _log_result(split.upper(), name, values)
        return result

    # ------------------------------------------------------------------- fit

    def fit(self, checkpoint_path: Path, checkpoint_metadata: dict) -> dict:
        """Train, select the round with the best Validation Overall NDCG@10, restore it.

        Test data is never touched here.  After restoring the best checkpoint
        Validation is evaluated again and must reproduce the stored score.
        """
        config = self.config
        history = []
        for round_index in range(config.pretrain_rounds):
            stats = self._train_round(round_index, _PRETRAIN_SEED_OFFSET, None)
            history.append({"stage": "pretrain", "round": round_index, "train": stats})
            _log_losses("Pretrain", round_index, stats)

        best_score, best_round, best_validation, stale = float("-inf"), -1, None, 0
        for round_index in range(config.transfer_rounds):
            packages = self._exchange("train", round_index)
            stats = self._train_round(round_index, _TRANSFER_SEED_OFFSET, packages)
            _log_losses("Transfer", round_index, stats)
            validation = self.evaluate("vali")
            score = validation["mean"]["overall"]["metrics"]["ndcg@10"]
            history.append({"stage": "transfer", "round": round_index, "train": stats,
                            "validation_ndcg@10": score})
            if score > best_score + config.min_improvement:
                best_score, best_round, best_validation, stale = score, round_index, validation, 0
                save_checkpoint(checkpoint_path, self.model_a, self.model_b, {
                    "round": round_index, "score": score, "validation": validation,
                    **checkpoint_metadata,
                })
                logging.info("New best validation NDCG@10=%.6f at transfer round %d",
                             score, round_index)
            else:
                stale += 1
                if stale >= config.patience:
                    logging.info("Early stopping at transfer round %d", round_index)
                    break
        if best_round < 0:
            raise RuntimeError("no Validation checkpoint was produced (transfer_rounds=0?)")

        load_checkpoint(checkpoint_path, self.model_a, self.model_b, self.device)
        restored = self.evaluate("vali")
        check_restored_validation(best_score, restored)
        logging.info("Restored best checkpoint: transfer round %d, Validation NDCG@10=%.6f",
                     best_round, best_score)
        return {"best_round": best_round, "best_validation": best_validation, "history": history}

    def test(self, checkpoint_path: Path) -> dict:
        """Evaluate a fixed checkpoint on Test (after re-checking its Validation score)."""
        checkpoint = load_checkpoint(checkpoint_path, self.model_a, self.model_b, self.device)
        check_restored_validation(float(checkpoint["score"]), self.evaluate("vali"))
        return {"best_round": int(checkpoint["round"]), "test": self.evaluate("test")}


def check_restored_validation(expected: float, restored: dict, tolerance: float = 1e-6) -> None:
    value = restored["mean"]["overall"]["metrics"]["ndcg@10"]
    if abs(value - expected) > tolerance:
        raise RuntimeError(
            f"restored checkpoint gives Validation NDCG@10={value:.8f}, expected {expected:.8f}; "
            "the data, code or configuration differ from the run that saved it"
        )


def _log_losses(stage: str, round_index: int, stats: Dict[str, Dict[str, float]]) -> None:
    logging.info("%s %d | %s", stage, round_index,
                 " | ".join(f"{name} loss={values['loss']:.6f}" for name, values in stats.items()))


def _log_result(prefix: str, name: str, result: dict) -> None:
    for group in GROUPS:
        values = ", ".join(
            f"{key}={value:.6f}" if value is not None else f"{key}=n/a"
            for key, value in sorted(result[group]["metrics"].items())
        )
        logging.info("%s %s %s count=%d | %s", prefix, name, group, result[group]["count"], values)
    if "diagnostics" in result:
        logging.info("%s %s OT | %s", prefix, name, ", ".join(
            f"{key}={value:.6f}" for key, value in sorted(result["diagnostics"].items())))
