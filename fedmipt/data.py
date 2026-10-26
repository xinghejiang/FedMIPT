"""Data loading for one pair of domains with shared user IDs.

Each domain directory contains ``train/vali/test.csv`` (leave-one-out split),
item-aligned CLIP features, a modality availability mask, and the fixed
Validation/Test candidate sets.  See ``data/README.md`` for the file contract.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_interactions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["uid", "iid", "rating"]
    if any(column not in frame for column in required):
        raise ValueError(f"{path} must contain columns {required}")
    frame = frame[required].copy()
    frame["uid"] = frame["uid"].astype(np.int64)
    frame["iid"] = frame["iid"].astype(np.int64)
    frame["rating"] = frame["rating"].astype(np.float32)
    if frame.empty or (frame[["uid", "iid"]].to_numpy() < 0).any():
        raise ValueError(f"{path} is empty or contains negative IDs")
    return frame


def contiguous_size(values: pd.Series, label: str) -> int:
    actual = np.sort(np.unique(values))
    expected = np.arange(int(actual[-1]) + 1)
    if not np.array_equal(actual, expected):
        raise ValueError(f"{label} IDs must be contiguous and start at 0")
    return len(expected)


@dataclass
class History:
    """Padded interaction histories: ``item_ids`` [U, H] (pad = num_items), ``mask`` [U, H]."""

    item_ids: torch.Tensor
    mask: torch.Tensor


class DomainData:
    def __init__(
        self,
        name: str,
        root: Path,
        clip_dim: int,
        use_text: bool = True,
        use_image: bool = True,
        content_keep_ratio: float = 1.0,
        content_mask_seed: int = 2026,
        validation_only: bool = False,
    ):
        self.name = name
        self.root = Path(root)
        self.train = read_interactions(self.root / "train.csv")
        self.vali = read_interactions(self.root / "vali.csv")
        self.validation_only = validation_only
        self.test = self.vali.iloc[:0].copy() if validation_only else read_interactions(self.root / "test.csv")
        all_interactions = pd.concat([self.train, self.vali, self.test], ignore_index=True)
        self.num_users = contiguous_size(all_interactions["uid"], f"{name} user")
        self.num_items = (np.load(self.root / "modality_mask.npy", mmap_mode="r").shape[0]
                          if validation_only else contiguous_size(all_interactions["iid"], f"{name} item"))
        if all_interactions.iid.max() >= self.num_items:
            raise ValueError("interaction item ID exceeds the feature catalog")

        self.text_features = self._load_feature("text_clip.npy", clip_dim)
        self.image_features = self._load_feature("image_clip.npy", clip_dim)
        self.modality_mask = torch.from_numpy(
            self._modality_mask(use_text, use_image, content_keep_ratio, content_mask_seed)
        )

        seen = np.zeros(self.num_items, dtype=np.bool_)
        seen[self.train["iid"].unique()] = True
        seen_path = self.root / "item_seen_in_train_mask.npy"
        if seen_path.exists():
            stored = np.load(seen_path, allow_pickle=False).astype(np.bool_, copy=False)
            if not np.array_equal(stored, seen):
                raise ValueError(f"{seen_path} disagrees with train.csv")
        self.item_seen_in_train = seen
        self.item_seen_in_train_tensor = torch.from_numpy(seen.copy())

        # Validation is scored against train-only histories; Test histories
        # additionally contain the Validation interaction.
        self.train_history = self._history(include_validation=False)
        self.test_history = self._history(include_validation=True)
        self.train_observed = self.train.groupby("uid")["iid"].apply(set).to_dict()

    def _load_feature(self, filename: str, clip_dim: int) -> torch.Tensor:
        array = np.load(self.root / filename, allow_pickle=False).astype(np.float32, copy=False)
        if array.shape != (self.num_items, clip_dim):
            raise ValueError(
                f"{self.name} {filename} has shape {array.shape}, "
                f"expected ({self.num_items}, {clip_dim})"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{self.name} {filename} contains NaN or Inf")
        return torch.from_numpy(array)

    def _modality_mask(self, use_text: bool, use_image: bool,
                       keep_ratio: float, seed: int) -> np.ndarray:
        """Return the [num_items, 2] (text, image) availability mask.

        Missing content is expressed only through this mask; feature rows of
        unavailable modalities are never read as zero vectors.
        """
        mask = np.load(self.root / "modality_mask.npy", allow_pickle=False)
        if mask.shape != (self.num_items, 2):
            raise ValueError(f"{self.name} modality_mask.npy has shape {mask.shape}")
        mask = mask.astype(np.bool_, copy=True)
        if not use_text:
            mask[:, 0] = False
        if not use_image:
            mask[:, 1] = False
        if keep_ratio < 1.0:
            rng = np.random.default_rng(seed)
            keep = np.zeros(self.num_items, dtype=np.bool_)
            keep[rng.permutation(self.num_items)[: int(round(self.num_items * keep_ratio))]] = True
            mask &= keep[:, None]
        return mask

    def _history(self, include_validation: bool) -> History:
        frame = self.train
        if include_validation:
            frame = pd.concat([self.train, self.vali], ignore_index=True)
        grouped = frame.groupby("uid")["iid"].apply(list).to_dict()
        if any(not grouped.get(uid) for uid in range(self.num_users)):
            raise ValueError(f"{self.name} contains a user without training history")
        length = max(len(items) for items in grouped.values())
        item_ids = torch.full((self.num_users, length), self.num_items, dtype=torch.long)
        mask = torch.zeros((self.num_users, length), dtype=torch.bool)
        for uid in range(self.num_users):
            items = torch.tensor(grouped[uid], dtype=torch.long)
            item_ids[uid, : len(items)] = items
            mask[uid, : len(items)] = True
        return History(item_ids, mask)

    def history(self, split: str) -> History:
        if split in ("train", "vali"):
            return self.train_history
        if split == "test":
            if self.validation_only:
                raise ValueError("Test is unavailable during Validation-only training")
            return self.test_history
        raise ValueError(f"unknown split: {split!r}")

    def make_train_loader(self, num_negatives: int, batch_size: int, seed: int) -> DataLoader:
        """Pointwise BCE samples: each positive with ``num_negatives`` unobserved items."""
        rng = np.random.default_rng(seed)
        samples = []
        for row in self.train.itertuples(index=False):
            uid, iid = int(row.uid), int(row.iid)
            samples.append((uid, iid, 1.0))
            observed = self.train_observed[uid]
            if len(observed) >= self.num_items:
                continue
            unique_target = min(num_negatives, self.num_items - len(observed))
            negatives = []
            selected = set()
            while len(negatives) < unique_target:
                candidate = int(rng.integers(0, self.num_items))
                if candidate not in observed and candidate not in selected:
                    selected.add(candidate)
                    negatives.append(candidate)
            if len(negatives) < num_negatives:
                negatives.extend(
                    rng.choice(negatives, size=num_negatives - len(negatives), replace=True).tolist()
                )
            samples.extend((uid, item, 0.0) for item in negatives)
        array = np.asarray(samples)
        dataset = TensorDataset(
            torch.from_numpy(array[:, 0].astype(np.int64)),
            torch.from_numpy(array[:, 1].astype(np.int64)),
            torch.from_numpy(array[:, 2].astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    def _positives(self, split: str) -> np.ndarray:
        frame = self.vali if split == "vali" else self.test
        counts = frame.groupby("uid").size().reindex(range(self.num_users), fill_value=0)
        if not (counts == 1).all():
            raise ValueError(f"{self.name} {split}.csv must contain exactly one positive per user")
        return frame.set_index("uid")["iid"].reindex(range(self.num_users)).to_numpy(np.int64)

    def evaluation_candidates(self, split: str, num_negatives: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load the fixed candidate set; column 0 is the held-out positive.

        Candidates are part of the released split.  They are never resampled,
        so every method and seed ranks the same items.
        """
        if split not in ("vali", "test"):
            raise ValueError(f"evaluation split must be 'vali' or 'test', got {split!r}")
        if self.validation_only and split == "test":
            raise ValueError("Test is unavailable during Validation-only training")
        path = self.root / f"{split}_candidates.npz"
        if not path.exists():
            raise FileNotFoundError(f"fixed candidate file is required: {path}")
        with np.load(path, allow_pickle=False) as payload:
            items = payload["items"].astype(np.int64, copy=False)
        if items.shape != (self.num_users, num_negatives + 1):
            raise ValueError(
                f"{path} has shape {items.shape}, expected ({self.num_users}, {num_negatives + 1})"
            )
        if not np.array_equal(items[:, 0], self._positives(split)):
            raise ValueError(f"{path} column 0 must contain the {split} positives")
        labels = np.zeros_like(items)
        labels[:, 0] = 1
        return items, labels

    def positive_is_cold(self, split: str) -> np.ndarray:
        """Cold = the held-out positive never occurs in this domain's train.csv."""
        path = self.root / f"{split}_positive_is_cold.npy"
        if path.exists():
            values = np.load(path, allow_pickle=False).astype(np.bool_, copy=False)
            if values.shape != (self.num_users,):
                raise ValueError(f"{path} has shape {values.shape}, expected ({self.num_users},)")
            return values
        return ~self.item_seen_in_train[self._positives(split)]


class DualDomainData:
    def __init__(self, data_dir: Path, domain_a: Dict[str, str], domain_b: Dict[str, str],
                 clip_dim: int, use_text: bool = True, use_image: bool = True,
                 content_keep_ratio: float = 1.0, content_mask_seed: int = 2026,
                 validation_only: bool = False):
        data_dir = Path(data_dir)
        common = dict(clip_dim=clip_dim, use_text=use_text, use_image=use_image,
                      content_keep_ratio=content_keep_ratio, validation_only=validation_only)
        # Domain B uses the next content-mask seed so the two masks are independent.
        self.domain_a = DomainData(domain_a["name"], data_dir / domain_a["subdir"],
                                   content_mask_seed=content_mask_seed, **common)
        self.domain_b = DomainData(domain_b["name"], data_dir / domain_b["subdir"],
                                   content_mask_seed=content_mask_seed + 1, **common)
        if self.domain_a.num_users != self.domain_b.num_users:
            raise ValueError("both domains must use the same contiguous user IDs")
