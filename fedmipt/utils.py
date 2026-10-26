"""Seeding, devices, logging, run metadata."""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cpu_threads(num_threads: int) -> None:
    if num_threads <= 0:
        raise ValueError(f"cpu_threads must be positive, got {num_threads}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(num_threads)
    torch.set_num_threads(num_threads)


def resolve_device(name: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not mps_available:
            raise RuntimeError("--device mps was requested but MPS is unavailable")
        return torch.device("mps")
    if name != "auto":
        raise ValueError(f"unknown device: {name!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if mps_available else torch.device("cpu")


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def save_json(data: Any, path: Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_fingerprint(data_dir: Path, subdirs) -> Dict[str, Any]:
    """SHA-256 of every file in the domain directories, plus one combined digest.

    Stored with each run and checked again before Test evaluation, so a
    checkpoint is never scored on a different split.
    """
    files = {}
    for subdir in subdirs:
        root = Path(data_dir) / subdir
        if not root.is_dir():
            raise FileNotFoundError(f"domain directory not found: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"):
            files[path.relative_to(data_dir).as_posix()] = file_sha256(path)
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": combined, "files": files}


def code_version() -> Dict[str, Any]:
    """Git commit and dirty flag of the repository, if it is a Git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    files = {path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
             for path in [REPO_ROOT / "main.py", *sorted((REPO_ROOT / "fedmipt").glob("*.py")),
                          *sorted((REPO_ROOT / "configs").glob("*.json"))]}
    source_sha256 = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "."))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source_sha256": source_sha256, "files": files}
    return {"commit": commit, "dirty": dirty, "source_sha256": source_sha256, "files": files}


def environment() -> Dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
