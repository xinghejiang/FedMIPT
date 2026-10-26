"""FedMIPT entry point.

    python main.py train --config configs/task1_phone_sports.json --seed 42
    python main.py test  --run_dir runs/task1_phone_sports_seed42

``train`` selects the best round on Validation and never reads Test labels.
``test`` evaluates that fixed checkpoint once and refuses to overwrite an
existing Test result.
"""

import argparse
import datetime
import logging
import time
from pathlib import Path

from fedmipt.config import Config, config_from_dict, load_config
from fedmipt.data import DualDomainData
from fedmipt.trainer import Trainer
from fedmipt.utils import (
    REPO_ROOT, code_version, configure_cpu_threads, data_fingerprint, environment, file_sha256,
    load_json, resolve_device, save_json, set_seed, setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedMIPT: federated multi-interest prototype transport")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and select a checkpoint on Validation")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--seed", required=True, type=int, help="training seed")
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="override one config value (JSON literal), e.g. --set num_interests=4")
    train.add_argument("--run_dir", type=Path, default=None,
                       help="output directory (default: runs/<task>[_<tag>]_seed<seed>)")
    train.add_argument("--tag", default=None, help="label added to the default run directory")

    test = commands.add_parser("test", help="evaluate the best checkpoint of a finished run on Test")
    test.add_argument("--run_dir", required=True, type=Path)

    for sub in (train, test):
        sub.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        sub.add_argument("--cpu_threads", type=int, default=4)
    return parser.parse_args()


def resolve_data_dir(config: Config) -> Path:
    path = Path(config.data_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def load_data(config: Config, validation_only: bool = False) -> DualDomainData:
    return DualDomainData(
        resolve_data_dir(config),
        {"name": config.domain_a_name, "subdir": config.domain_a_subdir},
        {"name": config.domain_b_name, "subdir": config.domain_b_subdir},
        clip_dim=config.clip_dim, use_text=config.use_text, use_image=config.use_image,
        content_keep_ratio=config.content_keep_ratio, content_mask_seed=config.content_mask_seed,
        validation_only=validation_only,
    )


def runtime_info(args: argparse.Namespace, device) -> dict:
    return {"device": str(device), "cpu_threads": args.cpu_threads, "environment": environment()}


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides, seed=args.seed)
    name = config.task + (f"_{args.tag}" if args.tag else "") + f"_seed{config.seed}"
    run_dir = (args.run_dir or REPO_ROOT / "runs" / name).resolve()
    # Never reuse a run directory: a repeated run needs a new --tag or --run_dir.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    setup_logging(run_dir / "train.log")
    started = datetime.datetime.now()
    clock = time.perf_counter()

    configure_cpu_threads(args.cpu_threads)
    set_seed(config.seed)
    device = resolve_device(args.device)
    fingerprint = data_fingerprint(resolve_data_dir(config),
                                   (config.domain_a_subdir, config.domain_b_subdir))
    save_json(config.to_dict(), run_dir / "config.json")
    logging.info("Run directory: %s", run_dir)
    logging.info("Device: %s | data sha256: %s", device, fingerprint["sha256"])

    data = load_data(config, validation_only=True)
    trainer = Trainer(config, data, device)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    result = trainer.fit(checkpoint_path, {
        "config": config.to_dict(), "data_sha256": fingerprint["sha256"],
    })
    save_json({
        "task": config.task,
        "tag": args.tag,
        "release": "release3-original-task1-fixed-null",
        "overrides": args.overrides,
        "seeds": {"training": config.seed, "content_mask": config.content_mask_seed,
                  "noise": config.noise_seed if config.noise_seed is not None else config.seed},
        "data": {"dir": config.data_dir, "sha256": fingerprint["sha256"]},
        "code": code_version(),
        "config": config.to_dict(),
        "best_round": result["best_round"],
        "validation": result["best_validation"],
        "checkpoint": {"path": "checkpoints/best.pt", "sha256": file_sha256(checkpoint_path)},
        "history": result["history"],
        "test": None,
        "runtime": {**runtime_info(args, device), "started_at": started.isoformat(),
                    "elapsed_seconds": time.perf_counter() - clock},
    }, run_dir / "summary.json")
    save_json(fingerprint, run_dir / "data_files.json")
    (run_dir / "status").mkdir(exist_ok=True)
    save_json({"rc": 0, "test": None}, run_dir / "status/complete.json")
    logging.info("Training finished; Test was not evaluated.")


def test(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    summary = load_json(run_dir / "summary.json")
    output = run_dir / "test_summary.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; Test is evaluated only once per run")
    config = config_from_dict(load_json(run_dir / "config.json"))
    config.validate()
    checkpoint_path = run_dir / summary["checkpoint"]["path"]
    if file_sha256(checkpoint_path) != summary["checkpoint"]["sha256"]:
        raise RuntimeError(f"{checkpoint_path} differs from the checkpoint recorded in summary.json")
    fingerprint = data_fingerprint(resolve_data_dir(config),
                                   (config.domain_a_subdir, config.domain_b_subdir))
    if fingerprint["sha256"] != summary["data"]["sha256"]:
        raise RuntimeError(f"data in {config.data_dir} differ from the data used for training")

    setup_logging(run_dir / "test.log")
    configure_cpu_threads(args.cpu_threads)
    set_seed(config.seed)
    device = resolve_device(args.device)
    trainer = Trainer(config, load_data(config), device)
    result = trainer.test(checkpoint_path)
    save_json({
        "task": config.task,
        "seed": config.seed,
        "best_round": result["best_round"],
        "validation": summary["validation"],
        "test": result["test"],
        "checkpoint_sha256": summary["checkpoint"]["sha256"],
        "data_sha256": fingerprint["sha256"],
        "code": code_version(),
        "runtime": {**runtime_info(args, device),
                    "evaluated_at": datetime.datetime.now().isoformat()},
    }, output)
    save_json({"rc": 0, "test": True}, run_dir / "status/test_complete.json")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
