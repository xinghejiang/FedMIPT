"""Aggregate finished runs into mean ± sample standard deviation over seeds.

    python scripts/summarize.py runs/ [--split test] [--output results.json]

Reads only ``summary.json`` (Validation) and ``test_summary.json`` (Test);
no log parsing.  Runs are grouped by (task, tag).
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = ("overall", "warm", "cold")
METRICS = ("hr@10", "ndcg@10")


def load_runs(root: Path, split: str):
    runs = defaultdict(list)
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if split == "validation":
            result = summary["validation"]
        else:
            test_path = summary_path.with_name("test_summary.json")
            if not test_path.exists():
                print(f"skip (no Test result): {summary_path.parent.name}", file=sys.stderr)
                continue
            result = json.loads(test_path.read_text())["test"]
        runs[(summary["task"], summary.get("tag"))].append((summary["config"]["seed"], result))
    return runs


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(results):
    table = {}
    for scope in results[0]:
        if scope == "diagnostics" or not isinstance(results[0][scope], dict):
            continue
        for group in GROUPS:
            for metric in METRICS:
                mean, std = mean_std([r[scope][group]["metrics"][metric] for r in results])
                table[f"{scope}/{group}/{metric}"] = {"mean": mean, "std": std}
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = {}
    for (task, tag), entries in sorted(load_runs(args.runs, args.split).items(), key=str):
        seeds = sorted(seed for seed, _ in entries)
        table = aggregate([result for _, result in entries])
        name = task + (f" [{tag}]" if tag else "")
        output[name] = {"seeds": seeds, "metrics": table}
        print(f"\n## {name}  ({args.split}, seeds={seeds})")
        print("| scope | group | HR@10 | NDCG@10 |\n|---|---|---|---|")
        for key in table:
            scope, group, metric = key.split("/")
            if metric != "hr@10":
                continue
            cells = []
            for m in METRICS:
                value = table[f"{scope}/{group}/{m}"]
                cells.append("n/a" if value["mean"] is None
                             else f"{value['mean']:.4f} ± {value['std']:.4f}")
            print(f"| {scope} | {group} | {cells[0]} | {cells[1]} |")
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
