#!/usr/bin/env python3
"""
Optional orchestrator for Top-K baseline pairs.

This script does not contain training/evaluation logic; it only dispatches
the decoupled scripts:
  - scripts/training/train_multiattention_probes.py
  - scripts/evaluation/evaluate_ood_multiattention.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def normalize_pooling(value: str) -> str:
    x = str(value).strip().lower()
    mapping = {"mean": "mean", "max": "max", "last": "last", "attn": "attn", "attention": "attn"}
    if x not in mapping:
        raise ValueError(f"Unsupported pooling value: {value}")
    return mapping[x]


def parse_source_probe(source_probe: str) -> Tuple[str, str]:
    s = source_probe.strip()
    if s.startswith("Roleplaying "):
        return "Deception-Roleplaying", s[len("Roleplaying ") :].strip().lower()
    if s.startswith("AI Liar "):
        return "Deception-AILiar", s[len("AI Liar ") :].strip().lower()
    raise ValueError(f"Unrecognized Source Probe format: {source_probe}")


def map_source_probe_to_train_dataset_name(source_probe: str) -> str:
    base_dataset, segment = parse_source_probe(source_probe)
    return base_dataset if segment == "full" else f"{base_dataset}-{segment}"


def target_to_dataset_name(value: str) -> str:
    v = str(value).strip().lower()
    if "insidertrading" in v:
        base = "Deception-InsiderTrading"
    elif "roleplaying" in v:
        base = "Deception-Roleplaying"
    elif "ai liar" in v or "ailiar" in v:
        base = "Deception-AILiar"
    else:
        raise ValueError(f"Unsupported target dataset value: {value}")

    segment = "full"
    if "completion" in v:
        segment = "completion"
    elif "system" in v:
        segment = "system"
    elif "user" in v:
        segment = "user"
    elif "prompt" in v:
        segment = "prompt"
    elif "full" in v:
        segment = "full"

    return base if segment == "full" else f"{base}-{segment}"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_topk(rows: List[Dict[str, str]], top_k: int) -> List[Dict[str, str]]:
    for row in rows:
        row["Best AUC"] = float(row["Best AUC"])
        row["Best Accuracy"] = float(row["Best Accuracy"])
    ordered = sorted(rows, key=lambda r: (r["Best AUC"], r["Best Accuracy"]), reverse=True)
    dedup = []
    seen = set()
    for row in ordered:
        source = row["Source Probe"]
        if source in seen:
            continue
        seen.add(source)
        dedup.append(row)
        if len(dedup) >= top_k:
            break
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description="Run top-k multiattention train/eval orchestration")
    parser.add_argument("--pairs_csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--probe_family", type=str, default="gmha")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--input_layers", type=str, default="")

    parser.add_argument("--activations_root", type=str, default="data/activations")
    parser.add_argument("--probes_root", type=str, default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes")
    parser.add_argument("--ood_results_root", type=str, default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--target_split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    rows = select_topk(read_rows(Path(args.pairs_csv)), args.top_k)
    model_dir = model_to_dir(args.model)
    summary = []

    for row in rows:
        source_probe = row["Source Probe"]
        target_label = row["Target Dataset (Test)"]
        train_dataset = map_source_probe_to_train_dataset_name(source_probe)
        target_dataset = target_to_dataset_name(target_label)
        run_name = f"{args.probe_family}_topk_{source_probe.lower().replace(' ', '_')}"

        if not args.skip_training:
            train_cmd = [
                "python",
                "scripts/training/train_multiattention_probes.py",
                "--model", args.model,
                "--dataset", train_dataset,
                "--probe_family", args.probe_family,
                "--num_heads", str(args.num_heads),
                "--activations_root", args.activations_root,
                "--output_root", args.probes_root,
                "--train_split", args.train_split,
                "--val_split", args.val_split,
                "--run_name", run_name,
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--patience", str(args.patience),
            ]
            if args.probe_family.startswith("multilayer_"):
                train_cmd.extend(["--input_layers", args.input_layers])
            else:
                train_cmd.extend(["--layer", str(args.layer)])
            subprocess.check_call(train_cmd)

        probe_run_dir = (
            Path(args.probes_root)
            / model_dir
            / "MultiAttentionProbes"
            / train_dataset
            / run_name
        )

        if not args.skip_eval:
            eval_cmd = [
                "python",
                "scripts/evaluation/evaluate_ood_multiattention.py",
                "--probe_run_dir", str(probe_run_dir),
                "--model", args.model,
                "--source_dataset", train_dataset,
                "--target_dataset", target_dataset,
                "--target_split", args.target_split,
                "--activations_root", args.activations_root,
                "--output_root", args.ood_results_root,
                "--batch_size", str(args.batch_size),
            ]
            subprocess.check_call(eval_cmd)

        summary.append(
            {
                "source_probe": source_probe,
                "train_dataset": train_dataset,
                "target_dataset": target_dataset,
                "best_auc_baseline": row["Best AUC"],
                "best_accuracy_baseline": row["Best Accuracy"],
                "probe_run_dir": str(probe_run_dir),
            }
        )

    summary_path = (
        Path(args.ood_results_root)
        / model_dir
        / "MultiAttention_probes"
        / "topk_orchestration_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
