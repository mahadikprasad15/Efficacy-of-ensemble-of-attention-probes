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
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def source_probe_slug(source_probe: str) -> str:
    slug = source_probe.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


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


def parse_heads_list(value: str) -> List[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("--num_heads_list produced an empty list")
    return out


def select_topk(rows: List[Dict[str, str]], top_k: int) -> List[Dict[str, str]]:
    for row in rows:
        row["Best AUC"] = float(row["Best AUC"])
        row["Best Accuracy"] = float(row["Best Accuracy"])
        row["Best Layer"] = int(float(row["Best Layer"]))
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


def read_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_id_metrics_from_train(probe_run_dir: Path) -> Dict[str, Optional[float]]:
    val_metrics = read_json_if_exists(probe_run_dir / "results" / "val_metrics.json")
    if not val_metrics:
        return {"id_best_auc": None, "id_best_f1_macro": None, "id_best_accuracy": None}
    history = val_metrics.get("history", [])
    if not history:
        return {"id_best_auc": None, "id_best_f1_macro": None, "id_best_accuracy": None}

    best_auc = None
    best_f1 = None
    best_acc = None
    for row in history:
        if "auc" in row:
            best_auc = float(row["auc"]) if best_auc is None else max(best_auc, float(row["auc"]))
        if "f1_macro" in row:
            best_f1 = float(row["f1_macro"]) if best_f1 is None else max(best_f1, float(row["f1_macro"]))
        if "accuracy" in row:
            best_acc = float(row["accuracy"]) if best_acc is None else max(best_acc, float(row["accuracy"]))
    return {
        "id_best_auc": best_auc,
        "id_best_f1_macro": best_f1,
        "id_best_accuracy": best_acc,
    }


def extract_ood_metrics(
    ood_results_root: Path,
    model_dir: str,
    train_dataset: str,
    target_dataset: str,
    probe_family: str,
    run_name: str,
) -> Dict[str, Optional[float]]:
    metrics_path = (
        ood_results_root
        / model_dir
        / "MultiAttention_probes"
        / f"{train_dataset}_{target_dataset}"
        / probe_family
        / run_name
        / "results"
        / "metrics.json"
    )
    payload = read_json_if_exists(metrics_path)
    if not payload:
        return {"ood_auc": None, "ood_f1_macro": None, "ood_accuracy": None}
    return {
        "ood_auc": float(payload["auc"]) if "auc" in payload else None,
        "ood_f1_macro": float(payload["f1_macro"]) if "f1_macro" in payload else None,
        "ood_accuracy": float(payload["accuracy"]) if "accuracy" in payload else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run top-k multiattention train/eval orchestration")
    parser.add_argument("--pairs_csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--probe_family", type=str, default="gmha")
    parser.add_argument("--num_heads", type=int, default=None, help="Single value fallback if --num_heads_list is omitted")
    parser.add_argument("--num_heads_list", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--use_baseline_layer", action="store_true", default=True)
    parser.add_argument("--no_use_baseline_layer", dest="use_baseline_layer", action="store_false")
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

    if args.num_heads_list:
        heads = parse_heads_list(args.num_heads_list)
    elif args.num_heads is not None:
        heads = [int(args.num_heads)]
    else:
        raise ValueError("Provide --num_heads_list or --num_heads")

    rows = select_topk(read_rows(Path(args.pairs_csv)), args.top_k)
    model_dir = model_to_dir(args.model)
    ood_root = Path(args.ood_results_root)
    summary = []

    for n_heads in heads:
        for row in rows:
            source_probe = row["Source Probe"]
            target_label = row["Target Dataset (Test)"]
            train_dataset = map_source_probe_to_train_dataset_name(source_probe)
            target_dataset = target_to_dataset_name(target_label)
            baseline_layer = int(row["Best Layer"])
            layer_used = baseline_layer if (args.use_baseline_layer and not args.probe_family.startswith("multilayer_")) else args.layer

            run_name = f"{args.probe_family}_h{n_heads}_{source_probe_slug(source_probe)}_L{layer_used}"

            if not args.skip_training:
                train_cmd = [
                    "python",
                    "scripts/training/train_multiattention_probes.py",
                    "--model", args.model,
                    "--dataset", train_dataset,
                    "--probe_family", args.probe_family,
                    "--num_heads", str(n_heads),
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
                    train_cmd.extend(["--layer", str(layer_used)])
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

            id_metrics = extract_id_metrics_from_train(probe_run_dir)
            ood_metrics = extract_ood_metrics(
                ood_results_root=ood_root,
                model_dir=model_dir,
                train_dataset=train_dataset,
                target_dataset=target_dataset,
                probe_family=args.probe_family,
                run_name=run_name,
            )

            summary.append(
                {
                    "source_probe": source_probe,
                    "target_label": target_label,
                    "train_dataset": train_dataset,
                    "target_dataset": target_dataset,
                    "probe_family": args.probe_family,
                    "num_heads": n_heads,
                    "baseline_layer": baseline_layer,
                    "layer_used": layer_used if not args.probe_family.startswith("multilayer_") else None,
                    "best_auc_baseline": row["Best AUC"],
                    "best_accuracy_baseline": row["Best Accuracy"],
                    **id_metrics,
                    **ood_metrics,
                    "probe_run_dir": str(probe_run_dir),
                }
            )

    summary_root = Path(args.ood_results_root) / model_dir / "MultiAttention_probes"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_json = summary_root / "topk_orchestration_summary.json"
    summary_csv = summary_root / "topk_orchestration_summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if summary:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Saved summary json: {summary_json}")
    print(f"Saved summary csv: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
