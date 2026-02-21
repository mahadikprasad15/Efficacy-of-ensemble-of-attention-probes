#!/usr/bin/env python3
"""
Top-5 Layer-Agnostic Follow-Up from Existing Top-20 Baselines.

This pipeline:
1) Loads two baseline top-20 CSVs (Roleplaying->Insider, AI Liar->Insider)
2) Builds a deduplicated global top-K by Source Probe
3) Trains layer-agnostic probes for selected Source Probe + Pooling pairs
4) Evaluates each trained probe on all discovered InsiderTrading test segments
5) Compares LA results against baseline union top-20 (segment-best and global-best)

Primary output root:
  <results_root>/<model_dir>/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, roc_auc_score


REQUIRED_BASELINE_COLUMNS = [
    "Source Probe",
    "Target Dataset (Test)",
    "Best Pooling",
    "Best Layer",
    "Best AUC",
    "Best Accuracy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def ensure_model_root(base: Path, model_dir: str) -> Path:
    return base if base.name == model_dir else (base / model_dir)


def normalize_pooling(value: str) -> str:
    x = str(value).strip().lower()
    mapping = {
        "mean": "mean",
        "max": "max",
        "last": "last",
        "attn": "attn",
        "attention": "attn",
    }
    if x not in mapping:
        raise ValueError(f"Unsupported pooling value: {value}")
    return mapping[x]


def display_pooling(pooling: str) -> str:
    return "Attn" if pooling == "attn" else pooling.capitalize()


def segment_to_title(segment: str) -> str:
    return segment.capitalize()


def source_probe_slug(source_probe: str) -> str:
    slug = source_probe.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


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


def resolve_existing_train_dataset_name(
    requested_dataset: str,
    activations_model_root: Path,
    train_split: str = "train",
) -> Optional[str]:
    candidates = [requested_dataset]
    if requested_dataset.endswith("-full"):
        candidates.append(requested_dataset[: -len("-full")])
    else:
        candidates.append(f"{requested_dataset}-full")
    for ds in candidates:
        split_dir = activations_model_root / ds / train_split
        if split_dir.exists():
            return ds
    return None


def parse_target_segment(value: str) -> Optional[str]:
    v = str(value).strip().lower()
    if "insidertrading" not in v:
        return None
    if "completion" in v:
        return "completion"
    if "system" in v:
        return "system"
    if "user" in v:
        return "user"
    if "prompt" in v:
        return "prompt"
    if "full" in v:
        return "full"
    compact = v.replace(" ", "")
    if compact in {"insidertrading", "deception-insidertrading"}:
        return "full"
    return None


def target_display_from_segment(segment: str) -> str:
    return f"InsiderTrading {segment_to_title(segment)}"


def validate_baseline_df(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_BASELINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


@dataclass
class Candidate:
    rank: int
    source_probe: str
    target_dataset_test: str
    best_pooling: str
    best_layer: int
    best_auc: float
    best_accuracy: float
    source_file: str
    train_dataset_name: str
    train_pooling: str


def build_candidates(
    roleplaying_csv: Path,
    ailiar_csv: Path,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_r = pd.read_csv(roleplaying_csv)
    validate_baseline_df(df_r, "roleplaying top20")
    df_r = df_r.copy()
    df_r["source_file"] = "roleplaying_top20"

    df_a = pd.read_csv(ailiar_csv)
    validate_baseline_df(df_a, "ai liar top20")
    df_a = df_a.copy()
    df_a["source_file"] = "ai_liar_top20"

    union = pd.concat([df_r, df_a], ignore_index=True)

    union["Best AUC"] = pd.to_numeric(union["Best AUC"], errors="coerce")
    union["Best Accuracy"] = pd.to_numeric(union["Best Accuracy"], errors="coerce")
    union["Best Layer"] = pd.to_numeric(union["Best Layer"], errors="coerce").fillna(-1).astype(int)
    union["train_pooling"] = union["Best Pooling"].apply(normalize_pooling)
    union["target_segment"] = union["Target Dataset (Test)"].apply(parse_target_segment)

    # Keep only InsiderTrading targets.
    union = union[union["target_segment"].notna()].copy()
    if union.empty:
        raise RuntimeError("Union top20 is empty after InsiderTrading filtering.")

    union_sorted = union.sort_values(
        by=["Best AUC", "Best Accuracy", "Source Probe", "train_pooling"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    dedup = union_sorted.drop_duplicates(subset=["Source Probe"], keep="first").reset_index(drop=True)
    selected = dedup.head(top_k).copy()
    selected["rank"] = np.arange(1, len(selected) + 1)

    selected["train_dataset_name"] = selected["Source Probe"].apply(
        map_source_probe_to_train_dataset_name
    )

    # Reorder/export columns exactly as requested.
    cols = [
        "rank",
        "Source Probe",
        "Target Dataset (Test)",
        "Best Pooling",
        "Best Layer",
        "Best AUC",
        "Best Accuracy",
        "source_file",
        "train_dataset_name",
        "train_pooling",
    ]
    selected = selected[cols].copy()

    return selected, union_sorted


def discover_insider_targets(activations_model_root: Path, split: str) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    if not activations_model_root.exists():
        return targets

    for child in sorted(activations_model_root.iterdir()):
        if not child.is_dir():
            continue
        ds = child.name
        if ds == "Deception-InsiderTrading":
            segment = "full"
        elif ds.startswith("Deception-InsiderTrading-"):
            segment = ds[len("Deception-InsiderTrading-") :].strip().lower()
        else:
            continue

        split_dir = child / split
        manifest = split_dir / "manifest.jsonl"
        shard_exists = any(split_dir.glob("shard_*.safetensors"))
        if split_dir.exists() and manifest.exists() and shard_exists:
            targets.append(
                {
                    "dataset_dir": ds,
                    "segment": segment,
                    "split_dir": str(split_dir),
                }
            )

    targets.sort(key=lambda x: (x["segment"] != "full", x["segment"]))
    return targets


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class LearnedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.query)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class AttentionProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = LearnedAttentionPooling(input_dim)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.attention(x)
        return self.classifier(pooled)


def evaluate_layer_agnostic_probe(
    probe_path: Path,
    pooling: str,
    activations_split_dir: Path,
    device: torch.device,
) -> Dict:
    manifest_path = activations_split_dir / "manifest.jsonl"
    shard_paths = sorted(activations_split_dir.glob("shard_*.safetensors"))
    if not manifest_path.exists() or not shard_paths:
        raise FileNotFoundError(f"Missing manifest/shards in {activations_split_dir}")

    label_map: Dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            label_map[obj["id"]] = int(obj.get("label", -1))

    model: Optional[nn.Module] = None
    n_layers: Optional[int] = None
    probs_by_layer: List[List[float]] = []
    labels_by_layer: List[List[int]] = []
    n_examples = 0

    state_dict = torch.load(str(probe_path), map_location=device)

    with torch.no_grad():
        for shard_path in shard_paths:
            tensors = load_file(str(shard_path))
            for eid, tensor in tensors.items():
                label = label_map.get(eid, -1)
                if label == -1:
                    continue
                if tensor.ndim != 3:
                    continue

                x = tensor.float()
                if model is None:
                    n_layers = int(x.shape[0])
                    hidden_dim = int(x.shape[-1])
                    if pooling == "attn":
                        model = AttentionProbe(hidden_dim).to(device)
                    else:
                        model = LinearProbe(hidden_dim).to(device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    probs_by_layer = [[] for _ in range(n_layers)]
                    labels_by_layer = [[] for _ in range(n_layers)]

                x_dev = x.to(device)
                if pooling == "mean":
                    pooled = x_dev.mean(dim=1)  # (L, D)
                    logits = model(pooled).squeeze(-1)  # (L,)
                elif pooling == "max":
                    pooled = x_dev.max(dim=1).values
                    logits = model(pooled).squeeze(-1)
                elif pooling == "last":
                    pooled = x_dev[:, -1, :]
                    logits = model(pooled).squeeze(-1)
                elif pooling == "attn":
                    logits = model(x_dev).squeeze(-1)  # treat layers as batch
                else:
                    raise ValueError(f"Unsupported pooling for eval: {pooling}")

                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                for layer_idx, p in enumerate(probs):
                    probs_by_layer[layer_idx].append(float(p))
                    labels_by_layer[layer_idx].append(int(label))
                n_examples += 1

    if model is None or n_layers is None or n_examples == 0:
        raise RuntimeError(f"No valid labeled examples found in {activations_split_dir}")

    aucs: List[float] = []
    accs: List[float] = []
    layers = list(range(n_layers))

    for layer_idx in layers:
        y_true = np.array(labels_by_layer[layer_idx], dtype=np.int64)
        y_prob = np.array(probs_by_layer[layer_idx], dtype=np.float32)
        y_pred = (y_prob > 0.5).astype(np.int64)

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        acc = float(accuracy_score(y_true, y_pred))
        aucs.append(auc)
        accs.append(acc)

    best_layer = max(layers, key=lambda l: (aucs[l], accs[l], -l))
    return {
        "pooling": pooling,
        "layers": layers,
        "aucs": aucs,
        "accuracies": accs,
        "best_layer": int(best_layer),
        "best_auc": float(aucs[best_layer]),
        "best_accuracy": float(accs[best_layer]),
        "n_examples": int(n_examples),
        "probe_path": str(probe_path),
        "activations_split_dir": str(activations_split_dir),
        "evaluated_at_utc": utc_now(),
    }


def resolve_device(device_arg: str) -> torch.device:
    x = device_arg.strip().lower()
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_arg}")
    if x == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-5 LA follow-up from union top-20 baselines")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--probes_root", type=str, required=True)
    parser.add_argument("--top20_roleplaying_csv", type=str, required=True)
    parser.add_argument("--top20_ailiar_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--target_split", type=str, default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already completed probe training/evaluations",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--train_results_root",
        type=str,
        default=None,
        help="Optional explicit results root passed to training script --results_dir",
    )
    args = parser.parse_args()

    run_started = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model_dir = model_to_dir(args.model)
    results_model_root = ensure_model_root(Path(args.results_root), model_dir)
    activations_model_root = ensure_model_root(Path(args.activations_root), model_dir)
    probes_model_root = ensure_model_root(Path(args.probes_root), model_dir)

    results_model_root.mkdir(parents=True, exist_ok=True)

    if args.train_results_root:
        train_results_root = Path(args.train_results_root)
    else:
        # Common case: .../results/ood_evaluation/<model_dir> -> sibling .../results/probes_layer_agnostic_top5
        if results_model_root.parent.name == "ood_evaluation":
            train_results_root = results_model_root.parent.parent / "probes_layer_agnostic_top5"
        else:
            train_results_root = results_model_root.parent / "probes_layer_agnostic_top5"

    candidate_csv_out = results_model_root / "layer_agnostic_top5_candidates_from_union_top20.csv"
    manifest_out = results_model_root / "layer_agnostic_top5_selection_manifest.json"
    train_status_out = results_model_root / "layer_agnostic_top5_train_status.jsonl"
    eval_status_out = results_model_root / "layer_agnostic_top5_ood_eval_status.jsonl"
    ood_summary_out = results_model_root / "layer_agnostic_top5_ood_segment_summary.csv"
    comparison_out = results_model_root / "layer_agnostic_vs_baseline_comparison.csv"
    leaderboard_out = results_model_root / "layer_agnostic_global_leaderboard_vs_union_top20.csv"

    reset_file(train_status_out)
    reset_file(eval_status_out)

    roleplaying_csv = Path(args.top20_roleplaying_csv)
    ailiar_csv = Path(args.top20_ailiar_csv)
    if not roleplaying_csv.exists():
        raise FileNotFoundError(f"Missing roleplaying top20 CSV: {roleplaying_csv}")
    if not ailiar_csv.exists():
        raise FileNotFoundError(f"Missing AI liar top20 CSV: {ailiar_csv}")

    print("=" * 88)
    print("Top-5 Layer-Agnostic Insider Follow-Up")
    print("=" * 88)
    print(f"Model: {args.model}")
    print(f"Results root: {results_model_root}")
    print(f"Activations root (model): {activations_model_root}")
    print(f"Probes root (model): {probes_model_root}")
    print(f"Run id: {run_id}")

    # ------------------------------------------------------------------ #
    # 1) Build candidates
    # ------------------------------------------------------------------ #
    candidates_df, baseline_union_df = build_candidates(
        roleplaying_csv=roleplaying_csv,
        ailiar_csv=ailiar_csv,
        top_k=args.top_k,
    )
    candidates_df.to_csv(candidate_csv_out, index=False)

    print(f"\n[Candidates] Selected {len(candidates_df)} source probes (requested top_k={args.top_k})")
    for _, row in candidates_df.iterrows():
        print(
            f"  - Rank {int(row['rank'])}: {row['Source Probe']} | "
            f"pool={row['train_pooling']} | train_ds={row['train_dataset_name']} | "
            f"AUC={float(row['Best AUC']):.6f}"
        )

    manifest: Dict = {
        "run_id": run_id,
        "run_started_utc": run_started,
        "run_completed_utc": None,
        "model": args.model,
        "model_dir": model_dir,
        "paths": {
            "results_model_root": str(results_model_root),
            "activations_model_root": str(activations_model_root),
            "probes_model_root": str(probes_model_root),
            "roleplaying_top20_csv": str(roleplaying_csv),
            "ai_liar_top20_csv": str(ailiar_csv),
            "candidate_csv": str(candidate_csv_out),
            "train_status_jsonl": str(train_status_out),
            "eval_status_jsonl": str(eval_status_out),
            "ood_summary_csv": str(ood_summary_out),
            "comparison_csv": str(comparison_out),
            "leaderboard_csv": str(leaderboard_out),
            "train_results_root": str(train_results_root),
        },
        "config": {
            "top_k": int(args.top_k),
            "target_split": args.target_split,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "skip_existing": bool(args.skip_existing),
            "device": args.device,
        },
        "counts": {
            "candidate_rows": int(len(candidates_df)),
            "train_ok": 0,
            "train_failed": 0,
            "train_skipped": 0,
            "target_segments_discovered": 0,
            "ood_eval_ok": 0,
            "ood_eval_failed": 0,
            "ood_eval_skipped": 0,
            "ood_summary_rows": 0,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 2) Train layer-agnostic probes
    # ------------------------------------------------------------------ #
    repo_root = Path(__file__).resolve().parents[2]
    train_script = repo_root / "scripts" / "training" / "train_layer_agnostic_probe.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    trained_rows: List[Dict] = []
    total_train = len(candidates_df)
    print(f"\n[Training] Starting {total_train} candidate runs")

    for idx, row in candidates_df.iterrows():
        rank = int(row["rank"])
        source_probe = str(row["Source Probe"])
        requested_train_dataset_name = str(row["train_dataset_name"])
        resolved_train_dataset_name = resolve_existing_train_dataset_name(
            requested_train_dataset_name,
            activations_model_root,
            train_split="train",
        )
        train_dataset_name = (
            resolved_train_dataset_name if resolved_train_dataset_name else requested_train_dataset_name
        )
        pooling = str(row["train_pooling"])
        probe_path = probes_model_root / train_dataset_name / pooling / "probe.pt"
        train_split_dir = activations_model_root / train_dataset_name / "train"

        event = {
            "run_id": run_id,
            "event": "train",
            "timestamp_utc": utc_now(),
            "candidate_rank": rank,
            "source_probe": source_probe,
            "train_dataset_name": train_dataset_name,
            "requested_train_dataset_name": requested_train_dataset_name,
            "train_pooling": pooling,
            "train_split_dir": str(train_split_dir),
            "probe_path": str(probe_path),
            "status": None,
            "returncode": None,
            "error": None,
        }

        if resolved_train_dataset_name is None or not train_split_dir.exists():
            event["status"] = "skipped_missing_train_data"
            append_jsonl(train_status_out, event)
            manifest["counts"]["train_skipped"] += 1
            print(
                f"[Train {idx+1}/{total_train}] SKIP missing train dir: "
                f"requested={requested_train_dataset_name}"
            )
            continue

        if args.skip_existing and probe_path.exists():
            event["status"] = "skipped_existing"
            append_jsonl(train_status_out, event)
            manifest["counts"]["train_skipped"] += 1
            trained_rows.append(
                {
                    **row.to_dict(),
                    "train_dataset_name": train_dataset_name,
                    "probe_path": str(probe_path),
                    "train_status": "skipped_existing",
                }
            )
            print(f"[Train {idx+1}/{total_train}] SKIP existing probe: {probe_path}")
            continue

        cmd = [
            sys.executable,
            str(train_script),
            "--model",
            args.model,
            "--dataset",
            train_dataset_name,
            "--activations_dir",
            str(Path(args.activations_root)),
            "--pooling",
            pooling,
            "--output_dir",
            str(Path(args.probes_root)),
            "--results_dir",
            str(train_results_root),
            "--ood_dataset",
            "Deception-InsiderTrading",
            "--ood_split",
            args.target_split,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch_size",
            str(args.batch_size),
        ]

        print(
            f"[Train {idx+1}/{total_train}] RUN rank={rank} "
            f"{source_probe} | ds={train_dataset_name} | pool={pooling}"
        )
        completed = subprocess.run(cmd, cwd=str(repo_root), check=False)

        event["returncode"] = int(completed.returncode)
        if completed.returncode == 0 and probe_path.exists():
            event["status"] = "ok"
            manifest["counts"]["train_ok"] += 1
            trained_rows.append(
                {
                    **row.to_dict(),
                    "train_dataset_name": train_dataset_name,
                    "probe_path": str(probe_path),
                    "train_status": "ok",
                }
            )
            print(f"[Train {idx+1}/{total_train}] OK  probe={probe_path}")
        else:
            event["status"] = "failed"
            event["error"] = "training command failed or probe.pt missing"
            manifest["counts"]["train_failed"] += 1
            print(
                f"[Train {idx+1}/{total_train}] FAIL rc={completed.returncode} "
                f"probe_exists={probe_path.exists()}"
            )

        append_jsonl(train_status_out, event)

    # ------------------------------------------------------------------ #
    # 3) OOD evaluation on all Insider segments
    # ------------------------------------------------------------------ #
    targets = discover_insider_targets(activations_model_root, args.target_split)
    manifest["counts"]["target_segments_discovered"] = len(targets)
    print(f"\n[OOD] Discovered {len(targets)} Insider targets with split={args.target_split}")
    for t in targets:
        print(f"  - {t['dataset_dir']} ({t['segment']}) -> {t['split_dir']}")

    device = resolve_device(args.device)
    print(f"[OOD] Using device: {device}")

    ood_rows: List[Dict] = []
    total_pairs = max(1, len(trained_rows) * len(targets))
    pair_idx = 0

    for tr in trained_rows:
        source_probe = str(tr["Source Probe"])
        pooling = str(tr["train_pooling"])
        probe_path = Path(str(tr["probe_path"]))
        rank = int(tr["rank"])
        from_slug = source_probe_slug(source_probe)

        if not probe_path.exists():
            print(f"[OOD] SKIP missing probe for rank={rank}: {probe_path}")
            continue

        for target in targets:
            pair_idx += 1
            segment = target["segment"]
            split_dir = Path(target["split_dir"])
            out_dir = (
                results_model_root
                / "layer_agnostic_top5"
                / f"from-{from_slug}-pool-{pooling}"
                / f"to-Deception-InsiderTrading-{segment}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            result_json = out_dir / "results.json"

            event = {
                "run_id": run_id,
                "event": "ood_eval",
                "timestamp_utc": utc_now(),
                "candidate_rank": rank,
                "source_probe": source_probe,
                "train_dataset_name": str(tr["train_dataset_name"]),
                "train_pooling": pooling,
                "target_dataset_dir": str(target["dataset_dir"]),
                "target_segment": segment,
                "split_dir": str(split_dir),
                "result_json_path": str(result_json),
                "status": None,
                "error": None,
            }

            if args.skip_existing and result_json.exists():
                try:
                    existing = json.loads(result_json.read_text(encoding="utf-8"))
                    ood_rows.append(
                        {
                            "candidate_rank": rank,
                            "Source Probe": source_probe,
                            "train_dataset_name": str(tr["train_dataset_name"]),
                            "train_pooling": pooling,
                            "target_segment": segment,
                            "best_layer": int(existing["best_layer"]),
                            "best_auc": float(existing["best_auc"]),
                            "best_accuracy": float(existing["best_accuracy"]),
                            "results_json_path": str(result_json),
                        }
                    )
                except Exception:
                    pass
                event["status"] = "skipped_existing"
                append_jsonl(eval_status_out, event)
                manifest["counts"]["ood_eval_skipped"] += 1
                print(
                    f"[OOD {pair_idx}/{total_pairs}] SKIP existing "
                    f"{source_probe} -> InsiderTrading {segment_to_title(segment)}"
                )
                continue

            print(
                f"[OOD {pair_idx}/{total_pairs}] RUN {source_probe} "
                f"({pooling}) -> InsiderTrading {segment_to_title(segment)}"
            )
            try:
                result = evaluate_layer_agnostic_probe(
                    probe_path=probe_path,
                    pooling=pooling,
                    activations_split_dir=split_dir,
                    device=device,
                )
                result.update(
                    {
                        "candidate_rank": rank,
                        "source_probe": source_probe,
                        "train_dataset_name": str(tr["train_dataset_name"]),
                        "target_dataset_dir": str(target["dataset_dir"]),
                        "target_segment": segment,
                    }
                )
                result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

                ood_rows.append(
                    {
                        "candidate_rank": rank,
                        "Source Probe": source_probe,
                        "train_dataset_name": str(tr["train_dataset_name"]),
                        "train_pooling": pooling,
                        "target_segment": segment,
                        "best_layer": int(result["best_layer"]),
                        "best_auc": float(result["best_auc"]),
                        "best_accuracy": float(result["best_accuracy"]),
                        "results_json_path": str(result_json),
                    }
                )
                event["status"] = "ok"
                manifest["counts"]["ood_eval_ok"] += 1
                print(
                    f"[OOD {pair_idx}/{total_pairs}] OK  "
                    f"best={display_pooling(pooling)} L{result['best_layer']} "
                    f"AUC={result['best_auc']:.6f}"
                )
            except Exception as exc:
                event["status"] = "failed"
                event["error"] = str(exc)
                manifest["counts"]["ood_eval_failed"] += 1
                print(f"[OOD {pair_idx}/{total_pairs}] FAIL {exc}")

            append_jsonl(eval_status_out, event)

    # ------------------------------------------------------------------ #
    # 4) Save OOD summary
    # ------------------------------------------------------------------ #
    ood_summary_cols = [
        "candidate_rank",
        "Source Probe",
        "train_dataset_name",
        "train_pooling",
        "target_segment",
        "best_layer",
        "best_auc",
        "best_accuracy",
        "results_json_path",
    ]
    ood_summary_df = pd.DataFrame(ood_rows)
    if ood_summary_df.empty:
        ood_summary_df = pd.DataFrame(columns=ood_summary_cols)
    else:
        ood_summary_df = ood_summary_df.sort_values(
            by=["best_auc", "best_accuracy", "candidate_rank", "target_segment"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    ood_summary_df.to_csv(ood_summary_out, index=False)
    manifest["counts"]["ood_summary_rows"] = int(len(ood_summary_df))

    # ------------------------------------------------------------------ #
    # 5) Baseline comparisons
    # ------------------------------------------------------------------ #
    baseline_for_cmp = baseline_union_df.copy()
    baseline_for_cmp["target_segment"] = baseline_for_cmp["Target Dataset (Test)"].apply(parse_target_segment)
    baseline_for_cmp = baseline_for_cmp[baseline_for_cmp["target_segment"].notna()].copy()

    if baseline_for_cmp.empty:
        baseline_segment_best: Dict[str, float] = {}
        baseline_global_best = np.nan
    else:
        baseline_segment_best = (
            baseline_for_cmp.groupby("target_segment")["Best AUC"].max().astype(float).to_dict()
        )
        baseline_global_best = float(baseline_for_cmp["Best AUC"].max())

    cmp_df = ood_summary_df.copy()
    if not cmp_df.empty:
        cmp_df["baseline_segment_best_auc"] = cmp_df["target_segment"].map(baseline_segment_best)
        cmp_df["baseline_global_best_auc"] = baseline_global_best
        cmp_df["delta_vs_segment_best"] = cmp_df["best_auc"] - cmp_df["baseline_segment_best_auc"]
        cmp_df["delta_vs_global_best"] = cmp_df["best_auc"] - cmp_df["baseline_global_best_auc"]
        cmp_df["beats_segment_best"] = cmp_df["delta_vs_segment_best"] > 0
        cmp_df["beats_global_best"] = cmp_df["delta_vs_global_best"] > 0
    else:
        cmp_df = pd.DataFrame(
            columns=ood_summary_cols
            + [
                "baseline_segment_best_auc",
                "baseline_global_best_auc",
                "delta_vs_segment_best",
                "delta_vs_global_best",
                "beats_segment_best",
                "beats_global_best",
            ]
        )
    cmp_df.to_csv(comparison_out, index=False)

    # Combined leaderboard: baseline union top20 + LA rows
    baseline_lb = baseline_for_cmp[
        ["Source Probe", "Target Dataset (Test)", "Best Pooling", "Best Layer", "Best AUC", "Best Accuracy"]
    ].copy()
    baseline_lb["System"] = "baseline_union_top20"

    la_lb = ood_summary_df.copy()
    if la_lb.empty:
        la_lb = pd.DataFrame(
            columns=[
                "Source Probe",
                "Target Dataset (Test)",
                "Best Pooling",
                "Best Layer",
                "Best AUC",
                "Best Accuracy",
                "System",
            ]
        )
    else:
        la_lb["Target Dataset (Test)"] = la_lb["target_segment"].apply(target_display_from_segment)
        la_lb["Best Pooling"] = la_lb["train_pooling"].apply(display_pooling)
        la_lb["Best Layer"] = la_lb["best_layer"]
        la_lb["Best AUC"] = la_lb["best_auc"]
        la_lb["Best Accuracy"] = la_lb["best_accuracy"]
        la_lb["System"] = "layer_agnostic_top5"
        la_lb = la_lb[
            ["Source Probe", "Target Dataset (Test)", "Best Pooling", "Best Layer", "Best AUC", "Best Accuracy", "System"]
        ]

    leaderboard = pd.concat(
        [
            baseline_lb[
                ["Source Probe", "Target Dataset (Test)", "Best Pooling", "Best Layer", "Best AUC", "Best Accuracy", "System"]
            ],
            la_lb,
        ],
        ignore_index=True,
    )
    leaderboard = leaderboard.sort_values(
        by=["Best AUC", "Best Accuracy", "System", "Source Probe", "Target Dataset (Test)"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    leaderboard.insert(0, "Rank", np.arange(1, len(leaderboard) + 1))
    leaderboard.to_csv(leaderboard_out, index=False)

    # Finalize manifest
    manifest["run_completed_utc"] = utc_now()
    manifest["baseline_stats"] = {
        "segment_best_auc": baseline_segment_best,
        "global_best_auc": None if pd.isna(baseline_global_best) else float(baseline_global_best),
        "union_rows": int(len(baseline_union_df)),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Candidates CSV:   {candidate_csv_out}")
    print(f"Train status:     {train_status_out}")
    print(f"OOD eval status:  {eval_status_out}")
    print(f"OOD summary CSV:  {ood_summary_out}")
    print(f"Comparison CSV:   {comparison_out}")
    print(f"Leaderboard CSV:  {leaderboard_out}")
    print(f"Manifest JSON:    {manifest_out}")


if __name__ == "__main__":
    main()
