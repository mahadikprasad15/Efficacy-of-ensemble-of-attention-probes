#!/usr/bin/env python3
"""
Decoupled Stage B pipeline:
- Ingest Stage A layerwise results
- Select top-K (pooling, layer) by macro mean test AUC
- For each seed, build an orthogonal chain of probes (K steps)
- Evaluate every chain step on all test datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from combined_probe_pipeline import (  # noqa: E402
    LayerSampleDataset,
    SplitIndex,
    dataset_segment_name,
    default_run_id,
    ensure_dir,
    model_dir_name,
    read_json,
    utc_now_iso,
    write_json,
)

sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from orthogonal_probe import (  # noqa: E402
    evaluate_probe_with_projection,
    max_cos_to_previous,
    train_probe_with_projection,
    update_q_basis,
)

DEFAULT_TRAIN_DATASETS = [
    "Deception-AILiar",
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-Roleplaying",
]

POOLING_ORDER = ["attn", "last", "max", "mean"]


class BasicFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{self.formatTime(record)} | {record.levelname} | {record.getMessage()}"


def configure_logger(log_path: Path, verbose: bool) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("combined_top3_orthogonal")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = BasicFormatter()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)
    return logger


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def read_layerwise_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["layer"] = int(row["layer"])
            row["auc"] = float(row["auc"])
            row["accuracy"] = float(row.get("accuracy", 0.0))
            row["f1"] = float(row.get("f1", 0.0))
            rows.append(row)
    return rows


def select_top_seeds(rows: Sequence[Dict], top_k: int) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        key = (str(r["pooling"]), int(r["layer"]))
        grouped.setdefault(key, []).append(r)

    scored: List[Dict] = []
    for (pooling, layer), vals in grouped.items():
        aucs = [float(v["auc"]) for v in vals]
        accs = [float(v.get("accuracy", 0.0)) for v in vals]
        f1s = [float(v.get("f1", 0.0)) for v in vals]

        try:
            pidx = POOLING_ORDER.index(pooling)
        except ValueError:
            pidx = len(POOLING_ORDER)

        scored.append(
            {
                "pooling": pooling,
                "layer": int(layer),
                "macro_auc": float(np.mean(aucs)) if aucs else 0.0,
                "macro_accuracy": float(np.mean(accs)) if accs else 0.0,
                "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
                "num_test_datasets": len(vals),
                "pooling_order": pidx,
            }
        )

    scored.sort(
        key=lambda r: (
            float(r["macro_auc"]),
            float(r["macro_accuracy"]),
            float(r["macro_f1"]),
            -int(r["layer"]),
            -int(r["pooling_order"]),
        ),
        reverse=True,
    )
    top = scored[: int(top_k)]
    for i, row in enumerate(top, start=1):
        row["seed_rank"] = i
    return top


def materialize_layer_tensors(split_index: SplitIndex, layer: int, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = LayerSampleDataset(split_index, layer=layer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for xb, yb in loader:
        xs.append(xb.float())
        ys.append(yb.float())

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_progress(progress_path: Path, progress: Dict) -> None:
    progress["updated_at"] = utc_now_iso()
    write_json(progress_path, progress)


def extract_probe_vector(model: LayerProbe) -> torch.Tensor:
    return model.classifier.weight.detach().reshape(-1).clone()


def load_probe(probe_path: Path, pooling: str, d_model: int, device: torch.device) -> LayerProbe:
    state = torch.load(str(probe_path), map_location=device)
    model = LayerProbe(input_dim=d_model, pooling_type=pooling).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_on_test_indices(
    *,
    model: LayerProbe,
    layer: int,
    seed_rank: int,
    pooling: str,
    k: int,
    q_basis: Optional[torch.Tensor],
    test_indices: Dict[str, SplitIndex],
    device: torch.device,
    batch_size: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for test_base, split_index in test_indices.items():
        ds = LayerSampleDataset(split_index, layer=layer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        m = evaluate_probe_with_projection(model=model, loader=loader, device=device, q_basis=q_basis)
        rows.append(
            {
                "timestamp": utc_now_iso(),
                "seed_rank": int(seed_rank),
                "pooling": pooling,
                "layer": int(layer),
                "k": int(k),
                "test_dataset_base": test_base,
                "test_dataset": split_index.split_dirs[0].parent.name,
                "auc": float(m["auc"]),
                "accuracy": float(m["accuracy"]),
                "count": int(m.get("count", 0)),
            }
        )
    return rows


def make_seed_heatmap(
    seed_rank: int,
    rows: Sequence[Dict],
    test_labels: Sequence[str],
    chain_k: int,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    mat = np.full((chain_k, len(test_labels)), np.nan, dtype=np.float64)

    test_to_col = {name: i for i, name in enumerate(test_labels)}
    for row in rows:
        k = int(row["k"])
        if k < 1 or k > chain_k:
            continue
        t = row["test_dataset"]
        if t not in test_to_col:
            continue
        mat[k - 1, test_to_col[t]] = float(row["auc"])

    fig, ax = plt.subplots(figsize=(max(8, len(test_labels) * 1.2), 5.2))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(test_labels)))
    ax.set_xticklabels(test_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(chain_k))
    ax.set_yticklabels([f"k={i}" for i in range(1, chain_k + 1)])
    ax.set_title(f"Seed {seed_rank}: AUC by Chain Step x Test Dataset")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = "NA" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run top3 orthogonal chains from Stage A outputs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--segment", type=str, choices=["completion", "full"], required=True)

    parser.add_argument("--stage_a_results_dir", type=str, required=True)
    parser.add_argument("--stage_a_probe_dir", type=str, default="")

    parser.add_argument("--train_datasets", type=str, default=",".join(DEFAULT_TRAIN_DATASETS))
    parser.add_argument("--test_only_datasets", type=str, default="Deception-InsiderTrading-SallyConcat")

    parser.add_argument("--orth_train_split", type=str, default="train")
    parser.add_argument("--orth_val_split", type=str, default="validation")
    parser.add_argument("--orth_test_split", type=str, default="test")

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--chain_k", type=int, default=10)

    parser.add_argument("--orth_output_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name.strip() or f"combined_datasets_all_{args.segment}s_orth_top{args.top_k}_k{args.chain_k}"

    stage_a_results_dir = Path(args.stage_a_results_dir)
    stage_a_layerwise_csv = stage_a_results_dir / "results" / "layerwise_eval.csv"
    if not stage_a_layerwise_csv.exists():
        raise FileNotFoundError(f"Missing Stage A layerwise CSV: {stage_a_layerwise_csv}")

    stage_a_summary_path = stage_a_results_dir / "results" / "summary.json"
    stage_a_summary = read_json(stage_a_summary_path, default={})

    stage_a_probe_dir = Path(args.stage_a_probe_dir) if args.stage_a_probe_dir.strip() else None
    if stage_a_probe_dir is None:
        probe_hint = stage_a_summary.get("probe_run_dir", "")
        if not probe_hint:
            raise ValueError("Stage A probe dir not provided and not found in Stage A summary.json")
        stage_a_probe_dir = Path(probe_hint)

    model_dir = model_dir_name(args.model)
    activations_root = Path(args.activations_root)

    run_dir = Path(args.orth_output_root) / run_name
    meta_dir = run_dir / "meta"
    ckpt_dir = run_dir / "checkpoints"
    out_dir = run_dir / "results"
    log_dir = run_dir / "logs"
    ensure_dir(meta_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    logger = configure_logger(log_dir / "run.log", verbose=args.verbose)

    status_path = meta_dir / "status.json"
    progress_path = ckpt_dir / "progress.json"
    train_metrics_path = out_dir / "orth_train_metrics.jsonl"
    eval_long_path = out_dir / "orth_chain_eval_long.jsonl"

    progress = read_json(progress_path, default={
        "train_done": [],
        "eval_done": [],
    })
    train_done = set(progress.get("train_done", []))
    eval_done = set(progress.get("eval_done", []))

    train_bases = parse_csv_list(args.train_datasets)
    test_only_bases = parse_csv_list(args.test_only_datasets)
    test_bases: List[str] = []
    for d in train_bases + test_only_bases:
        if d not in test_bases:
            test_bases.append(d)

    run_manifest = {
        "run_name": run_name,
        "created_at": utc_now_iso(),
        "model": args.model,
        "segment": args.segment,
        "stage_a_results_dir": str(stage_a_results_dir),
        "stage_a_probe_dir": str(stage_a_probe_dir),
        "train_datasets": train_bases,
        "test_only_datasets": test_only_bases,
        "test_datasets": test_bases,
        "top_k": int(args.top_k),
        "chain_k": int(args.chain_k),
        "splits": {
            "train": args.orth_train_split,
            "validation": args.orth_val_split,
            "test": args.orth_test_split,
        },
    }
    write_json(meta_dir / "run_manifest.json", run_manifest)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    write_json(status_path, {
        "state": "running",
        "stage": "select_top3",
        "updated_at": utc_now_iso(),
        "message": "selecting top seeds from stage A",
    })

    all_rows = [r for r in read_layerwise_rows(stage_a_layerwise_csv) if str(r.get("segment", args.segment)) == args.segment]
    if not all_rows:
        raise RuntimeError(f"No Stage A rows for segment={args.segment}")

    top_seeds = select_top_seeds(all_rows, top_k=args.top_k)
    write_csv(
        out_dir / "top3_seeds.csv",
        rows=top_seeds,
        fieldnames=["seed_rank", "pooling", "layer", "macro_auc", "macro_accuracy", "macro_f1", "num_test_datasets", "pooling_order"],
    )
    write_json(out_dir / "top3_selection_summary.json", {
        "segment": args.segment,
        "selection_metric": "macro_mean_auc_across_7_tests",
        "top_seeds": top_seeds,
        "updated_at": utc_now_iso(),
    })

    # Build split indices once
    write_json(status_path, {
        "state": "running",
        "stage": "index_data",
        "updated_at": utc_now_iso(),
        "message": "indexing train/val/test splits",
    })

    train_dirs = [
        activations_root / model_dir / dataset_segment_name(d, args.segment) / args.orth_train_split
        for d in train_bases
    ]
    val_dirs = [
        activations_root / model_dir / dataset_segment_name(d, args.segment) / args.orth_val_split
        for d in train_bases
    ]
    test_dirs = {
        d: activations_root / model_dir / dataset_segment_name(d, args.segment) / args.orth_test_split
        for d in test_bases
    }

    for p in train_dirs + val_dirs + list(test_dirs.values()):
        if not p.exists():
            raise FileNotFoundError(f"Missing split dir: {p}")

    train_index = SplitIndex(train_dirs)
    val_index = SplitIndex(val_dirs)
    test_indices = {d: SplitIndex([p]) for d, p in test_dirs.items()}

    d_model = int(train_index.d_model or 0)

    if train_index.num_layers != val_index.num_layers:
        raise ValueError("Train/val layers mismatch")

    # Cache layer tensors by layer id to avoid repeated IO across seeds.
    layer_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    write_json(status_path, {
        "state": "running",
        "stage": "orth_train_eval",
        "updated_at": utc_now_iso(),
        "message": "training/evaluating orthogonal chains",
    })

    for seed in top_seeds:
        seed_rank = int(seed["seed_rank"])
        pooling = str(seed["pooling"])
        layer = int(seed["layer"])

        logger.info(f"[seed {seed_rank}] pooling={pooling} layer={layer} macro_auc={seed['macro_auc']:.4f}")

        # Load materialized train/val tensors for this layer once.
        if layer not in layer_cache:
            x_train, y_train = materialize_layer_tensors(train_index, layer=layer, batch_size=max(64, args.batch_size))
            x_val, y_val = materialize_layer_tensors(val_index, layer=layer, batch_size=max(64, args.batch_size))
            layer_cache[layer] = (x_train, y_train, x_val, y_val)
        else:
            x_train, y_train, x_val, y_val = layer_cache[layer]

        seed_dir = run_dir / f"seed_rank_{seed_rank}_pooling_{pooling}_layer_{layer}"
        probes_dir = seed_dir / "probes"
        ensure_dir(probes_dir)

        # k=1: seed probe from Stage A
        seed_probe_src = stage_a_probe_dir / pooling / f"probe_layer_{layer}.pt"
        if not seed_probe_src.exists():
            raise FileNotFoundError(f"Missing Stage A seed probe: {seed_probe_src}")

        probe_k1_path = probes_dir / "probe_k001.pt"
        train_unit_k1 = f"seed{seed_rank}:k1"
        if not (args.resume and probe_k1_path.exists() and train_unit_k1 in train_done):
            shutil.copy2(seed_probe_src, probe_k1_path)
            train_done.add(train_unit_k1)
            progress["train_done"] = sorted(train_done)
            progress["eval_done"] = sorted(eval_done)
            write_progress(progress_path, progress)

        # Build basis and previous vectors incrementally.
        q_basis: Optional[torch.Tensor] = None
        prev_vecs: List[torch.Tensor] = []

        # k=1 evaluation with no projection
        model_k1 = load_probe(probe_k1_path, pooling=pooling, d_model=d_model, device=device)
        w1 = extract_probe_vector(model_k1).detach().cpu()
        q_basis, norm1 = update_q_basis(q_basis, w1)
        prev_vecs.append(w1)

        eval_rows_k1 = evaluate_on_test_indices(
            model=model_k1,
            layer=layer,
            seed_rank=seed_rank,
            pooling=pooling,
            k=1,
            q_basis=None,
            test_indices=test_indices,
            device=device,
            batch_size=args.batch_size,
        )
        for r in eval_rows_k1:
            eunit = f"seed{seed_rank}:k1:{r['test_dataset_base']}"
            if args.resume and eunit in eval_done:
                continue
            with eval_long_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
            eval_done.add(eunit)

        progress["train_done"] = sorted(train_done)
        progress["eval_done"] = sorted(eval_done)
        write_progress(progress_path, progress)

        # k >= 2
        for k in range(2, int(args.chain_k) + 1):
            train_unit = f"seed{seed_rank}:k{k}"
            probe_path = probes_dir / f"probe_k{k:03d}.pt"

            # The projection basis for this k is previous directions only.
            q_train = q_basis.clone().to(device) if q_basis is not None else None

            if args.resume and probe_path.exists() and train_unit in train_done:
                model_k = load_probe(probe_path, pooling=pooling, d_model=d_model, device=device)
                val_metrics = {"auc": None, "accuracy": None, "best_epoch": None}
                logger.info(f"[seed {seed_rank}] resume probe k={k}")
            else:
                model_k = LayerProbe(input_dim=d_model, pooling_type=pooling).to(device)
                model_k, val_metrics = train_probe_with_projection(
                    model=model_k,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    device=device,
                    q_basis=q_train,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                torch.save(model_k.state_dict(), str(probe_path))

                w_k = extract_probe_vector(model_k).detach().cpu()
                max_cos_prev = max_cos_to_previous(w_k, prev_vecs)
                q_basis, w_norm = update_q_basis(q_basis, w_k)
                prev_vecs.append(w_k)

                train_rec = {
                    "timestamp": utc_now_iso(),
                    "seed_rank": seed_rank,
                    "pooling": pooling,
                    "layer": layer,
                    "k": k,
                    "val_auc": float(val_metrics.get("auc", 0.5)),
                    "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
                    "best_epoch": int(val_metrics.get("best_epoch", 0)),
                    "w_norm": float(w_norm),
                    "max_abs_cos_to_previous": float(max_cos_prev),
                    "probe_path": str(probe_path),
                }
                with train_metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(train_rec, ensure_ascii=True) + "\n")

                train_done.add(train_unit)
                progress["train_done"] = sorted(train_done)
                progress["eval_done"] = sorted(eval_done)
                write_progress(progress_path, progress)

            # Resume branch still needs basis extension for subsequent k.
            if args.resume and probe_path.exists() and train_unit in train_done:
                # For resumed probes where we skipped training, ensure basis is updated now.
                w_k = extract_probe_vector(model_k).detach().cpu()
                if len(prev_vecs) < k:
                    q_basis, _ = update_q_basis(q_basis, w_k)
                    prev_vecs.append(w_k)

            eval_rows = evaluate_on_test_indices(
                model=model_k,
                layer=layer,
                seed_rank=seed_rank,
                pooling=pooling,
                k=k,
                q_basis=q_train,
                test_indices=test_indices,
                device=device,
                batch_size=args.batch_size,
            )
            for r in eval_rows:
                eunit = f"seed{seed_rank}:k{k}:{r['test_dataset_base']}"
                if args.resume and eunit in eval_done:
                    continue
                with eval_long_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=True) + "\n")
                eval_done.add(eunit)

            progress["train_done"] = sorted(train_done)
            progress["eval_done"] = sorted(eval_done)
            write_progress(progress_path, progress)

    # Aggregate outputs
    write_json(status_path, {
        "state": "running",
        "stage": "aggregate",
        "updated_at": utc_now_iso(),
        "message": "aggregating orthogonal results",
    })

    eval_rows_all: List[Dict] = []
    if eval_long_path.exists():
        with eval_long_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                eval_rows_all.append(json.loads(line))

    # dedupe and sort
    dedup = {}
    for r in eval_rows_all:
        key = (int(r["seed_rank"]), int(r["k"]), str(r["test_dataset_base"]))
        dedup[key] = r
    eval_rows_all = sorted(dedup.values(), key=lambda r: (int(r["seed_rank"]), int(r["k"]), str(r["test_dataset_base"])))

    eval_long_csv = out_dir / "orth_chain_eval_long.csv"
    write_csv(
        eval_long_csv,
        rows=eval_rows_all,
        fieldnames=["seed_rank", "pooling", "layer", "k", "test_dataset", "test_dataset_base", "auc", "accuracy", "count", "timestamp"],
    )

    # Macro by (seed,k)
    macro_rows: List[Dict] = []
    grouped: Dict[Tuple[int, int], List[Dict]] = {}
    for r in eval_rows_all:
        key = (int(r["seed_rank"]), int(r["k"]))
        grouped.setdefault(key, []).append(r)

    for (seed_rank, k), vals in sorted(grouped.items()):
        aucs = [float(v["auc"]) for v in vals]
        accs = [float(v["accuracy"]) for v in vals]
        macro_rows.append(
            {
                "seed_rank": seed_rank,
                "k": k,
                "macro_auc": float(np.mean(aucs)) if aucs else 0.0,
                "macro_accuracy": float(np.mean(accs)) if accs else 0.0,
                "num_test_datasets": len(vals),
            }
        )

    write_csv(
        out_dir / "orth_macro_auc_by_k.csv",
        rows=macro_rows,
        fieldnames=["seed_rank", "k", "macro_auc", "macro_accuracy", "num_test_datasets"],
    )

    best_rows: List[Dict] = []
    for seed_rank in sorted({int(r["seed_rank"]) for r in macro_rows}):
        vals = [r for r in macro_rows if int(r["seed_rank"]) == seed_rank]
        if not vals:
            continue
        best = max(vals, key=lambda r: (float(r["macro_auc"]), float(r["macro_accuracy"]), -int(r["k"])))
        best_rows.append(best)

    write_csv(
        out_dir / "orth_best_k_per_seed.csv",
        rows=best_rows,
        fieldnames=["seed_rank", "k", "macro_auc", "macro_accuracy", "num_test_datasets"],
    )

    # Heatmaps per seed (k x test)
    test_labels = [dataset_segment_name(d, args.segment) for d in test_bases]
    for seed_rank in sorted({int(r["seed_rank"]) for r in eval_rows_all}):
        seed_rows = [r for r in eval_rows_all if int(r["seed_rank"]) == seed_rank]
        make_seed_heatmap(
            seed_rank=seed_rank,
            rows=seed_rows,
            test_labels=test_labels,
            chain_k=int(args.chain_k),
            output_path=out_dir / f"heatmap_seed{seed_rank}_k_vs_test_auc.png",
        )

    write_json(
        out_dir / "orth_summary.json",
        {
            "run_name": run_name,
            "segment": args.segment,
            "model": args.model,
            "stage_a_results_dir": str(stage_a_results_dir),
            "stage_a_probe_dir": str(stage_a_probe_dir),
            "top_seeds": top_seeds,
            "chain_k": int(args.chain_k),
            "n_eval_rows": len(eval_rows_all),
            "artifacts": {
                "top3_seeds_csv": str(out_dir / "top3_seeds.csv"),
                "top3_summary_json": str(out_dir / "top3_selection_summary.json"),
                "orth_chain_eval_long_csv": str(eval_long_csv),
                "orth_macro_auc_by_k_csv": str(out_dir / "orth_macro_auc_by_k.csv"),
                "orth_best_k_per_seed_csv": str(out_dir / "orth_best_k_per_seed.csv"),
            },
            "updated_at": utc_now_iso(),
        },
    )

    write_json(status_path, {
        "state": "completed",
        "stage": "done",
        "updated_at": utc_now_iso(),
        "message": "completed successfully",
    })

    logger.info(f"[done] outputs -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
