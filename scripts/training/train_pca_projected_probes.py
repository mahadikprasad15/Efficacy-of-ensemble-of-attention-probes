#!/usr/bin/env python3
"""
Train probes on PCA-projected activations.

Projects pooled activations (N, D) onto K-dimensional subspaces via PCA
components, then trains linear probes on the (N, K) projections.

Three conditions:
  - top:    Project onto the K highest-variance PCA directions.
  - bottom: Project onto the K lowest-variance directions (from fitted set).
  - random: Project onto K random orthogonal directions (control).

Sweeps over K values and layers for a given pooling strategy.

Usage:
    python scripts/training/train_pca_projected_probes.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset Deception-Roleplaying \
        --train_activations_dir data/activations/.../train \
        --val_activations_dir data/activations/.../validation \
        --pca_artifacts_dir data/pca_ablation/mean/pca_artifacts \
        --pooling mean \
        --conditions top,bottom,random \
        --k_values 1,2,3,5,10,20,50,100 \
        --layers all \
        --seed 42
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – match codebase convention
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
sys.path.append(os.path.join(os.getcwd(), "scripts", "evaluation"))

from actprobe.features.pca_projection import (
    generate_random_orthogonal,
    get_bottom_k_components,
    get_top_k_components,
    load_pca_artifact,
    project_activations,
)
from actprobe.probes.models import LayerProbe
from evaluate_pca_ablation_sweep import load_pooled_split, parse_k_values, parse_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def evaluate_probe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a probe, returning (AUC, accuracy)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            targets.extend(y.numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    try:
        auc = roc_auc_score(targets, preds)
    except Exception:
        auc = 0.5
    acc = accuracy_score(targets, (preds > 0.5).astype(int))
    return auc, acc


def train_single_probe(
    probe: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    save_path: str,
) -> Tuple[float, float, int]:
    """Train one probe with early stopping.

    Returns (best_val_auc, best_val_acc, best_epoch).
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        probe.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        val_auc, val_acc = evaluate_probe(probe, val_loader, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(probe.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    if os.path.exists(save_path):
        probe.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return best_val_auc, best_val_acc, best_epoch


# ---------------------------------------------------------------------------
# PCA artifacts discovery
# ---------------------------------------------------------------------------

def find_pca_artifacts(pca_dir: str) -> Dict[int, str]:
    """Find layer_{l}.npz files in a PCA artifacts directory.

    Returns dict mapping layer index -> file path.
    """
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(pca_dir, "layer_*.npz")))
    out: Dict[int, str] = {}
    for path in files:
        name = os.path.basename(path)
        layer_idx = int(name.replace("layer_", "").replace(".npz", ""))
        out[layer_idx] = path
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep_for_pooling(
    pooling: str,
    train_activations_dir: str,
    val_activations_dir: str,
    pca_artifacts_dir: str,
    output_dir: str,
    conditions: List[str],
    k_values: List[int],
    layer_spec: str,
    seed: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    skip_existing: bool,
    device: torch.device,
) -> List[dict]:
    """Run the full condition × K × layer sweep for one pooling strategy."""

    # Discover PCA artifacts
    pca_map = find_pca_artifacts(pca_artifacts_dir)
    if not pca_map:
        raise FileNotFoundError(f"No layer_*.npz found in {pca_artifacts_dir}")
    available_layers = sorted(pca_map.keys())
    layers = parse_layers(layer_spec, available_layers)
    if not layers:
        raise ValueError(f"No valid layers selected. Available: {available_layers}")

    logger.info(f"Pooling: {pooling} | Layers ({len(layers)}): {layers}")
    logger.info(f"PCA artifacts dir: {pca_artifacts_dir}")

    # Load pooled activations
    x_train, y_train, _ = load_pooled_split(
        train_activations_dir, layers, pooling, desc="train",
    )
    x_val, y_val, _ = load_pooled_split(
        val_activations_dir, layers, pooling, desc="validation",
    )

    # Load PCA artifacts
    pca_by_layer: Dict[int, dict] = {}
    for layer in layers:
        pca_by_layer[layer] = load_pca_artifact(pca_map[layer])
    sample_layer = layers[0]
    D = pca_by_layer[sample_layer]["dim"]
    K_max = pca_by_layer[sample_layer]["components"].shape[0]

    max_K = max(k_values)
    if max_K > K_max and any(c in conditions for c in ("top", "bottom")):
        raise ValueError(
            f"max K={max_K} exceeds fitted PCA components K_max={K_max}. "
            f"Either reduce --k_values or refit PCA with more components."
        )
    if "bottom" in conditions:
        logger.warning(
            f"Bottom-K uses the K smallest-eigenvalue components from the fitted "
            f"PCA set (K_max={K_max}), not from all D={D} possible components."
        )

    logger.info(f"D={D}, K_max={K_max}, K_values={k_values}, conditions={conditions}")
    logger.info(f"Train: N={len(y_train)}, Val: N={len(y_val)}")

    os.makedirs(output_dir, exist_ok=True)

    all_results: List[dict] = []
    total_combos = len(conditions) * len(k_values) * len(layers)
    pbar = tqdm(total=total_combos, desc=f"Sweep ({pooling})")

    for condition in conditions:
        for K in k_values:
            run_dir = os.path.join(output_dir, f"{condition}_K{K}")
            os.makedirs(run_dir, exist_ok=True)
            layer_results_path = os.path.join(run_dir, "layer_results.json")

            # Check skip_existing
            if skip_existing and os.path.exists(layer_results_path):
                logger.info(f"Skipping {condition}_K{K} (results exist)")
                with open(layer_results_path, "r") as f:
                    layer_results = json.load(f)
                for r in layer_results:
                    all_results.append(r)
                pbar.update(len(layers))
                continue

            layer_results = []
            for layer in layers:
                pca_art = pca_by_layer[layer]

                # Build projection matrix V: (K, D)
                if condition == "top":
                    V = get_top_k_components(pca_art["components"], K)
                elif condition == "bottom":
                    V = get_bottom_k_components(pca_art["components"], K)
                elif condition == "random":
                    V = generate_random_orthogonal(D, K, seed=seed * 10000 + layer)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                # Project
                X_train_proj = project_activations(x_train[layer], pca_art["mean"], V)
                X_val_proj = project_activations(x_val[layer], pca_art["mean"], V)

                # Build datasets
                train_ds = TensorDataset(
                    torch.from_numpy(X_train_proj),
                    torch.from_numpy(y_train.astype(np.float32)),
                )
                val_ds = TensorDataset(
                    torch.from_numpy(X_val_proj),
                    torch.from_numpy(y_val.astype(np.float32)),
                )

                # Train probe
                probe = LayerProbe(input_dim=K, pooling_type="none").to(device)
                probe_save_path = os.path.join(run_dir, f"probe_layer_{layer}.pt")

                best_auc, best_acc, best_epoch = train_single_probe(
                    probe=probe,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    device=device,
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    patience=patience,
                    batch_size=batch_size,
                    save_path=probe_save_path,
                )

                result = {
                    "condition": condition,
                    "K": K,
                    "pooling": pooling,
                    "layer": layer,
                    "val_auc": round(best_auc, 6),
                    "val_acc": round(best_acc, 6),
                    "epoch": best_epoch,
                }
                layer_results.append(result)
                all_results.append(result)
                pbar.update(1)

            # Save per-condition-K results
            with open(layer_results_path, "w") as f:
                json.dump(layer_results, f, indent=2)

    pbar.close()
    return all_results


def save_sweep_summary(output_dir: str, results: List[dict], config: dict) -> None:
    """Save JSON and CSV summaries of the sweep."""
    # JSON summary
    summary = {"config": config, "results": results}
    json_path = os.path.join(output_dir, "sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved JSON summary: {json_path}")

    # CSV summary (one row per condition × K × layer)
    csv_path = os.path.join(output_dir, "sweep_summary.csv")
    fieldnames = ["condition", "K", "pooling", "layer", "val_auc", "val_acc", "epoch"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    logger.info(f"Saved CSV summary: {csv_path}")

    # Best-per-condition-K summary
    best_csv_path = os.path.join(output_dir, "sweep_best_per_k.csv")
    # Group by (condition, K, pooling)
    groups: Dict[tuple, List[dict]] = {}
    for r in results:
        key = (r["condition"], r["K"], r["pooling"])
        groups.setdefault(key, []).append(r)

    best_rows = []
    for key, rows in sorted(groups.items()):
        best = max(rows, key=lambda x: x["val_auc"])
        best_rows.append({
            "condition": key[0],
            "K": key[1],
            "pooling": key[2],
            "best_layer": best["layer"],
            "best_val_auc": best["val_auc"],
            "best_val_acc": best["val_acc"],
            "best_epoch": best["epoch"],
        })
    best_fieldnames = ["condition", "K", "pooling", "best_layer", "best_val_auc", "best_val_acc", "best_epoch"]
    with open(best_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)
    logger.info(f"Saved best-per-K summary: {best_csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train probes on PCA-projected activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (for logging)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (for logging)")
    parser.add_argument("--train_activations_dir", type=str, required=True,
                        help="Directory with train shard_*.safetensors + manifest.jsonl")
    parser.add_argument("--val_activations_dir", type=str, required=True,
                        help="Directory with validation shard_*.safetensors + manifest.jsonl")
    parser.add_argument("--pca_artifacts_dir", type=str, required=True,
                        help="Directory containing layer_*.npz PCA artifacts. "
                             "When --pooling all, expects {dir}/{pooling}/pca_artifacts/ structure.")
    parser.add_argument("--pooling", type=str, default="mean",
                        help="Pooling strategy: mean, max, last, or 'all' to loop through all three")
    parser.add_argument("--conditions", type=str, default="top,bottom,random",
                        help="Comma-separated conditions to run (top, bottom, random)")
    parser.add_argument("--k_values", type=str, default="1,2,3,5,10,20,50,100",
                        help="Comma-separated K values for the projection dimension sweep")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layers or 'all'")
    parser.add_argument("--output_dir", type=str, default="data/probes/pca_projected",
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip (condition, K) combos that already have layer_results.json")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    conditions = [c.strip() for c in args.conditions.split(",")]
    k_values = parse_k_values(args.k_values)

    # Determine pooling strategies to run
    if args.pooling.lower() == "all":
        poolings = ["mean", "max", "last"]
    else:
        poolings = [args.pooling]

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "conditions": conditions,
        "k_values": k_values,
        "layers_spec": args.layers,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }

    print("=" * 80)
    print("PCA-PROJECTED PROBE TRAINING SWEEP")
    print("=" * 80)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Pooling:    {poolings}")
    print(f"Conditions: {conditions}")
    print(f"K values:   {k_values}")
    print(f"Device:     {device}")
    print()

    for pooling in poolings:
        # Resolve PCA artifacts dir: if running multiple poolings, expect
        # {pca_artifacts_dir}/{pooling}/pca_artifacts/ structure
        if len(poolings) > 1:
            pca_dir = os.path.join(args.pca_artifacts_dir, pooling, "pca_artifacts")
        else:
            pca_dir = args.pca_artifacts_dir

        pool_output_dir = os.path.join(args.output_dir, pooling)

        results = run_sweep_for_pooling(
            pooling=pooling,
            train_activations_dir=args.train_activations_dir,
            val_activations_dir=args.val_activations_dir,
            pca_artifacts_dir=pca_dir,
            output_dir=pool_output_dir,
            conditions=conditions,
            k_values=k_values,
            layer_spec=args.layers,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            device=device,
        )

        config_with_pooling = {**config, "pooling": pooling}
        save_sweep_summary(pool_output_dir, results, config_with_pooling)

        # Print best result for this pooling
        if results:
            best = max(results, key=lambda r: r["val_auc"])
            print(
                f"\nBest for {pooling}: {best['condition']}_K{best['K']} "
                f"layer {best['layer']} | AUC={best['val_auc']:.4f}"
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
