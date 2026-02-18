#!/usr/bin/env python3
"""
Train L1-regularized probes on PCA-projected activations and evaluate consensus PCs.

Workflow (per pooling/layer/K):
1) Project pooled activations to top-K PCA components.
2) Train R L1 probes on bootstrap resamples (no retraining for consensus).
3) Define selected PCs per probe as |w| > eps.
4) Build consensus masks by agreement thresholds.
5) Score consensus by masking each probe and averaging logits.

Outputs:
- results/pca_consensus/<model>/<dataset>/<pooling>/consensus_summary.jsonl
- results/pca_consensus/<model>/<dataset>/<pooling>/consensus_summary.csv
- results/pca_consensus/<model>/<dataset>/<pooling>/best_by_k.csv
- results/pca_consensus/<model>/<dataset>/<pooling>/best_by_threshold.csv
- data/probes_pca_consensus/<model>/<dataset>/<pooling>/K_<k>/layer_<L>/probes/*.npz
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

# Path setup
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
sys.path.append(os.path.join(os.getcwd(), "scripts", "evaluation"))

from actprobe.features.pca_projection import (
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
logger = logging.getLogger("pca_consensus")


def parse_thresholds(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    vals = sorted(set(vals))
    for v in vals:
        if v <= 0 or v > 1:
            raise ValueError(f"Consensus thresholds must be in (0,1], got {v}")
    return vals


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    acc = float(accuracy_score(y_true, (probs >= 0.5).astype(np.int64)))
    return auc, acc


def bootstrap_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    n_boot = max(1, int(round(frac * n)))
    return rng.choice(n, size=n_boot, replace=True)


def extract_linear_params(model: nn.Module) -> Tuple[np.ndarray, float]:
    w = model.classifier.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])
    return w, b


def train_l1_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    l1_lambda: float,
    seed: int,
) -> Tuple[np.ndarray, float, float, float, int]:
    """Train a single L1-regularized probe. Returns (w, b, best_auc, best_acc, best_epoch)."""
    set_seeds(seed)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    probe = LayerProbe(input_dim=X_train.shape[1], pooling_type="none").to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = probe(xb).squeeze(-1)
            loss = criterion(logits, yb)
            # L1 on classifier weights only
            l1 = probe.classifier.weight.abs().sum()
            loss = loss + l1_lambda * l1
            loss.backward()
            optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = []
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = probe(xb).squeeze(-1)
                val_logits.append(logits.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_auc, val_acc = compute_metrics(y_val, val_logits)

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.detach().cpu() for k, v in probe.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    w, b = extract_linear_params(probe)
    return w, b, best_auc, best_acc, best_epoch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_symlink_ood(output_root: str) -> None:
    """Create symlink results/ood_evaluation/pca_consensus -> output_root if possible."""
    base = os.path.abspath(output_root)
    if not base.endswith(os.path.join("results", "pca_consensus")):
        return
    repo_root = os.path.abspath(os.getcwd())
    link_parent = os.path.join(repo_root, "results", "ood_evaluation")
    link_path = os.path.join(link_parent, "pca_consensus")
    try:
        ensure_dir(link_parent)
        if os.path.islink(link_path):
            # If correct, keep; else replace
            target = os.readlink(link_path)
            if os.path.abspath(os.path.join(link_parent, target)) == base:
                return
            os.unlink(link_path)
        elif os.path.exists(link_path):
            # Don't overwrite real dir
            return
        os.symlink(base, link_path)
    except Exception:
        # Best-effort only
        return


def load_existing_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def find_pca_layers(pca_dir: str) -> List[int]:
    layers = []
    for path in sorted(os.listdir(pca_dir)):
        if not path.startswith("layer_") or not path.endswith(".npz"):
            continue
        try:
            layer_idx = int(path.replace("layer_", "").replace(".npz", ""))
            layers.append(layer_idx)
        except Exception:
            continue
    return sorted(set(layers))


def load_progress(path: str) -> dict:
    if not os.path.exists(path):
        return {"completed_probes": [], "completed_thresholds": []}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return {"completed_probes": [], "completed_thresholds": []}
    payload.setdefault("completed_probes", [])
    payload.setdefault("completed_thresholds", [])
    return payload


def save_progress(path: str, payload: dict) -> None:
    payload.setdefault("completed_probes", [])
    payload.setdefault("completed_thresholds", [])
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consensus PCA probes (L1 + bootstrap)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_activations_dir", type=str, required=True)
    parser.add_argument("--val_activations_dir", type=str, required=True)
    parser.add_argument("--ood_activations_dir", type=str, default=None)
    parser.add_argument("--pca_artifacts_dir", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--k_values", type=str, default="5,10,20,30,50,80,100")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--num_probes", type=int, default=5)
    parser.add_argument("--bootstrap_frac", type=float, default=0.8)
    parser.add_argument("--l1_lambda", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--consensus_thresholds", type=str, default="0.6,0.7,0.8")
    parser.add_argument("--output_root", type=str, default="results/pca_consensus")
    parser.add_argument("--probes_output_root", type=str, default="data/probes_pca_consensus")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)

    k_values = parse_k_values(args.k_values)
    thresholds = parse_thresholds(args.consensus_thresholds)

    model_dir = args.model.replace("/", "_")

    if not os.path.isdir(args.pca_artifacts_dir):
        raise FileNotFoundError(f"Missing PCA artifacts dir: {args.pca_artifacts_dir}")

    available_layers = find_pca_layers(args.pca_artifacts_dir)
    if not available_layers:
        raise FileNotFoundError(f"No layer_*.npz found in {args.pca_artifacts_dir}")

    layers = parse_layers(args.layers, available_layers)
    if not layers:
        raise ValueError(f"No valid layers selected. Available: {available_layers}")

    # Load pooled activations for all layers
    x_train, y_train, _ = load_pooled_split(
        args.train_activations_dir, layers, args.pooling, desc="train"
    )
    x_val, y_val, _ = load_pooled_split(
        args.val_activations_dir, layers, args.pooling, desc="validation"
    )
    x_ood, y_ood = None, None
    if args.ood_activations_dir:
        x_ood, y_ood, _ = load_pooled_split(
            args.ood_activations_dir, layers, args.pooling, desc="ood"
        )

    results_dir = os.path.join(args.output_root, model_dir, args.dataset, args.pooling)
    probes_dir = os.path.join(args.probes_output_root, model_dir, args.dataset, args.pooling)
    ensure_dir(results_dir)
    ensure_dir(probes_dir)

    summary_path = os.path.join(results_dir, "consensus_summary.jsonl")
    existing_rows = load_existing_rows(summary_path)

    # Track completed (layer, K, threshold) to support resume
    completed = set()
    for r in existing_rows:
        key = (r.get("layer"), r.get("K"), r.get("threshold"))
        completed.add(key)

    all_rows = list(existing_rows)

    logger.info("CONSENSUS PCA PROBES")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Pooling: {args.pooling}")
    logger.info(f"Layers: {layers}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Probes per K: {args.num_probes} | Bootstrap frac: {args.bootstrap_frac}")
    logger.info(f"L1 lambda: {args.l1_lambda} | eps: {args.eps}")
    logger.info(f"Thresholds: {thresholds}")

    rng = np.random.default_rng(args.seed)

    for layer in tqdm(layers, desc="Layers", unit="layer"):
        # Load PCA artifact for this layer
        pca_path = os.path.join(args.pca_artifacts_dir, f"layer_{layer}.npz")
        if not os.path.exists(pca_path):
            logger.warning(f"Missing PCA artifact: {pca_path}. Skipping layer {layer}.")
            continue
        pca_art = load_pca_artifact(pca_path)
        components = pca_art["components"]
        mean = pca_art["mean"]
        k_max = components.shape[0]

        for K in tqdm(k_values, desc=f"Layer {layer} K sweep", unit="K", leave=False):
            if K > k_max:
                logger.warning(f"K={K} exceeds saved PCA components ({k_max}) for layer {layer}. Skipping.")
                continue

            # If all thresholds already done, skip
            if args.skip_existing:
                done = all((layer, K, t) in completed for t in thresholds)
                if done:
                    logger.info(f"Skipping layer {layer} K={K} (already done)")
                    continue

            V = get_top_k_components(components, K)
            X_train_proj = project_activations(x_train[layer], mean, V)
            X_val_proj = project_activations(x_val[layer], mean, V)
            X_ood_proj = None
            if x_ood is not None:
                X_ood_proj = project_activations(x_ood[layer], mean, V)

            # Train R probes
            probe_weights = []
            probe_biases = []
            selected_masks = []
            selected_counts = []
            layer_root = os.path.join(probes_dir, f"K_{K}", f"layer_{layer}")
            probe_dir = os.path.join(layer_root, "probes")
            sel_dir = os.path.join(layer_root, "selections")
            progress_path = os.path.join(layer_root, "progress.json")
            ensure_dir(probe_dir)
            ensure_dir(sel_dir)

            progress = load_progress(progress_path)
            completed_probes = set(progress.get("completed_probes", []))
            completed_thresholds = set(progress.get("completed_thresholds", []))

            for r in tqdm(range(args.num_probes), desc=f"L{layer} K{K} probes", unit="probe", leave=False):
                boot_idx = bootstrap_indices(len(y_train), args.bootstrap_frac, rng)
                Xb = X_train_proj[boot_idx]
                yb = y_train[boot_idx]

                probe_path = os.path.join(probe_dir, f"probe_r{r}.npz")
                sel_path = os.path.join(sel_dir, f"selected_r{r}.npy")

                if r in completed_probes and os.path.exists(probe_path) and os.path.exists(sel_path):
                    data = np.load(probe_path, allow_pickle=False)
                    w = data["w"].astype(np.float32)
                    b = float(np.asarray(data["b"]).reshape(-1)[0])
                else:
                    w, b, best_auc, best_acc, best_epoch = train_l1_probe(
                        X_train=Xb,
                        y_train=yb,
                        X_val=X_val_proj,
                        y_val=y_val,
                        device=device,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        l1_lambda=args.l1_lambda,
                        seed=args.seed + r + layer * 1000 + K * 10000,
                    )

                    np.savez_compressed(
                        probe_path,
                        w=w,
                        b=np.asarray([b], dtype=np.float32),
                        val_auc=np.asarray([best_auc], dtype=np.float32),
                        val_acc=np.asarray([best_acc], dtype=np.float32),
                        val_epoch=np.asarray([best_epoch], dtype=np.int64),
                    )

                    # Update completed probes
                    completed_probes.add(r)
                    progress["completed_probes"] = sorted(completed_probes)
                    save_progress(progress_path, progress)

                sel = (np.abs(w) > args.eps).astype(np.int8)
                selected_masks.append(sel)
                selected_counts.append(int(sel.sum()))
                probe_weights.append(w)
                probe_biases.append(b)

                if not os.path.exists(sel_path):
                    np.save(sel_path, np.where(sel > 0)[0].astype(np.int32))

            selected_masks = np.stack(selected_masks, axis=0)  # (R, K)
            freq = selected_masks.mean(axis=0)  # (K,)
            avg_selected = float(np.mean(selected_counts)) if selected_counts else 0.0

            # Evaluate consensus thresholds
            for t in thresholds:
                key = (layer, K, t)
                if args.skip_existing and key in completed:
                    continue
                if t in completed_thresholds:
                    continue

                consensus_mask = (freq >= t).astype(np.float32)
                num_consensus = int(consensus_mask.sum())

                row = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "pooling": args.pooling,
                    "layer": int(layer),
                    "K": int(K),
                    "threshold": float(t),
                    "num_probes": int(args.num_probes),
                    "bootstrap_frac": float(args.bootstrap_frac),
                    "l1_lambda": float(args.l1_lambda),
                    "eps": float(args.eps),
                    "num_consensus_pcs": num_consensus,
                    "avg_selected_per_probe": avg_selected,
                    "consensus_pc_indices": np.where(consensus_mask > 0)[0].tolist(),
                }

                if num_consensus == 0:
                    row.update({
                        "id_val_auc": None,
                        "id_val_acc": None,
                        "ood_test_auc": None,
                        "ood_test_acc": None,
                    })
                else:
                    # Mask each probe and average logits
                    val_logits_all = []
                    ood_logits_all = [] if X_ood_proj is not None else None
                    for w, b in zip(probe_weights, probe_biases):
                        w_masked = w * consensus_mask
                        val_logits_all.append(X_val_proj @ w_masked + b)
                        if X_ood_proj is not None:
                            ood_logits_all.append(X_ood_proj @ w_masked + b)

                    val_logits = np.mean(np.stack(val_logits_all, axis=0), axis=0)
                    id_auc, id_acc = compute_metrics(y_val, val_logits)
                    row["id_val_auc"] = float(id_auc)
                    row["id_val_acc"] = float(id_acc)

                    if X_ood_proj is not None and ood_logits_all is not None:
                        ood_logits = np.mean(np.stack(ood_logits_all, axis=0), axis=0)
                        ood_auc, ood_acc = compute_metrics(y_ood, ood_logits)
                        row["ood_test_auc"] = float(ood_auc)
                        row["ood_test_acc"] = float(ood_acc)
                    else:
                        row["ood_test_auc"] = None
                        row["ood_test_acc"] = None

                all_rows.append(row)
                completed.add(key)
                completed_thresholds.add(t)
                progress["completed_thresholds"] = sorted(float(x) for x in completed_thresholds)
                save_progress(progress_path, progress)

    # Write JSONL + CSV summaries (rewrite for now)
    with open(summary_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    csv_path = os.path.join(results_dir, "consensus_summary.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # Best-by-K and best-by-threshold
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        # best by K (max id_val_auc)
        best_by_k = (
            df.dropna(subset=["id_val_auc"])
              .sort_values(["K", "id_val_auc"], ascending=[True, False])
              .drop_duplicates(["K"], keep="first")
        )
        best_by_k.to_csv(os.path.join(results_dir, "best_by_k.csv"), index=False)

        # best by threshold
        best_by_t = (
            df.dropna(subset=["id_val_auc"])
              .sort_values(["threshold", "id_val_auc"], ascending=[True, False])
              .drop_duplicates(["threshold"], keep="first")
        )
        best_by_t.to_csv(os.path.join(results_dir, "best_by_threshold.csv"), index=False)

    maybe_symlink_ood(args.output_root)

    logger.info("Done.")
    logger.info(f"Results: {results_dir}")
    logger.info(f"Probes:  {probes_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
