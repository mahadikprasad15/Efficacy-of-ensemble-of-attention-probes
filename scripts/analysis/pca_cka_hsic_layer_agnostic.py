#!/usr/bin/env python3
"""
Layer-Agnostic Structure Analysis: PCA + Linear CKA/HSIC
========================================================

Compute per-layer and aggregate (concatenated) structure comparisons between
two domains using mean-pooled activations.

Outputs:
  - PCA explained variance curves and 2D scatter plots (per layer + aggregate)
  - Linear CKA + linear HSIC per layer + aggregate
  - summary_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.decomposition import PCA


def load_manifest(manifest_path: str) -> List[dict]:
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def select_ids(entries: List[dict], max_samples: int, seed: int) -> List[str]:
    valid = [e for e in entries if e.get("label", -1) != -1]
    if max_samples is None or max_samples <= 0 or max_samples >= len(valid):
        return [e["id"] for e in valid]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return [e["id"] for e in valid[:max_samples]]


def pool_tokens(x_layer: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return x_layer.mean(axis=0)
    if pooling == "max":
        return x_layer.max(axis=0)
    if pooling == "last":
        return x_layer[-1]
    raise ValueError(f"Unknown pooling: {pooling}")


def load_pooled_all_layers(
    activations_dir: str,
    pooling: str,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    entries = load_manifest(manifest_path)
    target_ids = set(select_ids(entries, max_samples, seed))
    if not target_ids:
        raise ValueError(f"No valid samples found in {manifest_path}")

    shards = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {activations_dir}")

    pooled_list: List[np.ndarray] = []
    ids_found: List[str] = []

    for shard_path in shards:
        tensors = load_file(shard_path)
        shard_ids = set(tensors.keys()) & target_ids
        if not shard_ids:
            continue
        for eid in shard_ids:
            tensor = tensors[eid].cpu().numpy()  # (L, T, D)
            L = tensor.shape[0]
            pooled = np.stack([pool_tokens(tensor[l], pooling) for l in range(L)], axis=0)  # (L, D)
            pooled_list.append(pooled)
            ids_found.append(eid)
            target_ids.remove(eid)
        if not target_ids:
            break

    if not pooled_list:
        raise ValueError(f"No activations loaded from {activations_dir}")

    X = np.stack(pooled_list, axis=0)  # (N, L, D)
    return X, ids_found


def center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic_linear(X: np.ndarray, Y: np.ndarray) -> float:
    K = X @ X.T
    L = Y @ Y.T
    Kc = center_gram(K)
    Lc = center_gram(L)
    n = X.shape[0]
    return float((Kc * Lc).sum() / ((n - 1) ** 2))


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    hsic_xy = hsic_linear(X, Y)
    hsic_xx = hsic_linear(X, X)
    hsic_yy = hsic_linear(Y, Y)
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(hsic_xy / denom)


def pca_fit_transform(
    X_a: np.ndarray,
    X_b: np.ndarray,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([X_a, X_b])
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(X)
    Z_a = Z[: X_a.shape[0]]
    Z_b = Z[X_a.shape[0] :]
    evr = pca.explained_variance_ratio_
    return Z_a, Z_b, evr


def save_evr_plot(evr: np.ndarray, out_path: str, title: str):
    cum = np.cumsum(evr)
    plt.figure(figsize=(6, 4))
    plt.plot(cum, color="#4c78a8", linewidth=2)
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_scatter_plot(Z_a: np.ndarray, Z_b: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.scatter(Z_a[:, 0], Z_a[:, 1], s=10, alpha=0.6, label="Roleplaying")
    plt.scatter(Z_b[:, 0], Z_b[:, 1], s=10, alpha=0.6, label="InsiderTrading")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def evr_at_k(evr: np.ndarray, k: int) -> float:
    if evr.size == 0:
        return 0.0
    k = min(k, evr.size)
    return float(np.cumsum(evr)[k - 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer-agnostic PCA/CKA/HSIC analysis")
    parser.add_argument("--base_data_dir", required=True, help="Base activations dir (contains model subdir)")
    parser.add_argument("--model", required=True, help="Model name used in activations path")
    parser.add_argument("--domain_a", default="Deception-Roleplaying")
    parser.add_argument("--domain_b", default="Deception-InsiderTrading")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "pca").mkdir(parents=True, exist_ok=True)
    (out_dir / "cka_hsic").mkdir(parents=True, exist_ok=True)

    act_a = os.path.join(args.base_data_dir, args.model, args.domain_a, args.split)
    act_b = os.path.join(args.base_data_dir, args.model, args.domain_b, args.split)

    print(f"Loading activations A: {act_a}")
    X_a, _ = load_pooled_all_layers(act_a, args.pooling, args.max_samples, args.seed)
    print(f"Loading activations B: {act_b}")
    X_b, _ = load_pooled_all_layers(act_b, args.pooling, args.max_samples, args.seed + 1)

    n = min(X_a.shape[0], X_b.shape[0])
    X_a = X_a[:n]
    X_b = X_b[:n]

    L = min(X_a.shape[1], X_b.shape[1])
    D = min(X_a.shape[2], X_b.shape[2])
    X_a = X_a[:, :L, :D]
    X_b = X_b[:, :L, :D]

    summary = []
    layerwise_cka = []
    layerwise_hsic = []

    for layer in range(L):
        Xa = X_a[:, layer, :]
        Xb = X_b[:, layer, :]

        Z_a, Z_b, evr = pca_fit_transform(Xa, Xb, args.pca_components, args.seed)
        save_evr_plot(
            evr,
            str(out_dir / "pca" / f"evr_layer_{layer}.png"),
            f"EVR Layer {layer} ({args.pooling})",
        )
        save_scatter_plot(
            Z_a,
            Z_b,
            str(out_dir / "pca" / f"pca_layer_{layer}.png"),
            f"PCA Layer {layer} ({args.pooling})",
        )

        cka = cka_linear(Xa, Xb)
        hsic = hsic_linear(Xa, Xb)
        layerwise_cka.append(cka)
        layerwise_hsic.append(hsic)

        summary.append({
            "layer": layer,
            "cka": cka,
            "hsic": hsic,
            "evr_10": evr_at_k(evr, 10),
            "evr_50": evr_at_k(evr, 50),
            "evr_90": evr_at_k(evr, 90),
        })

    Xa_agg = X_a.reshape(n, L * D)
    Xb_agg = X_b.reshape(n, L * D)

    Z_a, Z_b, evr = pca_fit_transform(Xa_agg, Xb_agg, args.pca_components, args.seed)
    save_evr_plot(
        evr,
        str(out_dir / "pca" / "evr_aggregate.png"),
        f"EVR Aggregate ({args.pooling})",
    )
    save_scatter_plot(
        Z_a,
        Z_b,
        str(out_dir / "pca" / "pca_aggregate.png"),
        f"PCA Aggregate ({args.pooling})",
    )

    cka_agg = cka_linear(Xa_agg, Xb_agg)
    hsic_agg = hsic_linear(Xa_agg, Xb_agg)

    plt.figure(figsize=(7, 4))
    plt.plot(layerwise_cka, marker="o")
    plt.title("Layerwise Linear CKA")
    plt.xlabel("Layer")
    plt.ylabel("CKA")
    plt.tight_layout()
    plt.savefig(out_dir / "cka_hsic" / "cka_layerwise.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(layerwise_hsic, marker="o", color="#f58518")
    plt.title("Layerwise Linear HSIC")
    plt.xlabel("Layer")
    plt.ylabel("HSIC")
    plt.tight_layout()
    plt.savefig(out_dir / "cka_hsic" / "hsic_layerwise.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_dir / "cka_hsic" / "cka_aggregate.json", "w") as f:
        json.dump({"cka_aggregate": cka_agg}, f, indent=2)
    with open(out_dir / "cka_hsic" / "hsic_aggregate.json", "w") as f:
        json.dump({"hsic_aggregate": hsic_agg}, f, indent=2)

    import csv
    with open(out_dir / "summary_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "cka", "hsic", "evr_10", "evr_50", "evr_90"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
        writer.writerow({
            "layer": "aggregate",
            "cka": cka_agg,
            "hsic": hsic_agg,
            "evr_10": evr_at_k(evr, 10),
            "evr_50": evr_at_k(evr, 50),
            "evr_90": evr_at_k(evr, 90),
        })

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"✓ Saved outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
