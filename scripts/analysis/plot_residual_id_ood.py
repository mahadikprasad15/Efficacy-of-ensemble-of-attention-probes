#!/usr/bin/env python3
"""
Residual ID vs OOD Evaluation (Per Pooling)
===========================================

Evaluate saved invariant (residual) probes on ID and OOD activations and
plot per-layer AUC curves for each pooling.

Default: ID = Roleplaying, OOD = InsiderTrading.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


POOLINGS = ["mean", "max", "last", "attn"]


def load_invariant_direction(invariant_dir: str, layer: int) -> np.ndarray:
    path = os.path.join(invariant_dir, f"invariant_layer_{layer}.pt")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu")
    if "classifier.weight" not in state:
        return None
    w = state["classifier.weight"].squeeze().cpu().numpy()
    return w


def load_activations(act_dir: str, layer: int, pooling: str) -> Tuple[np.ndarray, np.ndarray]:
    manifest_path = os.path.join(act_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return None, None

    with open(manifest_path, "r") as f:
        manifest = [json.loads(line) for line in f]

    shards = sorted(glob.glob(os.path.join(act_dir, "shard_*.safetensors")))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))

    activations, labels = [], []
    for entry in manifest:
        eid = entry.get("id")
        if eid not in all_tensors:
            continue
        label = entry.get("label", -1)
        if label == -1:
            continue
        tensor = all_tensors[eid]
        if layer >= tensor.shape[0]:
            continue
        x_layer = tensor[layer, :, :]
        if pooling in ["mean", "attn"]:
            pooled = x_layer.mean(dim=0)
        elif pooling == "max":
            pooled = x_layer.max(dim=0)[0]
        elif pooling == "last":
            pooled = x_layer[-1, :]
        else:
            pooled = x_layer.mean(dim=0)
        activations.append(pooled.numpy())
        labels.append(int(label))

    if len(activations) == 0:
        return None, None
    return np.array(activations), np.array(labels)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        auc_pos = roc_auc_score(labels, scores)
        auc_neg = roc_auc_score(labels, -scores)
        return max(auc_pos, auc_neg)
    except Exception:
        return 0.5


def extract_layers_from_sweep(sweep_results: Dict, pooling: str) -> List[int]:
    results = sweep_results.get("results", {}).get(pooling, [])
    layers = []
    for r in results:
        if "error" in r:
            continue
        layer = r.get("layer")
        if layer is not None:
            layers.append(int(layer))
    return sorted(list(set(layers)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual ID vs OOD evaluation")
    parser.add_argument("--base_data_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--invariant_root", required=True,
                        help="Path to invariant_core_sweep/invariant_probes")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pooling", default="all",
                        help="Pooling type (mean/max/last/attn) or 'all'")
    parser.add_argument("--id_domain", default="Deception-Roleplaying")
    parser.add_argument("--ood_domain", default="Deception-InsiderTrading")
    parser.add_argument("--id_split", default="validation")
    parser.add_argument("--ood_split", default="test")
    parser.add_argument("--min_layer", type=int, default=1)
    parser.add_argument("--max_layer", type=int, default=28)
    parser.add_argument("--sweep_results", default=None,
                        help="Optional sweep_results.json to infer layers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.pooling == "all":
        poolings = POOLINGS
    else:
        poolings = [args.pooling]

    sweep_results = None
    if args.sweep_results and os.path.exists(args.sweep_results):
        with open(args.sweep_results, "r") as f:
            sweep_results = json.load(f)

    for pooling in poolings:
        if sweep_results:
            layers = extract_layers_from_sweep(sweep_results, pooling)
        else:
            layers = list(range(args.min_layer, args.max_layer + 1))

        invariant_dir = os.path.join(args.invariant_root, pooling)
        id_dir = os.path.join(args.base_data_dir, "activations", args.model,
                              args.id_domain, args.id_split)
        ood_dir = os.path.join(args.base_data_dir, "activations", args.model,
                               args.ood_domain, args.ood_split)

        id_aucs = []
        ood_aucs = []
        valid_layers = []

        for layer in tqdm(layers, desc=f"{pooling} pooling"):
            w = load_invariant_direction(invariant_dir, layer)
            if w is None:
                continue
            w = w / (np.linalg.norm(w) + 1e-8)

            X_id, y_id = load_activations(id_dir, layer, pooling)
            X_ood, y_ood = load_activations(ood_dir, layer, pooling)
            if X_id is None or X_ood is None:
                continue

            scores_id = X_id @ w
            scores_ood = X_ood @ w
            id_auc = compute_auc(scores_id, y_id)
            ood_auc = compute_auc(scores_ood, y_ood)

            valid_layers.append(layer)
            id_aucs.append(id_auc)
            ood_aucs.append(ood_auc)

        # Save JSON
        out_json = os.path.join(args.output_dir, f"residual_id_ood_{pooling}.json")
        with open(out_json, "w") as f:
            json.dump({
                "pooling": pooling,
                "id_domain": args.id_domain,
                "ood_domain": args.ood_domain,
                "id_split": args.id_split,
                "ood_split": args.ood_split,
                "layers": valid_layers,
                "id_auc": id_aucs,
                "ood_auc": ood_aucs,
            }, f, indent=2)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(valid_layers, id_aucs, "o-", label=f"ID ({args.id_domain})", linewidth=2)
        plt.plot(valid_layers, ood_aucs, "s--", label=f"OOD ({args.ood_domain})", linewidth=2)
        plt.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        plt.xlabel("Layer")
        plt.ylabel("AUC")
        plt.title(f"Residual ID vs OOD (Pooling={pooling.upper()})")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.ylim(0.4, 1.0)

        out_png = os.path.join(args.output_dir, f"residual_id_ood_{pooling}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
