#!/usr/bin/env python3
"""
Residual Token Patching (Mean Pooling)
=====================================

For samples where residual is correct but BOTH single-domain probes fail,
compute token contributions and counterfactual deltas using a position-wise
non-deceptive baseline.

Residual direction is loaded from saved invariant probes produced by the
invariant_core_sweep pipeline.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def load_norm_stats(probe_dir: str, layer: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    norm_path = os.path.join(probe_dir, f"norm_layer_{layer}.npz")
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        return norm["mean"], norm["std"]
    return None, None


def load_invariant_direction(invariant_dir: str, layer: int) -> np.ndarray:
    path = os.path.join(invariant_dir, f"invariant_layer_{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invariant probe not found: {path}")
    state = torch.load(path, map_location="cpu")
    if "classifier.weight" not in state:
        raise ValueError(f"Missing classifier.weight in {path}")
    w = state["classifier.weight"].squeeze().cpu().numpy()
    return w


def load_linear_probe(probe_dir: str, layer: int) -> Tuple[np.ndarray, float]:
    path = os.path.join(probe_dir, f"probe_layer_{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probe not found: {path}")
    state = torch.load(path, map_location="cpu")
    if "classifier.weight" in state:
        w = state["classifier.weight"].squeeze().cpu().numpy()
        b = float(state.get("classifier.bias", torch.zeros(1)).squeeze().cpu().numpy())
        return w, b
    # Fallback: try first layer of MLP (direction only, bias=0)
    if "net.0.weight" in state:
        w = state["net.0.weight"].cpu().numpy()
        if w.ndim == 2 and w.shape[0] == 1:
            w = w.squeeze()
        else:
            # Use first right-singular vector as direction
            _, _, vt = np.linalg.svd(w, full_matrices=False)
            w = vt[0]
        return w, 0.0
    raise ValueError(f"Unsupported probe format: {path}")


def load_activations_with_manifest(
    activations_dir: str,
    layer: int,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int], List[str], List[Optional[List[str]]]]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = []
    with open(manifest_path, "r") as f:
        for line in f:
            manifest.append(json.loads(line))

    if max_samples is not None:
        manifest = manifest[:max_samples]

    shards = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))

    activations = []
    labels = []
    ids = []
    tokens = []

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

        x_layer = tensor[layer].cpu().numpy()
        activations.append(x_layer)
        labels.append(int(label))
        ids.append(eid)
        tokens.append(entry.get("tokens"))

    return activations, labels, ids, tokens


def normalize_tokens(x: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return x
    return (x - mean) / (std + 1e-8)


def pooled_score(x_tokens: np.ndarray, w: np.ndarray, b: float = 0.0,
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> float:
    x_norm = normalize_tokens(x_tokens, mean, std)
    pooled = x_norm.mean(axis=0)
    return float(np.dot(w, pooled) + b)


def compute_position_baseline(
    activations: List[np.ndarray],
    labels: List[int],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> List[np.ndarray]:
    max_len = max(x.shape[0] for x in activations)
    sums = [None] * max_len
    counts = [0] * max_len

    for x, y in zip(activations, labels):
        if y != 0:
            continue  # non-deceptive only
        x_norm = normalize_tokens(x, mean, std)
        for t in range(x_norm.shape[0]):
            if sums[t] is None:
                sums[t] = x_norm[t].copy()
            else:
                sums[t] += x_norm[t]
            counts[t] += 1

    baseline = []
    for t in range(max_len):
        if counts[t] == 0:
            baseline.append(None)
        else:
            baseline.append(sums[t] / counts[t])
    return baseline


def select_top_layers(sweep_results: Dict, pooling: str, top_k: int) -> List[int]:
    results = sweep_results.get("results", {}).get(pooling, [])
    scored = []
    for r in results:
        if "error" in r:
            continue
        layer = r.get("layer")
        if layer is None:
            continue
        auc = None
        if "eval_on_insider" in r:
            auc = r["eval_on_insider"].get("invariant_core")
        elif "ood_auc" in r:
            if isinstance(r["ood_auc"], dict):
                auc = r["ood_auc"].get("invariant_core")
            else:
                try:
                    auc = float(r["ood_auc"])
                except Exception:
                    auc = None
        if auc is None:
            continue
        scored.append((auc, int(layer)))
    scored.sort(reverse=True)
    return [layer for _, layer in scored[:top_k]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Residual token patching (mean pooling)")
    parser.add_argument("--base_data_dir", required=True, help="Base data directory")
    parser.add_argument("--model", required=True, help="Model name used in activations path")
    parser.add_argument("--sweep_results", required=True, help="Path to sweep_results.json")
    parser.add_argument("--invariant_probes_dir", required=True, help="Invariant probes dir (from sweep)")
    parser.add_argument("--combined_probes_dir", required=True, help="Combined probes dir (for norm stats)")
    parser.add_argument("--probes_a_dir", required=True, help="Domain A probes dir (mean pooling)")
    parser.add_argument("--probes_b_dir", required=True, help="Domain B probes dir (mean pooling)")
    parser.add_argument("--domain_a", default="Deception-Roleplaying")
    parser.add_argument("--domain_b", default="Deception-InsiderTrading")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--filter_mode", default="both_fail",
                        choices=["both_fail", "either_fail", "residual_only"],
                        help="Sample filter: residual correct AND (both fail | either fail | no constraint)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.sweep_results, "r") as f:
        sweep_results = json.load(f)

    layers = select_top_layers(sweep_results, args.pooling, args.top_k)
    if not layers:
        print("No layers found from sweep_results.")
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    for layer in layers:
        print(f"\n=== Layer {layer} ===")

        w_res = load_invariant_direction(args.invariant_probes_dir, layer)
        res_norm = np.linalg.norm(w_res)
        if res_norm < 1e-8:
            print(f"  Warning: near-zero residual vector at layer {layer}")
        w_res = w_res / (res_norm + 1e-8)

        # Load single-domain probes
        w_a, b_a = load_linear_probe(args.probes_a_dir, layer)
        w_b, b_b = load_linear_probe(args.probes_b_dir, layer)

        # Norm stats
        mean_res, std_res = load_norm_stats(args.combined_probes_dir, layer)
        if mean_res is None:
            print("  Warning: combined norm stats not found; using raw activations for residual.")
        mean_a, std_a = load_norm_stats(args.probes_a_dir, layer)
        mean_b, std_b = load_norm_stats(args.probes_b_dir, layer)

        # Sanity check dimension
        if mean_res is not None and w_res.shape[-1] != mean_res.shape[-1]:
            raise ValueError(f"Residual dim {w_res.shape[-1]} != norm dim {mean_res.shape[-1]}")

        for split in splits:
            for domain, domain_dir in [(args.domain_a, args.domain_a), (args.domain_b, args.domain_b)]:
                act_dir = os.path.join(args.base_data_dir, "activations", args.model, domain_dir, split)
                print(f"  Loading {domain}/{split} from {act_dir}")
                acts, labels, ids, tokens = load_activations_with_manifest(
                    act_dir, layer, max_samples=args.max_samples
                )
                if not acts:
                    print("  No activations found.")
                    continue

                # Baseline in residual-normalized space
                baseline = compute_position_baseline(acts, labels, mean_res, std_res)

                # Evaluate and select samples
                selected = []
                for x, y, eid, toks in zip(acts, labels, ids, tokens):
                    res_score = pooled_score(x, w_res, 0.0, mean_res, std_res)
                    pred_res = int(res_score > 0)

                    score_a = pooled_score(x, w_a, b_a, mean_a, std_a)
                    score_b = pooled_score(x, w_b, b_b, mean_b, std_b)
                    pred_a = int(score_a > 0)
                    pred_b = int(score_b > 0)

                    keep = False
                    if pred_res == y:
                        if args.filter_mode == "both_fail":
                            keep = (pred_a != y and pred_b != y)
                        elif args.filter_mode == "either_fail":
                            keep = (pred_a != y or pred_b != y)
                        elif args.filter_mode == "residual_only":
                            keep = True
                    if keep:
                        selected.append((x, y, eid, toks, res_score, score_a, score_b))

                if not selected:
                    print("  No samples matching residual-correct and both single-domain wrong.")
                    continue

                out_dir = os.path.join(args.output_dir, split, domain)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"layer_{layer}.jsonl")
                summary_path = os.path.join(out_dir, f"layer_{layer}_summary.json")

                pos_counts = {}
                token_counts = {}
                deltas_all = []

                with open(out_path, "w") as f:
                    for x, y, eid, toks, res_score, score_a, score_b in tqdm(selected, desc="Patching"):
                        x_norm = normalize_tokens(x, mean_res, std_res)
                        T = x_norm.shape[0]
                        contrib = (x_norm @ w_res) / T

                        top_entries = []
                        for t in range(T):
                            b_t = baseline[t] if t < len(baseline) else None
                            if b_t is None:
                                continue
                            delta = float(np.dot(w_res, (b_t - x_norm[t])) / T)
                            token = None
                            if toks and t < len(toks):
                                token = toks[t]
                            top_entries.append({
                                "index": t,
                                "token": token,
                                "contribution": float(contrib[t]),
                                "delta_score": delta
                            })

                        top_entries.sort(key=lambda r: abs(r["delta_score"]), reverse=True)
                        top_entries = top_entries[:20]

                        for te in top_entries:
                            pos_counts[str(te["index"])] = pos_counts.get(str(te["index"]), 0) + 1
                            if te["token"] is not None:
                                token_counts[te["token"]] = token_counts.get(te["token"], 0) + 1
                            deltas_all.append(te["delta_score"])

                        record = {
                            "id": eid,
                            "label": y,
                            "residual_score": float(res_score),
                            "roleplaying_score": float(score_a),
                            "insider_score": float(score_b),
                            "length": int(T),
                            "top_tokens": top_entries,
                        }
                        f.write(json.dumps(record) + "\n")

                summary = {
                    "layer": layer,
                    "domain": domain,
                    "split": split,
                    "num_samples": len(selected),
                    "top_positions": sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:20],
                    "top_tokens": sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20],
                    "delta_stats": {
                        "mean_abs_delta": float(np.mean(np.abs(deltas_all))) if deltas_all else 0.0,
                        "median_abs_delta": float(np.median(np.abs(deltas_all))) if deltas_all else 0.0,
                    },
                }

                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"  ✓ Saved: {out_path}")
                print(f"  ✓ Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
