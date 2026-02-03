#!/usr/bin/env python3
"""
Plot Cosine Similarity Across Layers for Invariant Core Sweep
=============================================================

Generates cosine similarity plots between:
  - Combined vs Roleplaying (w_C vs w_R)
  - Combined vs InsiderTrading (w_C vs w_I)
  - Roleplaying vs InsiderTrading (w_R vs w_I)

Optionally creates a side-by-side panel with OOD AUC curves from sweep_results.

Usage (Colab):
  python scripts/analysis/plot_invariant_core_cosine.py \
      --base_data_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data \
      --sweep_results /content/drive/MyDrive/results/invariant_core_sweep/sweep_results.json \
      --output_dir /content/drive/MyDrive/results/invariant_core_sweep/plots
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_probe_direction(probe_path: str) -> np.ndarray:
    """Load probe and extract a direction vector."""
    if not os.path.exists(probe_path):
        return None

    state_dict = torch.load(probe_path, map_location="cpu")

    # Prefer classifier weights; fall back to other known keys.
    priority_keys = [
        "classifier.weight",
        "net.0.weight",
        "pooling.weight",
        "pooling.query",
        "attn.weight",
    ]

    for key in priority_keys:
        if key in state_dict:
            w = state_dict[key].cpu().numpy()
            if w.ndim == 2:
                _, _, vt = np.linalg.svd(w, full_matrices=False)
                return vt[0]
            if w.ndim == 1:
                return w

    for key in sorted(state_dict.keys()):
        if "weight" in key and state_dict[key].ndim == 2:
            w = state_dict[key].cpu().numpy()
            _, _, vt = np.linalg.svd(w, full_matrices=False)
            return vt[0]

    return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with numerical safety."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def load_cosines_for_pooling(
    base_data_dir: str,
    sweep_results: Dict,
    model: str,
    domain_a: str,
    domain_b: str,
    pooling: str,
    signed: bool,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Compute cosine similarities across layers for a pooling type."""
    results = sweep_results.get("results", {}).get(pooling, [])

    layers, cos_cr, cos_ci, cos_ri = [], [], [], []

    probes_a_base = os.path.join(base_data_dir, "probes", model, domain_a, pooling)
    probes_b_base = os.path.join(base_data_dir, "probes_flipped", model, domain_b, pooling)
    probes_c_base = os.path.join(base_data_dir, "probes_combined", model, "Deception-Combined", pooling)

    for r in results:
        if "error" in r:
            continue
        layer = r.get("layer")
        if layer is None:
            continue

        probe_a_path = os.path.join(probes_a_base, f"probe_layer_{layer}.pt")
        probe_b_path = os.path.join(probes_b_base, f"probe_layer_{layer}.pt")
        probe_c_path = os.path.join(probes_c_base, f"probe_layer_{layer}.pt")

        w_r = get_probe_direction(probe_a_path)
        w_i = get_probe_direction(probe_b_path)
        w_c = get_probe_direction(probe_c_path)

        if w_r is None or w_i is None or w_c is None:
            continue

        c_cr = cosine(w_c, w_r)
        c_ci = cosine(w_c, w_i)
        c_ri = cosine(w_r, w_i)

        if not signed:
            c_cr = abs(c_cr)
            c_ci = abs(c_ci)
            c_ri = abs(c_ri)

        layers.append(layer)
        cos_cr.append(c_cr)
        cos_ci.append(c_ci)
        cos_ri.append(c_ri)

    # Sort by layer
    order = np.argsort(layers)
    layers = [layers[i] for i in order]
    cos_cr = [cos_cr[i] for i in order]
    cos_ci = [cos_ci[i] for i in order]
    cos_ri = [cos_ri[i] for i in order]

    return layers, cos_cr, cos_ci, cos_ri


def plot_cosines_per_pooling(
    layers: List[int],
    cos_cr: List[float],
    cos_ci: List[float],
    cos_ri: List[float],
    pooling: str,
    output_dir: str,
    signed: bool,
) -> str:
    """Plot cosine similarities for a single pooling."""
    fig, ax = plt.subplots(figsize=(11, 5))

    label_prefix = "" if signed else "|"
    label_suffix = "" if signed else "|"

    ax.plot(layers, cos_cr, "o-", label=f"{label_prefix}cos(w_C, w_R){label_suffix}", linewidth=2)
    ax.plot(layers, cos_ci, "s-", label=f"{label_prefix}cos(w_C, w_I){label_suffix}", linewidth=2)
    ax.plot(layers, cos_ri, "^-", label=f"{label_prefix}cos(w_R, w_I){label_suffix}", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Cosine Similarity Across Layers ({pooling.upper()} pooling)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(-1.0 if signed else 0.0, 1.0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cosine_similarity_{pooling}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _extract_eval_dict(r: Dict) -> Tuple[Dict, str]:
    """Find the evaluation dict in a sweep result entry."""
    # Common legacy formats
    if "ood_auc" in r:
        # ood_auc may be a dict of metrics or a single float
        if isinstance(r["ood_auc"], dict):
            return r["ood_auc"], "ood_auc"
        try:
            return {"invariant_core": float(r["ood_auc"])}, "ood_auc_scalar"
        except Exception:
            pass
    if "id_auc" in r:
        if isinstance(r["id_auc"], dict):
            return r["id_auc"], "id_auc"
        try:
            return {"invariant_core": float(r["id_auc"])}, "id_auc_scalar"
        except Exception:
            pass

    for key in ["eval_on_insider", "eval_on_roleplaying", "eval_on_ood", "eval_on_id", "eval"]:
        if key in r and isinstance(r[key], dict):
            return r[key], key
    # Some formats may store metrics at the top level
    if "invariant_core" in r or "combined" in r:
        return r, "root"
    return {}, ""


def plot_auc_and_cosine_panel(
    sweep_results: Dict,
    pooling: str,
    layers: List[int],
    cos_cr: List[float],
    cos_ci: List[float],
    cos_ri: List[float],
    output_dir: str,
    signed: bool,
) -> str:
    """Plot OOD AUC curves and cosine similarities side-by-side."""
    layer_results = sweep_results.get("results", {}).get(pooling, [])
    valid = [r for r in layer_results if "error" not in r]
    if not valid:
        return ""

    # Determine which eval dict key exists
    eval_dict, eval_key = _extract_eval_dict(valid[0])
    if not eval_dict:
        print(f"  ⚠ No eval dict found for pooling={pooling}; skipping AUC panel.")
        return ""

    auc_layers = [r["layer"] for r in valid]
    # Safely extract metrics, fall back to NaN if missing
    auc_invariant = [(_extract_eval_dict(r)[0].get("invariant_core", float("nan"))) for r in valid]
    auc_roleplaying = [(_extract_eval_dict(r)[0].get("roleplaying_OOD", float("nan"))) for r in valid]
    auc_insider = [(_extract_eval_dict(r)[0].get("insider_ID", float("nan"))) for r in valid]
    auc_combined = [(_extract_eval_dict(r)[0].get("combined", float("nan"))) for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # AUC panel
    ax = axes[0]
    ax.plot(auc_layers, auc_invariant, "o-", label="Invariant Core", linewidth=2)
    if not all(np.isnan(auc_roleplaying)):
        ax.plot(auc_layers, auc_roleplaying, "s--", label="Roleplaying (ID→OOD)", linewidth=1.5, alpha=0.8)
    if not all(np.isnan(auc_insider)):
        ax.plot(auc_layers, auc_insider, "^--", label="Insider (ID→OOD)", linewidth=1.5, alpha=0.8)
    if not all(np.isnan(auc_combined)):
        ax.plot(auc_layers, auc_combined, "d-", label="Combined", linewidth=2, alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("OOD AUC")
    ax.set_title(f"OOD AUC Across Layers ({pooling.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)

    # Cosine panel
    ax = axes[1]
    label_prefix = "" if signed else "|"
    label_suffix = "" if signed else "|"
    ax.plot(layers, cos_cr, "o-", label=f"{label_prefix}cos(w_C, w_R){label_suffix}", linewidth=2)
    ax.plot(layers, cos_ci, "s-", label=f"{label_prefix}cos(w_C, w_I){label_suffix}", linewidth=2)
    ax.plot(layers, cos_ri, "^-", label=f"{label_prefix}cos(w_R, w_I){label_suffix}", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarities")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(-1.0 if signed else 0.0, 1.0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ood_auc_and_cosine_{pooling}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_cosines_grid(
    all_cosines: Dict[str, Tuple[List[int], List[float], List[float], List[float]]],
    output_dir: str,
    signed: bool,
) -> str:
    """Create a 2x2 grid plot for all poolings."""
    poolings = ["mean", "max", "last", "attn"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    label_prefix = "" if signed else "|"
    label_suffix = "" if signed else "|"

    for ax, pooling in zip(axes.flatten(), poolings):
        if pooling not in all_cosines:
            ax.set_visible(False)
            continue
        layers, cos_cr, cos_ci, cos_ri = all_cosines[pooling]
        ax.plot(layers, cos_cr, "o-", label=f"{label_prefix}cos(w_C, w_R){label_suffix}", linewidth=2)
        ax.plot(layers, cos_ci, "s-", label=f"{label_prefix}cos(w_C, w_I){label_suffix}", linewidth=2)
        ax.plot(layers, cos_ri, "^-", label=f"{label_prefix}cos(w_R, w_I){label_suffix}", linewidth=2)
        ax.set_title(f"{pooling.upper()} pooling")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.0 if signed else 0.0, 1.0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Cosine Similarity Across Layers (All Pooling Types)", y=0.98)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cosine_similarity_all_pooling.png")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot cosine similarity across layers from sweep results.")
    parser.add_argument("--base_data_dir", type=str, required=True, help="Base data directory (contains probes/)")
    parser.add_argument("--sweep_results", type=str, required=True, help="Path to sweep_results.json")
    parser.add_argument("--model", type=str, default="meta-llama_Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--domain_a", type=str, default="Deception-Roleplaying", help="Domain A")
    parser.add_argument("--domain_b", type=str, default="Deception-InsiderTrading", help="Domain B")
    parser.add_argument("--output_dir", type=str, default="results/invariant_core_sweep/plots", help="Output directory")
    parser.add_argument("--signed", action="store_true", help="Plot signed cosine similarities (default: abs)")
    parser.add_argument("--no_auc_panel", action="store_true", help="Skip AUC+cosine side-by-side panels")
    args = parser.parse_args()

    with open(args.sweep_results, "r") as f:
        sweep_results = json.load(f)

    all_cosines = {}
    for pooling in ["mean", "max", "last", "attn"]:
        layers, cos_cr, cos_ci, cos_ri = load_cosines_for_pooling(
            args.base_data_dir,
            sweep_results,
            args.model,
            args.domain_a,
            args.domain_b,
            pooling,
            signed=args.signed,
        )
        if not layers:
            continue

        all_cosines[pooling] = (layers, cos_cr, cos_ci, cos_ri)
        out_path = plot_cosines_per_pooling(
            layers, cos_cr, cos_ci, cos_ri, pooling, args.output_dir, signed=args.signed
        )
        print(f"✓ Saved: {out_path}")

        if not args.no_auc_panel:
            panel_path = plot_auc_and_cosine_panel(
                sweep_results, pooling, layers, cos_cr, cos_ci, cos_ri, args.output_dir, signed=args.signed
            )
            if panel_path:
                print(f"✓ Saved: {panel_path}")

    grid_path = plot_cosines_grid(all_cosines, args.output_dir, signed=args.signed)
    print(f"✓ Saved: {grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
