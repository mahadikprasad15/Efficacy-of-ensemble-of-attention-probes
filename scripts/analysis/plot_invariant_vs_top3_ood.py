#!/usr/bin/env python3
"""
Compare Invariant (Residual) vs Top-3 OOD Probes (Per Pooling)
==============================================================

Uses existing residual_id_ood_<pooling>.json files and OOD results JSONs
to produce per-pooling bar charts and a summary table.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


POOLINGS = ["mean", "max", "last", "attn"]


def load_residual_id_ood(residual_json: str) -> Tuple[float, float]:
    with open(residual_json, "r") as f:
        data = json.load(f)
    id_list = data.get("id_auc", [])
    ood_list = data.get("ood_auc", [])
    if not id_list or not ood_list:
        return None, None
    # Use best layer AUCs
    id_auc = float(np.max(id_list))
    ood_auc = float(np.max(ood_list))
    return id_auc, ood_auc


def extract_top3_from_ood_json(ood_json: str) -> Dict[str, Dict]:
    """
    Returns:
      pooling -> {
         top3_layers: [layer,...],
         top3_aucs: [auc,...],
         top3_avg: float
      }
    """
    with open(ood_json, "r") as f:
        data = json.load(f)

    results = {}
    for pooling in POOLINGS:
        # Common formats
        if pooling in data and isinstance(data[pooling], dict):
            # Array format: layers + aucs
            if "layers" in data[pooling] and "aucs" in data[pooling]:
                layer_aucs = list(zip(data[pooling]["layers"], data[pooling]["aucs"]))
            # Try per-layer data if present
            else:
                layer_results = data[pooling].get("per_layer_results")
                if layer_results and isinstance(layer_results, list):
                    layer_aucs = [(r.get("layer"), r.get("ood_auc", r.get("auc", 0.0))) for r in layer_results]
                else:
                    # Some results store per-layer under "results" or "layers"
                    layer_aucs = []
                    for k, v in data[pooling].items():
                        if isinstance(v, dict) and "ood_auc" in v:
                            try:
                                layer = int(str(k).replace("layer_", ""))
                            except Exception:
                                continue
                            layer_aucs.append((layer, v["ood_auc"]))
        elif "results" in data and pooling in data["results"]:
            # sweep-like format
            layer_aucs = []
            for r in data["results"][pooling]:
                if "error" in r:
                    continue
                auc = None
                if "ood_auc" in r:
                    auc = r["ood_auc"]
                elif "eval_on_insider" in r:
                    auc = r["eval_on_insider"].get("roleplaying_OOD") or r["eval_on_insider"].get("insider_ID")
                if auc is None:
                    continue
                layer_aucs.append((r.get("layer"), auc))
        else:
            layer_aucs = []

        if not layer_aucs:
            continue

        # sort by AUC
        layer_aucs = [(l, float(a)) for l, a in layer_aucs if l is not None]
        layer_aucs.sort(key=lambda x: x[1], reverse=True)
        top3 = layer_aucs[:3]
        results[pooling] = {
            "top3_layers": [l for l, _ in top3],
            "top3_aucs": [a for _, a in top3],
            "top3_avg": float(np.mean([a for _, a in top3])),
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot invariant vs top-3 OOD probes per pooling")
    parser.add_argument("--residual_dir", required=True,
                        help="Directory with residual_id_ood_<pooling>.json")
    parser.add_argument("--ood_results_a", required=True,
                        help="OOD results for Roleplaying->Insider")
    parser.add_argument("--ood_results_b", required=True,
                        help="OOD results for Insider->Role")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    top3_a = extract_top3_from_ood_json(args.ood_results_a)
    top3_b = extract_top3_from_ood_json(args.ood_results_b)

    summary_rows = []
    layers_summary = {}

    for pooling in POOLINGS:
        residual_json = os.path.join(args.residual_dir, f"residual_id_ood_{pooling}.json")
        if not os.path.exists(residual_json):
            continue

        id_auc, ood_auc = load_residual_id_ood(residual_json)

        a_top = top3_a.get(pooling)
        b_top = top3_b.get(pooling)
        if a_top is None or b_top is None:
            continue

        # Save summary
        summary_rows.append({
            "pooling": pooling,
            "residual_role_id_auc": id_auc,
            "residual_insider_ood_auc": ood_auc,
            "top3_role_ood_avg": a_top["top3_avg"],
            "top3_insider_ood_avg": b_top["top3_avg"],
            "top3_role_layers": a_top["top3_layers"],
            "top3_insider_layers": b_top["top3_layers"],
        })

        layers_summary[pooling] = {
            "role_top3_layers": a_top["top3_layers"],
            "role_top3_aucs": a_top["top3_aucs"],
            "insider_top3_layers": b_top["top3_layers"],
            "insider_top3_aucs": b_top["top3_aucs"],
        }

        # Plot
        labels = [
            "Residual (Role ID)",
            "Residual (Insider OOD)",
            "Top3 Role→Insider OOD",
            "Top3 Insider→Role OOD",
        ]
        values = [id_auc, ood_auc, a_top["top3_avg"], b_top["top3_avg"]]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f28e2b", "#e15759"])
        plt.ylabel("AUC")
        plt.ylim(0.4, 1.0)
        plt.title(f"Invariant vs Top‑3 OOD Probes ({pooling.upper()})")
        plt.grid(True, axis="y", alpha=0.3)

        # annotate
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

        out_path = os.path.join(args.output_dir, f"ood_vs_invariant_{pooling}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")

    # Save summary CSV
    out_csv = os.path.join(args.output_dir, "ood_vs_invariant_summary.csv")
    with open(out_csv, "w") as f:
        f.write("pooling,residual_role_id_auc,residual_insider_ood_auc,top3_role_ood_avg,top3_insider_ood_avg,top3_role_layers,top3_insider_layers\n")
        for row in summary_rows:
            f.write(
                f"{row['pooling']},{row['residual_role_id_auc']},{row['residual_insider_ood_auc']},"
                f"{row['top3_role_ood_avg']},{row['top3_insider_ood_avg']},"
                f"\"{row['top3_role_layers']}\",\"{row['top3_insider_layers']}\"\n"
            )
    print(f"✓ Saved: {out_csv}")

    out_json = os.path.join(args.output_dir, "ood_vs_invariant_layers.json")
    with open(out_json, "w") as f:
        json.dump(layers_summary, f, indent=2)
    print(f"✓ Saved: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
