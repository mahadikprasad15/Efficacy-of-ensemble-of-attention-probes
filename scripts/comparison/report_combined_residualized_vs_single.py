#!/usr/bin/env python3
"""
One-command comparison report:
raw combined vs residualized combined vs single-domain baselines.

Inputs:
1) Raw combined OOD JSON (from evaluate_ood_all_pooling.py)
2) Residualized OOD JSON (same format)
3) Optional single-domain baseline CSVs (top20 summaries)

Outputs:
- comparison_report.json
- per_pooling_comparison.csv
- selected_single_baselines.csv (if baseline CSVs provided)
- summary.txt
- status.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


POOLINGS = ["mean", "max", "last", "attn"]


@dataclass
class PoolingBest:
    pooling: str
    best_layer: int
    best_auc: float
    best_accuracy: Optional[float]


def load_ood_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON structure: {path}")
    return data


def pooling_rows(ood_json: Dict, pooling: str) -> pd.DataFrame:
    entry = ood_json.get(pooling, {})
    layers = entry.get("layers", [])
    aucs = entry.get("aucs", [])
    accs = entry.get("accuracies", [])
    if not layers or not aucs:
        return pd.DataFrame(columns=["layer", "auc", "accuracy"])
    n = min(len(layers), len(aucs))
    if not accs:
        accs = [None] * n
    else:
        accs = accs[:n]
    return pd.DataFrame(
        {
            "layer": [int(x) for x in layers[:n]],
            "auc": [float(x) for x in aucs[:n]],
            "accuracy": [None if x is None else float(x) for x in accs],
        }
    )


def pooling_best(ood_json: Dict, pooling: str) -> Optional[PoolingBest]:
    df = pooling_rows(ood_json, pooling)
    if df.empty:
        return None
    idx = df["auc"].idxmax()
    row = df.loc[idx]
    return PoolingBest(
        pooling=pooling,
        best_layer=int(row["layer"]),
        best_auc=float(row["auc"]),
        best_accuracy=None if pd.isna(row["accuracy"]) else float(row["accuracy"]),
    )


def auc_at_layer(ood_json: Dict, pooling: str, layer: int) -> Optional[float]:
    df = pooling_rows(ood_json, pooling)
    if df.empty:
        return None
    m = df[df["layer"] == int(layer)]
    if m.empty:
        return None
    return float(m.iloc[0]["auc"])


def pick_best_single_baseline(
    csv_path: str,
    target_contains: Optional[str],
    source_probe_contains: Optional[str],
    tag: str,
) -> Optional[Dict]:
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    required = [
        "Source Probe",
        "Target Dataset (Test)",
        "Best Pooling",
        "Best Layer",
        "Best AUC",
        "Best Accuracy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{tag}: missing columns {missing} in {csv_path}")

    if target_contains:
        t = target_contains.lower()
        df = df[df["Target Dataset (Test)"].astype(str).str.lower().str.contains(t, na=False)]
    if source_probe_contains:
        s = source_probe_contains.lower()
        df = df[df["Source Probe"].astype(str).str.lower().str.contains(s, na=False)]
    if df.empty:
        return None

    df = df.copy()
    df["Best AUC"] = pd.to_numeric(df["Best AUC"], errors="coerce")
    df["Best Accuracy"] = pd.to_numeric(df["Best Accuracy"], errors="coerce")
    df["Best Layer"] = pd.to_numeric(df["Best Layer"], errors="coerce")
    df = df.dropna(subset=["Best AUC", "Best Layer"])
    if df.empty:
        return None

    df = df.sort_values(["Best AUC", "Best Accuracy"], ascending=[False, False]).reset_index(drop=True)
    row = df.iloc[0]
    return {
        "tag": tag,
        "csv_path": csv_path,
        "source_probe": str(row["Source Probe"]),
        "target_dataset_test": str(row["Target Dataset (Test)"]),
        "best_pooling": str(row["Best Pooling"]).strip().lower(),
        "best_layer": int(row["Best Layer"]),
        "best_auc": float(row["Best AUC"]),
        "best_accuracy": None if pd.isna(row["Best Accuracy"]) else float(row["Best Accuracy"]),
    }


def format_summary(
    raw_overall: Dict,
    resid_overall: Dict,
    per_pool_df: pd.DataFrame,
    baselines: List[Dict],
) -> str:
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("COMBINED RAW vs RESIDUALIZED vs SINGLE-DOMAIN BASELINES")
    lines.append("=" * 90)
    lines.append("")
    lines.append(
        f"Overall best raw:          pooling={raw_overall['pooling']} layer={raw_overall['layer']} auc={raw_overall['auc']:.4f}"
    )
    lines.append(
        f"Overall best residualized: pooling={resid_overall['pooling']} layer={resid_overall['layer']} auc={resid_overall['auc']:.4f}"
    )
    lines.append(f"Overall delta (resid - raw): {resid_overall['auc'] - raw_overall['auc']:+.4f}")
    lines.append("")
    lines.append("Per-pooling:")
    for _, r in per_pool_df.iterrows():
        lines.append(
            f"- {r['pooling']:>4s} | raw_best L{int(r['raw_best_layer']):>2d} {r['raw_best_auc']:.4f} | "
            f"resid_best L{int(r['resid_best_layer']):>2d} {r['resid_best_auc']:.4f} | "
            f"delta_best {r['delta_resid_best_minus_raw_best']:+.4f} | "
            f"delta_at_raw_best {r['delta_resid_at_raw_best_minus_raw_best']:+.4f}"
        )

    if baselines:
        lines.append("")
        lines.append("Selected single-domain baselines:")
        for b in baselines:
            lines.append(
                f"- {b['tag']}: auc={b['best_auc']:.4f} pooling={b['best_pooling']} layer={b['best_layer']} | "
                f"{b['source_probe']} -> {b['target_dataset_test']}"
            )
        best_single = max(b["best_auc"] for b in baselines)
        lines.append("")
        lines.append(
            f"Best single baseline AUC: {best_single:.4f} | "
            f"raw delta={raw_overall['auc'] - best_single:+.4f}, "
            f"resid delta={resid_overall['auc'] - best_single:+.4f}"
        )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw vs residualized combined probes and single-domain baselines.")
    parser.add_argument("--raw_ood_json", type=str, required=True)
    parser.add_argument("--residualized_ood_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_csv_a", type=str, default=None)
    parser.add_argument("--baseline_csv_b", type=str, default=None)
    parser.add_argument("--target_contains", type=str, default="InsiderTrading-completion")
    parser.add_argument("--source_probe_a_contains", type=str, default=None)
    parser.add_argument("--source_probe_b_contains", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "comparison_report.json")
    status_path = os.path.join(args.output_dir, "status.json")
    per_pool_csv = os.path.join(args.output_dir, "per_pooling_comparison.csv")
    summary_txt = os.path.join(args.output_dir, "summary.txt")
    baselines_csv = os.path.join(args.output_dir, "selected_single_baselines.csv")

    if args.resume and os.path.exists(report_path):
        with open(report_path, "r") as f:
            existing = json.load(f)
        print(json.dumps(existing.get("overall", {}), indent=2))
        return 0

    with open(status_path, "w") as f:
        json.dump({"status": "running"}, f, indent=2)

    raw = load_ood_json(args.raw_ood_json)
    resid = load_ood_json(args.residualized_ood_json)

    rows = []
    raw_global = {"auc": -1.0, "pooling": None, "layer": None}
    resid_global = {"auc": -1.0, "pooling": None, "layer": None}

    for pooling in POOLINGS:
        b_raw = pooling_best(raw, pooling)
        b_resid = pooling_best(resid, pooling)
        if b_raw is None or b_resid is None:
            continue

        resid_at_raw_best = auc_at_layer(resid, pooling, b_raw.best_layer)
        raw_at_resid_best = auc_at_layer(raw, pooling, b_resid.best_layer)

        rows.append(
            {
                "pooling": pooling,
                "raw_best_layer": b_raw.best_layer,
                "raw_best_auc": b_raw.best_auc,
                "resid_best_layer": b_resid.best_layer,
                "resid_best_auc": b_resid.best_auc,
                "resid_auc_at_raw_best_layer": resid_at_raw_best,
                "raw_auc_at_resid_best_layer": raw_at_resid_best,
                "delta_resid_best_minus_raw_best": b_resid.best_auc - b_raw.best_auc,
                "delta_resid_at_raw_best_minus_raw_best": None
                if resid_at_raw_best is None
                else resid_at_raw_best - b_raw.best_auc,
            }
        )

        if b_raw.best_auc > raw_global["auc"]:
            raw_global = {"auc": b_raw.best_auc, "pooling": pooling, "layer": b_raw.best_layer}
        if b_resid.best_auc > resid_global["auc"]:
            resid_global = {"auc": b_resid.best_auc, "pooling": pooling, "layer": b_resid.best_layer}

    if not rows:
        raise RuntimeError("No overlapping pooling results found between raw and residualized JSONs.")

    per_pool_df = pd.DataFrame(rows).sort_values("pooling").reset_index(drop=True)
    per_pool_df.to_csv(per_pool_csv, index=False)

    baselines: List[Dict] = []
    b_a = pick_best_single_baseline(
        args.baseline_csv_a,
        target_contains=args.target_contains,
        source_probe_contains=args.source_probe_a_contains,
        tag="baseline_a",
    ) if args.baseline_csv_a else None
    b_b = pick_best_single_baseline(
        args.baseline_csv_b,
        target_contains=args.target_contains,
        source_probe_contains=args.source_probe_b_contains,
        tag="baseline_b",
    ) if args.baseline_csv_b else None
    if b_a:
        baselines.append(b_a)
    if b_b:
        baselines.append(b_b)
    if baselines:
        pd.DataFrame(baselines).to_csv(baselines_csv, index=False)

    overall = {
        "raw_best": raw_global,
        "residualized_best": resid_global,
        "delta_residualized_minus_raw": resid_global["auc"] - raw_global["auc"],
    }

    if baselines:
        best_single_auc = max(x["best_auc"] for x in baselines)
        overall["best_single_baseline_auc"] = best_single_auc
        overall["delta_raw_minus_best_single"] = raw_global["auc"] - best_single_auc
        overall["delta_residualized_minus_best_single"] = resid_global["auc"] - best_single_auc

    report = {
        "inputs": {
            "raw_ood_json": args.raw_ood_json,
            "residualized_ood_json": args.residualized_ood_json,
            "baseline_csv_a": args.baseline_csv_a,
            "baseline_csv_b": args.baseline_csv_b,
            "target_contains": args.target_contains,
            "source_probe_a_contains": args.source_probe_a_contains,
            "source_probe_b_contains": args.source_probe_b_contains,
        },
        "overall": overall,
        "per_pooling": rows,
        "selected_single_baselines": baselines,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    summary = format_summary(raw_global, resid_global, per_pool_df, baselines)
    with open(summary_txt, "w") as f:
        f.write(summary + "\n")
    print(summary)

    with open(status_path, "w") as f:
        json.dump(
            {
                "status": "completed",
                "report_path": report_path,
                "per_pooling_csv": per_pool_csv,
                "summary_txt": summary_txt,
                "selected_single_baselines_csv": baselines_csv if baselines else None,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
