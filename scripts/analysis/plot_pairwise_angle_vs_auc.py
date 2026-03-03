#!/usr/bin/env python3
"""
Plot pairwise probe-angle vs AUC scatter charts from existing matrix artifacts.

Inputs:
  - Pairwise evaluation score matrices (wide CSV):
      matrix_completion_auc.csv, matrix_full_auc.csv
  - Pairwise Mahalanobis probe-angle matrices:
      probe_angle_v1_completion.csv, probe_angle_v1_full.csv
      probe_angle_v2_completion.csv, probe_angle_v2_full.csv

Outputs:
  - Scatter plots (with best-fit line) for v1/v2 on completion/full and combined
  - Aligned point tables per plot
  - Summary JSON with fit statistics and paths

Default output root:
  artifacts/runs/pairwise_angle_vs_auc_scatter/<model_dir>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Dict | None = None) -> Dict:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_status(path: Path, state: str, message: str) -> None:
    payload = read_json(path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def load_wide_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix CSV: {path}")
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Matrix CSV has too few columns: {path}")
    row_col = df.columns[0]
    df = df.set_index(row_col)
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def aligned_points(
    score: pd.DataFrame,
    angle: pd.DataFrame,
    include_diagonal: bool = False,
) -> pd.DataFrame:
    common_rows = [r for r in score.index if r in set(angle.index)]
    common_cols = [c for c in score.columns if c in set(angle.columns)]

    rows: List[Dict] = []
    for r in common_rows:
        for c in common_cols:
            if (not include_diagonal) and (r == c):
                continue
            auc = score.at[r, c]
            ang = angle.at[r, c]
            if pd.isna(auc) or pd.isna(ang):
                continue
            rows.append(
                {
                    "row": r,
                    "col": c,
                    "auc": float(auc),
                    "angle": float(ang),
                }
            )
    return pd.DataFrame(rows)


def fit_stats(points: pd.DataFrame) -> Dict:
    n = int(len(points))
    if n == 0:
        return {"n_points": 0, "slope": None, "intercept": None, "pearson_r": None}
    x = points["angle"].to_numpy(dtype=np.float64)
    y = points["auc"].to_numpy(dtype=np.float64)

    if n >= 2 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        r = float(np.corrcoef(x, y)[0, 1])
        return {
            "n_points": n,
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": r,
        }
    return {"n_points": n, "slope": None, "intercept": None, "pearson_r": None}


def plot_scatter(points: pd.DataFrame, out_path: Path, title: str) -> Dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = fit_stats(points)

    plt.figure(figsize=(7, 6))
    if len(points) > 0:
        x = points["angle"].to_numpy(dtype=np.float64)
        y = points["auc"].to_numpy(dtype=np.float64)
        plt.scatter(x, y, alpha=0.7, s=28)
        if stats["slope"] is not None and stats["intercept"] is not None:
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            ys = stats["slope"] * xs + stats["intercept"]
            plt.plot(xs, ys, color="black", linewidth=1.7)
    plt.xlabel("Probe Mahalanobis cosine")
    plt.ylabel("AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pairwise probe-angle vs AUC scatter charts.")
    p.add_argument("--pairwise_eval_results_dir", type=str, required=True)
    p.add_argument("--pairwise_alignment_results_dir", type=str, required=True)
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional direct run root override. If set, outputs go to <output_root>/<run_id>/...",
    )
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--include_diagonal", action="store_true", help="Include diagonal cells (row==col).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()

    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = (
            Path(args.artifact_root)
            / "runs"
            / "pairwise_angle_vs_auc_scatter"
            / model_dir
            / run_id
        )

    meta_dir = run_root / "meta"
    out_dir = run_root / "results"
    plots_dir = out_dir / "plots"
    points_dir = out_dir / "points"
    for d in [meta_dir, out_dir, plots_dir, points_dir]:
        d.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress_path = meta_dir / "progress.json"
    summary_path = out_dir / "summary.json"
    progress = read_json(progress_path, default={"completed_steps": []})

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "pairwise_eval_results_dir": args.pairwise_eval_results_dir,
            "pairwise_alignment_results_dir": args.pairwise_alignment_results_dir,
            "include_diagonal": bool(args.include_diagonal),
            "run_root": str(run_root),
        },
    )
    update_status(status_path, "running", "starting")

    def mark(step: str) -> None:
        done = set(progress.get("completed_steps", []))
        done.add(step)
        progress["completed_steps"] = sorted(done)
        progress["updated_at"] = utc_now()
        write_json(progress_path, progress)

    if args.resume and "finished" in set(progress.get("completed_steps", [])) and summary_path.exists():
        update_status(status_path, "completed", "resume: already finished")
        print(f"[resume] already complete: {run_root}")
        return 0

    eval_dir = Path(args.pairwise_eval_results_dir)
    align_dir = Path(args.pairwise_alignment_results_dir)

    score_comp = load_wide_matrix(eval_dir / "matrix_completion_auc.csv")
    score_full = load_wide_matrix(eval_dir / "matrix_full_auc.csv")
    angle_v1_comp = load_wide_matrix(align_dir / "probe_angle_v1_completion.csv")
    angle_v1_full = load_wide_matrix(align_dir / "probe_angle_v1_full.csv")
    angle_v2_comp = load_wide_matrix(align_dir / "probe_angle_v2_completion.csv")
    angle_v2_full = load_wide_matrix(align_dir / "probe_angle_v2_full.csv")
    mark("load_inputs")

    point_sets: Dict[str, pd.DataFrame] = {
        "v1_completion": aligned_points(score_comp, angle_v1_comp, include_diagonal=args.include_diagonal),
        "v1_full": aligned_points(score_full, angle_v1_full, include_diagonal=args.include_diagonal),
        "v2_completion": aligned_points(score_comp, angle_v2_comp, include_diagonal=args.include_diagonal),
        "v2_full": aligned_points(score_full, angle_v2_full, include_diagonal=args.include_diagonal),
    }
    point_sets["v1_combined"] = pd.concat(
        [point_sets["v1_completion"], point_sets["v1_full"]],
        ignore_index=True,
    )
    point_sets["v2_combined"] = pd.concat(
        [point_sets["v2_completion"], point_sets["v2_full"]],
        ignore_index=True,
    )
    mark("build_points")

    plot_stats: Dict[str, Dict] = {}
    plot_titles = {
        "v1_completion": "Probe Angle v1 vs AUC (Completion)",
        "v1_full": "Probe Angle v1 vs AUC (Full)",
        "v1_combined": "Probe Angle v1 vs AUC (Combined)",
        "v2_completion": "Probe Angle v2 vs AUC (Completion)",
        "v2_full": "Probe Angle v2 vs AUC (Full)",
        "v2_combined": "Probe Angle v2 vs AUC (Combined)",
    }
    for key, points in point_sets.items():
        points_path = points_dir / f"{key}.csv"
        points.to_csv(points_path, index=False)
        stats = plot_scatter(points, plots_dir / f"scatter_{key}.png", plot_titles[key])
        stats["points_csv"] = str(points_path)
        stats["plot_path"] = str(plots_dir / f"scatter_{key}.png")
        plot_stats[key] = stats
    mark("plots")

    summary = {
        "run_id": run_id,
        "completed_at": utc_now(),
        "model": args.model,
        "inputs": {
            "pairwise_eval_results_dir": str(eval_dir),
            "pairwise_alignment_results_dir": str(align_dir),
            "score_completion_csv": str(eval_dir / "matrix_completion_auc.csv"),
            "score_full_csv": str(eval_dir / "matrix_full_auc.csv"),
            "probe_angle_v1_completion_csv": str(align_dir / "probe_angle_v1_completion.csv"),
            "probe_angle_v1_full_csv": str(align_dir / "probe_angle_v1_full.csv"),
            "probe_angle_v2_completion_csv": str(align_dir / "probe_angle_v2_completion.csv"),
            "probe_angle_v2_full_csv": str(align_dir / "probe_angle_v2_full.csv"),
        },
        "include_diagonal": bool(args.include_diagonal),
        "outputs": {
            "plots_dir": str(plots_dir),
            "points_dir": str(points_dir),
            "plot_stats": plot_stats,
        },
    }
    write_json(summary_path, summary)
    mark("finished")
    update_status(status_path, "completed", "finished")
    print(f"[done] artifacts -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

