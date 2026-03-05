#!/usr/bin/env python3
"""
Run probe-angle Mahalanobis alignment (Sigma_test, v1 only) from principled matrices.

This wrapper reads principled matrix-long CSVs produced by:
  scripts/analysis/build_pairwise_score_matrices_from_artifacts.py

and invokes:
  scripts/analysis/run_pairwise_mahalanobis_alignment.py

in a probe-angle-only mode:
  - --probe_angle_mode v1
  - --skip_activation_angles
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run principled probe-angle Mahalanobis (Sigma_test, v1 only)."
    )
    p.add_argument(
        "--principled_results_dir",
        type=str,
        required=True,
        help=(
            "Path to results dir from build_pairwise_score_matrices_from_artifacts.py "
            "(contains completion/full subdirs)."
        ),
    )
    p.add_argument(
        "--activations_root",
        type=str,
        default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt",
    )
    p.add_argument(
        "--probes_root",
        type=str,
        default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes",
    )
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--target_split", type=str, default="test")
    p.add_argument("--target_split_fallback", type=str, default="validation")
    p.add_argument(
        "--cov_backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
    )
    p.add_argument(
        "--covariance_scope",
        type=str,
        default="required",
        choices=["all", "required"],
        help="required is typically sufficient/faster for principled matrix configs.",
    )
    p.add_argument("--progress_every", type=int, default=200)
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.principled_results_dir)
    comp_csv = results_dir / "completion" / "matrix_long_source_val_selected.csv"
    full_csv = results_dir / "full" / "matrix_long_source_val_selected.csv"

    if not comp_csv.exists():
        raise FileNotFoundError(f"Missing completion principled matrix-long CSV: {comp_csv}")
    if not full_csv.exists():
        raise FileNotFoundError(f"Missing full principled matrix-long CSV: {full_csv}")

    cmd = [
        sys.executable,
        "scripts/analysis/run_pairwise_mahalanobis_alignment.py",
        "--matrix_completion_csv",
        str(comp_csv),
        "--matrix_full_csv",
        str(full_csv),
        "--activations_root",
        args.activations_root,
        "--probes_root",
        args.probes_root,
        "--model",
        args.model,
        "--artifact_root",
        args.artifact_root,
        "--target_split",
        args.target_split,
        "--target_split_fallback",
        args.target_split_fallback,
        "--cov_backend",
        args.cov_backend,
        "--covariance_scope",
        args.covariance_scope,
        "--probe_angle_mode",
        "v1",
        "--skip_activation_angles",
        "--progress_every",
        str(args.progress_every),
    ]
    if args.output_root:
        cmd.extend(["--output_root", args.output_root])
    if args.run_id:
        cmd.extend(["--run_id", args.run_id])
    if args.resume:
        cmd.append("--resume")
    if args.no_tqdm:
        cmd.append("--no_tqdm")
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run] invoking probe-angle wrapper command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
