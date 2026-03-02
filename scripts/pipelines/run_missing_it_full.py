#!/usr/bin/env python3
"""
Detect missing IT-full evaluations and run them via the pairwise matrix pipeline.

Default target: Deception-InsiderTrading-full
Default sources: full-segment row datasets from the matrix spec.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines.run_pairwise_eval_matrix import (  # noqa: E402
    parse_existing_pair_summary,
    split_root_and_model,
    stage_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing IT-full evaluations only.")
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--probes_root", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target", type=str, default="Deception-InsiderTrading-full")
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Optional comma-separated list of sources to consider (default: full matrix rows).",
    )
    parser.add_argument("--poolings", type=str, default="mean,max,last,attn")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_diagonals",
        action="store_true",
        help="Skip diagonal validation stage in the invoked pipeline.",
    )
    parser.add_argument(
        "--rebuild_matrices",
        action="store_true",
        help="After running missing evals, rebuild full matrices via the pipeline (resume + skip_training).",
    )
    return parser.parse_args()


def missing_sources(
    results_model_root: Path,
    sources: Sequence[str],
    target: str,
) -> List[str]:
    missing: List[str] = []
    for src in sources:
        if src == target:
            continue
        pair_dir = results_model_root / f"from-{src}" / f"to-{target}"
        if (pair_dir / "pair_summary.json").exists():
            continue
        # Accept older formats as "existing"
        if parse_existing_pair_summary(pair_dir) is not None:
            continue
        missing.append(src)
    return missing


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    _, results_model_root = split_root_and_model(Path(args.results_root), model_dir)
    results_model_root.mkdir(parents=True, exist_ok=True)

    rows_map, _, _, _, _ = stage_spec()
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        sources = rows_map["full"]

    missing = missing_sources(results_model_root, sources, args.target)
    print(f"[scan] target={args.target}")
    print(f"[scan] sources_considered={len(sources)}")
    print(f"[scan] missing_sources={missing}")

    if not missing:
        print("[scan] no missing evaluations detected.")
        if not args.rebuild_matrices:
            return 0

    cmd = [
        sys.executable,
        "scripts/pipelines/run_pairwise_eval_matrix.py",
        "--activations_root",
        args.activations_root,
        "--probes_root",
        args.probes_root,
        "--results_root",
        args.results_root,
        "--model",
        args.model,
        "--only_segments",
        "full",
        "--only_targets",
        args.target,
        "--only_sources",
        ",".join(missing),
        "--poolings",
        args.poolings,
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--progress_every",
        str(args.progress_every),
        "--skip_training",
    ]
    if args.no_tqdm:
        cmd.append("--no_tqdm")
    if args.resume:
        cmd.append("--resume")
    if args.skip_diagonals:
        cmd.append("--skip_diagonals")
    if args.dry_run:
        cmd.append("--dry_run")

    print("[run] invoking pairwise eval pipeline:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        return proc.returncode

    if args.rebuild_matrices:
        rebuild_cmd = [
            sys.executable,
            "scripts/pipelines/run_pairwise_eval_matrix.py",
            "--activations_root",
            args.activations_root,
            "--probes_root",
            args.probes_root,
            "--results_root",
            args.results_root,
            "--model",
            args.model,
            "--poolings",
            args.poolings,
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--progress_every",
            str(args.progress_every),
            "--skip_training",
            "--resume",
        ]
        if args.no_tqdm:
            rebuild_cmd.append("--no_tqdm")
        if args.skip_diagonals:
            rebuild_cmd.append("--skip_diagonals")
        print("[rebuild] invoking pipeline to rebuild matrices:")
        print(" ".join(rebuild_cmd))
        proc = subprocess.run(rebuild_cmd, text=True)
        return proc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
