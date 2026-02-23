#!/usr/bin/env python3
"""
Sweep GLP denoise settings by calling run_glp_target_denoise_metrics.py
for multiple (noise_scale, start_timestep_mode) combinations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_list(value: str) -> List[str]:
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v) for v in parse_list(value)]


def build_modes(start_timestep_modes: str | None, start_timestep_fracs: str | None) -> List[str]:
    if start_timestep_fracs:
        fracs = parse_float_list(start_timestep_fracs)
        return [f"frac:{frac}" for frac in fracs]
    if start_timestep_modes:
        return parse_list(start_timestep_modes)
    return ["half"]


def tag_value(val: str) -> str:
    return val.replace(":", "").replace(".", "p").replace("/", "_")


def iter_combos(noise_scales: Iterable[float], start_modes: Iterable[str]) -> Iterable[tuple[float, str]]:
    for noise_scale in noise_scales:
        for mode in start_modes:
            yield noise_scale, mode


def resolve_run_dir(output_root: Path, target_dir: Path, layer: int, run_id: str) -> Path:
    split = target_dir.name
    dataset = target_dir.parent.name
    model_dir = target_dir.parent.parent.name
    return (
        output_root
        / "glp_target_denoise"
        / model_dir
        / dataset
        / split
        / f"layer_{layer}"
        / run_id
    )


def load_results_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def mean_start_timestep(start_values: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in start_values.values() if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def extract_summary_rows(run_id: str, target_dir: Path, results: Dict) -> List[Dict]:
    rows: List[Dict] = []
    by_timestep = results.get("results_by_timestep", {})
    for key, payload in by_timestep.items():
        start_vals = payload.get("start_timestep_values", {})
        rows.append(
            {
                "run_id": run_id,
                "target_dir": str(target_dir),
                "timestep_key": key,
                "num_timesteps": payload.get("num_timesteps"),
                "start_timestep_mode": payload.get("start_timestep_mode"),
                "start_timestep_value_mean": mean_start_timestep(start_vals),
                "noise_scale": payload.get("noise_scale"),
                "num_seeds": payload.get("num_seeds"),
                "num_samples": payload.get("num_samples"),
                "auc_x": payload.get("auc_x"),
                "auc_xprime_mean": payload.get("auc_xprime_mean"),
                "auc_xprime_seed_mean": payload.get("auc_xprime_seed_mean"),
                "separation_l2_x": payload.get("separation_l2_x"),
                "separation_l2_xprime": payload.get("separation_l2_xprime"),
                "separation_norm_x": payload.get("separation_norm_x"),
                "separation_norm_xprime": payload.get("separation_norm_xprime"),
                "cov_trace_x_pos": payload.get("cov_trace_x_pos"),
                "cov_trace_x_neg": payload.get("cov_trace_x_neg"),
                "cov_trace_xprime_pos": payload.get("cov_trace_xprime_pos"),
                "cov_trace_xprime_neg": payload.get("cov_trace_xprime_neg"),
                "delta_norm_mean": payload.get("delta_norm_all", {}).get("mean"),
                "delta_parallel_mean": payload.get("delta_parallel_all", {}).get("mean"),
                "drift_norm_mean": payload.get("drift_norm_all", {}).get("mean"),
                "var_trace_mean": payload.get("var_trace_all", {}).get("mean"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep GLP target denoise settings")
    parser.add_argument("--target_dir", type=str, action="append", required=True, help="Target activations split dir. Repeatable.")
    parser.add_argument("--probe_dir", type=str, required=True, help="Directory containing probe_layer_*.pt")
    parser.add_argument("--probe_layer", type=int, default=7)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--glp_model", type=str, default="generative-latent-prior/glp-llama1b-d12")
    parser.add_argument("--glp_checkpoint", type=str, default="final")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--num_timesteps", type=str, default="100")
    parser.add_argument("--start_timestep_modes", type=str, default=None, help="Comma list: none|half|idx:<n>|frac:<0..1>")
    parser.add_argument("--start_timestep_fracs", type=str, default=None, help="Comma list of fracs in [0,1], converted to frac:<f>")
    parser.add_argument("--noise_scales", type=str, default="0.1,0.3,1.0")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_root", type=str, default="results/GLP_Experiments")
    parser.add_argument("--run_id_prefix", type=str, default="sweep")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).parent / "run_glp_target_denoise_metrics.py"
    noise_scales = parse_float_list(args.noise_scales)
    start_modes = build_modes(args.start_timestep_modes, args.start_timestep_fracs)

    if not noise_scales:
        raise SystemExit("No noise_scales provided.")
    if not start_modes:
        raise SystemExit("No start_timestep_modes provided.")

    output_root = Path(args.output_root).resolve()
    summary_rows_by_target: Dict[str, List[Dict]] = {t: [] for t in args.target_dir}

    for noise_scale, start_mode in iter_combos(noise_scales, start_modes):
        run_id = f"{args.run_id_prefix}_ns{tag_value(str(noise_scale))}_st{tag_value(start_mode)}"
        cmd = [
            sys.executable,
            str(script_path),
            "--probe_dir",
            args.probe_dir,
            "--probe_layer",
            str(args.probe_layer),
            "--model",
            args.model,
            "--glp_model",
            args.glp_model,
            "--glp_checkpoint",
            args.glp_checkpoint,
            "--layer",
            str(args.layer),
            "--num_timesteps",
            args.num_timesteps,
            "--start_timestep_mode",
            start_mode,
            "--noise_scale",
            str(noise_scale),
            "--num_seeds",
            str(args.num_seeds),
            "--max_samples",
            str(args.max_samples),
            "--sample_seed",
            str(args.sample_seed),
            "--batch_size",
            str(args.batch_size),
            "--output_root",
            args.output_root,
            "--run_id",
            run_id,
            "--device",
            args.device,
        ]
        if args.resume:
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")
        if args.verbose:
            cmd.append("--verbose")
        for target in args.target_dir:
            cmd.extend(["--target_dir", target])

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        for target in args.target_dir:
            target_dir = Path(target).resolve()
            run_dir = resolve_run_dir(output_root, target_dir, args.layer, run_id)
            results_path = run_dir / "results" / "results.json"
            if not results_path.exists():
                raise FileNotFoundError(f"Missing results.json at {results_path}")
            results = load_results_json(results_path)
            summary_rows_by_target[target].extend(extract_summary_rows(run_id, target_dir, results))

    # Write summary CSV per target.
    for target, rows in summary_rows_by_target.items():
        if not rows:
            continue
        target_dir = Path(target).resolve()
        base_dir = resolve_run_dir(output_root, target_dir, args.layer, run_id="dummy").parent
        summary_path = base_dir / f"sweep_summary_{args.run_id_prefix}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote sweep summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
