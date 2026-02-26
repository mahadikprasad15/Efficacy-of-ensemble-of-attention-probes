#!/usr/bin/env python3
"""
Residualize combined probes against single-domain probe span.

For each pooling/layer:
    w_resid = w_comb - Proj_span(w_a, w_b)(w_comb)

This script edits probe weights only (no activation residualization).
It writes residualized probes to a new output tree so downstream evaluation
can be run with existing OOD evaluation scripts.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


POOLINGS_DEFAULT = ["mean", "max", "last", "attn"]
KEY_CANDIDATES = ["classifier.weight", "net.0.weight", "pooling.weight", "pooling.query"]


@dataclass
class VectorSpec:
    key: str
    mode: str  # "row" or "flat"
    shape: Tuple[int, ...]
    dtype: torch.dtype


def parse_poolings(poolings_str: str) -> List[str]:
    vals = [p.strip() for p in poolings_str.split(",") if p.strip()]
    return vals or POOLINGS_DEFAULT


def parse_layers(layers_arg: Optional[str]) -> Optional[List[int]]:
    if not layers_arg:
        return None
    out: List[int] = []
    chunks = [x.strip() for x in layers_arg.split(",") if x.strip()]
    for chunk in chunks:
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def list_probe_layers(pooling_dir: str) -> List[int]:
    paths = glob.glob(os.path.join(pooling_dir, "probe_layer_*.pt"))
    layers: List[int] = []
    for p in paths:
        m = re.search(r"probe_layer_(\d+)\.pt$", p)
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def load_state(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def find_vector_spec(
    s_comb: Dict[str, torch.Tensor],
    s_a: Dict[str, torch.Tensor],
    s_b: Dict[str, torch.Tensor],
) -> Optional[VectorSpec]:
    for key in KEY_CANDIDATES:
        if key not in s_comb or key not in s_a or key not in s_b:
            continue
        tc, ta, tb = s_comb[key], s_a[key], s_b[key]
        if tc.shape != ta.shape or tc.shape != tb.shape:
            continue
        if tc.ndim == 2 and tc.shape[0] == 1:
            return VectorSpec(key=key, mode="row", shape=tuple(tc.shape), dtype=tc.dtype)
        if tc.ndim == 1:
            return VectorSpec(key=key, mode="flat", shape=tuple(tc.shape), dtype=tc.dtype)
    return None


def extract_vec(state: Dict[str, torch.Tensor], spec: VectorSpec) -> np.ndarray:
    t = state[spec.key].detach().cpu().float().numpy()
    if spec.mode == "row":
        return t[0].astype(np.float64)
    return t.astype(np.float64)


def write_vec(state: Dict[str, torch.Tensor], spec: VectorSpec, vec: np.ndarray) -> None:
    t = torch.from_numpy(vec.astype(np.float32))
    if spec.mode == "row":
        out = state[spec.key].detach().clone()
        out[0] = t.to(out.dtype)
        state[spec.key] = out
    else:
        state[spec.key] = t.to(spec.dtype)


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def residualize(
    w_comb: np.ndarray,
    basis_vecs: List[np.ndarray],
    eps: float = 1e-10,
) -> np.ndarray:
    nonzero_basis = [w for w in basis_vecs if float(np.linalg.norm(w)) > eps]
    if not nonzero_basis:
        return w_comb.copy()
    bmat = np.column_stack(nonzero_basis)  # (D, k)
    q, _ = np.linalg.qr(bmat)
    rank = int(np.linalg.matrix_rank(bmat))
    q = q[:, :rank]
    proj = q @ (q.T @ w_comb)
    return w_comb - proj


def rescale(vec: np.ndarray, ref_norm: float, mode: str, eps: float = 1e-10) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if mode == "none":
        return vec
    if n < eps:
        return vec
    if mode == "unit":
        return vec / n
    if mode == "preserve_norm":
        return vec * (ref_norm / n)
    raise ValueError(f"Unknown rescale mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Residualize combined probes by single-domain span.")
    parser.add_argument("--combined_probes_root", type=str, required=True)
    parser.add_argument("--single_probes_a_root", type=str, required=True)
    parser.add_argument("--single_probes_b_root", type=str, required=True)
    parser.add_argument("--output_probes_root", type=str, required=True)
    parser.add_argument("--poolings", type=str, default="mean,max,last,attn")
    parser.add_argument("--layers", type=str, default=None, help="Examples: 0-31 or 5,7,12")
    parser.add_argument("--rescale_mode", type=str, default="preserve_norm", choices=["preserve_norm", "unit", "none"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    poolings = parse_poolings(args.poolings)
    layers_filter = parse_layers(args.layers)
    os.makedirs(args.output_probes_root, exist_ok=True)

    status_path = os.path.join(args.output_probes_root, "residualization_status.json")
    manifest_path = os.path.join(args.output_probes_root, "residualization_manifest.jsonl")

    status = {
        "status": "running",
        "combined_probes_root": args.combined_probes_root,
        "single_probes_a_root": args.single_probes_a_root,
        "single_probes_b_root": args.single_probes_b_root,
        "output_probes_root": args.output_probes_root,
        "poolings": poolings,
        "rescale_mode": args.rescale_mode,
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    total = 0
    done = 0
    skipped = 0
    missing = 0

    for pooling in poolings:
        comb_pool = os.path.join(args.combined_probes_root, pooling)
        a_pool = os.path.join(args.single_probes_a_root, pooling)
        b_pool = os.path.join(args.single_probes_b_root, pooling)
        out_pool = os.path.join(args.output_probes_root, pooling)
        os.makedirs(out_pool, exist_ok=True)

        if not os.path.isdir(comb_pool):
            if args.verbose:
                print(f"[skip] missing combined pooling dir: {comb_pool}")
            continue

        layers = list_probe_layers(comb_pool)
        if layers_filter is not None:
            layers = [l for l in layers if l in layers_filter]

        for layer in layers:
            total += 1
            out_probe = os.path.join(out_pool, f"probe_layer_{layer}.pt")
            if args.resume and os.path.exists(out_probe):
                skipped += 1
                continue

            p_comb = os.path.join(comb_pool, f"probe_layer_{layer}.pt")
            p_a = os.path.join(a_pool, f"probe_layer_{layer}.pt")
            p_b = os.path.join(b_pool, f"probe_layer_{layer}.pt")

            if not (os.path.exists(p_comb) and os.path.exists(p_a) and os.path.exists(p_b)):
                missing += 1
                if args.verbose:
                    print(f"[missing] pooling={pooling} layer={layer}")
                continue

            try:
                s_comb = load_state(p_comb)
                s_a = load_state(p_a)
                s_b = load_state(p_b)

                spec = find_vector_spec(s_comb, s_a, s_b)
                if spec is None:
                    missing += 1
                    if args.verbose:
                        print(f"[unsupported] pooling={pooling} layer={layer}: no compatible vector key")
                    continue

                w_comb = extract_vec(s_comb, spec)
                w_a = extract_vec(s_a, spec)
                w_b = extract_vec(s_b, spec)

                w_resid = residualize(w_comb, [w_a, w_b])
                w_resid = rescale(w_resid, ref_norm=float(np.linalg.norm(w_comb)), mode=args.rescale_mode)

                s_out = copy.deepcopy(s_comb)
                write_vec(s_out, spec, w_resid)
                torch.save(s_out, out_probe)

                norm_src = os.path.join(comb_pool, f"norm_layer_{layer}.npz")
                norm_dst = os.path.join(out_pool, f"norm_layer_{layer}.npz")
                if os.path.exists(norm_src) and (not args.resume or not os.path.exists(norm_dst)):
                    shutil.copy2(norm_src, norm_dst)

                rec = {
                    "pooling": pooling,
                    "layer": layer,
                    "key": spec.key,
                    "mode": spec.mode,
                    "norm_comb": float(np.linalg.norm(w_comb)),
                    "norm_resid": float(np.linalg.norm(w_resid)),
                    "cos_resid_a": cos_sim(w_resid, w_a),
                    "cos_resid_b": cos_sim(w_resid, w_b),
                    "path_out": out_probe,
                }
                with open(manifest_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done += 1

            except Exception as exc:  # noqa: BLE001
                missing += 1
                if args.verbose:
                    print(f"[error] pooling={pooling} layer={layer}: {exc}")

    summary = {
        **status,
        "status": "completed",
        "total_candidates": total,
        "written": done,
        "skipped_existing": skipped,
        "missing_or_unsupported": missing,
        "manifest_path": manifest_path,
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
