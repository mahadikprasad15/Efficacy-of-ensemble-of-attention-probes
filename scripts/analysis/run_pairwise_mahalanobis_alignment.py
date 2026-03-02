#!/usr/bin/env python3
"""
Pairwise Mahalanobis alignment for 6x7 completion/full matrices.

Inputs:
  - matrix_completion.csv, matrix_full.csv from pairwise eval pipeline
Outputs per segment:
  - score_matrix (AUC)
  - probe_angle_matrix (Mahalanobis cosine between scoring probe and canonical target probe)
  - activation_mean_angle_matrix (Mahalanobis cosine between train/test activation means)

Shared covariance Σ is computed per segment using pooled train activations across all datasets.
Attn pooling uses the attention weights from the source dataset's attn probe at that layer.
Missing pooling defaults to mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from tqdm import tqdm

# Add src to path for LayerProbe
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402


POOLING_ORDER = ["attn", "last", "max", "mean"]
ALL_POOLINGS = ["mean", "max", "last", "attn"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_status(status_path: Path, state: str, message: str) -> None:
    current = read_json(status_path, default={})
    current["state"] = state
    current["message"] = message
    current["updated_at"] = utc_now()
    write_json(status_path, current)


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


def dataset_base(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return dataset_name[: -len("-completion")]
    if dataset_name.endswith("-full"):
        return dataset_name[: -len("-full")]
    return dataset_name


def segment_of(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return "completion"
    if dataset_name.endswith("-full"):
        return "full"
    raise ValueError(f"Dataset has no segment suffix: {dataset_name}")


def short_name(dataset_name: str) -> str:
    seg = segment_of(dataset_name)
    base = dataset_base(dataset_name).replace("Deception-", "")
    m = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "Roleplaying": "RP",
    }
    return f"{m.get(base, base)}-{'c' if seg == 'completion' else 'f'}"


def sort_datasets(datasets: Sequence[str]) -> List[str]:
    order = [
        "Deception-ConvincingGame",
        "Deception-HarmPressureChoice",
        "Deception-InstructedDeception",
        "Deception-Mask",
        "Deception-AILiar",
        "Deception-InsiderTrading",
        "Deception-Roleplaying",
    ]
    order_map = {name: i for i, name in enumerate(order)}

    def key(ds: str) -> Tuple[int, str]:
        base = dataset_base(ds)
        return (order_map.get(base, 999), ds)

    return sorted(datasets, key=key)


class CovAccumulator:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_xx = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N,{self.dim}), got {x.shape}")
        if x.shape[0] == 0:
            return
        self.n += int(x.shape[0])
        self.sum += x.sum(axis=0, dtype=np.float64)
        self.sum_xx += x.T @ x

    def finalize(self, eps: float) -> np.ndarray:
        if self.n < 2:
            raise ValueError("Need at least 2 samples for covariance")
        mean_outer = np.outer(self.sum, self.sum) / float(self.n)
        cov = (self.sum_xx - mean_outer) / float(self.n - 1)
        cov += np.eye(self.dim, dtype=np.float64) * float(eps)
        return cov


def mahalanobis_cosine(a: np.ndarray, b: np.ndarray, sigma: np.ndarray, eps: float = 1e-12) -> float:
    num = float(a.T @ sigma @ b)
    da = float(a.T @ sigma @ a)
    db = float(b.T @ sigma @ b)
    den = max(np.sqrt(max(da, 0.0)) * np.sqrt(max(db, 0.0)), eps)
    return float(num / den)


def iter_labeled_activations(split_dir: Path) -> Iterable[Tuple[torch.Tensor, int]]:
    manifest = split_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    label_map: Dict[str, int] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            label_map[entry["id"]] = int(entry.get("label", -1))
    shards = sorted(split_dir.glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No shards in {split_dir}")
    yielded = 0
    for shard in shards:
        shard_data = load_file(str(shard))
        for sid, tensor in shard_data.items():
            y = label_map.get(sid, -1)
            if y == -1:
                continue
            yielded += 1
            yield tensor, y
    if yielded == 0:
        raise RuntimeError(f"No labeled activations in {split_dir}")


class AttentionLinearProbeCompat(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.classifier(pooled)


@dataclass(frozen=True)
class CellMeta:
    row_dataset: str
    col_dataset: str
    pooling: str
    layer: int
    auc: Optional[float]


def normalize_pooling(pooling: Optional[str]) -> str:
    if pooling is None or str(pooling).strip() == "" or str(pooling).lower() == "none":
        return "mean"
    return str(pooling).strip().lower()


def resolve_probe_path(probes_model_root: Path, dataset_name: str, pooling: str, layer: int) -> Path:
    base = dataset_base(dataset_name)
    return probes_model_root / f"{base}_slices" / dataset_name / pooling / f"probe_layer_{layer}.pt"


def probe_dataset_root(probes_model_root: Path, dataset_name: str) -> Path:
    base = dataset_base(dataset_name)
    return probes_model_root / f"{base}_slices" / dataset_name


def dataset_has_any_probes(probes_model_root: Path, dataset_name: str) -> bool:
    root = probe_dataset_root(probes_model_root, dataset_name)
    if not root.exists():
        return False
    return any(root.glob("*/probe_layer_*.pt"))


def dataset_has_attn_probes(probes_model_root: Path, dataset_name: str) -> bool:
    root = probe_dataset_root(probes_model_root, dataset_name) / "attn"
    if not root.exists():
        return False
    return any(root.glob("probe_layer_*.pt"))


def load_classifier_weight(path: Path) -> np.ndarray:
    state = torch.load(str(path), map_location="cpu")
    w = state["classifier.weight"].detach().cpu().numpy().reshape(-1)
    return w.astype(np.float64)


class Pooler:
    def __init__(
        self,
        pooling: str,
        attn_model: Optional[AttentionLinearProbeCompat] = None,
        layer_probe_model: Optional[LayerProbe] = None,
    ):
        self.pooling = pooling
        self.attn_model = attn_model
        self.layer_probe_model = layer_probe_model

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,D) or (D,)
        if x.dim() == 1:
            return x
        if self.pooling == "mean":
            return x.mean(dim=0)
        if self.pooling == "max":
            return x.max(dim=0).values
        if self.pooling == "last":
            return x[-1]
        if self.pooling == "attn":
            if self.attn_model is not None:
                model_device = next(self.attn_model.parameters()).device
                x_model = x.to(model_device)
                with torch.no_grad():
                    weights = torch.softmax(self.attn_model.attn(x_model), dim=0)  # (T,1)
                    return (x_model * weights).sum(dim=0)
            if self.layer_probe_model is not None:
                model_device = next(self.layer_probe_model.parameters()).device
                x_model = x.to(model_device)
                with torch.no_grad():
                    return self.layer_probe_model.pooling(x_model.unsqueeze(0)).squeeze(0)
            if self.attn_model is None and self.layer_probe_model is None:
                raise RuntimeError("Attn pooling requested but attn_model is None.")
        raise ValueError(f"Unsupported pooling: {self.pooling}")


def load_attn_pooler(probe_path: Path, input_dim: int, device: torch.device) -> Pooler:
    state = torch.load(str(probe_path), map_location=device)
    if any(k.startswith("attn.") for k in state.keys()):
        model = AttentionLinearProbeCompat(input_dim=input_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        return Pooler("attn", attn_model=model)
    model = LayerProbe(input_dim=input_dim, pooling_type="attn").to(device)
    model.load_state_dict(state)
    model.eval()
    return Pooler("attn", layer_probe_model=model)


def ensure_split(dataset_dir: Path, split: str, fallback: Optional[str]) -> Path:
    candidate = dataset_dir / split
    if candidate.exists():
        return candidate
    if fallback:
        fb = dataset_dir / fallback
        if fb.exists():
            return fb
    raise FileNotFoundError(f"Missing split {split} for {dataset_dir}")


def read_matrix_csv(path: Path) -> Dict[Tuple[str, str], CellMeta]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cells: Dict[Tuple[str, str], CellMeta] = {}
    for row in rows:
        r = row.get("row_dataset")
        c = row.get("col_dataset")
        if not r or not c:
            continue
        pooling = normalize_pooling(row.get("pooling"))
        layer = row.get("layer")
        try:
            layer_i = int(layer) if layer not in [None, ""] else -1
        except Exception:
            layer_i = -1
        auc = None
        try:
            if row.get("auc") not in [None, ""]:
                auc = float(row.get("auc"))
        except Exception:
            auc = None
        cells[(r, c)] = CellMeta(row_dataset=r, col_dataset=c, pooling=pooling, layer=layer_i, auc=auc)
    return cells


def matrix_row_col_lists(cells: Dict[Tuple[str, str], CellMeta]) -> Tuple[List[str], List[str]]:
    rows = sort_datasets({r for r, _ in cells.keys()})
    cols = sort_datasets({c for _, c in cells.keys()})
    return rows, cols


def save_matrix_csv(path: Path, mat: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row"] + list(col_labels))
        for i, r in enumerate(row_labels):
            writer.writerow([r] + [mat[i, j] for j in range(len(col_labels))])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise Mahalanobis alignment for 6x7 matrices.")
    p.add_argument("--matrix_completion_csv", type=str, required=True)
    p.add_argument("--matrix_full_csv", type=str, required=True)
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
    p.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Optional direct output root for runs. "
            "If set, outputs go to <output_root>/<run_id>/... instead of artifact_root/runs/..."
        ),
    )
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cov_eps", type=float, default=1e-5)
    p.add_argument("--progress_every", type=int, default=200)
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--allow_missing_attn", action="store_true")
    p.add_argument(
        "--covariance_scope",
        type=str,
        default="all",
        choices=["all", "required"],
        help="all: use all layers+poolings for all datasets; required: only combos used by cells.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_tqdm = not args.no_tqdm

    # Resolve roots (supports passing either .../<model_dir> or its parent)
    acts_base, acts_model_root = split_root_and_model(Path(args.activations_root), model_dir)
    probes_base, probes_model_root = split_root_and_model(Path(args.probes_root), model_dir)

    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = (
            Path(args.artifact_root)
            / "runs"
            / "pairwise_mahalanobis_alignment"
            / model_dir
            / "matrix6x7"
            / "v1"
            / run_id
        )
    meta_dir = run_root / "meta"
    chk_dir = run_root / "checkpoints"
    out_dir = run_root / "results"
    for d in [meta_dir, chk_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "run_manifest.json"
    status_path = meta_dir / "status.json"
    progress_path = chk_dir / "progress.json"
    progress = read_json(
        progress_path,
        default={"completed_steps": [], "cov_completed": {}, "mean_completed": []},
    )

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "activations_base": str(acts_base),
                "activations_model_root": str(acts_model_root),
                "probes_base": str(probes_base),
                "probes_model_root": str(probes_model_root),
                "run_root": str(run_root),
            },
            "matrix_completion_csv": args.matrix_completion_csv,
            "matrix_full_csv": args.matrix_full_csv,
            "covariance_scope": args.covariance_scope,
        },
    )
    update_status(status_path, "running", "starting")
    print(f"[start] run_id={run_id}")
    print(f"[start] model={args.model} device={device}")

    def mark(step: str) -> None:
        done = set(progress.get("completed_steps", []))
        done.add(step)
        progress["completed_steps"] = sorted(done)
        progress["updated_at"] = utc_now()
        write_json(progress_path, progress)

    # Load matrices
    update_status(status_path, "running", "loading matrices")
    print("[stage 1/6] loading matrices")
    comp_cells = read_matrix_csv(Path(args.matrix_completion_csv))
    full_cells = read_matrix_csv(Path(args.matrix_full_csv))
    comp_rows, comp_cols = matrix_row_col_lists(comp_cells)
    full_rows, full_cols = matrix_row_col_lists(full_cells)
    all_datasets = sort_datasets(set(comp_rows) | set(comp_cols) | set(full_rows) | set(full_cols))
    no_probe_datasets = {ds for ds in all_datasets if not dataset_has_any_probes(probes_model_root, ds)}
    if no_probe_datasets:
        print(f"[info] datasets with no probes (probe-angle NA where applicable): {sorted(no_probe_datasets)}")
    mark("load_matrices")
    print("[done] matrices loaded")

    def build_score_matrix(cells: Dict[Tuple[str, str], CellMeta], rows: List[str], cols: List[str]) -> np.ndarray:
        mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                cell = cells.get((r, c))
                if cell and cell.auc is not None:
                    mat[i, j] = float(cell.auc)
        return mat

    comp_score = build_score_matrix(comp_cells, comp_rows, comp_cols)
    full_score = build_score_matrix(full_cells, full_rows, full_cols)
    save_matrix_csv(out_dir / "score_matrix_completion.csv", comp_score, [short_name(r) for r in comp_rows], [short_name(c) for c in comp_cols])
    save_matrix_csv(out_dir / "score_matrix_full.csv", full_score, [short_name(r) for r in full_rows], [short_name(c) for c in full_cols])

    # Canonical probes from diagonal
    update_status(status_path, "running", "resolving canonical probes")
    print("[stage 2/6] resolving canonical probes (diagonal)")

    def canonical_probes(cells: Dict[Tuple[str, str], CellMeta], rows: List[str], cols: List[str]) -> Dict[str, Dict[str, Any]]:
        canon: Dict[str, Dict[str, Any]] = {}
        for ds in rows:
            if ds not in cols:
                continue
            cell = cells.get((ds, ds))
            if not cell:
                continue
            if cell.layer < 0:
                continue
            if ds in no_probe_datasets:
                continue
            pooling = normalize_pooling(cell.pooling)
            probe_path = resolve_probe_path(probes_model_root, ds, pooling, cell.layer)
            if not probe_path.exists():
                continue
            canon[ds] = {"pooling": pooling, "layer": cell.layer, "path": probe_path}
        return canon

    comp_canon = canonical_probes(comp_cells, comp_rows, comp_cols)
    full_canon = canonical_probes(full_cells, full_rows, full_cols)
    print(f"[info] canonical probes (completion)={len(comp_canon)} (full)={len(full_canon)}")
    mark("canonical_probes")

    # Shared covariance per segment
    update_status(status_path, "running", "computing covariances")
    print("[stage 3/6] computing shared covariances per segment")

    def segment_covariance(
        segment: str,
        rows: List[str],
        cols: List[str],
        cells: Dict[Tuple[str, str], CellMeta],
    ) -> np.ndarray:
        cov_path = chk_dir / f"cov_{segment}.npz"
        if args.resume and cov_path.exists():
            ckpt = np.load(cov_path, allow_pickle=True)
            sigma = ckpt["cov"].astype(np.float64)
            print(f"[resume] using cached covariance for {segment}")
            return sigma

        datasets = sort_datasets(set(rows) | set(cols))
        completed: List[str] = []
        acc: Optional[CovAccumulator] = None
        cov_sum = None
        cov_sum_xx = None
        cov_n = 0
        cov_ckpt_path = chk_dir / f"cov_{segment}_checkpoint.npz"

        if args.resume and cov_ckpt_path.exists():
            ckpt = np.load(cov_ckpt_path, allow_pickle=True)
            completed = [str(x) for x in ckpt["completed"].tolist()]
            cov_n = int(ckpt["cov_n"])
            cov_sum = ckpt["cov_sum"].astype(np.float64)
            cov_sum_xx = ckpt["cov_sum_xx"].astype(np.float64)

        # Determine input dim from first dataset
        d_model = None
        for ds in datasets:
            split_dir = ensure_split(acts_model_root / ds, "train", None)
            tensor, _ = next(iter_labeled_activations(split_dir))
            d_model = int(tensor.shape[-1])
            break
        if d_model is None:
            raise RuntimeError(f"No activations found for segment {segment}")

        acc = CovAccumulator(d_model)
        if cov_sum is not None and cov_sum_xx is not None:
            acc.sum = cov_sum
            acc.sum_xx = cov_sum_xx
            acc.n = cov_n

        def needed_poolings_for_layer(ds_name: str, layer: int) -> List[str]:
            if args.covariance_scope == "all":
                if dataset_has_attn_probes(probes_model_root, ds_name):
                    return ALL_POOLINGS
                return ["mean", "max", "last"]
            # required: poolings used when this dataset is the source row.
            # Attn pooling is source-probe dependent, so target-only datasets (e.g. IT)
            # should not force attn requirements.
            pset = set()
            for (r, c), cell in cells.items():
                if r == ds_name and cell.layer == layer:
                    pset.add(normalize_pooling(cell.pooling))
            if "attn" in pset and not dataset_has_attn_probes(probes_model_root, ds_name):
                pset.remove("attn")
            return sorted(pset) if pset else []

        for ds in datasets:
            if ds in completed:
                print(f"[cov] skip {segment}:{ds} (completed)")
                continue
            split_dir = ensure_split(acts_model_root / ds, "train", None)
            sample, _ = next(iter_labeled_activations(split_dir))
            layers = int(sample.shape[0])
            input_dim = int(sample.shape[-1])

            # Cache attn poolers per layer for this dataset
            attn_poolers: Dict[int, Pooler] = {}

            def get_attn_pooler(layer: int) -> Optional[Pooler]:
                if layer in attn_poolers:
                    return attn_poolers[layer]
                probe_path = resolve_probe_path(probes_model_root, ds, "attn", layer)
                if not probe_path.exists():
                    print(f"[warn] missing attn probe for {ds} layer {layer}; using mean fallback")
                    return None
                attn_poolers[layer] = load_attn_pooler(probe_path, input_dim=input_dim, device=device)
                return attn_poolers[layer]

            t0 = time.time()
            sample_count = 0
            iterator = iter_labeled_activations(split_dir)
            if use_tqdm:
                iterator = tqdm(iterator, desc=f"cov:{segment}:{short_name(ds)}", unit="sample")
            for tensor, _ in iterator:
                for layer in range(layers):
                    x_layer = tensor[layer].float()
                    poolings = needed_poolings_for_layer(ds, layer)
                    for pooling in poolings:
                        if pooling == "attn":
                            attn_pooler = get_attn_pooler(layer)
                            if attn_pooler is None:
                                pooled = x_layer.mean(dim=0)
                            else:
                                pooled = attn_pooler.pool(x_layer)
                        else:
                            pooled = Pooler(pooling).pool(x_layer)
                        acc.update(pooled.detach().cpu().numpy().reshape(1, -1).astype(np.float64))
                sample_count += 1
                if args.progress_every and sample_count % args.progress_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    print(f"[cov] {segment}:{ds} processed {sample_count} samples ({sample_count/elapsed:.2f} samples/s)")

            completed.append(ds)
            np.savez(
                cov_ckpt_path,
                completed=np.array(completed, dtype=object),
                cov_n=np.array(acc.n, dtype=np.int64),
                cov_sum=acc.sum,
                cov_sum_xx=acc.sum_xx,
            )

        sigma = acc.finalize(eps=args.cov_eps)
        np.savez(cov_path, cov=sigma)
        return sigma

    sigma_comp = segment_covariance("completion", comp_rows, comp_cols, comp_cells)
    sigma_full = segment_covariance("full", full_rows, full_cols, full_cells)
    mark("covariances")

    # Probe angle matrices
    update_status(status_path, "running", "computing probe angle matrices")
    print("[stage 4/6] computing probe angle matrices")

    def probe_angle_matrix(
        rows: List[str],
        cols: List[str],
        cells: Dict[Tuple[str, str], CellMeta],
        canon: Dict[str, Dict[str, Any]],
        sigma: np.ndarray,
    ) -> np.ndarray:
        mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if r == c:
                    mat[i, j] = 1.0
                    continue
                if c not in canon:
                    continue
                if r in no_probe_datasets:
                    continue
                cell = cells.get((r, c))
                if cell is None or cell.layer < 0:
                    continue
                pooling = normalize_pooling(cell.pooling)
                probe_path = resolve_probe_path(probes_model_root, r, pooling, cell.layer)
                if not probe_path.exists():
                    continue
                try:
                    w_a = load_classifier_weight(probe_path)
                    w_b = load_classifier_weight(Path(canon[c]["path"]))
                    mat[i, j] = mahalanobis_cosine(w_a, w_b, sigma)
                except Exception:
                    mat[i, j] = np.nan
        return mat

    def probe_angle_matrix_v2(
        rows: List[str],
        cols: List[str],
        cells: Dict[Tuple[str, str], CellMeta],
        sigma: np.ndarray,
    ) -> np.ndarray:
        mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if r == c:
                    mat[i, j] = 1.0
                    continue
                cell_ab = cells.get((r, c))
                cell_ba = cells.get((c, r))
                if cell_ab is None or cell_ba is None:
                    continue
                if r in no_probe_datasets or c in no_probe_datasets:
                    continue
                if cell_ab.layer < 0 or cell_ba.layer < 0:
                    continue
                pooling_ab = normalize_pooling(cell_ab.pooling)
                pooling_ba = normalize_pooling(cell_ba.pooling)
                probe_a = resolve_probe_path(probes_model_root, r, pooling_ab, cell_ab.layer)
                probe_b = resolve_probe_path(probes_model_root, c, pooling_ba, cell_ba.layer)
                if not probe_a.exists() or not probe_b.exists():
                    continue
                try:
                    w_a = load_classifier_weight(probe_a)
                    w_b = load_classifier_weight(probe_b)
                    mat[i, j] = mahalanobis_cosine(w_a, w_b, sigma)
                except Exception:
                    mat[i, j] = np.nan
        return mat

    comp_probe_angle = probe_angle_matrix(comp_rows, comp_cols, comp_cells, comp_canon, sigma_comp)
    full_probe_angle = probe_angle_matrix(full_rows, full_cols, full_cells, full_canon, sigma_full)
    comp_probe_angle_v2 = probe_angle_matrix_v2(comp_rows, comp_cols, comp_cells, sigma_comp)
    full_probe_angle_v2 = probe_angle_matrix_v2(full_rows, full_cols, full_cells, sigma_full)
    save_matrix_csv(
        out_dir / "probe_angle_v1_completion.csv",
        comp_probe_angle,
        [short_name(r) for r in comp_rows],
        [short_name(c) for c in comp_cols],
    )
    save_matrix_csv(
        out_dir / "probe_angle_v1_full.csv",
        full_probe_angle,
        [short_name(r) for r in full_rows],
        [short_name(c) for c in full_cols],
    )
    save_matrix_csv(
        out_dir / "probe_angle_v2_completion.csv",
        comp_probe_angle_v2,
        [short_name(r) for r in comp_rows],
        [short_name(c) for c in comp_cols],
    )
    save_matrix_csv(
        out_dir / "probe_angle_v2_full.csv",
        full_probe_angle_v2,
        [short_name(r) for r in full_rows],
        [short_name(c) for c in full_cols],
    )
    mark("probe_angles")

    # Activation mean angle matrices
    update_status(status_path, "running", "computing activation mean angle matrices")
    print("[stage 5/6] computing activation mean angle matrices")

    mean_cache_dir = chk_dir / "means"
    mean_cache_dir.mkdir(parents=True, exist_ok=True)

    def mean_cache_path(segment: str, dataset: str, split: str, source_for_pooler: str, layer: int, pooling: str) -> Path:
        key = f"{segment}__{dataset}__{split}__src-{source_for_pooler}__L{layer}__{pooling}.npy"
        return mean_cache_dir / key

    def compute_mean(
        segment: str,
        dataset: str,
        split: str,
        source_for_pooler: str,
        layer: int,
        pooling: str,
    ) -> np.ndarray:
        cache_path = mean_cache_path(segment, dataset, split, source_for_pooler, layer, pooling)
        if args.resume and cache_path.exists():
            return np.load(cache_path)

        split_dir = ensure_split(acts_model_root / dataset, split, "validation" if split == "val" else None)
        sample, _ = next(iter_labeled_activations(split_dir))
        input_dim = int(sample.shape[-1])
        pooling = normalize_pooling(pooling)

        if pooling == "attn":
            probe_path = resolve_probe_path(probes_model_root, source_for_pooler, "attn", layer)
            if not probe_path.exists():
                print(f"[warn] missing attn probe for {source_for_pooler} layer {layer}; using mean fallback")
                pooler = Pooler("mean")
            else:
                pooler = load_attn_pooler(probe_path, input_dim=input_dim, device=device)
        else:
            pooler = Pooler(pooling)

        acc_sum = np.zeros(input_dim, dtype=np.float64)
        acc_n = 0
        iterator = iter_labeled_activations(split_dir)
        if use_tqdm:
            iterator = tqdm(iterator, desc=f"mean:{short_name(dataset)}:{split}:L{layer}:{pooling}", unit="sample", leave=False)
        t0 = time.time()
        for tensor, _ in iterator:
            x_layer = tensor[layer].float()
            pooled = pooler.pool(x_layer)
            acc_sum += pooled.detach().cpu().numpy().astype(np.float64)
            acc_n += 1
            if args.progress_every and acc_n % args.progress_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(f"[mean] {dataset}:{split} L{layer} {pooling} processed {acc_n} ({acc_n/elapsed:.2f} samples/s)")
        if acc_n == 0:
            raise RuntimeError(f"No samples for mean: {dataset} {split} L{layer} {pooling}")
        mean_vec = acc_sum / float(acc_n)
        np.save(cache_path, mean_vec)
        return mean_vec

    def activation_angle_matrix(
        segment: str,
        rows: List[str],
        cols: List[str],
        cells: Dict[Tuple[str, str], CellMeta],
        sigma: np.ndarray,
    ) -> np.ndarray:
        mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                cell = cells.get((r, c))
                if cell is None or cell.layer < 0:
                    continue
                pooling = normalize_pooling(cell.pooling)
                if r == c:
                    # diagonal: train vs val
                    mu_a = compute_mean(segment, r, "train", r, cell.layer, pooling)
                    mu_b = compute_mean(segment, r, "val", r, cell.layer, pooling)
                else:
                    mu_a = compute_mean(segment, r, "train", r, cell.layer, pooling)
                    mu_b = compute_mean(segment, c, "test", r, cell.layer, pooling)
                mat[i, j] = mahalanobis_cosine(mu_a, mu_b, sigma)
        return mat

    comp_act_angle = activation_angle_matrix("completion", comp_rows, comp_cols, comp_cells, sigma_comp)
    full_act_angle = activation_angle_matrix("full", full_rows, full_cols, full_cells, sigma_full)
    save_matrix_csv(out_dir / "activation_mean_angle_completion.csv", comp_act_angle, [short_name(r) for r in comp_rows], [short_name(c) for c in comp_cols])
    save_matrix_csv(out_dir / "activation_mean_angle_full.csv", full_act_angle, [short_name(r) for r in full_rows], [short_name(c) for c in full_cols])
    mark("activation_angles")

    # Summary
    update_status(status_path, "running", "saving summary")
    print("[stage 6/6] saving summary")
    write_json(
        out_dir / "summary.json",
        {
            "run_id": run_id,
            "completed_at": utc_now(),
            "model": args.model,
            "outputs": {
                "score_matrix_completion": str(out_dir / "score_matrix_completion.csv"),
                "score_matrix_full": str(out_dir / "score_matrix_full.csv"),
                "probe_angle_v1_completion": str(out_dir / "probe_angle_v1_completion.csv"),
                "probe_angle_v1_full": str(out_dir / "probe_angle_v1_full.csv"),
                "probe_angle_v2_completion": str(out_dir / "probe_angle_v2_completion.csv"),
                "probe_angle_v2_full": str(out_dir / "probe_angle_v2_full.csv"),
                "activation_angle_completion": str(out_dir / "activation_mean_angle_completion.csv"),
                "activation_angle_full": str(out_dir / "activation_mean_angle_full.csv"),
            },
        },
    )
    update_status(status_path, "completed", "finished")
    mark("finished")
    print(f"[done] artifacts -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
