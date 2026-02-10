#!/usr/bin/env python3
"""
PCA -> SAE follow-up analysis on top-k OOD gain rows.

This script consumes a ranking CSV (typically top20_ood_gain_lowk_leq5.csv),
selects top-N rows, and runs two complementary analyses:

1) Decoder alignment branch:
   For each removed PC direction in each selected row, find top-M SAE latents
   whose decoder vectors are most aligned (by absolute cosine).

2) Activation delta branch:
   For each selected row, apply the exact top-k PCA subspace removal to pooled
   activations, encode before/after with SAE, and find top-M latents by mean
   absolute activation change.

Artifacts are persisted with resume metadata:
- meta/run_manifest.json
- meta/status.json
- checkpoints/progress.json
- decoder_pc_feature_top3.csv
- activation_subspace_latent_top5.csv
- coverage_summary.json
- logs/run.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import random
import string
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def make_run_id() -> str:
    token = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{token}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def init_logging(log_path: Path, verbose: bool) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("pca_sae_topk_followup")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)
    return logger


def set_status(meta_dir: Path, run_id: str, state: str, current_step: Optional[str]) -> None:
    payload = {
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "last_updated_utc": utc_now_iso(),
    }
    write_json(meta_dir / "status.json", payload)


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_decoder_source_rows": [],
            "completed_delta_source_rows": [],
            "skipped": [],
            "updated_at_utc": utc_now_iso(),
        }
    payload = read_json(path)
    payload.setdefault("completed_decoder_source_rows", [])
    payload.setdefault("completed_delta_source_rows", [])
    payload.setdefault("skipped", [])
    payload.setdefault("updated_at_utc", utc_now_iso())
    return payload


def save_progress(path: Path, progress: dict) -> None:
    progress["updated_at_utc"] = utc_now_iso()
    write_json(path, progress)


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    norm_to_col = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        c = norm_to_col.get(name.strip().lower())
        if c is not None:
            return c
    if required:
        raise KeyError(f"Missing required column; expected one of {list(candidates)}")
    return None


def read_top_rows(top_csv: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(top_csv)
    pooling_col = find_col(df, ["pooling"])
    layer_col = find_col(df, ["layer"])
    k_col = find_col(df, ["k"])

    out = df.copy()
    out = out.rename(
        columns={
            pooling_col: "pooling",
            layer_col: "layer",
            k_col: "k",
        }
    )
    out["pooling"] = out["pooling"].astype(str).str.strip().str.lower()
    out["layer"] = out["layer"].astype(int)
    out["k"] = out["k"].astype(int)
    out = out.head(int(top_n)).copy().reset_index(drop=True)
    out.insert(0, "source_rank", np.arange(1, len(out) + 1))
    out.insert(0, "source_row_id", [f"row_{i:03d}" for i in range(len(out))])
    return out


def normalize_pooling_name(pooling: str) -> str:
    return pooling.strip().lower()


def resolve_pca_npz_path(pca_root: Path, pooling: str, layer: int, pca_subdir: str) -> Path:
    pooling = normalize_pooling_name(pooling)
    candidates = [
        pca_root / pooling / pca_subdir / f"layer_{layer}.npz",
        pca_root / pooling / f"layer_{layer}.npz",
        pca_root / f"layer_{layer}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Missing PCA artifact for pooling={pooling}, layer={layer}. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def load_pca_artifact(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "components" not in data or "mean" not in data:
            raise KeyError(f"{npz_path} missing required arrays: components/mean")
        comps = np.asarray(data["components"], dtype=np.float32)
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        return {"components": comps, "mean": mean}


def load_label_map(activations_dir: Path) -> Dict[str, int]:
    manifest_path = activations_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {activations_dir}")

    label_map: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            if not sid:
                continue
            label_map[sid] = int(row.get("label", -1))
    return label_map


def pool_tokens(x_layer: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "mean":
        pooled = x_layer.mean(dim=0)
    elif pooling == "max":
        pooled = x_layer.max(dim=0).values
    elif pooling == "last":
        pooled = x_layer[-1, :]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def load_pooled_split(
    activations_dir: Path,
    layers: Sequence[int],
    pooling: str,
    desc: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(glob.glob(str(activations_dir / "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    layers_sorted = sorted(set(int(x) for x in layers))
    buckets: Dict[int, List[np.ndarray]] = {l: [] for l in layers_sorted}
    labels: List[int] = []
    sample_ids: List[str] = []
    loaded = 0
    skipped_unknown = 0

    for shard_path in tqdm(shard_paths, desc=f"Loading {desc} shards"):
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            sid = str(sid)
            if sid not in label_map:
                continue
            label = int(label_map[sid])
            if label == -1:
                skipped_unknown += 1
                continue
            if tensor.dim() != 3:
                raise ValueError(
                    f"Expected tensor (L,T,D), got {tuple(tensor.shape)} for sample {sid}"
                )
            if max(layers_sorted) >= int(tensor.shape[0]):
                raise ValueError(
                    f"Layer index out of range for sample {sid}: requested up to {max(layers_sorted)}, "
                    f"tensor has {int(tensor.shape[0])} layers"
                )
            for layer in layers_sorted:
                buckets[layer].append(pool_tokens(tensor[layer, :, :], pooling))
            labels.append(label)
            sample_ids.append(sid)
            loaded += 1

    if loaded == 0:
        raise ValueError(f"No labeled samples loaded from {activations_dir}")

    x_by_layer = {layer: np.stack(rows).astype(np.float32) for layer, rows in buckets.items()}
    y = np.asarray(labels, dtype=np.int64)
    print(
        f"Loaded {desc}: N={len(y)}, D={next(iter(x_by_layer.values())).shape[1]}, "
        f"unknown_skipped={skipped_unknown}"
    )
    return x_by_layer, y, sample_ids


def remove_top_k_pcs(x: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    x_centered = x - mean[None, :]
    u = components[:k, :]
    proj = (x_centered @ u.T) @ u
    return (x_centered - proj) + mean[None, :]


class SAELensAdapter:
    def __init__(self, release: str, sae_id: str, device: str):
        try:
            from sae_lens import SAE  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sae_lens is not available. Install it before running this script."
            ) from e

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        self.sae = loaded[0] if isinstance(loaded, tuple) else loaded
        self.device = device
        self.sae_id = sae_id

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.sae.encode(x.to(self.device).float())
        return self._normalize_encoded(z)

    @staticmethod
    def _normalize_encoded(z: Any) -> torch.Tensor:
        if isinstance(z, tuple):
            z = z[0]
        if not torch.is_tensor(z):
            raise TypeError(f"Unexpected SAE encode output type: {type(z)}")
        if z.is_sparse:
            z = z.to_dense()
        return z.float()

    def decoder_feature_matrix(self, d_model_expected: int) -> torch.Tensor:
        w = None
        if hasattr(self.sae, "W_dec"):
            w = getattr(self.sae, "W_dec")
        elif hasattr(self.sae, "decoder") and hasattr(self.sae.decoder, "weight"):
            w = self.sae.decoder.weight
        if w is None:
            raise AttributeError("Could not locate SAE decoder weights (W_dec / decoder.weight)")

        w = w.detach().float().to("cpu")
        if w.ndim != 2:
            raise ValueError(f"Expected 2D decoder weight, got shape {tuple(w.shape)}")

        # Return shape (n_latents, d_model)
        if w.shape[1] == d_model_expected:
            return w
        if w.shape[0] == d_model_expected:
            return w.t().contiguous()
        raise ValueError(
            f"Decoder weight shape {tuple(w.shape)} incompatible with d_model={d_model_expected}"
        )


class AutoInterpLookup:
    def __init__(self, mapping_path: Optional[Path]):
        self.by_sae_latent: Dict[Tuple[str, int], str] = {}
        self.by_layer_latent: Dict[Tuple[int, int], str] = {}
        self.by_latent: Dict[int, str] = {}
        self.source = ""
        if mapping_path is None:
            return
        self.source = str(mapping_path)
        payload = read_json(mapping_path)
        self._ingest(payload)

    def _ingest(self, payload: Any) -> None:
        if isinstance(payload, dict):
            if "latents" in payload and isinstance(payload["latents"], list):
                self._ingest_latent_list(payload["latents"])
                return
            # Case: {"0": {"12": "label"}}
            if all(isinstance(v, dict) for v in payload.values()):
                for layer_key, layer_map in payload.items():
                    try:
                        layer = int(layer_key)
                    except Exception:
                        continue
                    for latent_key, label in layer_map.items():
                        try:
                            latent = int(latent_key)
                        except Exception:
                            continue
                        self.by_layer_latent[(layer, latent)] = str(label)
                return
            # Case: {"123": "label"}
            if all(isinstance(v, (str, int, float)) for v in payload.values()):
                for k, v in payload.items():
                    try:
                        latent = int(k)
                    except Exception:
                        continue
                    self.by_latent[latent] = str(v)
                return
        if isinstance(payload, list):
            self._ingest_latent_list(payload)

    def _ingest_latent_list(self, rows: List[Any]) -> None:
        for item in rows:
            if not isinstance(item, dict):
                continue
            latent_val = item.get("latent_id", item.get("feature_id", item.get("id")))
            if latent_val is None:
                continue
            try:
                latent = int(latent_val)
            except Exception:
                continue
            label = (
                item.get("autointerp")
                or item.get("label")
                or item.get("description")
                or item.get("name")
                or ""
            )
            if not label:
                continue
            sae_id = item.get("sae_id")
            layer_val = item.get("layer")
            if sae_id is not None:
                self.by_sae_latent[(str(sae_id), latent)] = str(label)
            if layer_val is not None:
                try:
                    layer = int(layer_val)
                    self.by_layer_latent[(layer, latent)] = str(label)
                except Exception:
                    pass
            self.by_latent.setdefault(latent, str(label))

    def get(self, layer: int, latent_id: int, sae_id: str) -> Tuple[str, str]:
        key_sae = (sae_id, int(latent_id))
        if key_sae in self.by_sae_latent:
            return self.by_sae_latent[key_sae], "sae_id_latent"
        key_layer = (int(layer), int(latent_id))
        if key_layer in self.by_layer_latent:
            return self.by_layer_latent[key_layer], "layer_latent"
        if int(latent_id) in self.by_latent:
            return self.by_latent[int(latent_id)], "latent_only"
        return "", ""


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_df(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def cosine_topk_for_pc(
    pc: np.ndarray,
    decoder_features: torch.Tensor,
    top_m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # decoder_features: (n_latents, d_model)
    pc_t = torch.from_numpy(pc.astype(np.float32))
    pc_norm = torch.norm(pc_t) + 1e-12
    pc_unit = pc_t / pc_norm

    dec = decoder_features
    dec_norm = torch.norm(dec, dim=1, keepdim=True) + 1e-12
    dec_unit = dec / dec_norm
    scores = torch.matmul(dec_unit, pc_unit)  # (n_latents,)

    top_m = int(min(max(top_m, 1), scores.numel()))
    vals, idx = torch.topk(torch.abs(scores), k=top_m, largest=True, sorted=True)
    idx_np = idx.detach().cpu().numpy().astype(int)
    signed_np = scores[idx].detach().cpu().numpy().astype(np.float32)
    return idx_np, signed_np


def mean_latent_delta_topk(
    adapter: SAELensAdapter,
    x: np.ndarray,
    x_prime: np.ndarray,
    batch_size: int,
    top_m: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if x.shape != x_prime.shape:
        raise ValueError(f"x and x_prime shape mismatch: {x.shape} vs {x_prime.shape}")
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("No samples provided for activation delta analysis")

    sum_delta: Optional[torch.Tensor] = None
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.float32)
            xpb = torch.from_numpy(x_prime[start:end]).to(device=device, dtype=torch.float32)
            zb = adapter.encode(xb).detach().cpu()
            zpb = adapter.encode(xpb).detach().cpu()
            db = (zpb - zb).sum(dim=0)  # (n_latents,)
            if sum_delta is None:
                sum_delta = db
            else:
                sum_delta = sum_delta + db

    if sum_delta is None:
        raise RuntimeError("Failed to compute latent deltas")

    mean_delta = sum_delta / float(n)
    top_m = int(min(max(top_m, 1), mean_delta.numel()))
    vals, idx = torch.topk(torch.abs(mean_delta), k=top_m, largest=True, sorted=True)
    idx_np = idx.numpy().astype(int)
    mean_delta_np = mean_delta[idx].numpy().astype(np.float32)
    return idx_np, mean_delta_np, n


def summarize_expected_counts(selected_df: pd.DataFrame, decoder_top_m: int, delta_top_m: int) -> Dict[str, int]:
    expected_decoder = int((selected_df["k"].astype(int) * int(decoder_top_m)).sum())
    expected_delta = int(len(selected_df) * int(delta_top_m))
    return {
        "expected_decoder_rows": expected_decoder,
        "expected_delta_rows": expected_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA -> SAE follow-up analysis on top-k rows")
    parser.add_argument("--top_csv", type=str, required=True, help="Input ranking CSV path")
    parser.add_argument("--pca_root", type=str, required=True, help="PCA root directory")
    parser.add_argument("--id_val_activations_dir", type=str, required=True, help="ID-val activations dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated artifacts",
    )
    parser.add_argument("--top_n", type=int, default=10, help="Number of top rows to process")
    parser.add_argument(
        "--pca_subdir",
        type=str,
        default="pca_artifacts",
        help="Subdirectory under each pooling containing layer_*.npz",
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default="chanind/sae-llama-3.2-1b-topk-res",
        help="SAE release/repo id",
    )
    parser.add_argument(
        "--sae_l0_tag",
        type=str,
        default="l0-10",
        help="SAE suffix tag. SAE id resolves to blocks.{layer}.hook_resid_post/{sae_l0_tag}",
    )
    parser.add_argument("--decoder_top_m", type=int, default=3, help="Top latents per PC in decoder branch")
    parser.add_argument("--delta_top_m", type=int, default=5, help="Top changed latents per row in delta branch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for SAE encoding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--autointerp_json", type=str, default=None, help="Optional JSON mapping for latent labels")
    parser.add_argument("--force_rebuild", action="store_true", help="Recompute even if completed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    top_csv = Path(args.top_csv).resolve()
    pca_root = Path(args.pca_root).resolve()
    id_val_dir = Path(args.id_val_activations_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    meta_dir = output_dir / "meta"
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"

    ensure_dir(output_dir)
    ensure_dir(meta_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    logger = init_logging(logs_dir / "run.log", verbose=bool(args.verbose))

    run_id = make_run_id()
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"

    if status_path.exists() and not args.force_rebuild:
        status = read_json(status_path)
        if status.get("state") == "completed":
            logger.info("Status already completed at %s; use --force_rebuild to rerun.", status_path)
            return

    if args.force_rebuild:
        for p in [
            output_dir / "decoder_pc_feature_top3.csv",
            output_dir / "activation_subspace_latent_top5.csv",
            output_dir / "coverage_summary.json",
            progress_path,
        ]:
            if p.exists():
                p.unlink()

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "top_csv": str(top_csv),
            "pca_root": str(pca_root),
            "id_val_activations_dir": str(id_val_dir),
            "autointerp_json": str(args.autointerp_json) if args.autointerp_json else None,
        },
        "config": {
            "top_n": int(args.top_n),
            "pca_subdir": args.pca_subdir,
            "sae_repo": args.sae_repo,
            "sae_l0_tag": args.sae_l0_tag,
            "decoder_top_m": int(args.decoder_top_m),
            "delta_top_m": int(args.delta_top_m),
            "batch_size": int(args.batch_size),
            "device": args.device,
            "force_rebuild": bool(args.force_rebuild),
        },
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    set_status(meta_dir, run_id=run_id, state="running", current_step="load_inputs")

    try:
        selected = read_top_rows(top_csv, top_n=int(args.top_n))
        if selected.empty:
            raise ValueError(f"No rows available in {top_csv}")
        selected.to_csv(meta_dir / "input_snapshot_topN.csv", index=False)
        logger.info("Selected %d rows from %s", len(selected), top_csv)

        auto_lookup = AutoInterpLookup(Path(args.autointerp_json) if args.autointerp_json else None)
        progress = load_progress(progress_path)
        done_decoder: Set[str] = set(progress.get("completed_decoder_source_rows", []))
        done_delta: Set[str] = set(progress.get("completed_delta_source_rows", []))
        skipped: List[dict] = list(progress.get("skipped", []))

        decoder_csv = output_dir / "decoder_pc_feature_top3.csv"
        delta_csv = output_dir / "activation_subspace_latent_top5.csv"
        decoder_existing_df = maybe_read_csv(decoder_csv)
        delta_existing_df = maybe_read_csv(delta_csv)
        decoder_rows: List[dict] = decoder_existing_df.to_dict("records")
        delta_rows: List[dict] = delta_existing_df.to_dict("records")
        if "source_row_id" in decoder_existing_df.columns:
            done_decoder |= set(str(x) for x in decoder_existing_df["source_row_id"].dropna().tolist())
        if "source_row_id" in delta_existing_df.columns:
            done_delta |= set(str(x) for x in delta_existing_df["source_row_id"].dropna().tolist())

        # Preload PCA artifacts used by selected rows.
        set_status(meta_dir, run_id=run_id, state="running", current_step="load_pca_artifacts")
        pca_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        for _, row in selected.iterrows():
            pooling = str(row["pooling"])
            layer = int(row["layer"])
            key = (pooling, layer)
            if key in pca_cache:
                continue
            npz_path = resolve_pca_npz_path(pca_root, pooling, layer, pca_subdir=args.pca_subdir)
            pca_cache[key] = load_pca_artifact(npz_path)
            logger.info("Loaded PCA artifact: pooling=%s layer=%d path=%s", pooling, layer, npz_path)

        # Preload pooled activations by pooling+layers for delta branch.
        set_status(meta_dir, run_id=run_id, state="running", current_step="load_pooled_activations")
        layers_by_pooling: Dict[str, Set[int]] = {}
        for _, row in selected.iterrows():
            pooling = str(row["pooling"])
            layer = int(row["layer"])
            layers_by_pooling.setdefault(pooling, set()).add(layer)

        pooled_cache: Dict[Tuple[str, int], np.ndarray] = {}
        for pooling, layers in sorted(layers_by_pooling.items()):
            x_by_layer, _, _ = load_pooled_split(
                id_val_dir,
                layers=sorted(layers),
                pooling=pooling,
                desc=f"id_val/{pooling}",
            )
            for layer in sorted(layers):
                pooled_cache[(pooling, layer)] = x_by_layer[layer]
                logger.info(
                    "Loaded pooled vectors: pooling=%s layer=%d shape=%s",
                    pooling,
                    layer,
                    tuple(x_by_layer[layer].shape),
                )

        # SAE cache by layer.
        set_status(meta_dir, run_id=run_id, state="running", current_step="load_saes")
        sae_cache: Dict[int, SAELensAdapter] = {}
        for layer in sorted(set(int(x) for x in selected["layer"].tolist())):
            sae_id = f"blocks.{layer}.hook_resid_post/{args.sae_l0_tag}"
            sae_cache[layer] = SAELensAdapter(release=args.sae_repo, sae_id=sae_id, device=args.device)
            logger.info("Loaded SAE layer=%d id=%s", layer, sae_id)

        # Branch A: Decoder alignment
        set_status(meta_dir, run_id=run_id, state="running", current_step="decoder_alignment")
        for _, row in selected.iterrows():
            row_id = str(row["source_row_id"])
            if row_id in done_decoder:
                continue
            try:
                pooling = str(row["pooling"])
                layer = int(row["layer"])
                k = int(row["k"])
                pca_art = pca_cache[(pooling, layer)]
                comps = pca_art["components"]
                d_model = int(comps.shape[1])
                if k > int(comps.shape[0]):
                    raise ValueError(
                        f"Requested k={k} but only {comps.shape[0]} PCA components available "
                        f"for pooling={pooling}, layer={layer}"
                    )

                adapter = sae_cache[layer]
                sae_id = adapter.sae_id
                decoder_mat = adapter.decoder_feature_matrix(d_model_expected=d_model)

                source_payload = {f"source_{c}": row[c] for c in row.index if c not in ("source_row_id",)}

                for pc_idx in range(k):
                    pc = comps[pc_idx].astype(np.float32, copy=False)
                    latent_ids, signed_scores = cosine_topk_for_pc(
                        pc=pc,
                        decoder_features=decoder_mat,
                        top_m=int(args.decoder_top_m),
                    )
                    for rank, (latent_id, signed_score) in enumerate(
                        zip(latent_ids.tolist(), signed_scores.tolist()), start=1
                    ):
                        label, label_source = auto_lookup.get(layer=layer, latent_id=int(latent_id), sae_id=sae_id)
                        out_row = {
                            "source_row_id": row_id,
                            "pooling": pooling,
                            "layer": layer,
                            "k": k,
                            "pc_index": int(pc_idx),
                            "pc_rank": int(pc_idx + 1),
                            "feature_rank": int(rank),
                            "latent_id": int(latent_id),
                            "cosine": float(signed_score),
                            "abs_cosine": float(abs(signed_score)),
                            "sae_repo": args.sae_repo,
                            "sae_id": sae_id,
                            "autointerp_label": label,
                            "autointerp_source": label_source,
                        }
                        out_row.update(source_payload)
                        decoder_rows.append(out_row)

                done_decoder.add(row_id)
                progress["completed_decoder_source_rows"] = sorted(done_decoder)
                save_progress(progress_path, progress)
                save_df(decoder_csv, decoder_rows)
                logger.info("Decoder branch complete for %s", row_id)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.error("Decoder branch failed for %s: %s", row_id, reason)
                skipped.append({"branch": "decoder", "source_row_id": row_id, "reason": reason})
                progress["skipped"] = skipped
                save_progress(progress_path, progress)

        # Branch B: Activation deltas under row-level subspace removal.
        set_status(meta_dir, run_id=run_id, state="running", current_step="activation_delta")
        for _, row in selected.iterrows():
            row_id = str(row["source_row_id"])
            if row_id in done_delta:
                continue
            try:
                pooling = str(row["pooling"])
                layer = int(row["layer"])
                k = int(row["k"])
                pca_art = pca_cache[(pooling, layer)]
                comps = pca_art["components"]
                mean = pca_art["mean"]
                if k > int(comps.shape[0]):
                    raise ValueError(
                        f"Requested k={k} but only {comps.shape[0]} PCA components available "
                        f"for pooling={pooling}, layer={layer}"
                    )

                x = pooled_cache[(pooling, layer)]
                x_clean = remove_top_k_pcs(x=x, mean=mean, components=comps, k=k)

                adapter = sae_cache[layer]
                sae_id = adapter.sae_id
                latent_ids, mean_deltas, n_samples = mean_latent_delta_topk(
                    adapter=adapter,
                    x=x,
                    x_prime=x_clean,
                    batch_size=int(args.batch_size),
                    top_m=int(args.delta_top_m),
                    device=args.device,
                )
                source_payload = {f"source_{c}": row[c] for c in row.index if c not in ("source_row_id",)}
                for rank, (latent_id, delta_val) in enumerate(
                    zip(latent_ids.tolist(), mean_deltas.tolist()), start=1
                ):
                    label, label_source = auto_lookup.get(layer=layer, latent_id=int(latent_id), sae_id=sae_id)
                    out_row = {
                        "source_row_id": row_id,
                        "pooling": pooling,
                        "layer": layer,
                        "k": k,
                        "latent_rank": int(rank),
                        "latent_id": int(latent_id),
                        "delta_mean": float(delta_val),
                        "delta_abs": float(abs(delta_val)),
                        "delta_direction": "increase" if float(delta_val) >= 0.0 else "decrease",
                        "n_samples": int(n_samples),
                        "batch_size": int(args.batch_size),
                        "sae_repo": args.sae_repo,
                        "sae_id": sae_id,
                        "autointerp_label": label,
                        "autointerp_source": label_source,
                    }
                    out_row.update(source_payload)
                    delta_rows.append(out_row)

                done_delta.add(row_id)
                progress["completed_delta_source_rows"] = sorted(done_delta)
                save_progress(progress_path, progress)
                save_df(delta_csv, delta_rows)
                logger.info("Delta branch complete for %s", row_id)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.error("Delta branch failed for %s: %s", row_id, reason)
                skipped.append({"branch": "delta", "source_row_id": row_id, "reason": reason})
                progress["skipped"] = skipped
                save_progress(progress_path, progress)

        counts = summarize_expected_counts(
            selected_df=selected,
            decoder_top_m=int(args.decoder_top_m),
            delta_top_m=int(args.delta_top_m),
        )
        coverage = {
            "generated_at_utc": utc_now_iso(),
            "input_top_csv": str(top_csv),
            "num_input_rows_in_csv": int(pd.read_csv(top_csv).shape[0]),
            "num_selected_rows": int(selected.shape[0]),
            "selected_source_row_ids": selected["source_row_id"].tolist(),
            "unique_poolings": sorted(set(selected["pooling"].tolist())),
            "unique_layers": sorted(set(int(x) for x in selected["layer"].tolist())),
            "expected_decoder_rows": counts["expected_decoder_rows"],
            "expected_delta_rows": counts["expected_delta_rows"],
            "written_decoder_rows": int(len(decoder_rows)),
            "written_delta_rows": int(len(delta_rows)),
            "completed_decoder_source_rows": sorted(done_decoder),
            "completed_delta_source_rows": sorted(done_delta),
            "skipped": skipped,
        }
        write_json(output_dir / "coverage_summary.json", coverage)

        set_status(meta_dir, run_id=run_id, state="completed", current_step=None)
        logger.info("Completed. Output dir: %s", output_dir)
    except Exception:
        set_status(meta_dir, run_id=run_id, state="failed", current_step=None)
        logger.error("Run failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
