#!/usr/bin/env python3
"""
Run SAE feature analysis on shortlisted influential sample groups.

Expected input CSV columns:
- sample_id (required)
- group (optional; defaults to "all")
- layer / target_layer / best_target_layer (optional; else use --default_layer)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sae_influential_samples")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_layer_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for c in ("layer", "target_layer", "best_target_layer"):
        if c in df.columns:
            return c
    return None


def pool_layer_tensor(layer_tensor: torch.Tensor, pooling: str) -> torch.Tensor:
    # layer_tensor: (T, D) or (D,)
    if layer_tensor.dim() == 1:
        return layer_tensor
    if pooling == "mean":
        return layer_tensor.mean(dim=0)
    if pooling == "max":
        return layer_tensor.max(dim=0).values
    if pooling == "last":
        return layer_tensor[-1]
    if pooling == "none":
        if layer_tensor.shape[0] == 1:
            return layer_tensor[0]
        raise ValueError("pooling='none' expects per-layer tensor with T=1")
    raise ValueError(f"Unsupported pooling: {pooling}")


def load_selected_tensors(activations_dir: str, selected_ids: set) -> Dict[str, torch.Tensor]:
    shard_paths = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {activations_dir}")

    tensors: Dict[str, torch.Tensor] = {}
    for p in shard_paths:
        shard = load_file(p)
        for sid in selected_ids:
            if sid in shard and sid not in tensors:
                tensors[sid] = shard[sid]
        if len(tensors) == len(selected_ids):
            break
    return tensors


class SAEAdapter:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CustomLinearSAEAdapter(SAEAdapter):
    """
    Supports simple custom checkpoints exposing either:
    - W_enc (+ optional b_enc)
    - encoder.weight (+ optional encoder.bias)
    """

    def __init__(self, checkpoint_path: str, device: str):
        obj = torch.load(checkpoint_path, map_location=device)
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise ValueError("Custom SAE checkpoint must load to a dict-like object")

        w = None
        b = None
        if "W_enc" in obj:
            w = obj["W_enc"]
            b = obj.get("b_enc")
        elif "encoder.weight" in obj:
            w = obj["encoder.weight"]
            b = obj.get("encoder.bias")

        if w is None:
            raise ValueError("Missing encoder weights: expected keys 'W_enc' or 'encoder.weight'")
        if b is None:
            b = torch.zeros(w.shape[0], dtype=w.dtype)

        self.w = w.detach().to(device).float()
        self.b = b.detach().to(device).float()
        self.device = device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        if self.w.shape[1] == x.shape[1]:
            z = x @ self.w.t()
        elif self.w.shape[0] == x.shape[1]:
            z = x @ self.w
        else:
            raise ValueError(f"Dimension mismatch: input D={x.shape[1]}, W={tuple(self.w.shape)}")
        z = z + self.b
        return torch.relu(z)


class SAELensAdapter(SAEAdapter):
    def __init__(self, release: str, sae_id: str, device: str):
        try:
            from sae_lens import SAE  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sae_lens is not available. Install it in Colab or use --sae_source custom"
            ) from e

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        self.sae = loaded[0] if isinstance(loaded, tuple) else loaded
        self.device = device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        with torch.no_grad():
            z = self.sae.encode(x)
        return z


def build_adapter(args: argparse.Namespace) -> SAEAdapter:
    if args.sae_source == "custom":
        if not args.sae_checkpoint:
            raise ValueError("--sae_checkpoint is required for --sae_source custom")
        return CustomLinearSAEAdapter(args.sae_checkpoint, args.device)

    if not args.sae_release or not args.sae_id:
        raise ValueError("--sae_release and --sae_id are required for --sae_source pretrained")
    return SAELensAdapter(args.sae_release, args.sae_id, args.device)


def parse_group_filter(raw: Optional[str]) -> Optional[set]:
    if raw is None or raw.strip() == "":
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def extract_grouped_vectors(
    groups_df: pd.DataFrame,
    tensors: Dict[str, torch.Tensor],
    layer_col: Optional[str],
    default_layer: Optional[int],
    group_col: str,
    pooling: str,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    rows = []
    vecs = []

    for _, r in groups_df.iterrows():
        sid = str(r["sample_id"])
        t = tensors.get(sid)
        if t is None:
            continue

        if layer_col is not None:
            layer = int(r[layer_col])
        elif default_layer is not None:
            layer = int(default_layer)
        else:
            continue

        if t.dim() == 3:
            if layer < 0 or layer >= t.shape[0]:
                continue
            lv = pool_layer_tensor(t[layer], pooling=pooling)
        elif t.dim() == 2:
            if layer < 0 or layer >= t.shape[0]:
                continue
            lv = t[layer]
        else:
            continue

        group = str(r[group_col]) if group_col in groups_df.columns else "all"
        rows.append({"sample_id": sid, "group": group, "layer": layer})
        vecs.append(lv.float())

    if not vecs:
        return pd.DataFrame(columns=["sample_id", "group", "layer"]), torch.empty((0, 0))

    return pd.DataFrame(rows), torch.stack(vecs)


def compute_top_features(meta_df: pd.DataFrame, acts: torch.Tensor, top_k: int) -> pd.DataFrame:
    acts_np = acts.detach().cpu().numpy()
    rows = []
    for i in range(acts_np.shape[0]):
        row = acts_np[i]
        top_idx = np.argsort(row)[::-1][:top_k]
        for rank, fid in enumerate(top_idx, start=1):
            rows.append(
                {
                    "sample_id": meta_df.iloc[i]["sample_id"],
                    "group": meta_df.iloc[i]["group"],
                    "layer": int(meta_df.iloc[i]["layer"]),
                    "feature_id": int(fid),
                    "activation": float(row[fid]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def compute_group_enrichment(meta_df: pd.DataFrame, acts: torch.Tensor) -> pd.DataFrame:
    acts_np = acts.detach().cpu().numpy()
    groups = meta_df["group"].tolist()
    unique_groups = sorted(set(groups))
    rows = []

    for g in unique_groups:
        mask = np.array([x == g for x in groups])
        if mask.sum() == 0:
            continue
        g_acts = acts_np[mask]
        rest_acts = acts_np[~mask] if (~mask).sum() > 0 else np.zeros_like(g_acts)

        mean_g = g_acts.mean(axis=0)
        mean_rest = rest_acts.mean(axis=0) if rest_acts.size > 0 else np.zeros_like(mean_g)
        freq_g = (g_acts > 0).mean(axis=0)

        for fid in range(mean_g.shape[0]):
            rows.append(
                {
                    "group": g,
                    "feature_id": fid,
                    "mean_activation": float(mean_g[fid]),
                    "frequency_active": float(freq_g[fid]),
                    "mean_activation_lift_vs_rest": float(mean_g[fid] - mean_rest[fid]),
                }
            )

    return pd.DataFrame(rows)


def plot_group_contrast(enrich: pd.DataFrame, out_path: str, top_plot_features: int) -> None:
    if enrich.empty:
        return

    overall = enrich.groupby("feature_id", as_index=False)["mean_activation"].mean()
    top = overall.sort_values("mean_activation", ascending=False).head(top_plot_features)["feature_id"].tolist()
    sub = enrich[enrich["feature_id"].isin(top)].copy()

    groups = sorted(sub["group"].unique().tolist())
    feat_order = top

    mat = np.zeros((len(groups), len(feat_order)), dtype=float)
    for gi, g in enumerate(groups):
        gsub = sub[sub["group"] == g].set_index("feature_id")
        for fi, fid in enumerate(feat_order):
            if fid in gsub.index:
                mat[gi, fi] = gsub.loc[fid, "mean_activation"]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_order) * 0.25), max(4, len(groups) * 0.7)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title("SAE Mean Activation by Group (Top Features)")
    ax.set_xlabel("Feature ID")
    ax.set_ylabel("Group")
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    fig.colorbar(im, ax=ax, label="Mean activation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAE feature analysis on influential sample groups")
    p.add_argument("--sample_groups_csv", type=str, required=True)
    p.add_argument("--activations_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)

    p.add_argument("--group_column", type=str, default="group")
    p.add_argument("--layer_column", type=str, default=None)
    p.add_argument("--default_layer", type=int, default=None)
    p.add_argument("--groups", type=str, default=None, help="Comma-separated group names to include")

    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last", "none"])

    p.add_argument("--sae_source", type=str, default="pretrained", choices=["pretrained", "custom"])
    p.add_argument("--sae_release", type=str, default=None)
    p.add_argument("--sae_id", type=str, default=None)
    p.add_argument("--sae_checkpoint", type=str, default=None)

    p.add_argument("--top_k_features", type=int, default=20)
    p.add_argument("--top_plot_features", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> int:
    args = make_parser().parse_args()

    ensure_dir(args.output_root)
    tables_dir = os.path.join(args.output_root, "tables")
    figs_dir = os.path.join(args.output_root, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    groups_df = pd.read_csv(args.sample_groups_csv)
    if "sample_id" not in groups_df.columns:
        raise ValueError("sample_groups_csv must contain 'sample_id'")
    if args.group_column not in groups_df.columns:
        groups_df[args.group_column] = "all"

    include_groups = parse_group_filter(args.groups)
    if include_groups is not None:
        groups_df = groups_df[groups_df[args.group_column].isin(include_groups)].copy()

    layer_col = infer_layer_column(groups_df, args.layer_column)
    if layer_col is None and args.default_layer is None:
        raise ValueError("No layer column found and --default_layer was not provided")

    selected_ids = set(groups_df["sample_id"].astype(str).tolist())
    tensors = load_selected_tensors(args.activations_dir, selected_ids)
    logger.info("Loaded tensors for %d/%d selected IDs", len(tensors), len(selected_ids))

    meta_df, vectors = extract_grouped_vectors(
        groups_df=groups_df,
        tensors=tensors,
        layer_col=layer_col,
        default_layer=args.default_layer,
        group_col=args.group_column,
        pooling=args.pooling,
    )
    if vectors.numel() == 0:
        raise ValueError("No valid vectors extracted from selected samples/layers")

    adapter = build_adapter(args)
    with torch.no_grad():
        acts = adapter.encode(vectors.to(args.device)).detach().cpu()

    top_df = compute_top_features(meta_df, acts, top_k=args.top_k_features)
    enrich_df = compute_group_enrichment(meta_df, acts)

    top_df.to_csv(os.path.join(tables_dir, "sae_top_features_per_sample.csv"), index=False)
    enrich_df.to_csv(os.path.join(tables_dir, "sae_feature_group_enrichment.csv"), index=False)
    meta_df.to_csv(os.path.join(tables_dir, "sae_input_samples_resolved.csv"), index=False)

    plot_group_contrast(
        enrich_df,
        out_path=os.path.join(figs_dir, "sae_feature_activation_contrast.png"),
        top_plot_features=args.top_plot_features,
    )

    run_meta = {
        "num_input_rows": int(len(groups_df)),
        "num_resolved_rows": int(len(meta_df)),
        "num_groups": int(meta_df["group"].nunique()),
        "num_features": int(acts.shape[1]),
        "sae_source": args.sae_source,
        "pooling": args.pooling,
    }
    with open(os.path.join(args.output_root, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("SAE analysis complete. Output: %s", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
