#!/usr/bin/env python3
"""
PCA direction-removal sweep for pretrained probes.

Workflow:
1. Load pooled activations for ID-train, ID-validation, OOD-test.
2. Fit PCA per layer on ID-train only.
3. Save PCA artifacts (mean, components, EV, EVR, metadata).
4. Evaluate pretrained probes on baseline (k=0) and top-k PC-removed activations.
5. Save JSON/CSV summaries and plots.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))

from actprobe.probes.models import LayerProbe


def parse_k_values(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    vals = sorted(set(vals))
    if any(k <= 0 for k in vals):
        raise ValueError(f"All k values must be positive. Got: {vals}")
    return vals


def find_probe_files(probes_dir: str) -> Dict[int, str]:
    probe_files = sorted(glob.glob(os.path.join(probes_dir, "probe_layer_*.pt")))
    out = {}
    for path in probe_files:
        name = os.path.basename(path)
        layer_idx = int(name.replace("probe_layer_", "").replace(".pt", ""))
        out[layer_idx] = path
    return out


def parse_layers(raw: str, available_layers: List[int]) -> List[int]:
    if raw.strip().lower() == "all":
        return sorted(available_layers)

    selected = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start = int(a)
            end = int(b)
            for i in range(start, end + 1):
                selected.add(i)
        else:
            selected.add(int(token))
    layers = sorted(selected)
    return [l for l in layers if l in set(available_layers)]


def load_label_map(activations_dir: str) -> Dict[str, int]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest.jsonl in {activations_dir}")

    label_map = {}
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            label_map[entry["id"]] = int(entry.get("label", -1))
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
    activations_dir: str,
    layers: List[int],
    pooling: str,
    desc: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    feature_buckets: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    labels: List[int] = []
    sample_ids: List[str] = []
    skipped_unknown = 0
    loaded = 0

    for shard_path in tqdm(shard_paths, desc=f"Loading {desc} shards"):
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            if sid not in label_map:
                continue
            label = label_map[sid]
            if label == -1:
                skipped_unknown += 1
                continue

            if tensor.dim() != 3:
                raise ValueError(
                    f"Expected tensor shape (L,T,D), got {tuple(tensor.shape)} for sample {sid}"
                )

            if max(layers) >= tensor.shape[0]:
                raise ValueError(
                    f"Layer index out of range for sample {sid}: max layer {max(layers)} "
                    f"but tensor has {tensor.shape[0]} layers."
                )

            for layer in layers:
                x_layer = tensor[layer, :, :]
                feature_buckets[layer].append(pool_tokens(x_layer, pooling))

            labels.append(label)
            sample_ids.append(sid)
            loaded += 1

    if loaded == 0:
        raise ValueError(f"No labeled examples loaded from {activations_dir}")

    x_by_layer = {layer: np.stack(rows).astype(np.float32) for layer, rows in feature_buckets.items()}
    y = np.asarray(labels, dtype=np.int64)

    print(
        f"Loaded {desc}: N={len(y)}, D={next(iter(x_by_layer.values())).shape[1]}, "
        f"unknown_skipped={skipped_unknown}"
    )
    return x_by_layer, y, sample_ids


def load_token_split(
    activations_dir: str,
    layers: List[int],
    desc: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    feature_buckets: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    labels: List[int] = []
    sample_ids: List[str] = []
    skipped_unknown = 0
    loaded = 0
    token_len: Optional[int] = None
    hidden_dim: Optional[int] = None

    for shard_path in tqdm(shard_paths, desc=f"Loading {desc} shards"):
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            if sid not in label_map:
                continue
            label = label_map[sid]
            if label == -1:
                skipped_unknown += 1
                continue

            if tensor.dim() != 3:
                raise ValueError(
                    f"Expected tensor shape (L,T,D), got {tuple(tensor.shape)} for sample {sid}"
                )
            if max(layers) >= tensor.shape[0]:
                raise ValueError(
                    f"Layer index out of range for sample {sid}: max layer {max(layers)} "
                    f"but tensor has {tensor.shape[0]} layers."
                )

            for layer in layers:
                x_layer = tensor[layer, :, :].detach().cpu().numpy().astype(np.float32, copy=False)
                if token_len is None:
                    token_len = int(x_layer.shape[0])
                    hidden_dim = int(x_layer.shape[1])
                elif x_layer.shape[0] != token_len or x_layer.shape[1] != hidden_dim:
                    raise ValueError(
                        "Inconsistent token activation shape detected. "
                        f"Expected (T,D)=({token_len},{hidden_dim}), got {x_layer.shape} for sample {sid}"
                    )
                feature_buckets[layer].append(x_layer)

            labels.append(label)
            sample_ids.append(sid)
            loaded += 1

    if loaded == 0:
        raise ValueError(f"No labeled examples loaded from {activations_dir}")

    x_by_layer = {layer: np.stack(rows).astype(np.float32) for layer, rows in feature_buckets.items()}
    y = np.asarray(labels, dtype=np.int64)

    print(
        f"Loaded {desc}: N={len(y)}, T={token_len}, D={hidden_dim}, "
        f"unknown_skipped={skipped_unknown}"
    )
    return x_by_layer, y, sample_ids


def maybe_subsample_train(
    x_by_layer: Dict[int, np.ndarray],
    y: np.ndarray,
    ids: List[str],
    max_samples: Optional[int],
    seed: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str], Optional[np.ndarray]]:
    if max_samples is None or len(y) <= max_samples:
        return x_by_layer, y, ids, None

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    idx = np.sort(idx)
    x_sub = {layer: x[idx] for layer, x in x_by_layer.items()}
    y_sub = y[idx]
    ids_sub = [ids[i] for i in idx]
    return x_sub, y_sub, ids_sub, idx


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def extract_classifier_params(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, float]:
    weight_key = None
    bias_key = None
    for k in state_dict:
        if k.endswith("classifier.weight"):
            weight_key = k
        if k.endswith("classifier.bias"):
            bias_key = k
    if weight_key is None or bias_key is None:
        raise KeyError("Could not find classifier.weight/classifier.bias in probe state dict")

    w = state_dict[weight_key].detach().cpu().numpy().reshape(-1).astype(np.float32)
    b = float(state_dict[bias_key].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    probs = sigmoid(logits)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    acc = float(accuracy_score(y_true, (probs >= 0.5).astype(np.int64)))
    return auc, acc


def remove_top_k_pcs(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    k: int,
) -> np.ndarray:
    x_centered = x - mean[None, :]
    u = components[:k, :]  # (k, D)
    proj = (x_centered @ u.T) @ u
    return (x_centered - proj) + mean[None, :]


def remove_top_k_pcs_tokens(
    x_ntd: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    k: int,
) -> np.ndarray:
    n, t, d = x_ntd.shape
    x_flat = x_ntd.reshape(n * t, d)
    x_centered = x_flat - mean[None, :]
    u = components[:k, :]
    proj = (x_centered @ u.T) @ u
    x_clean = (x_centered - proj) + mean[None, :]
    return x_clean.reshape(n, t, d)


def run_attn_probe_logits(
    probe: LayerProbe,
    x_ntd: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probe.eval()
    logits_all: List[np.ndarray] = []
    n = x_ntd.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(x_ntd[start:end]).to(device=device, dtype=torch.float32)
            logits = probe(batch).squeeze(-1).detach().cpu().numpy()
            logits_all.append(logits)
    return np.concatenate(logits_all, axis=0)


@dataclass
class PcaLayerArtifact:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    sign_flips: np.ndarray
    n_train_samples: int
    dim: int


def fit_layer_pca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_components_to_save: int,
    pca_solver: str,
    seed: int,
    pc_sign_rule: str,
) -> PcaLayerArtifact:
    n_samples, dim = x_train.shape
    max_allowed = min(n_samples, dim)
    n_components = min(n_components_to_save, max_allowed)
    if n_components < 1:
        raise ValueError("PCA needs at least one component")

    pca = PCA(n_components=n_components, svd_solver=pca_solver, random_state=seed)
    pca.fit(x_train)

    components = pca.components_.astype(np.float32, copy=True)
    sign_flips = np.zeros(n_components, dtype=np.int8)

    if pc_sign_rule == "deceptive_positive":
        deceptive_mask = y_train == 1
        if deceptive_mask.any():
            centered = x_train - pca.mean_[None, :]
            proj = centered @ components.T
            deceptive_means = proj[deceptive_mask].mean(axis=0)
            flips = deceptive_means < 0
            components[flips, :] *= -1.0
            sign_flips = flips.astype(np.int8)

    return PcaLayerArtifact(
        mean=pca.mean_.astype(np.float32),
        components=components,
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        sign_flips=sign_flips,
        n_train_samples=n_samples,
        dim=dim,
    )


def fit_layer_pca_tokens(
    x_train_ntd: np.ndarray,
    y_train: np.ndarray,
    n_components_to_save: int,
    pca_solver: str,
    seed: int,
    pc_sign_rule: str,
) -> PcaLayerArtifact:
    n, t, d = x_train_ntd.shape
    x_flat = x_train_ntd.reshape(n * t, d)

    max_allowed = min(x_flat.shape[0], d)
    n_components = min(n_components_to_save, max_allowed)
    if n_components < 1:
        raise ValueError("PCA needs at least one component")

    pca = PCA(n_components=n_components, svd_solver=pca_solver, random_state=seed)
    pca.fit(x_flat)

    components = pca.components_.astype(np.float32, copy=True)
    sign_flips = np.zeros(n_components, dtype=np.int8)

    if pc_sign_rule == "deceptive_positive":
        deceptive_mask = y_train == 1
        if deceptive_mask.any():
            centered = x_flat - pca.mean_[None, :]
            proj = centered @ components.T  # (N*T, K)
            proj_per_sample = proj.reshape(n, t, n_components).mean(axis=1)  # (N, K)
            deceptive_means = proj_per_sample[deceptive_mask].mean(axis=0)
            flips = deceptive_means < 0
            components[flips, :] *= -1.0
            sign_flips = flips.astype(np.int8)

    return PcaLayerArtifact(
        mean=pca.mean_.astype(np.float32),
        components=components,
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        sign_flips=sign_flips,
        n_train_samples=int(x_flat.shape[0]),
        dim=d,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pca_artifact(
    out_path: str,
    artifact: PcaLayerArtifact,
) -> None:
    np.savez_compressed(
        out_path,
        mean=artifact.mean,
        components=artifact.components,
        explained_variance=artifact.explained_variance,
        explained_variance_ratio=artifact.explained_variance_ratio,
        sign_flips=artifact.sign_flips,
        n_train_samples=np.asarray([artifact.n_train_samples], dtype=np.int64),
        dim=np.asarray([artifact.dim], dtype=np.int64),
    )


def plot_delta_heatmap(
    matrix: np.ndarray,
    layers: List[int],
    k_values: List[int],
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(max(8, len(layers) * 0.35), max(5, len(k_values) * 0.35)))
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 0.01
    vmax = max(vmax, 1e-3)
    plt.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Delta AUC vs baseline")
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.yticks(range(len(k_values)), k_values)
    plt.xlabel("Layer")
    plt.ylabel("k removed PCs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tradeoff(
    k_values: List[int],
    mean_id_auc: List[float],
    mean_ood_auc: List[float],
    baseline_id_auc: float,
    baseline_ood_auc: float,
    out_path: str,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, mean_id_auc, marker="o", linewidth=2.0, label="Mean ID Val AUC")
    plt.plot(k_values, mean_ood_auc, marker="s", linewidth=2.0, label="Mean OOD Test AUC")
    plt.axhline(baseline_id_auc, linestyle="--", alpha=0.5, color="#1f77b4", label="Baseline ID mean")
    plt.axhline(baseline_ood_auc, linestyle="--", alpha=0.5, color="#ff7f0e", label="Baseline OOD mean")
    plt.xlabel("k removed PCs")
    plt.ylabel("AUC")
    plt.title("ID vs OOD Tradeoff Across PCA Direction Removal")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA direction-removal sweep for probe evaluation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--id_train_activations_dir", type=str, required=True)
    parser.add_argument("--id_val_activations_dir", type=str, required=True)
    parser.add_argument("--ood_test_activations_dir", type=str, required=True)
    parser.add_argument("--probes_dir", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last", "attn"])
    parser.add_argument("--k_values", type=str, default="1,2,3,4,5,7,10,15,20")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pca_components_to_save", type=int, default=128)
    parser.add_argument("--pca_solver", type=str, default="randomized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples_for_pca", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pc_sign_rule",
        type=str,
        default="deceptive_positive",
        choices=["none", "deceptive_positive"],
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    probe_map = find_probe_files(args.probes_dir)
    if not probe_map:
        raise FileNotFoundError(f"No probe_layer_*.pt found in {args.probes_dir}")
    available_layers = sorted(probe_map.keys())

    layers = parse_layers(args.layers, available_layers)
    if not layers:
        raise ValueError(f"No valid layers selected. Available layers: {available_layers}")

    ensure_dir(args.output_dir)
    pca_artifacts_dir = os.path.join(args.output_dir, "pca_artifacts")
    ensure_dir(pca_artifacts_dir)

    print("=" * 90)
    print("PCA ABLATION SWEEP")
    print("=" * 90)
    print(f"Model: {args.model_name}")
    print(f"Pooling: {args.pooling}")
    print(f"Layers ({len(layers)}): {layers[:10]}{' ...' if len(layers) > 10 else ''}")
    print(f"Probe dir: {args.probes_dir}")
    print()

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)
    is_attn = args.pooling == "attn"
    pca_fit_level = "token" if is_attn else "pooled"
    evaluation_mode = "full_probe_forward" if is_attn else "linear_head_shortcut"

    # ---------------------------------------------------------------------
    # Load activations
    # ---------------------------------------------------------------------
    if is_attn:
        x_train, y_train, train_ids = load_token_split(
            args.id_train_activations_dir, layers, desc="ID train"
        )
        x_val, y_val, _ = load_token_split(
            args.id_val_activations_dir, layers, desc="ID validation"
        )
        x_ood, y_ood, _ = load_token_split(
            args.ood_test_activations_dir, layers, desc="OOD test"
        )
    else:
        x_train, y_train, train_ids = load_pooled_split(
            args.id_train_activations_dir, layers, args.pooling, desc="ID train"
        )
        x_val, y_val, _ = load_pooled_split(
            args.id_val_activations_dir, layers, args.pooling, desc="ID validation"
        )
        x_ood, y_ood, _ = load_pooled_split(
            args.ood_test_activations_dir, layers, args.pooling, desc="OOD test"
        )

    x_train, y_train, train_ids, sampled_idx = maybe_subsample_train(
        x_train, y_train, train_ids, args.max_train_samples_for_pca, args.seed
    )
    if sampled_idx is not None:
        print(f"Subsampled ID train for PCA: {len(y_train)} samples")

    # ---------------------------------------------------------------------
    # Fit PCA per layer + save artifacts
    # ---------------------------------------------------------------------
    pca_by_layer: Dict[int, PcaLayerArtifact] = {}
    pca_manifest = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "pca_fit_level": pca_fit_level,
        "evaluation_mode": evaluation_mode,
        "fit_split": "id_train",
        "fit_path": args.id_train_activations_dir,
        "pca_solver": args.pca_solver,
        "pc_sign_rule": args.pc_sign_rule,
        "seed": args.seed,
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "layers": [],
    }

    for layer in tqdm(layers, desc="Fitting PCA by layer"):
        if is_attn:
            artifact = fit_layer_pca_tokens(
                x_train[layer],
                y_train,
                n_components_to_save=args.pca_components_to_save,
                pca_solver=args.pca_solver,
                seed=args.seed,
                pc_sign_rule=args.pc_sign_rule,
            )
        else:
            artifact = fit_layer_pca(
                x_train[layer],
                y_train,
                n_components_to_save=args.pca_components_to_save,
                pca_solver=args.pca_solver,
                seed=args.seed,
                pc_sign_rule=args.pc_sign_rule,
            )
        if artifact.components.shape[0] < max_k:
            raise ValueError(
                f"Layer {layer} has only {artifact.components.shape[0]} saved PCs, "
                f"but max k={max_k}. Increase --pca_components_to_save."
            )

        artifact_path = os.path.join(pca_artifacts_dir, f"layer_{layer}.npz")
        save_pca_artifact(artifact_path, artifact)
        pca_by_layer[layer] = artifact

        pca_manifest["layers"].append(
            {
                "layer": layer,
                "artifact_file": os.path.relpath(artifact_path, args.output_dir),
                "n_components_saved": int(artifact.components.shape[0]),
                "dim": int(artifact.dim),
                "n_train_samples": int(artifact.n_train_samples),
                "num_sign_flips": int(artifact.sign_flips.sum()),
            }
        )

    with open(os.path.join(pca_artifacts_dir, "manifest.json"), "w") as f:
        json.dump(pca_manifest, f, indent=2)

    # ---------------------------------------------------------------------
    # Evaluate baseline and sweep
    # ---------------------------------------------------------------------
    baseline = {}
    sweep = {str(k): {} for k in k_values}

    if is_attn:
        for layer in tqdm(layers, desc="Evaluating baseline + k sweep (attn)"):
            state = torch.load(probe_map[layer], map_location="cpu")
            dim = x_val[layer].shape[2]
            probe = LayerProbe(input_dim=dim, pooling_type="attn").to(device)
            probe.load_state_dict(state)
            probe.eval()

            val_logits_base = run_attn_probe_logits(probe, x_val[layer], args.batch_size, device)
            ood_logits_base = run_attn_probe_logits(probe, x_ood[layer], args.batch_size, device)
            val_auc_base, val_acc_base = compute_metrics(y_val, val_logits_base)
            ood_auc_base, ood_acc_base = compute_metrics(y_ood, ood_logits_base)

            baseline[layer] = {
                "id_val_auc": val_auc_base,
                "id_val_acc": val_acc_base,
                "ood_test_auc": ood_auc_base,
                "ood_test_acc": ood_acc_base,
                "generalization_gap": val_auc_base - ood_auc_base,
            }

            pca_art = pca_by_layer[layer]
            for k in k_values:
                x_val_clean = remove_top_k_pcs_tokens(x_val[layer], pca_art.mean, pca_art.components, k)
                x_ood_clean = remove_top_k_pcs_tokens(x_ood[layer], pca_art.mean, pca_art.components, k)

                val_logits = run_attn_probe_logits(probe, x_val_clean, args.batch_size, device)
                ood_logits = run_attn_probe_logits(probe, x_ood_clean, args.batch_size, device)
                val_auc, val_acc = compute_metrics(y_val, val_logits)
                ood_auc, ood_acc = compute_metrics(y_ood, ood_logits)

                sweep[str(k)][layer] = {
                    "id_val_auc": val_auc,
                    "id_val_acc": val_acc,
                    "ood_test_auc": ood_auc,
                    "ood_test_acc": ood_acc,
                    "generalization_gap": val_auc - ood_auc,
                    "delta_id_auc_vs_baseline": val_auc - val_auc_base,
                    "delta_ood_auc_vs_baseline": ood_auc - ood_auc_base,
                    "delta_gap_vs_baseline": (val_auc - ood_auc) - (val_auc_base - ood_auc_base),
                }
    else:
        probe_params = {}
        for layer in layers:
            state = torch.load(probe_map[layer], map_location="cpu")
            w, b = extract_classifier_params(state)
            probe_params[layer] = (w, b)

        for layer in tqdm(layers, desc="Evaluating baseline + k sweep"):
            w, b = probe_params[layer]
            dim = x_val[layer].shape[1]
            if w.shape[0] != dim:
                raise ValueError(
                    f"Probe dim mismatch at layer {layer}: probe={w.shape[0]} vs features={dim}"
                )

            # Baseline (k=0)
            val_logits_base = x_val[layer] @ w + b
            ood_logits_base = x_ood[layer] @ w + b
            val_auc_base, val_acc_base = compute_metrics(y_val, val_logits_base)
            ood_auc_base, ood_acc_base = compute_metrics(y_ood, ood_logits_base)

            baseline[layer] = {
                "id_val_auc": val_auc_base,
                "id_val_acc": val_acc_base,
                "ood_test_auc": ood_auc_base,
                "ood_test_acc": ood_acc_base,
                "generalization_gap": val_auc_base - ood_auc_base,
            }

            # k sweep
            pca_art = pca_by_layer[layer]
            for k in k_values:
                x_val_clean = remove_top_k_pcs(x_val[layer], pca_art.mean, pca_art.components, k)
                x_ood_clean = remove_top_k_pcs(x_ood[layer], pca_art.mean, pca_art.components, k)

                val_logits = x_val_clean @ w + b
                ood_logits = x_ood_clean @ w + b
                val_auc, val_acc = compute_metrics(y_val, val_logits)
                ood_auc, ood_acc = compute_metrics(y_ood, ood_logits)

                sweep[str(k)][layer] = {
                    "id_val_auc": val_auc,
                    "id_val_acc": val_acc,
                    "ood_test_auc": ood_auc,
                    "ood_test_acc": ood_acc,
                    "generalization_gap": val_auc - ood_auc,
                    "delta_id_auc_vs_baseline": val_auc - val_auc_base,
                    "delta_ood_auc_vs_baseline": ood_auc - ood_auc_base,
                    "delta_gap_vs_baseline": (val_auc - ood_auc) - (val_auc_base - ood_auc_base),
                }

    # ---------------------------------------------------------------------
    # Summaries + plots
    # ---------------------------------------------------------------------
    baseline_mean_id = float(np.mean([baseline[l]["id_val_auc"] for l in layers]))
    baseline_mean_ood = float(np.mean([baseline[l]["ood_test_auc"] for l in layers]))

    # Build delta matrices (rows=k, cols=layer)
    delta_id_matrix = np.zeros((len(k_values), len(layers)), dtype=np.float32)
    delta_ood_matrix = np.zeros((len(k_values), len(layers)), dtype=np.float32)
    mean_id_auc_by_k = []
    mean_ood_auc_by_k = []

    summary_rows = []
    for i, k in enumerate(k_values):
        per_layer = sweep[str(k)]
        id_aucs = [per_layer[l]["id_val_auc"] for l in layers]
        ood_aucs = [per_layer[l]["ood_test_auc"] for l in layers]
        mean_id_auc_by_k.append(float(np.mean(id_aucs)))
        mean_ood_auc_by_k.append(float(np.mean(ood_aucs)))

        best_layer_id = max(layers, key=lambda l: per_layer[l]["id_val_auc"])
        best_layer_ood = max(layers, key=lambda l: per_layer[l]["ood_test_auc"])
        summary_rows.append(
            {
                "k": k,
                "best_layer_id": best_layer_id,
                "best_id_val_auc": per_layer[best_layer_id]["id_val_auc"],
                "ood_at_best_id_layer": per_layer[best_layer_id]["ood_test_auc"],
                "best_layer_ood": best_layer_ood,
                "best_ood_test_auc": per_layer[best_layer_ood]["ood_test_auc"],
                "id_at_best_ood_layer": per_layer[best_layer_ood]["id_val_auc"],
                "mean_id_val_auc": float(np.mean(id_aucs)),
                "mean_ood_test_auc": float(np.mean(ood_aucs)),
            }
        )

        for j, layer in enumerate(layers):
            delta_id_matrix[i, j] = per_layer[layer]["delta_id_auc_vs_baseline"]
            delta_ood_matrix[i, j] = per_layer[layer]["delta_ood_auc_vs_baseline"]

    plot_delta_heatmap(
        delta_id_matrix,
        layers=layers,
        k_values=k_values,
        title="ID Validation AUC Delta vs Baseline",
        out_path=os.path.join(args.output_dir, "delta_heatmap_id.png"),
    )
    plot_delta_heatmap(
        delta_ood_matrix,
        layers=layers,
        k_values=k_values,
        title="OOD Test AUC Delta vs Baseline",
        out_path=os.path.join(args.output_dir, "delta_heatmap_ood.png"),
    )
    plot_tradeoff(
        k_values=k_values,
        mean_id_auc=mean_id_auc_by_k,
        mean_ood_auc=mean_ood_auc_by_k,
        baseline_id_auc=baseline_mean_id,
        baseline_ood_auc=baseline_mean_ood,
        out_path=os.path.join(args.output_dir, "id_vs_ood_tradeoff.png"),
    )

    summary_csv_path = os.path.join(args.output_dir, "summary_best_layers.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    pca_stats = {}
    for layer in layers:
        art = pca_by_layer[layer]
        pca_stats[layer] = {
            "explained_variance_ratio": art.explained_variance_ratio.tolist(),
            "explained_variance_ratio_cumsum": np.cumsum(art.explained_variance_ratio).tolist(),
            "num_sign_flips": int(art.sign_flips.sum()),
        }

    with open(os.path.join(args.output_dir, "pca_stats_layerwise.json"), "w") as f:
        json.dump(pca_stats, f, indent=2)

    config = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "pca_fit_level": pca_fit_level,
        "evaluation_mode": evaluation_mode,
        "id_train_activations_dir": args.id_train_activations_dir,
        "id_val_activations_dir": args.id_val_activations_dir,
        "ood_test_activations_dir": args.ood_test_activations_dir,
        "probes_dir": args.probes_dir,
        "layers": layers,
        "k_values": k_values,
        "seed": args.seed,
        "pca_components_to_save": args.pca_components_to_save,
        "pca_solver": args.pca_solver,
        "pc_sign_rule": args.pc_sign_rule,
        "max_train_samples_for_pca": args.max_train_samples_for_pca,
        "batch_size": args.batch_size,
        "device": str(device),
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.output_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    with open(os.path.join(args.output_dir, "pca_ablation_results.json"), "w") as f:
        json.dump({"baseline": baseline, "sweep": sweep}, f, indent=2)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Baseline mean ID AUC:  {baseline_mean_id:.4f}")
    print(f"Baseline mean OOD AUC: {baseline_mean_ood:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
