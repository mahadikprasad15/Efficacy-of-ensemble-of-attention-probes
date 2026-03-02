#!/usr/bin/env python3
"""
Top-10 Mahalanobis alignment analysis for probes and activation means.

Pipeline:
1) Load two top-20 tables (CSV, or OCR images if CSVs are not provided)
2) Select top-5 by Best AUC from each table => top-10 rows
3) Build 10 probe definitions from source rows
4) Evaluate all 10 probes on all unique target test segments from top-10
5) Build score matrices:
   - probe x unique_target
   - probe x top10_target_row (10x10, with repeated target segments if present)
6) Compute 10x10 probe Mahalanobis-cosine angle matrix with shared Σ from pooled target-test activations
7) Compute 10x10 activation-mean Mahalanobis-cosine angle matrix with shared Σ from pooled source-train activations
8) Correlate matrices and save plots/artifacts

Artifacts are written under:
  artifacts/runs/mahalanobis_probe_alignment/<model_dir>/top10/v1/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402


REQUIRED_COLUMNS = [
    "Source Probe",
    "Target Dataset (Test)",
    "Best Pooling",
    "Best Layer",
    "Best AUC",
    "Best Accuracy",
]

POOLING_MAP = {
    "mean": "mean",
    "max": "max",
    "last": "last",
    "attn": "attn",
    "attention": "attn",
}


@dataclass
class ProbeRow:
    top10_index: int
    source_table: str
    source_probe: str
    source_dataset: str
    source_segment: str
    target_label: str
    target_dataset: str
    pooling: str
    layer: int
    best_auc: float
    best_accuracy: float
    probe_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{ts}-{token}"


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Optional[Dict] = None) -> Dict:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_status(status_path: Path, state: str, message: str) -> None:
    payload = read_json(status_path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now_iso()
    write_json(status_path, payload)


def normalize_pooling(value: str) -> str:
    x = str(value).strip().lower()
    if x not in POOLING_MAP:
        raise ValueError(f"Unsupported pooling value: {value}")
    return POOLING_MAP[x]


def parse_source_probe(source_probe: str) -> Tuple[str, str]:
    s = source_probe.strip()
    if s.startswith("Roleplaying "):
        seg = s[len("Roleplaying ") :].strip().lower()
        return "Deception-Roleplaying", seg
    if s.startswith("AI Liar "):
        seg = s[len("AI Liar ") :].strip().lower()
        return "Deception-AILiar", seg
    raise ValueError(f"Unsupported source probe format: {source_probe}")


def parse_target_dataset(label: str) -> str:
    v = label.strip().lower()
    if "insidertrading" not in v:
        raise ValueError(f"Only InsiderTrading targets supported, got: {label}")
    if "completion" in v:
        return "Deception-InsiderTrading-completion"
    if "system" in v:
        return "Deception-InsiderTrading-system"
    if "user" in v:
        return "Deception-InsiderTrading-user"
    if "prompt" in v:
        return "Deception-InsiderTrading-prompt"
    if "full" in v:
        return "Deception-InsiderTrading"
    compact = v.replace(" ", "")
    if compact in {"insidertrading", "deception-insidertrading"}:
        return "Deception-InsiderTrading"
    raise ValueError(f"Cannot parse target dataset from: {label}")


def validate_top20_df(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def try_ocr_to_dataframe(image_path: Path) -> pd.DataFrame:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OCR requested but dependencies are missing. Install with: pip install pytesseract pillow "
            "and ensure tesseract binary is available."
        ) from exc

    img = Image.open(str(image_path))
    text = pytesseract.image_to_string(img)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    parsed: List[Dict[str, str]] = []
    # Loose parser for rows like:
    # "0 Roleplaying User InsiderTrading Completion Attn 12 0.552188 0.505714"
    pattern = re.compile(
        r"^\d+\s+(Roleplaying|AI\s+Liar)\s+(\w+)\s+InsiderTrading\s+(\w+)\s+(\w+)\s+(\d+)\s+([0-9]*\.[0-9]+)\s+([0-9]*\.[0-9]+)$",
        re.IGNORECASE,
    )
    for ln in lines:
        m = pattern.match(ln)
        if not m:
            continue
        source_prefix = "AI Liar" if "ai" in m.group(1).lower() else "Roleplaying"
        source_seg = m.group(2).capitalize()
        tgt_seg = m.group(3).capitalize()
        pooling = m.group(4).capitalize()
        layer = m.group(5)
        auc = m.group(6)
        acc = m.group(7)
        parsed.append(
            {
                "Source Probe": f"{source_prefix} {source_seg}",
                "Target Dataset (Test)": f"InsiderTrading {tgt_seg}",
                "Best Pooling": pooling,
                "Best Layer": layer,
                "Best AUC": auc,
                "Best Accuracy": acc,
            }
        )

    if not parsed:
        raise RuntimeError(f"OCR parse produced no rows from: {image_path}")
    return pd.DataFrame(parsed)


def load_table(
    csv_path: Optional[Path],
    image_path: Optional[Path],
    table_name: str,
    inputs_dir: Path,
) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        validate_top20_df(df, table_name)
        return df
    if image_path is None:
        raise ValueError(f"{table_name}: must provide either CSV or image path")
    df = try_ocr_to_dataframe(image_path)
    validate_top20_df(df, f"{table_name} (OCR)")
    out_csv = inputs_dir / f"{table_name}_ocr.csv"
    df.to_csv(out_csv, index=False)
    return df


def select_top10(df_role: pd.DataFrame, df_ai: pd.DataFrame) -> pd.DataFrame:
    role = df_role.copy()
    ai = df_ai.copy()
    role["Best AUC"] = pd.to_numeric(role["Best AUC"], errors="coerce")
    role["Best Accuracy"] = pd.to_numeric(role["Best Accuracy"], errors="coerce")
    role["Best Layer"] = pd.to_numeric(role["Best Layer"], errors="coerce")
    ai["Best AUC"] = pd.to_numeric(ai["Best AUC"], errors="coerce")
    ai["Best Accuracy"] = pd.to_numeric(ai["Best Accuracy"], errors="coerce")
    ai["Best Layer"] = pd.to_numeric(ai["Best Layer"], errors="coerce")

    role = role.dropna(subset=["Best AUC", "Best Accuracy", "Best Layer"])
    ai = ai.dropna(subset=["Best AUC", "Best Accuracy", "Best Layer"])
    role["Best Layer"] = role["Best Layer"].astype(int)
    ai["Best Layer"] = ai["Best Layer"].astype(int)

    role = role.sort_values(["Best AUC", "Best Accuracy"], ascending=[False, False]).head(5).copy()
    ai = ai.sort_values(["Best AUC", "Best Accuracy"], ascending=[False, False]).head(5).copy()
    role["source_table"] = "roleplaying_top20"
    ai["source_table"] = "ailiar_top20"

    top10 = pd.concat([role, ai], ignore_index=True)
    top10["top10_index"] = np.arange(len(top10))
    return top10


def find_split_dir(base: Path, model_dir: str, dataset: str, split: str) -> Path:
    return base / model_dir / dataset / split


def ensure_split(base: Path, model_dir: str, dataset: str, split: str, fallback: Optional[str] = None) -> Path:
    primary = find_split_dir(base, model_dir, dataset, split)
    if primary.exists():
        return primary
    if fallback:
        alt = find_split_dir(base, model_dir, dataset, fallback)
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Missing split dir: {primary} (fallback={fallback})")


def infer_input_dim_from_probe(probe_path: Path) -> int:
    state = torch.load(str(probe_path), map_location="cpu")
    if "classifier.weight" not in state:
        raise ValueError(f"classifier.weight missing in {probe_path}")
    return int(state["classifier.weight"].shape[1])


def resolve_probe_path(
    probes_root: Path,
    model_dir: str,
    source_base: str,
    source_segment: str,
    pooling: str,
    layer: int,
) -> Path:
    # Build candidate dataset dirs under <base>_slices.
    # Segment "full" may be stored as base (no suffix) or base-full.
    candidates: List[str] = []
    if source_segment == "full":
        candidates = [source_base, f"{source_base}-full"]
    else:
        candidates = [f"{source_base}-{source_segment}"]

    tried: List[Path] = []
    for dataset_dir in candidates:
        path = (
            probes_root
            / model_dir
            / f"{source_base}_slices"
            / dataset_dir
            / pooling
            / f"probe_layer_{layer}.pt"
        )
        tried.append(path)
        if path.exists():
            return path

    tried_str = "\n".join(str(p) for p in tried)
    raise FileNotFoundError(f"Probe not found. Tried:\n{tried_str}")


def load_label_map(split_dir: Path) -> Dict[str, int]:
    manifest_path = split_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.jsonl missing: {manifest_path}")

    label_map: Dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            label_map[item["id"]] = int(item.get("label", -1))
    return label_map


def iter_labeled_activations(split_dir: Path) -> Iterable[Tuple[torch.Tensor, int]]:
    label_map = load_label_map(split_dir)
    shard_paths = sorted(split_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {split_dir}")

    yielded = 0
    for shard in shard_paths:
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


class ProbeHandle:
    def __init__(self, pooling: str, layer: int, path: Path, input_dim: int, device: torch.device):
        self.pooling = pooling
        self.layer = int(layer)
        self.path = path
        self.input_dim = input_dim
        self.device = device
        self.model = self._load_model()
        self.weight = self._extract_classifier_weight()

    def _load_model(self) -> nn.Module:
        state = torch.load(str(self.path), map_location=self.device)
        if self.pooling == "attn" and any(k.startswith("attn.") for k in state.keys()):
            model: nn.Module = AttentionLinearProbeCompat(self.input_dim).to(self.device)
        else:
            model = LayerProbe(input_dim=self.input_dim, pooling_type=self.pooling).to(self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def _extract_classifier_weight(self) -> np.ndarray:
        state = torch.load(str(self.path), map_location="cpu")
        w = state["classifier.weight"].detach().cpu().numpy().reshape(-1)
        return w.astype(np.float64)

    def _x_layer(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor[self.layer, :, :].float()
        if tensor.dim() == 2:
            return tensor[self.layer, :].float().unsqueeze(0)
        raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

    def pooled_vector(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self._x_layer(tensor).to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            if isinstance(self.model, AttentionLinearProbeCompat):
                weights = torch.softmax(self.model.attn(x), dim=1)
                pooled = (x * weights).sum(dim=1).squeeze(0)
            else:
                # LayerProbe pooling returns (B, D)
                pooled = self.model.pooling(x).squeeze(0)
        return pooled.detach().cpu()

    def predict_logit(self, tensor: torch.Tensor) -> float:
        x = self._x_layer(tensor).to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x).reshape(-1)[0].detach().cpu().item()
        return float(logit)


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


def matrix_offdiag_values(mat_a: np.ndarray, mat_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mat_a.shape != mat_b.shape:
        raise ValueError(f"Matrix shape mismatch: {mat_a.shape} vs {mat_b.shape}")
    mask = ~np.eye(mat_a.shape[0], dtype=bool)
    return mat_a[mask].reshape(-1), mat_b[mask].reshape(-1)


def compute_auc(labels: Sequence[int], logits: Sequence[float]) -> float:
    y = np.asarray(labels)
    s = np.asarray(logits)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return 0.5


def save_matrix_csv(path: Path, mat: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str]) -> None:
    df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
    df.to_csv(path)


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, vmin: float = -1.0, vmax: float = 1.0) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.7)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        plt.plot(xs, m * xs + b, color="black", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-10 Mahalanobis alignment analysis")
    p.add_argument("--roleplaying_csv", type=str, default=None)
    p.add_argument("--ailiar_csv", type=str, default=None)
    p.add_argument("--roleplaying_image", type=str, default=None)
    p.add_argument("--ailiar_image", type=str, default=None)
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
    p.add_argument("--target_split", type=str, default="test")
    p.add_argument("--target_split_fallback", type=str, default="validation")
    p.add_argument("--source_split", type=str, default="train")
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cov_eps", type=float, default=1e-5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--progress_every", type=int, default=200,
                   help="Print progress every N samples (0 to disable).")
    p.add_argument("--no_tqdm", action="store_true",
                   help="Disable tqdm progress bars.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    run_root = (
        Path(args.artifact_root)
        / "runs"
        / "mahalanobis_probe_alignment"
        / model_dir
        / "top10"
        / "v1"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"

    for d in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "run_manifest.json"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    progress = read_json(progress_path, default={"completed_steps": []})

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "activations_root": args.activations_root,
                "probes_root": args.probes_root,
                "run_root": str(run_root),
            },
            "splits": {
                "source_split": args.source_split,
                "target_split": args.target_split,
                "target_split_fallback": args.target_split_fallback,
            },
        },
    )
    update_status(status_path, "running", "starting")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[start] run_id={run_id}")
    print(f"[start] model={args.model} device={device}")
    print(f"[start] activations_root={args.activations_root}")
    print(f"[start] probes_root={args.probes_root}")

    def done(step: str) -> bool:
        return bool(args.resume and step in progress.get("completed_steps", []))

    def mark(step: str) -> None:
        steps = set(progress.get("completed_steps", []))
        steps.add(step)
        progress["completed_steps"] = sorted(steps)
        progress["updated_at"] = utc_now_iso()
        write_json(progress_path, progress)

    # Step 1: load tables + top10
    update_status(status_path, "running", "loading top20 tables")
    print("[step] loading top20 tables")
    top10_csv_path = results_dir / "top10_rows.csv"
    if done("top10_selection") and top10_csv_path.exists():
        print(f"[resume] top10 selection found at {top10_csv_path}")
        top10_df = pd.read_csv(top10_csv_path)
    else:
        df_role = load_table(
            Path(args.roleplaying_csv) if args.roleplaying_csv else None,
            Path(args.roleplaying_image) if args.roleplaying_image else None,
            "roleplaying_top20",
            inputs_dir,
        )
        df_ai = load_table(
            Path(args.ailiar_csv) if args.ailiar_csv else None,
            Path(args.ailiar_image) if args.ailiar_image else None,
            "ailiar_top20",
            inputs_dir,
        )
        top10_df = select_top10(df_role, df_ai)
        top10_df.to_csv(top10_csv_path, index=False)
        mark("top10_selection")
        print(f"[done] top10 selection -> {top10_csv_path}")

    # Step 2: build probe rows
    update_status(status_path, "running", "resolving top10 probes")
    print("[step] resolving probe paths")
    probe_rows: List[ProbeRow] = []
    for _, row in top10_df.iterrows():
        source_base, source_segment = parse_source_probe(str(row["Source Probe"]))
        source_dataset = source_base if source_segment == "full" else f"{source_base}-{source_segment}"
        target_dataset = parse_target_dataset(str(row["Target Dataset (Test)"]))
        pooling = normalize_pooling(str(row["Best Pooling"]))
        layer = int(row["Best Layer"])
        probe_path = resolve_probe_path(
            probes_root=Path(args.probes_root),
            model_dir=model_dir,
            source_base=source_base,
            source_segment=source_segment,
            pooling=pooling,
            layer=layer,
        )
        probe_rows.append(
            ProbeRow(
                top10_index=int(row["top10_index"]),
                source_table=str(row["source_table"]),
                source_probe=str(row["Source Probe"]),
                source_dataset=source_dataset,
                source_segment=source_segment,
                target_label=str(row["Target Dataset (Test)"]),
                target_dataset=target_dataset,
                pooling=pooling,
                layer=layer,
                best_auc=float(row["Best AUC"]),
                best_accuracy=float(row["Best Accuracy"]),
                probe_path=probe_path,
            )
        )
    if len(probe_rows) != 10:
        raise RuntimeError(f"Expected 10 probe rows, got {len(probe_rows)}")
    print("[done] resolved 10 probe rows")

    # Step 3: instantiate probes
    update_status(status_path, "running", "loading probe checkpoints")
    print("[step] loading probe checkpoints")
    probe_handles: List[ProbeHandle] = []
    for pr in probe_rows:
        input_dim = infer_input_dim_from_probe(pr.probe_path)
        probe_handles.append(
            ProbeHandle(
                pooling=pr.pooling,
                layer=pr.layer,
                path=pr.probe_path,
                input_dim=input_dim,
                device=device,
            )
        )
    d_model = probe_handles[0].input_dim
    print(f"[done] loaded probes (input_dim={d_model})")

    # Step 4: evaluate all probes on unique targets + stream covariance for target-test baseline
    update_status(status_path, "running", "evaluating probes on unique top10 targets")
    print("[step] evaluating probes on unique targets")
    unique_targets = sorted({r.target_dataset for r in probe_rows})
    print(f"[info] unique targets: {unique_targets}")

    probe_labels = [f"p{i}:{r.source_probe}|{r.pooling}|L{r.layer}" for i, r in enumerate(probe_rows)]
    target_labels = [t.replace("Deception-", "") for t in unique_targets]

    score_unique = np.zeros((len(probe_rows), len(unique_targets)), dtype=np.float64)
    cov_probe_path = checkpoints_dir / "cov_target_test.npy"
    need_cov_probe = not (done("cov_target_test") and cov_probe_path.exists())
    acc_target = CovAccumulator(d_model) if need_cov_probe else None

    use_tqdm = not args.no_tqdm
    for j, tgt in enumerate(unique_targets):
        split_dir = ensure_split(
            Path(args.activations_root),
            model_dir,
            tgt,
            args.target_split,
            fallback=args.target_split_fallback,
        )
        print(f"[targets] {tgt} split_dir={split_dir}")

        per_probe_logits: List[List[float]] = [[] for _ in range(len(probe_handles))]
        labels: List[int] = []
        sample_count = 0
        t0 = time.time()
        iterator = iter_labeled_activations(split_dir)
        if use_tqdm:
            iterator = tqdm(iterator, desc=f"targets:{tgt}", unit="sample")
        for tensor, y in iterator:
            labels.append(y)
            pooled_rows = []
            for i, ph in enumerate(probe_handles):
                per_probe_logits[i].append(ph.predict_logit(tensor))
                if acc_target is not None:
                    pooled_rows.append(ph.pooled_vector(tensor).numpy().astype(np.float64))
            if acc_target is not None and pooled_rows:
                acc_target.update(np.stack(pooled_rows, axis=0))
            sample_count += 1
            if args.progress_every and sample_count % args.progress_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate = sample_count / elapsed
                print(f"[targets] {tgt} processed {sample_count} samples ({rate:.2f} samples/s)")

        for i in range(len(probe_handles)):
            score_unique[i, j] = compute_auc(labels, per_probe_logits[i])
        elapsed = max(time.time() - t0, 1e-6)
        rate = sample_count / elapsed
        print(f"[targets] {tgt} done: {sample_count} samples ({rate:.2f} samples/s)")

        # Save partial score matrices after each target segment
        np.save(results_dir / "score_matrix_unique.partial.npy", score_unique)
        save_matrix_csv(results_dir / "score_matrix_unique.partial.csv", score_unique, probe_labels, target_labels)
        score_10x10_partial = np.zeros((10, 10), dtype=np.float64)
        for jj, row in enumerate(probe_rows):
            t = row.target_dataset
            if t in target_to_col:
                score_10x10_partial[:, jj] = score_unique[:, target_to_col[t]]
        np.save(results_dir / "score_matrix_10x10.partial.npy", score_10x10_partial)
        save_matrix_csv(results_dir / "score_matrix_10x10.partial.csv", score_10x10_partial, probe_labels, col_labels_10)
        update_status(status_path, "running", f"finished target {tgt}")

    np.save(results_dir / "score_matrix_unique.npy", score_unique)
    save_matrix_csv(results_dir / "score_matrix_unique.csv", score_unique, probe_labels, target_labels)
    print("[done] score matrices (unique targets)")

    # 10x10 matrix aligned to top10 target columns for correlation with 10x10 angles
    target_to_col = {t: idx for idx, t in enumerate(unique_targets)}
    score_10x10 = np.zeros((10, 10), dtype=np.float64)
    col_labels_10 = []
    for j, row in enumerate(probe_rows):
        t = row.target_dataset
        col_labels_10.append(f"t{j}:{t.replace('Deception-', '')}")
        score_10x10[:, j] = score_unique[:, target_to_col[t]]
    np.save(results_dir / "score_matrix_10x10.npy", score_10x10)
    save_matrix_csv(results_dir / "score_matrix_10x10.csv", score_10x10, probe_labels, col_labels_10)
    print("[done] score matrix 10x10")

    # Step 5: shared Σ for probe angles from pooled target-test activations
    update_status(status_path, "running", "computing shared target-test covariance for probe angles")
    print("[step] computing target-test covariance for probe angles")
    if done("cov_target_test") and cov_probe_path.exists():
        print(f"[resume] using cached target covariance {cov_probe_path}")
        sigma_probe = np.load(cov_probe_path)
    else:
        if acc_target is None:
            raise RuntimeError("Target covariance accumulator is unexpectedly missing.")
        sigma_probe = acc_target.finalize(eps=args.cov_eps)
        np.save(cov_probe_path, sigma_probe)
        mark("cov_target_test")
        print(f"[done] target-test covariance saved -> {cov_probe_path}")

    # Step 6: probe angle matrix
    update_status(status_path, "running", "computing probe angle matrix")
    print("[step] computing probe angle matrix")
    probe_angle = np.zeros((10, 10), dtype=np.float64)
    for i in range(10):
        wi = probe_handles[i].weight
        for j in range(10):
            wj = probe_handles[j].weight
            probe_angle[i, j] = mahalanobis_cosine(wi, wj, sigma_probe)
    np.save(results_dir / "probe_angle_matrix.npy", probe_angle)
    save_matrix_csv(results_dir / "probe_angle_matrix.csv", probe_angle, probe_labels, probe_labels)
    print("[done] probe angle matrix")

    # Step 7: source-train means + Σ for activation means
    update_status(status_path, "running", "computing source-train activation means and covariance")
    print("[step] computing source-train means and covariance")
    cov_source_path = checkpoints_dir / "cov_source_train.npy"
    means_path = checkpoints_dir / "source_means.npy"
    if done("source_means_cov") and cov_source_path.exists() and means_path.exists():
        print(f"[resume] using cached source means/cov at {cov_source_path}")
        sigma_source = np.load(cov_source_path)
        source_means = np.load(means_path)
    else:
        acc_source = CovAccumulator(d_model)
        source_sums = np.zeros((10, d_model), dtype=np.float64)
        source_counts = np.zeros(10, dtype=np.int64)

        source_to_indices: Dict[str, List[int]] = {}
        for i, pr in enumerate(probe_rows):
            source_to_indices.setdefault(pr.source_dataset, []).append(i)

        for src_dataset, indices in source_to_indices.items():
            split_dir = ensure_split(Path(args.activations_root), model_dir, src_dataset, args.source_split)
            print(f"[sources] {src_dataset} split_dir={split_dir}")
            sample_count = 0
            iterator = iter_labeled_activations(split_dir)
            if use_tqdm:
                iterator = tqdm(iterator, desc=f"sources:{src_dataset}", unit="sample")
            for tensor, _ in iterator:
                pooled_rows = []
                for idx in indices:
                    pooled = probe_handles[idx].pooled_vector(tensor).numpy().astype(np.float64)
                    source_sums[idx] += pooled
                    source_counts[idx] += 1
                    pooled_rows.append(pooled)
                if pooled_rows:
                    acc_source.update(np.stack(pooled_rows, axis=0))
                sample_count += 1
                if args.progress_every and sample_count % args.progress_every == 0:
                    print(f"[sources] {src_dataset} processed {sample_count} samples")

        if np.any(source_counts == 0):
            bad = np.where(source_counts == 0)[0].tolist()
            raise RuntimeError(f"No source-train samples found for top10 indices: {bad}")
        source_means = source_sums / source_counts[:, None]
        sigma_source = acc_source.finalize(eps=args.cov_eps)
        np.save(cov_source_path, sigma_source)
        np.save(means_path, source_means)
        mark("source_means_cov")
        print(f"[done] source-train covariance saved -> {cov_source_path}")

    # Step 8: activation-mean angle matrix
    update_status(status_path, "running", "computing activation mean angle matrix")
    print("[step] computing activation mean angle matrix")
    act_angle = np.zeros((10, 10), dtype=np.float64)
    for i in range(10):
        vi = source_means[i]
        for j in range(10):
            vj = source_means[j]
            act_angle[i, j] = mahalanobis_cosine(vi, vj, sigma_source)
    np.save(results_dir / "activation_mean_angle_matrix.npy", act_angle)
    save_matrix_csv(results_dir / "activation_mean_angle_matrix.csv", act_angle, probe_labels, probe_labels)
    print("[done] activation mean angle matrix")

    # Step 9: correlations + plots
    update_status(status_path, "running", "correlating matrices and plotting")
    print("[step] correlating matrices and plotting")
    x_pa, y_sc = matrix_offdiag_values(probe_angle, score_10x10)
    x_aa, y_pa = matrix_offdiag_values(act_angle, probe_angle)
    x_aa2, y_sc2 = matrix_offdiag_values(act_angle, score_10x10)

    correlations = {
        "probe_angle_vs_score_10x10": {
            "pearson_r": float(pearsonr(x_pa, y_sc)[0]),
            "pearson_p": float(pearsonr(x_pa, y_sc)[1]),
            "spearman_rho": float(spearmanr(x_pa, y_sc)[0]),
            "spearman_p": float(spearmanr(x_pa, y_sc)[1]),
        },
        "activation_angle_vs_probe_angle": {
            "pearson_r": float(pearsonr(x_aa, y_pa)[0]),
            "pearson_p": float(pearsonr(x_aa, y_pa)[1]),
            "spearman_rho": float(spearmanr(x_aa, y_pa)[0]),
            "spearman_p": float(spearmanr(x_aa, y_pa)[1]),
        },
        "activation_angle_vs_score_10x10": {
            "pearson_r": float(pearsonr(x_aa2, y_sc2)[0]),
            "pearson_p": float(pearsonr(x_aa2, y_sc2)[1]),
            "spearman_rho": float(spearmanr(x_aa2, y_sc2)[0]),
            "spearman_p": float(spearmanr(x_aa2, y_sc2)[1]),
        },
    }
    write_json(results_dir / "correlations.json", correlations)
    print("[done] correlations")

    plot_heatmap(results_dir / "plots" / "probe_angle_matrix.png", probe_angle, "Probe Mahalanobis Cosine (10x10)")
    plot_heatmap(results_dir / "plots" / "score_matrix_10x10.png", score_10x10, "Score Matrix AUC (10x10)", vmin=0.0, vmax=1.0)
    plot_heatmap(
        results_dir / "plots" / "activation_mean_angle_matrix.png",
        act_angle,
        "Activation Mean Mahalanobis Cosine (10x10)",
    )
    plot_scatter(
        results_dir / "plots" / "scatter_probe_angle_vs_score.png",
        x_pa,
        y_sc,
        xlabel="Probe Mahalanobis cosine",
        ylabel="AUC score (row-aligned target column)",
        title="Probe angle vs score",
    )
    plot_scatter(
        results_dir / "plots" / "scatter_activation_angle_vs_probe_angle.png",
        x_aa,
        y_pa,
        xlabel="Activation mean Mahalanobis cosine",
        ylabel="Probe Mahalanobis cosine",
        title="Activation angle vs probe angle",
    )

    write_json(
        results_dir / "results.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "model_dir": model_dir,
            "top10_size": 10,
            "unique_target_count": len(unique_targets),
            "unique_targets": unique_targets,
            "probe_labels": probe_labels,
            "target_labels_unique": target_labels,
            "target_labels_top10_columns": col_labels_10,
            "correlations": correlations,
            "notes": {
                "score_matrix_unique_shape": list(score_unique.shape),
                "score_matrix_10x10_shape": list(score_10x10.shape),
                "probe_angle_shape": list(probe_angle.shape),
                "activation_angle_shape": list(act_angle.shape),
                "score_10x10_definition": "score[i,j] = AUC(probe_i on target segment from top10 row j)",
                "cov_probe_definition": "Shared covariance from pooled target-test activations across all unique top10 targets and all top10 probe poolers",
                "cov_activation_definition": "Shared covariance from pooled source-train activations across top10 source segments and row-specific poolers",
            },
        },
    )

    update_status(status_path, "completed", "finished")
    mark("finished")
    print(f"[DONE] Artifacts written to: {run_root}")
    print(f"[DONE] Unique targets in top10: {len(unique_targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
