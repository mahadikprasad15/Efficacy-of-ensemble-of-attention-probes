#!/usr/bin/env python3
"""
Utilities for orthogonal probe training with leakage-safe projection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


def project_batch(x: torch.Tensor, q_basis: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Project activations into complement of span(Q).

    Args:
      x:
        - (B, D) for final-token features
        - (B, T, D) for token-level features
      q_basis: (D, K) orthonormal basis, or None/empty.
    """
    if q_basis is None or q_basis.numel() == 0:
        return x

    if x.dim() == 2:
        coeff = x @ q_basis
        return x - (coeff @ q_basis.T)

    if x.dim() == 3:
        bsz, tlen, dim = x.shape
        flat = x.reshape(bsz * tlen, dim)
        coeff = flat @ q_basis
        flat_proj = flat - (coeff @ q_basis.T)
        return flat_proj.reshape(bsz, tlen, dim)

    raise ValueError(f"Unsupported batch shape for projection: {tuple(x.shape)}")


def orthogonalize_vector(
    w: torch.Tensor,
    q_basis: Optional[torch.Tensor],
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    Remove components of w in span(Q), return residual direction and its norm.
    """
    if q_basis is None or q_basis.numel() == 0:
        norm = torch.linalg.vector_norm(w).item()
        return w, norm
    w_res = w - q_basis @ (q_basis.T @ w)
    norm = torch.linalg.vector_norm(w_res).item()
    if norm < tol:
        return torch.zeros_like(w), norm
    return w_res, norm


def update_q_basis(
    q_basis: Optional[torch.Tensor],
    w: torch.Tensor,
    tol: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Append normalized residual of w to orthonormal basis Q.
    """
    w_res, norm = orthogonalize_vector(w, q_basis=q_basis, tol=tol)
    if norm < tol:
        return q_basis, norm

    q_new = (w_res / norm).reshape(-1, 1)
    if q_basis is None or q_basis.numel() == 0:
        return q_new, norm
    return torch.cat([q_basis, q_new], dim=1), norm


def max_cos_to_previous(
    w: torch.Tensor,
    previous_w: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> float:
    if not previous_w:
        return 0.0
    wn = torch.linalg.vector_norm(w).item()
    if wn < eps:
        return 0.0
    vals: List[float] = []
    for w_prev in previous_w:
        pn = torch.linalg.vector_norm(w_prev).item()
        if pn < eps:
            continue
        cos = float(torch.dot(w, w_prev).item() / (wn * pn))
        vals.append(abs(cos))
    return float(max(vals)) if vals else 0.0


def evaluate_probe_with_projection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    q_basis: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    probs: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            xb = project_batch(xb, q_basis)
            logits = model(xb)
            p = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(yb.detach().cpu().numpy().astype(np.int64).tolist())

    labels_arr = np.asarray(labels, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float32)
    preds = (probs_arr >= 0.5).astype(np.int64)

    return {
        "auc": safe_auc(labels_arr, probs_arr),
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "count": int(labels_arr.size),
    }


def train_probe_with_projection(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    q_basis: Optional[torch.Tensor],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_epoch = 0
    patience_count = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            xb = project_batch(xb, q_basis)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_probe_with_projection(
            model=model,
            loader=val_loader,
            device=device,
            q_basis=q_basis,
        )
        val_auc = float(val_metrics["auc"])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_probe_with_projection(
        model=model,
        loader=val_loader,
        device=device,
        q_basis=q_basis,
    )
    final_val["best_epoch"] = int(best_epoch)
    return model, final_val
