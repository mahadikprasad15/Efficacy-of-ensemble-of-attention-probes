"""
Train a Layer-Agnostic Probe for Deception Detection.

Key Idea:
    Instead of training 28 separate probes (one per layer), train ONE probe
    on pooled activations from ALL layers, treating each (sample, layer) pair
    as a separate training datapoint.

Algorithm:
    For each sample i with label y_i:
        Load tensor: (28, 64, D)
        For each layer l in 0..27:
            Pool tokens: mean over dim=1 → (D,)
            Create training pair: (pooled_vector_l, y_i)
    
    Train single probe w on all (28 × N_samples) training pairs

Evaluation:
    At eval time, apply same probe to each layer separately and report
    per-layer AUC to see which layers the probe generalizes to.

Usage:
    python scripts/training/train_layer_agnostic_probe.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --activations_dir data/activations \
        --pooling mean \
        --output_dir data/probes_layer_agnostic
"""

import argparse
import os
import sys
import json
import glob
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Utilities
# ============================================================================

def infer_num_layers(activations_dir: str) -> int:
    """Infer number of layers from the first shard in an activations directory."""
    shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
    shards = sorted(glob.glob(shard_pattern))
    if not shards:
        raise FileNotFoundError(f"No shard files found for layer inference: {shard_pattern}")
    tensors = load_file(shards[0])
    if not tensors:
        raise ValueError(f"Empty shard file: {shards[0]}")
    first_tensor = next(iter(tensors.values()))
    return int(first_tensor.shape[0])

# ============================================================================
# Dataset: Load (L, T, D) activations and flatten to (L*N, D) for training
# ============================================================================

class LayerAgnosticDataset(Dataset):
    """
    Loads cached activations and expands them for layer-agnostic training.
    
    Each original sample (L, T, D) becomes L training samples, one per layer.
    Each layer is mean-pooled over tokens to get (D,) vector.
    """
    
    def __init__(self, activations_dir: str, pooling: str = "mean"):
        """
        Args:
            activations_dir: Path to directory with shards and manifest
            pooling: How to pool tokens within each layer ("mean", "max", "last")
        """
        self.pooling = pooling
        self.items = []  # List of (vector: (D,), label: int, layer: int, sample_id: str)
        
        # Find all shard files
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        
        if not shards:
            raise FileNotFoundError(f"No shard files found: {shard_pattern}")
        
        logger.info(f"Loading {len(shards)} shard(s) from {activations_dir}...")
        
        # Load manifest
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                meta = json.loads(line)
                manifest[meta['id']] = meta
        
        logger.info(f"Loaded manifest with {len(manifest)} entries")
        
        # Load tensors and expand
        n_samples = 0
        n_layers = None
        
        for shard_path in tqdm(shards, desc="Loading shards"):
            try:
                tensors = load_file(shard_path)
                
                for eid, tensor in tensors.items():
                    if eid not in manifest:
                        logger.warning(f"ID {eid} in shard but not in manifest, skipping")
                        continue
                    
                    meta = manifest[eid]
                    label = meta.get('label', -1)
                    
                    # Skip unknown labels
                    if label == -1:
                        continue
                    
                    # tensor shape: (L, T, D)
                    L, T, D = tensor.shape
                    if n_layers is None:
                        n_layers = L
                        self.hidden_dim = D
                        logger.info(f"Detected shape: L={L}, T={T}, D={D}")
                    
                    # Pool each layer and create separate training samples
                    for layer_idx in range(L):
                        layer_activations = tensor[layer_idx]  # (T, D)
                        
                        if self.pooling == "mean":
                            pooled = layer_activations.mean(dim=0)  # (D,)
                        elif self.pooling == "max":
                            pooled = layer_activations.max(dim=0).values  # (D,)
                        elif self.pooling == "last":
                            pooled = layer_activations[-1]  # (D,)
                        else:
                            raise ValueError(f"Unknown pooling: {self.pooling}")
                        
                        self.items.append({
                            "vector": pooled,
                            "label": label,
                            "layer": layer_idx,
                            "sample_id": eid
                        })
                    
                    n_samples += 1
                    
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")
        
        self.n_layers = n_layers
        logger.info(f"✓ Loaded {n_samples} samples → {len(self.items)} training pairs ({n_layers} layers × {n_samples} samples)")
        
        # Log label distribution
        labels = [item['label'] for item in self.items]
        honest_count = sum(1 for l in labels if l == 0)
        deceptive_count = sum(1 for l in labels if l == 1)
        logger.info(f"  • Honest pairs: {honest_count} ({100*honest_count/len(labels):.1f}%)")
        logger.info(f"  • Deceptive pairs: {deceptive_count} ({100*deceptive_count/len(labels):.1f}%)")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['vector'].float(), torch.tensor(item['label'], dtype=torch.float32), item['layer'], item['sample_id']


class LayerAgnosticAttnDataset(Dataset):
    """
    Dataset for ATTENTION pooling - returns raw (T, D) tensors.
    Pooling is done by the model (learned).
    """
    
    def __init__(self, activations_dir: str):
        self.items = []  # List of (tensor: (T, D), label: int, layer: int)
        
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        
        if not shards:
            raise FileNotFoundError(f"No shard files found: {shard_pattern}")
        
        logger.info(f"Loading {len(shards)} shard(s) for attention pooling...")
        
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                meta = json.loads(line)
                manifest[meta['id']] = meta
        
        n_samples = 0
        n_layers = None
        
        for shard_path in tqdm(shards, desc="Loading shards"):
            try:
                tensors = load_file(shard_path)
                
                for eid, tensor in tensors.items():
                    if eid not in manifest:
                        continue
                    
                    meta = manifest[eid]
                    label = meta.get('label', -1)
                    
                    if label == -1:
                        continue
                    
                    L, T, D = tensor.shape
                    if n_layers is None:
                        n_layers = L
                        self.hidden_dim = D
                        self.seq_len = T
                        logger.info(f"Detected shape: L={L}, T={T}, D={D}")
                    
                    # Store raw tensors for each layer (no pre-pooling)
                    for layer_idx in range(L):
                        self.items.append({
                            "tensor": tensor[layer_idx],  # (T, D)
                            "label": label,
                            "layer": layer_idx,
                            "sample_id": eid
                        })
                    
                    n_samples += 1
                    
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")
        
        self.n_layers = n_layers
        logger.info(f"✓ Loaded {n_samples} samples → {len(self.items)} training pairs")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32), item['layer'], item['sample_id']


class PerLayerEvalAttnDataset(Dataset):
    """
    Per-layer eval dataset for attention - returns raw (T, D) tensors.
    """
    
    def __init__(self, activations_dir: str, layer_idx: int):
        self.layer_idx = layer_idx
        self.items = []
        
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        
        if not shards:
            raise FileNotFoundError(f"No shard files found: {shard_pattern}")
        
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                meta = json.loads(line)
                manifest[meta['id']] = meta
        
        for shard_path in shards:
            try:
                tensors = load_file(shard_path)
                
                for eid, tensor in tensors.items():
                    if eid not in manifest:
                        continue
                    
                    meta = manifest[eid]
                    label = meta.get('label', -1)
                    
                    if label == -1:
                        continue
                    
                    self.items.append({
                        "tensor": tensor[layer_idx],  # (T, D)
                        "label": label,
                        "id": eid
                    })
                    
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)


class PerLayerEvalDataset(Dataset):
    """
    Dataset that returns activations for a SPECIFIC layer only.
    Used for per-layer evaluation of the trained probe.
    """
    
    def __init__(self, activations_dir: str, layer_idx: int, pooling: str = "mean"):
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.items = []
        
        # Find all shard files
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        
        if not shards:
            raise FileNotFoundError(f"No shard files found: {shard_pattern}")
        
        # Load manifest
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                meta = json.loads(line)
                manifest[meta['id']] = meta
        
        # Load tensors
        for shard_path in shards:
            try:
                tensors = load_file(shard_path)
                
                for eid, tensor in tensors.items():
                    if eid not in manifest:
                        continue
                    
                    meta = manifest[eid]
                    label = meta.get('label', -1)
                    
                    if label == -1:
                        continue
                    
                    # Extract specific layer
                    layer_activations = tensor[layer_idx]  # (T, D)
                    
                    if self.pooling == "mean":
                        pooled = layer_activations.mean(dim=0)
                    elif self.pooling == "max":
                        pooled = layer_activations.max(dim=0).values
                    elif self.pooling == "last":
                        pooled = layer_activations[-1]
                    else:
                        raise ValueError(f"Unknown pooling: {self.pooling}")
                    
                    self.items.append({
                        "vector": pooled,
                        "label": label,
                        "id": eid
                    })
                    
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['vector'].float(), torch.tensor(item['label'], dtype=torch.float32)


# ============================================================================
# Simple Linear Probe
# ============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe: Linear(D, 1) → logits"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class LearnedAttentionPooling(nn.Module):
    """
    Learns a query vector q to compute attention over tokens.
    Shared across all layers in layer-agnostic training.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = torch.matmul(x, self.query)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)  # (B, D)
        return pooled


class AttentionProbe(nn.Module):
    """
    Attention pooling + linear probe.
    Shared attention learns which tokens are important across ALL layers.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = LearnedAttentionPooling(input_dim)
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        pooled = self.attention(x)  # (B, D)
        return self.classifier(pooled)  # (B, 1)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_probe(model, train_loader, val_loader, device, epochs=20, lr=1e-3, checkpoint_dir=None, checkpoint_every=1):
    """Train the probe and return best model based on validation AUC."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    best_state = None
    
    checkpoints = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 4:
                x, y, layers, _ = batch
            else:
                x, y, layers = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x, y, layers, _ = batch
                else:
                    x, y, layers = batch
                x = x.to(device)
                logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except:
            val_auc = 0.5
        
        val_acc = accuracy_score(val_targets, (val_preds > 0.5).astype(int))
        
        logger.info(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            checkpoints.append((epoch + 1, ckpt_path))

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_auc, checkpoints


# ============================================================================
# Attribution Helpers
# ============================================================================

def _flatten_params(model: nn.Module) -> torch.Tensor:
    params = []
    for p in model.parameters():
        params.append(p.detach().flatten().cpu())
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten().cpu())
        else:
            grads.append(p.grad.detach().flatten().cpu())
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def _write_csv(path: str, header: list, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _compute_per_sample_grads(model, x, y, loss_fn):
    logits = model(x).squeeze(-1)
    losses = loss_fn(logits, y)
    grads = []
    for i in range(len(losses)):
        model.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=True)
        grads.append(_flatten_grads(model))
    model.zero_grad(set_to_none=True)
    return grads


def _summarize_per_layer(results: dict):
    best_layer = max(results.keys(), key=lambda l: results[l]['auc'])
    return {
        "best_layer": best_layer,
        "best_auc": results[best_layer]["auc"],
        "best_acc": results[best_layer]["accuracy"]
    }


def run_layer_agnostic_attribution(
    model: nn.Module,
    checkpoints: list,
    w_star: torch.Tensor,
    train_loader: DataLoader,
    attr_every_n: int,
    attr_top_n: int,
    attr_out_dir: str,
    device: str,
    val_path: str,
    ood_path: str,
    pooling: str,
    n_layers: int
):
    training_dynamics = []
    checkpoint_metrics = []

    sample_progress = {}
    sample_influence = {}
    layer_progress = {}
    layer_influence = {}
    layer_counts = {}

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_id_auc = -1.0
    best_ood_auc = -1.0

    for epoch, ckpt_path in checkpoints:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        w_t = _flatten_params(model)
        cos_to_final = _cosine(w_t, w_star)
        w_norm = float(torch.norm(w_t).item()) if w_t.numel() else 0.0
        training_dynamics.append([epoch, cos_to_final, w_norm])

        # Checkpoint evaluation
        id_results = evaluate_per_layer(model, val_path, pooling, device, n_layers)
        id_best = _summarize_per_layer(id_results)
        best_id_auc = max(best_id_auc, id_best["best_auc"])
        checkpoint_metrics.append([epoch, "id", id_best["best_auc"], id_best["best_acc"], id_best["best_layer"], 1 if id_best["best_auc"] >= best_id_auc else 0])

        if ood_path and os.path.exists(ood_path):
            ood_results = evaluate_per_layer(model, ood_path, pooling, device, n_layers)
            ood_best = _summarize_per_layer(ood_results)
            best_ood_auc = max(best_ood_auc, ood_best["best_auc"])
            checkpoint_metrics.append([epoch, "ood", ood_best["best_auc"], ood_best["best_acc"], ood_best["best_layer"], 1 if ood_best["best_auc"] >= best_ood_auc else 0])

        # Attribution pass
        delta = w_star - w_t
        for batch_idx, batch in enumerate(train_loader):
            if attr_every_n > 1 and batch_idx % attr_every_n != 0:
                continue
            if len(batch) == 4:
                x, y, layers, ids = batch
            else:
                x, y, layers = batch
                ids = [f"idx_{batch_idx}_{i}" for i in range(len(y))]

            x = x.to(device)
            y = y.to(device)

            grads = _compute_per_sample_grads(model, x, y, loss_fn)
            for i, g in enumerate(grads):
                sample_id = ids[i]
                layer_id = int(layers[i])
                alignment = -_cosine(g, w_star)
                influence = float(torch.dot(g, delta)) if g.numel() else 0.0

                sample_progress[sample_id] = sample_progress.get(sample_id, 0.0) + alignment
                sample_influence[sample_id] = sample_influence.get(sample_id, 0.0) + influence

                layer_progress[layer_id] = layer_progress.get(layer_id, 0.0) + alignment
                layer_influence[layer_id] = layer_influence.get(layer_id, 0.0) + influence
                layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1

    # Write outputs
    _write_csv(
        os.path.join(attr_out_dir, "training_dynamics.csv"),
        ["epoch", "cos_to_final", "w_norm"],
        training_dynamics
    )
    _write_csv(
        os.path.join(attr_out_dir, "checkpoint_metrics.csv"),
        ["epoch", "split", "auc", "acc", "best_layer", "best_so_far"],
        checkpoint_metrics
    )

    layer_rows = []
    for l, total in layer_progress.items():
        count = max(layer_counts.get(l, 1), 1)
        layer_rows.append([l, total / count, count])
    _write_csv(
        os.path.join(attr_out_dir, "layer_progress.csv"),
        ["layer", "mean_grad_alignment", "count"],
        layer_rows
    )

    layer_inf_rows = []
    for l, total in layer_influence.items():
        count = max(layer_counts.get(l, 1), 1)
        layer_inf_rows.append([l, total / count, count])
    _write_csv(
        os.path.join(attr_out_dir, "layer_influence.csv"),
        ["layer", "mean_influence", "count"],
        layer_inf_rows
    )

    top_prog = sorted(sample_progress.items(), key=lambda x: abs(x[1]), reverse=True)[:attr_top_n]
    _write_csv(
        os.path.join(attr_out_dir, f"sample_progress_top{attr_top_n}.csv"),
        ["sample_id", "grad_alignment"],
        [[sid, score] for sid, score in top_prog]
    )

    top_inf = sorted(sample_influence.items(), key=lambda x: abs(x[1]), reverse=True)[:attr_top_n]
    _write_csv(
        os.path.join(attr_out_dir, f"sample_influence_top{attr_top_n}.csv"),
        ["sample_id", "influence"],
        [[sid, score] for sid, score in top_inf]
    )


def evaluate_per_layer(model, activations_dir: str, pooling: str, device: str, n_layers: int = 28):
    """
    Evaluate the trained probe on each layer separately.
    Returns dict mapping layer -> (AUC, accuracy).
    """
    model.eval()
    results = {}
    
    logger.info(f"\nEvaluating per-layer performance on {activations_dir}...")
    
    for layer_idx in tqdm(range(n_layers), desc="Evaluating layers"):
        # Use different dataset class for attention pooling
        if pooling == "attn":
            ds = PerLayerEvalAttnDataset(activations_dir, layer_idx)
        else:
            ds = PerLayerEvalDataset(activations_dir, layer_idx, pooling=pooling)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        
        preds, targets = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                targets.extend(y.numpy())
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        try:
            auc = roc_auc_score(targets, preds)
        except:
            auc = 0.5
        
        acc = accuracy_score(targets, (preds > 0.5).astype(int))
        results[layer_idx] = {"auc": auc, "accuracy": acc}
    
    return results


def plot_results(id_results: dict, ood_results: dict, output_path: str, title: str):
    """Plot per-layer AUC for ID and OOD."""
    import matplotlib.pyplot as plt
    
    layers = sorted(id_results.keys())
    id_aucs = [id_results[l]["auc"] for l in layers]
    ood_aucs = [ood_results[l]["auc"] for l in layers]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, id_aucs, 'b-o', label='ID (Roleplaying)', linewidth=2)
    plt.plot(layers, ood_aucs, 'r-s', label='OOD (InsiderTrading)', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved plot: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train layer-agnostic deception probe")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (for directory structure)")
    parser.add_argument("--dataset", type=str, default="Deception-Roleplaying", help="Training dataset")
    parser.add_argument("--ood_dataset", type=str, default="Deception-InsiderTrading", help="OOD evaluation dataset")
    parser.add_argument("--activations_dir", type=str, default="data/activations", help="Base activations directory")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last", "attn"], help="Token pooling method")
    parser.add_argument("--output_dir", type=str, default="data/probes_layer_agnostic", help="Output directory for probes")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory for results/plots (default: results/probes_layer_agnostic)")
    parser.add_argument("--train_split", type=str, default="train", help="Training split")
    parser.add_argument("--val_split", type=str, default="validation", help="Validation split")
    parser.add_argument("--ood_split", type=str, default="test", help="OOD evaluation split")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--split_train_for_val", action="store_true", help="Use 80/20 split of train for validation")
    parser.add_argument("--n_layers", type=int, default=None, help="Override number of layers (auto-detect if not set)")
    # Attribution / checkpointing
    parser.add_argument("--attr_enable", action="store_true", help="Enable attribution logging")
    parser.add_argument("--attr_every_n", type=int, default=50, help="Batch interval for attribution aggregation")
    parser.add_argument("--attr_top_n", type=int, default=100, help="Top-N samples to save")
    parser.add_argument("--attr_checkpoint_every", type=int, default=1, help="Checkpoint frequency (epochs)")
    parser.add_argument("--attr_out_dir", type=str, default=None, help="Output directory for attribution results")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_dir = args.model.replace("/", "_")
    
    # ========================================================================
    # 1. Load Training Data
    # ========================================================================
    
    train_path = os.path.join(args.activations_dir, model_dir, args.dataset, args.train_split)
    logger.info(f"\n{'='*70}")
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"{'='*70}")
    
    # Use different dataset class for attention pooling
    if args.pooling == "attn":
        train_ds = LayerAgnosticAttnDataset(train_path)
    else:
        train_ds = LayerAgnosticDataset(train_path, pooling=args.pooling)
    
    # ========================================================================
    # 2. Create Train/Val Split or Load Validation
    # ========================================================================
    
    if args.split_train_for_val:
        # 80/20 split of training data
        n_train = int(0.8 * len(train_ds))
        n_val = len(train_ds) - n_train
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])
        logger.info(f"Split training data: {n_train} train, {n_val} val")
    else:
        val_path = os.path.join(args.activations_dir, model_dir, args.dataset, args.val_split)
        if os.path.exists(val_path):
            if args.pooling == "attn":
                val_ds = LayerAgnosticAttnDataset(val_path)
            else:
                val_ds = LayerAgnosticDataset(val_path, pooling=args.pooling)
        else:
            logger.warning(f"Validation path not found: {val_path}, using 80/20 split")
            n_train = int(0.8 * len(train_ds))
            n_val = len(train_ds) - n_train
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # ========================================================================
    # 3. Create Probe and Train
    # ========================================================================
    
    # Get hidden dim from first batch
    first_batch = next(iter(train_loader))
    if len(first_batch) == 4:
        sample_x, _, _, _ = first_batch
    else:
        sample_x, _, _ = first_batch
    hidden_dim = sample_x.shape[-1]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Layer-Agnostic Probe")
    logger.info(f"{'='*70}")
    logger.info(f"Hidden dim: {hidden_dim}")
    logger.info(f"Pooling: {args.pooling}")
    
    # Create appropriate probe based on pooling type
    if args.pooling == "attn":
        probe = AttentionProbe(hidden_dim)
        logger.info(f"Using AttentionProbe (shared attention across all layers)")
    else:
        probe = LinearProbe(hidden_dim)
    
    # Attribution output dir
    if args.attr_enable:
        attr_out_dir = args.attr_out_dir or os.path.join(
            "results/layer_agnostic_attribution",
            model_dir,
            args.dataset,
            args.pooling
        )
        os.makedirs(attr_out_dir, exist_ok=True)
    else:
        attr_out_dir = None

    checkpoint_dir = os.path.join(attr_out_dir, "checkpoints") if args.attr_enable else None

    probe, best_val_auc, checkpoints = train_probe(
        probe,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.attr_checkpoint_every
    )
    
    logger.info(f"\n✓ Training complete! Best Val AUC: {best_val_auc:.4f}")
    
    # ========================================================================
    # 4. Evaluate Per-Layer on ID Validation
    # ========================================================================
    
    val_eval_path = os.path.join(args.activations_dir, model_dir, args.dataset, args.val_split)

    # Infer number of layers if not provided
    n_layers = args.n_layers
    if n_layers is None:
        infer_dir = val_eval_path if os.path.exists(val_eval_path) else train_path
        try:
            n_layers = infer_num_layers(infer_dir)
            logger.info(f"Inferred n_layers={n_layers} from {infer_dir}")
        except Exception as e:
            n_layers = 28
            logger.warning(f"Failed to infer n_layers ({e}); defaulting to {n_layers}")
    if os.path.exists(val_eval_path):
        id_results = evaluate_per_layer(probe, val_eval_path, args.pooling, device, n_layers)
    else:
        # If no separate val, use train (for per-layer analysis only)
        id_results = evaluate_per_layer(probe, train_path, args.pooling, device, n_layers)
    
    # Log ID results
    logger.info(f"\nPer-Layer ID Validation Results:")
    logger.info(f"{'Layer':<6} {'AUC':<8} {'Acc':<8}")
    logger.info("-" * 24)
    for l in sorted(id_results.keys()):
        logger.info(f"{l:<6} {id_results[l]['auc']:<8.4f} {id_results[l]['accuracy']:<8.4f}")
    
    best_id_layer = max(id_results.keys(), key=lambda l: id_results[l]['auc'])
    logger.info(f"\nBest ID layer: {best_id_layer} (AUC: {id_results[best_id_layer]['auc']:.4f})")
    
    # ========================================================================
    # 5. Evaluate Per-Layer on OOD
    # ========================================================================
    
    ood_path = os.path.join(args.activations_dir, model_dir, args.ood_dataset, args.ood_split)
    ood_results = None
    
    if os.path.exists(ood_path):
        ood_results = evaluate_per_layer(probe, ood_path, args.pooling, device, n_layers)
        
        logger.info(f"\nPer-Layer OOD Results ({args.ood_dataset}):")
        logger.info(f"{'Layer':<6} {'AUC':<8} {'Acc':<8}")
        logger.info("-" * 24)
        for l in sorted(ood_results.keys()):
            logger.info(f"{l:<6} {ood_results[l]['auc']:<8.4f} {ood_results[l]['accuracy']:<8.4f}")
        
        best_ood_layer = max(ood_results.keys(), key=lambda l: ood_results[l]['auc'])
        logger.info(f"\nBest OOD layer: {best_ood_layer} (AUC: {ood_results[best_ood_layer]['auc']:.4f})")
    else:
        logger.warning(f"OOD path not found: {ood_path}")

    # ========================================================================
    # 6. Attribution (Post-hoc)
    # ========================================================================
    if args.attr_enable and checkpoints:
        w_star = _flatten_params(probe)
        attr_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        run_layer_agnostic_attribution(
            model=probe,
            checkpoints=checkpoints,
            w_star=w_star,
            train_loader=attr_train_loader,
            attr_every_n=args.attr_every_n,
            attr_top_n=args.attr_top_n,
            attr_out_dir=attr_out_dir,
            device=device,
            val_path=val_eval_path if os.path.exists(val_eval_path) else train_path,
            ood_path=ood_path if os.path.exists(ood_path) else None,
            pooling=args.pooling,
            n_layers=n_layers
        )
    
    # ========================================================================
    # 6. Save Results
    # ========================================================================
    
    save_dir = os.path.join(args.output_dir, model_dir, args.dataset, args.pooling)
    os.makedirs(save_dir, exist_ok=True)

    # Results directory (separate from probes)
    if args.results_dir:
        results_dir = os.path.join(args.results_dir, model_dir, args.dataset, args.pooling)
    else:
        results_dir = os.path.join("results/probes_layer_agnostic", model_dir, args.dataset, args.pooling)
    os.makedirs(results_dir, exist_ok=True)

    # Save probe
    probe_path = os.path.join(save_dir, "probe.pt")
    torch.save(probe.state_dict(), probe_path)
    logger.info(f"\n✓ Saved probe: {probe_path}")

    # Save best_probe.json for eval_ood.py compatibility
    best_probe_info = {
        'probe_type': 'layer_agnostic',
        'probe_path': probe_path,
        'best_id_layer': best_id_layer,
        'best_id_auc': id_results[best_id_layer]['auc'],
        'best_ood_layer': best_ood_layer if ood_results else None,
        'best_ood_auc': ood_results[best_ood_layer]['auc'] if ood_results else None,
        'pooling': args.pooling,
        'model': args.model,
        'dataset': args.dataset,
        'hidden_dim': hidden_dim,
        'note': 'Layer-agnostic probe - single probe.pt applied to any layer'
    }
    best_probe_path = os.path.join(save_dir, "best_probe.json")
    with open(best_probe_path, 'w') as f:
        json.dump(best_probe_info, f, indent=2)
    logger.info(f"✓ Saved best probe info: {best_probe_path}")

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "pooling": args.pooling,
        "hidden_dim": hidden_dim,
        "best_val_auc": best_val_auc,
        "id_per_layer": id_results,
        "ood_per_layer": ood_results,
        "best_id_layer": best_id_layer,
        "best_ood_layer": best_ood_layer if ood_results else None
    }
    
    results_path = os.path.join(results_dir, "layer_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Saved results: {results_path}")

    # Plot if OOD available
    if ood_results:
        plot_path = os.path.join(results_dir, "per_layer_auc.png")
        plot_results(
            id_results, ood_results, plot_path,
            f"Layer-Agnostic Probe ({args.pooling} pooling) - Per-Layer AUC"
        )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ All done!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
