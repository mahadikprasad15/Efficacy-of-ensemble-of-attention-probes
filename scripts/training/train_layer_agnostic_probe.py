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
        return item['vector'].float(), torch.tensor(item['label'], dtype=torch.float32), item['layer']


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
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32), item['layer']


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

def train_probe(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train the probe and return best model based on validation AUC."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for x, y, layers in train_loader:
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
            for x, y, layers in val_loader:
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
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_auc


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
    sample_x, _, _ = next(iter(train_loader))
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
    
    probe, best_val_auc = train_probe(probe, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    
    logger.info(f"\n✓ Training complete! Best Val AUC: {best_val_auc:.4f}")
    
    # ========================================================================
    # 4. Evaluate Per-Layer on ID Validation
    # ========================================================================
    
    n_layers = 28  # Default for Llama-3B
    
    val_eval_path = os.path.join(args.activations_dir, model_dir, args.dataset, args.val_split)
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
