#!/usr/bin/env python3
"""
Train probes on COMBINED prompted-probing activations (Domain A + Domain B).

Supports:
- Training on merged data from two domains
- Separate evaluation on each domain's validation/test set
- Layer-wise AUC/Accuracy plotting

Usage:
    python scripts/training/train_combined_prompted_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --suffix_condition suffix_deception_yesno \
        --dataset_a Deception-Roleplaying \
        --dataset_b Deception-InsiderTrading \
        --activations_dir data/prompted_activations \
        --limit_a 100 \
        --limit_b 100
"""

import argparse
import os
import sys
import json
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class CachedPromptedDataset(Dataset):
    """Load cached prompted-probing activations (L, D format)."""
    
    def __init__(self, activations_dir: str, limit: int = None):
        self.items = []
        
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        
        if not shards:
            raise FileNotFoundError(f"No shards found in {activations_dir}")
        
        # Load manifest
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
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
                    
                    self.items.append({
                        "id": eid,
                        "tensor": tensor,
                        "label": label
                    })
            except Exception as e:
                logger.error(f"Error loading {shard_path}: {e}")
        
        # Apply limit
        if limit and limit < len(self.items):
            import random
            random.seed(42)
            self.items = random.sample(self.items, limit)
        
        logger.info(f"Loaded {len(self.items)} examples from {activations_dir}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)


class LayerDataset(Dataset):
    """Extract a specific layer from (L, D) tensors."""
    
    def __init__(self, base_dataset, layer_idx):
        self.base = base_dataset
        self.layer_idx = layer_idx
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x[self.layer_idx], y


# ============================================================================
# Training
# ============================================================================

def train_probe(train_loader, input_dim, device, epochs=50, lr=1e-3, patience=5):
    """Train a linear probe."""
    model = LayerProbe(input_dim=input_dim, pooling_type='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model


def evaluate_probe(model, dataloader, device):
    """Evaluate probe and return metrics."""
    model.eval()
    preds = []
    targets = []
    probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(prob)
            preds.extend((prob > 0.5).astype(int))
            targets.extend(y.numpy())
    
    preds = np.array(preds)
    targets = np.array(targets)
    probs = np.array(probs)
    
    auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5
    acc = accuracy_score(targets, preds)
    
    return {'auc': auc, 'accuracy': acc}


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results_a, results_b, label_a, label_b, output_path):
    """Generate layer-wise plot comparing performance on two domains."""
    import matplotlib.pyplot as plt
    
    layers = [r['layer'] for r in results_a]
    auc_a = [r['auc'] for r in results_a]
    auc_b = [r['auc'] for r in results_b]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, auc_a, 'o-', label=f'{label_a} (Val)', color='#1f77b4', linewidth=2)
    ax.plot(layers, auc_b, 's--', label=f'{label_b} (Val)', color='#ff7f0e', linewidth=2)
    
    # Mark best for each
    best_a_idx = np.argmax(auc_a)
    best_b_idx = np.argmax(auc_b)
    ax.plot(layers[best_a_idx], auc_a[best_a_idx], '*', markersize=15, color='#1f77b4')
    ax.plot(layers[best_b_idx], auc_b[best_b_idx], '*', markersize=15, color='#ff7f0e')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Combined Training: Per-Domain Validation Performance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate(f"Best {label_a}: L{layers[best_a_idx]} ({auc_a[best_a_idx]:.3f})",
                xy=(layers[best_a_idx], auc_a[best_a_idx]),
                xytext=(5, 10), textcoords='offset points', fontsize=9)
    ax.annotate(f"Best {label_b}: L{layers[best_b_idx]} ({auc_b[best_b_idx]:.3f})",
                xy=(layers[best_b_idx], auc_b[best_b_idx]),
                xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train probes on combined prompted-probing activations"
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (for directory structure)")
    parser.add_argument("--suffix_condition", type=str, required=True,
                        help="Suffix condition (e.g., suffix_deception_yesno)")
    parser.add_argument("--dataset_a", type=str, required=True,
                        help="First dataset name (e.g., Deception-Roleplaying)")
    parser.add_argument("--dataset_b", type=str, required=True,
                        help="Second dataset name (e.g., Deception-InsiderTrading)")
    parser.add_argument("--activations_dir", type=str, default="data/prompted_activations",
                        help="Base directory for prompted activations")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val_split", type=str, default="test",
                        help="Validation split name")
    parser.add_argument("--limit_a", type=int, default=None,
                        help="Limit samples from dataset A")
    parser.add_argument("--limit_b", type=int, default=None,
                        help="Limit samples from dataset B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--probes_dir", type=str, default="data/prompted_probes_combined",
                        help="Directory to save trained probes")
    parser.add_argument("--results_dir", type=str, default="results/prompted_probes_combined",
                        help="Directory to save results and plots")
    parser.add_argument("--split_train_for_test", action="store_true",
                        help="If set, uses a portion of train data for testing if test split is missing")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_dir = args.model.replace("/", "_")
    
    # ========================================================================
    # Load Training Data (Combined)
    # ========================================================================
    
    train_dir_a = os.path.join(
        args.activations_dir, model_dir, args.suffix_condition,
        args.dataset_a, args.train_split
    )
    train_dir_b = os.path.join(
        args.activations_dir, model_dir, args.suffix_condition,
        args.dataset_b, args.train_split
    )
    
    logger.info(f"Loading training data...")
    logger.info(f"  {args.dataset_a}: {train_dir_a}")
    logger.info(f"  {args.dataset_b}: {train_dir_b}")
    
    train_dataset_a = CachedPromptedDataset(train_dir_a, limit=args.limit_a)
    train_dataset_b = CachedPromptedDataset(train_dir_b, limit=args.limit_b)
    
    logger.info(f"Combined training: {len(train_dataset_a)} + {len(train_dataset_b)} = {len(train_dataset_a) + len(train_dataset_b)} samples")
    
    # ========================================================================
    # Load Validation Data (Separate)
    # ========================================================================
    
    val_dir_a = os.path.join(
        args.activations_dir, model_dir, args.suffix_condition,
        args.dataset_a, args.val_split
    )
    val_dir_b = os.path.join(
        args.activations_dir, model_dir, args.suffix_condition,
        args.dataset_b, args.val_split
    )
    
    logger.info(f"Loading validation data...")
    
    # helper to load or split
    def load_or_split(data_dir, train_dataset, limit_split):
        try:
            return CachedPromptedDataset(data_dir, limit=None)
        except (FileNotFoundError, IndexError):
            if args.split_train_for_test:
                logger.warning(f"Validation data not found at {data_dir}. Splitting from TRAIN.")
                if len(train_dataset.items) < 20:
                    raise ValueError("Train dataset too small to split.")
                
                # Take last N samples for validation (max 20% of total)
                total_samples = len(train_dataset.items)
                max_split = int(total_samples * 0.2)
                
                if limit_split and limit_split <= max_split:
                    split_size = limit_split
                else:
                    if limit_split:
                        logger.warning(f"Requested split {limit_split} is > 20% of {total_samples}. Capping at {max_split} to preserve training data.")
                    split_size = max(1, max_split)
                
                val_items = train_dataset.items[-split_size:]
                train_dataset.items = train_dataset.items[:-split_size]
                
                logger.info(f"  Split {len(val_items)} samples for validation. Train now has {len(train_dataset.items)}.")
                
                # Create new dataset object manually
                val_ds = CachedPromptedDataset.__new__(CachedPromptedDataset)
                val_ds.items = val_items
                return val_ds
            else:
                raise

    val_dataset_a = load_or_split(val_dir_a, train_dataset_a, args.limit_a)
    val_dataset_b = load_or_split(val_dir_b, train_dataset_b, args.limit_b)
    
    # Get dimensions
    sample_tensor = train_dataset_a.items[0]['tensor']
    num_layers, hidden_dim = sample_tensor.shape
    logger.info(f"Activation shape: ({num_layers} layers, {hidden_dim} dim)")
    
    # ========================================================================
    # Train for Each Layer
    # ========================================================================
    
    results_a = []
    results_b = []
    
    # Probes directory
    probes_dir = os.path.join(
        args.probes_dir, model_dir, args.suffix_condition,
        f"{args.dataset_a}+{args.dataset_b}"
    )
    os.makedirs(probes_dir, exist_ok=True)
    
    # Results directory
    results_output_dir = os.path.join(
        args.results_dir, model_dir, args.suffix_condition,
        f"{args.dataset_a}+{args.dataset_b}"
    )
    os.makedirs(results_output_dir, exist_ok=True)
    
    for layer_idx in range(num_layers):
        logger.info(f"Training Layer {layer_idx}...")
        
        # Create combined training dataset for this layer
        layer_train_a = LayerDataset(train_dataset_a, layer_idx)
        layer_train_b = LayerDataset(train_dataset_b, layer_idx)
        combined_train = ConcatDataset([layer_train_a, layer_train_b])
        
        train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True)
        
        # Train
        model = train_probe(train_loader, hidden_dim, device)
        
        # Evaluate on each domain separately
        val_loader_a = DataLoader(LayerDataset(val_dataset_a, layer_idx), batch_size=args.batch_size)
        val_loader_b = DataLoader(LayerDataset(val_dataset_b, layer_idx), batch_size=args.batch_size)
        
        metrics_a = evaluate_probe(model, val_loader_a, device)
        metrics_b = evaluate_probe(model, val_loader_b, device)
        
        results_a.append({"layer": layer_idx, **metrics_a})
        results_b.append({"layer": layer_idx, **metrics_b})
        
        print(f"  L{layer_idx}: {args.dataset_a} AUC={metrics_a['auc']:.4f} | {args.dataset_b} AUC={metrics_b['auc']:.4f}")
        
        # Save probe
        probe_path = os.path.join(probes_dir, f"probe_layer_{layer_idx}.pt")
        torch.save(model.state_dict(), probe_path)
    
    # ========================================================================
    # Save Results & Plot
    # ========================================================================
    
    # Find best layers
    best_a = max(results_a, key=lambda x: x['auc'])
    best_b = max(results_b, key=lambda x: x['auc'])
    
    summary = {
        "dataset_a": args.dataset_a,
        "dataset_b": args.dataset_b,
        "best_layer_a": best_a['layer'],
        "best_auc_a": best_a['auc'],
        "best_layer_b": best_b['layer'],
        "best_auc_b": best_b['auc'],
        "results_a": results_a,
        "results_b": results_b
    }
    
    results_path = os.path.join(results_output_dir, "combined_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot
    plot_path = os.path.join(results_output_dir, "combined_layerwise_auc.png")
    plot_results(results_a, results_b, args.dataset_a, args.dataset_b, plot_path)
    
    print("\n" + "=" * 70)
    print("✓ COMBINED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best {args.dataset_a}: Layer {best_a['layer']} (AUC: {best_a['auc']:.4f})")
    print(f"Best {args.dataset_b}: Layer {best_b['layer']} (AUC: {best_b['auc']:.4f})")
    print(f"Results: {results_path}")
    print(f"Plot: {plot_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
