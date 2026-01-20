"""
Train probes for deception detection using cached activations.

Simple training script (no LODO):
    - Train on one dataset (e.g., Deception-Roleplaying)
    - Validate on same dataset
    - Test on same dataset or different split

Usage:
    # Train mean pooling probe on single dataset
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling mean

    # Train with attention pooling
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling attn \
        --epochs 20

    # Train on specific layer only
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling mean \
        --layer 20
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Dataset loader
# ============================================================================

class CachedDeceptionDataset(Dataset):
    """
    Load cached activations from safetensors shards + manifest.

    Expected directory structure:
        activations_dir/
            ├── shard_000.safetensors
            ├── shard_001.safetensors
            └── manifest.jsonl
    """

    def __init__(self, activations_dir: str):
        """
        Args:
            activations_dir: Path to directory with shards and manifest
        """
        self.items = []

        # Find all shard files
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))

        if not shards:
            raise FileNotFoundError(
                f"No shard files found in {activations_dir}\n"
                f"Pattern: {shard_pattern}"
            )

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

        # Load tensors from shards
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

                    self.items.append({
                        "id": eid,
                        "tensor": tensor,  # (L, T, D)
                        "label": label
                    })

            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")

        logger.info(f"✓ Loaded {len(self.items)} examples")

        # Log label distribution
        labels = [item['label'] for item in self.items]
        honest_count = sum(1 for l in labels if l == 0)
        deceptive_count = sum(1 for l in labels if l == 1)
        logger.info(f"  • Honest: {honest_count} ({100*honest_count/len(labels):.1f}%)")
        logger.info(f"  • Deceptive: {deceptive_count} ({100*deceptive_count/len(labels):.1f}%)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Return tensor as float32 and label as float32 (for BCE loss)
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, dataloader, device):
    """
    Evaluate model on dataloader.

    Returns:
        auc: AUROC score
        acc: Accuracy
        preds: Predictions (for analysis)
        targets: Ground truth labels
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.flatten())
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    # Calculate metrics
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = 0.5

    acc = accuracy_score(targets, (preds > 0.5).astype(int))

    return auc, acc, preds, targets

# ============================================================================
# Training
# ============================================================================

def train_probe(model, train_loader, val_loader, device, args):
    """
    Train a single probe with early stopping.

    Args:
        model: LayerProbe model
        train_loader: Training data
        val_loader: Validation data
        device: Device to train on
        args: Training arguments

    Returns:
        best_val_auc: Best validation AUROC
        best_epoch: Epoch with best validation AUROC
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_targets.extend(y.cpu().numpy())

        train_loss /= len(train_loader)

        # Calculate train metrics
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except:
            train_auc = 0.5

        # Validation
        val_auc, val_acc, _, _ = evaluate(model, val_loader, device)

        # Logging
        logger.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), args.output_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"✓ Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    return best_val_auc, best_epoch

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train deception detection probes"
    )

    # Model & data
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Deception-Roleplaying)"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="data/activations",
        help="Base directory for cached activations"
    )

    # Probe configuration
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "last", "attn"],
        help="Token pooling strategy"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to probe (if None, trains on all layers)"
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/probes",
        help="Output directory for trained probes"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load data
    # ========================================================================

    model_dir = args.model.replace("/", "_")

    train_dir = os.path.join(args.activations_dir, model_dir, args.dataset, "train")
    val_dir = os.path.join(args.activations_dir, model_dir, args.dataset, "validation")
    test_dir = os.path.join(args.activations_dir, model_dir, args.dataset, "test")

    logger.info(f"{'='*70}")
    logger.info(f"Loading datasets...")
    logger.info(f"{'='*70}")

    # Check which splits exist
    splits_exist = {
        'train': os.path.exists(train_dir),
        'validation': os.path.exists(val_dir),
        'test': os.path.exists(test_dir)
    }

    logger.info(f"Train dir: {train_dir} {'✓' if splits_exist['train'] else '✗'}")
    logger.info(f"Val dir: {val_dir} {'✓' if splits_exist['validation'] else '✗'}")
    logger.info(f"Test dir: {test_dir} {'✓' if splits_exist['test'] else '✗'}")

    if not splits_exist['train']:
        logger.error(f"Training data not found! Run cache_deception_activations.py first.")
        return 1

    # Load datasets
    train_dataset = CachedDeceptionDataset(train_dir)

    if splits_exist['validation']:
        val_dataset = CachedDeceptionDataset(val_dir)
    else:
        logger.warning("Validation split not found, using train for validation")
        val_dataset = train_dataset

    if splits_exist['test']:
        test_dataset = CachedDeceptionDataset(test_dir)
    else:
        logger.warning("Test split not found")
        test_dataset = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Colab compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

    # Get tensor shape from first batch
    sample_x, _ = next(iter(train_loader))
    L, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]
    logger.info(f"\nTensor shape: (batch, {L} layers, {T} tokens, {D} dim)")

    # ========================================================================
    # Training
    # ========================================================================

    logger.info(f"\n{'='*70}")
    logger.info(f"Training Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"{'='*70}\n")

    # Prepare output directory
    output_dir = os.path.join(
        args.output_dir,
        model_dir,
        args.dataset,
        args.pooling
    )
    os.makedirs(output_dir, exist_ok=True)

    # Check if probes already exist (skip if so, unless training single layer)
    results_path = os.path.join(output_dir, "layer_results.json")
    if args.layer is None and os.path.exists(results_path):
        # Check if we have results for all layers
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            if len(existing_results) == L:
                logger.info(f"⚠️  Probes already trained in {output_dir}")
                logger.info(f"   Found layer_results.json with {len(existing_results)} layers")
                logger.info(f"   Skipping training to avoid overwriting existing probes.")
                logger.info(f"   To retrain, delete the directory or train specific layer with --layer flag.")
                return 0
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}, will retrain")

    if args.layer is not None:
        # Train single layer
        logger.info(f"Training probe for layer {args.layer}...")

        args.output_model_path = os.path.join(output_dir, f"probe_layer_{args.layer}.pt")

        model = LayerProbe(
            input_dim=D,
            pooling_type=args.pooling
        ).to(device)

        best_val_auc, best_epoch = train_probe(
            model, train_loader, val_loader, device, args
        )

        # Test
        if test_dataset:
            model.load_state_dict(torch.load(args.output_model_path))
            test_auc, test_acc, _, _ = evaluate(model, test_loader, device)
            logger.info(f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")

    else:
        # Train on all layers
        logger.info(f"Training probes for all {L} layers...")

        results = []

        for layer_idx in range(L):
            logger.info(f"\n{'='*70}")
            logger.info(f"Layer {layer_idx}/{L-1}")
            logger.info(f"{'='*70}")

            args.output_model_path = os.path.join(output_dir, f"probe_layer_{layer_idx}.pt")

            # Extract single layer from batch
            # Need to modify dataset to return single layer? Or slice in training loop?
            # For simplicity, train on all layers and probe will select via indexing

            model = LayerProbe(
                input_dim=D,
                pooling_type=args.pooling
            ).to(device)

            # Create dataloaders that select specific layer
            # This is a bit hacky but works
            class LayerDataset(Dataset):
                def __init__(self, base_dataset, layer_idx):
                    self.base = base_dataset
                    self.layer_idx = layer_idx

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    x, y = self.base[idx]
                    return x[self.layer_idx], y  # Select layer

            layer_train_dataset = LayerDataset(train_dataset, layer_idx)
            layer_val_dataset = LayerDataset(val_dataset, layer_idx)

            layer_train_loader = DataLoader(
                layer_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
            layer_val_loader = DataLoader(
                layer_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

            best_val_auc, best_epoch = train_probe(
                model, layer_train_loader, layer_val_loader, device, args
            )

            # Load best model and get final validation accuracy
            model.load_state_dict(torch.load(args.output_model_path))
            _, best_val_acc, _, _ = evaluate(model, layer_val_loader, device)

            results.append({
                'layer': layer_idx,
                'val_auc': best_val_auc,
                'val_acc': best_val_acc,
                'epoch': best_epoch
            })

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Complete - Per-Layer Results")
        logger.info(f"{'='*70}")

        best_layer = max(results, key=lambda x: x['val_auc'])
        logger.info(f"Best layer: {best_layer['layer']} (AUC: {best_layer['val_auc']:.4f})")

        # Save results
        results_path = os.path.join(output_dir, "layer_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Training Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}\n")

    return 0

if __name__ == "__main__":
    exit(main())
