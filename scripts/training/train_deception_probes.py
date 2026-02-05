"""
Train probes for deception detection using cached activations.

Supports two input formats:
    - 'pooled': Standard activations with shape (L, T, D), requires pooling
    - 'final_token': Prompted-probing activations with shape (L, D), no pooling needed

Usage:
    # Train with standard pooled activations
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling mean

    # Train with prompted-probing final-token activations
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --activations_dir data/prompted_activations \
        --suffix_condition suffix_deception_yesno \
        --input_format final_token

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
import csv
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

    def __init__(self, activations_dir: str, pool_before_batch: bool = False, pooling_type: str = "mean", return_ids: bool = False):
        """
        Args:
            activations_dir: Path to directory with shards and manifest
            pool_before_batch: If True, pool (L, T, D) -> (L, D) during load
            pooling_type: Pooling strategy for pre-batching ("mean", "max", "last")
            return_ids: If True, __getitem__ returns (tensor, label, sample_id)
        """
        self.items = []
        self.pool_before_batch = pool_before_batch
        self.pooling_type = pooling_type
        self.return_ids = return_ids

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

                    if self.pool_before_batch and len(tensor.shape) == 3:
                        if self.pooling_type == "mean":
                            tensor = tensor.mean(dim=1)
                        elif self.pooling_type == "max":
                            tensor = tensor.max(dim=1).values
                        elif self.pooling_type == "last":
                            tensor = tensor[:, -1, :]
                        else:
                            raise ValueError(f"pool_before_batch does not support pooling '{self.pooling_type}'")

                    self.items.append({
                        "id": eid,
                        "tensor": tensor,  # (L, T, D) or (L, D)
                        "label": label
                    })

            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")

        logger.info(f"✓ Loaded {len(self.items)} examples")
        
        # Detect input format from tensor shape
        if self.items:
            sample_shape = self.items[0]['tensor'].shape
            if len(sample_shape) == 2:
                self.input_format = 'final_token'  # (L, D)
                logger.info(f"  • Input format: final_token (L={sample_shape[0]}, D={sample_shape[1]})")
            elif len(sample_shape) == 3:
                self.input_format = 'pooled'  # (L, T, D)
                logger.info(f"  • Input format: pooled (L={sample_shape[0]}, T={sample_shape[1]}, D={sample_shape[2]})")
            else:
                raise ValueError(f"Unexpected tensor shape: {sample_shape}")

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
        if self.return_ids:
            return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32), item['id']
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
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
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

def train_probe(model, train_loader, val_loader, device, args, checkpoint_dir=None, checkpoint_every=1):
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

    checkpoints = []
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
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

        # Save checkpoint if requested
        if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            checkpoints.append((epoch + 1, ckpt_path))

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

    return best_val_auc, best_epoch, checkpoints


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
    """Return list of gradient vectors for each sample in batch."""
    logits = model(x).squeeze(-1)
    losses = loss_fn(logits, y)
    grads = []
    for i in range(len(losses)):
        model.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=True)
        grads.append(_flatten_grads(model))
    model.zero_grad(set_to_none=True)
    return grads


def run_attribution_for_layer(
    layer_idx: int,
    model: nn.Module,
    checkpoints: list,
    w_star: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ood_loader: DataLoader,
    attr_every_n: int,
    attr_top_n: int,
    attr_out_dir: str,
    device: str,
):
    """
    Post-hoc attribution and checkpoint eval for a single layer probe.
    """
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
        # Load checkpoint
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        w_t = _flatten_params(model)
        cos_to_final = _cosine(w_t, w_star)
        w_norm = float(torch.norm(w_t).item()) if w_t.numel() else 0.0
        training_dynamics.append([epoch, cos_to_final, w_norm])

        # Checkpoint evaluation (ID + OOD)
        id_auc, id_acc, _, _ = evaluate(model, val_loader, device)
        best_id_auc = max(best_id_auc, id_auc)
        checkpoint_metrics.append([epoch, "id", id_auc, id_acc, 1 if id_auc >= best_id_auc else 0])

        if ood_loader is not None:
            ood_auc, ood_acc, _, _ = evaluate(model, ood_loader, device)
            best_ood_auc = max(best_ood_auc, ood_auc)
            checkpoint_metrics.append([epoch, "ood", ood_auc, ood_acc, 1 if ood_auc >= best_ood_auc else 0])

        # Attribution pass (subset of batches)
        delta = w_star - w_t
        for batch_idx, batch in enumerate(train_loader):
            if attr_every_n > 1 and batch_idx % attr_every_n != 0:
                continue
            if len(batch) == 3:
                x, y, ids = batch
            else:
                x, y = batch
                ids = [f"idx_{batch_idx}_{i}" for i in range(len(y))]

            x = x.to(device)
            y = y.to(device)

            grads = _compute_per_sample_grads(model, x, y, loss_fn)
            for i, g in enumerate(grads):
                sample_id = ids[i]
                alignment = -_cosine(g, w_star)
                influence = float(torch.dot(g, delta)) if g.numel() else 0.0

                sample_progress[sample_id] = sample_progress.get(sample_id, 0.0) + alignment
                sample_influence[sample_id] = sample_influence.get(sample_id, 0.0) + influence

                layer_progress[layer_idx] = layer_progress.get(layer_idx, 0.0) + alignment
                layer_influence[layer_idx] = layer_influence.get(layer_idx, 0.0) + influence
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1

    # Write outputs
    _write_csv(
        os.path.join(attr_out_dir, f"training_dynamics_layer_{layer_idx}.csv"),
        ["epoch", "cos_to_final", "w_norm"],
        training_dynamics
    )
    _write_csv(
        os.path.join(attr_out_dir, f"checkpoint_metrics_layer_{layer_idx}.csv"),
        ["epoch", "split", "auc", "acc", "best_so_far"],
        checkpoint_metrics
    )

    # Layer progress/influence
    layer_rows = []
    for l, total in layer_progress.items():
        count = max(layer_counts.get(l, 1), 1)
        layer_rows.append([l, total / count, count])
    _write_csv(
        os.path.join(attr_out_dir, f"layer_progress_layer_{layer_idx}.csv"),
        ["layer", "mean_grad_alignment", "count"],
        layer_rows
    )

    layer_inf_rows = []
    for l, total in layer_influence.items():
        count = max(layer_counts.get(l, 1), 1)
        layer_inf_rows.append([l, total / count, count])
    _write_csv(
        os.path.join(attr_out_dir, f"layer_influence_layer_{layer_idx}.csv"),
        ["layer", "mean_influence", "count"],
        layer_inf_rows
    )

    # Sample top-N
    top_prog = sorted(sample_progress.items(), key=lambda x: abs(x[1]), reverse=True)[:attr_top_n]
    _write_csv(
        os.path.join(attr_out_dir, f"sample_progress_top{attr_top_n}_layer_{layer_idx}.csv"),
        ["sample_id", "grad_alignment"],
        [[sid, score] for sid, score in top_prog]
    )

    top_inf = sorted(sample_influence.items(), key=lambda x: abs(x[1]), reverse=True)[:attr_top_n]
    _write_csv(
        os.path.join(attr_out_dir, f"sample_influence_top{attr_top_n}_layer_{layer_idx}.csv"),
        ["sample_id", "influence"],
        [[sid, score] for sid, score in top_inf]
    )

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
        choices=["mean", "max", "last", "attn", "none"],
        help="Token pooling strategy ('none' for final_token input format)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to probe (if None, trains on all layers)"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="auto",
        choices=["auto", "pooled", "final_token"],
        help="Input format: 'pooled' (L,T,D), 'final_token' (L,D), or 'auto' to detect"
    )
    parser.add_argument(
        "--pool_before_batch",
        action="store_true",
        help="Pool (L,T,D) -> (L,D) before batching (for raw variable-length activations)."
    )
    # Attribution / checkpointing
    parser.add_argument("--attr_enable", action="store_true", help="Enable attribution logging")
    parser.add_argument("--attr_every_n", type=int, default=50, help="Batch interval for attribution aggregation")
    parser.add_argument("--attr_top_n", type=int, default=100, help="Top-N samples to save")
    parser.add_argument("--attr_checkpoint_every", type=int, default=1, help="Checkpoint frequency (epochs)")
    parser.add_argument("--attr_out_dir", type=str, default=None, help="Output directory for attribution results")
    parser.add_argument("--ood_dataset", type=str, default="Deception-InsiderTrading", help="OOD dataset for checkpoint eval")
    parser.add_argument("--ood_split", type=str, default="test", help="OOD split for checkpoint eval")
    parser.add_argument("--ood_activations_dir", type=str, default=None, help="Base activations dir for OOD (default: --activations_dir)")
    parser.add_argument(
        "--suffix_condition",
        type=str,
        default=None,
        help="Suffix condition subdirectory (e.g., suffix_deception_yesno) for prompted activations"
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
    parser.add_argument(
        "--split_train_for_test",
        action="store_true",
        help="If set, splits 20%% of training data for validation (if validation split missing)"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load data
    # ========================================================================

    model_dir = args.model.replace("/", "_")

    # Build paths - handle prompted activations with suffix_condition subdirectory
    if args.suffix_condition:
        train_dir = os.path.join(args.activations_dir, model_dir, args.suffix_condition, args.dataset, "train")
        val_dir = os.path.join(args.activations_dir, model_dir, args.suffix_condition, args.dataset, "validation")
        test_dir = os.path.join(args.activations_dir, model_dir, args.suffix_condition, args.dataset, "test")
    else:
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

    if args.pool_before_batch and args.pooling == "attn":
        logger.error("--pool_before_batch does not support pooling 'attn' (requires token dimension).")
        return 1

    # Load datasets
    train_dataset = CachedDeceptionDataset(
        train_dir,
        pool_before_batch=args.pool_before_batch,
        pooling_type=args.pooling,
        return_ids=args.attr_enable
    )

    if splits_exist['validation']:
        val_dataset = CachedDeceptionDataset(
            val_dir,
            pool_before_batch=args.pool_before_batch,
            pooling_type=args.pooling,
            return_ids=args.attr_enable
        )
    else:
        if args.split_train_for_test:
            logger.warning("Validation split not found. splitting 20% from TRAIN.")
            # Splitting logic
            total = len(train_dataset.items)
            val_size = int(total * 0.2)
            if val_size < 1: val_size = 1
            
            # Split items
            val_items = train_dataset.items[-val_size:]
            train_dataset.items = train_dataset.items[:-val_size]
            
            # Create val dataset
            val_dataset = CachedDeceptionDataset.__new__(CachedDeceptionDataset)
            val_dataset.items = val_items
            val_dataset.input_format = train_dataset.input_format
            val_dataset.pool_before_batch = train_dataset.pool_before_batch
            val_dataset.pooling_type = train_dataset.pooling_type
            val_dataset.return_ids = train_dataset.return_ids
            
            logger.info(f"  Train: {len(train_dataset)} | Val (split): {len(val_dataset)}")
        else:
            logger.warning("Validation split not found, using train for validation")
            val_dataset = train_dataset

    if splits_exist['test']:
        test_dataset = CachedDeceptionDataset(
            test_dir,
            pool_before_batch=args.pool_before_batch,
            pooling_type=args.pooling,
            return_ids=args.attr_enable
        )
    else:
        logger.warning("Test split not found")
        test_dataset = None

    # Optional OOD dataset for checkpoint eval
    ood_dataset = None
    ood_dir = None
    if args.attr_enable and args.ood_dataset:
        ood_base = args.ood_activations_dir or args.activations_dir
        ood_dir = os.path.join(ood_base, model_dir, args.ood_dataset, args.ood_split)
        if os.path.exists(ood_dir):
            ood_dataset = CachedDeceptionDataset(
                ood_dir,
                pool_before_batch=args.pool_before_batch,
                pooling_type=args.pooling,
                return_ids=args.attr_enable
            )
        else:
            logger.warning(f"OOD activations not found: {ood_dir}")

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

    # Get tensor shape and input format from first batch
    first_batch = next(iter(train_loader))
    if len(first_batch) == 3:
        sample_x, _, _ = first_batch
    else:
        sample_x, _ = first_batch
    input_format = train_dataset.input_format
    
    model_pooling_type = args.pooling
    if input_format == 'final_token' or args.pool_before_batch:
        # (B, L, D)
        L, D = sample_x.shape[1], sample_x.shape[2]
        T = 1  # No token dimension
        logger.info(f"\nTensor shape: (batch, {L} layers, {D} dim) - final_token format")
        
        # Force pooling to 'none' for final_token format
        if not args.pool_before_batch and args.pooling not in ['none', 'last']:
            logger.warning(f"Overriding pooling '{args.pooling}' -> 'none' for final_token format")
            model_pooling_type = 'none'
    else:
        # (B, L, T, D)
        L, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]
        logger.info(f"\nTensor shape: (batch, {L} layers, {T} tokens, {D} dim) - pooled format")

    if args.pool_before_batch:
        model_pooling_type = 'none'
        logger.info("Pooling before batch: dataset returns pre-pooled (L, D); using IdentityPooling in model.")

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
    if args.suffix_condition:
        output_dir = os.path.join(
            args.output_dir,
            model_dir,
            args.suffix_condition,
            args.dataset,
            args.pooling
        )
    else:
        output_dir = os.path.join(
            args.output_dir,
            model_dir,
            args.dataset,
            args.pooling
        )
    os.makedirs(output_dir, exist_ok=True)

    # Attribution output directory
    if args.attr_enable:
        attr_out_dir = args.attr_out_dir or os.path.join(
            "results/probe_attribution",
            model_dir,
            args.dataset,
            args.pooling
        )
        os.makedirs(attr_out_dir, exist_ok=True)
    else:
        attr_out_dir = None

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
        attr_checkpoint_dir = None
        if args.attr_enable:
            attr_checkpoint_dir = os.path.join(attr_out_dir, "checkpoints", f"layer_{args.layer}")

        class LayerDataset(Dataset):
            def __init__(self, base_dataset, layer_idx):
                self.base = base_dataset
                self.layer_idx = layer_idx

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                item = self.base[idx]
                if len(item) == 3:
                    x, y, eid = item
                    return x[self.layer_idx], y, eid
                x, y = item
                return x[self.layer_idx], y

        layer_train_dataset = LayerDataset(train_dataset, args.layer)
        layer_val_dataset = LayerDataset(val_dataset, args.layer)

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

        model = LayerProbe(
            input_dim=D,
            pooling_type=model_pooling_type
        ).to(device)

        best_val_auc, best_epoch, checkpoints = train_probe(
            model,
            layer_train_loader,
            layer_val_loader,
            device,
            args,
            checkpoint_dir=attr_checkpoint_dir if args.attr_enable else None,
            checkpoint_every=args.attr_checkpoint_every
        )

        # Test
        if test_dataset:
            model.load_state_dict(torch.load(args.output_model_path))
            layer_test_dataset = LayerDataset(test_dataset, args.layer)
            layer_test_loader = DataLoader(
                layer_test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )
            test_auc, test_acc, _, _ = evaluate(model, layer_test_loader, device)
            logger.info(f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")

        # Attribution + checkpoint eval
        if args.attr_enable and checkpoints:
            model.load_state_dict(torch.load(args.output_model_path))
            w_star = _flatten_params(model)

            attr_train_loader = DataLoader(
                layer_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

            layer_ood_loader = None
            if ood_dataset is not None:
                layer_ood_dataset = LayerDataset(ood_dataset, args.layer)
                layer_ood_loader = DataLoader(
                    layer_ood_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0
                )

            run_attribution_for_layer(
                layer_idx=args.layer,
                model=model,
                checkpoints=checkpoints,
                w_star=w_star,
                train_loader=attr_train_loader,
                val_loader=layer_val_loader,
                ood_loader=layer_ood_loader,
                attr_every_n=args.attr_every_n,
                attr_top_n=args.attr_top_n,
                attr_out_dir=attr_out_dir,
                device=device
            )

    else:
        # Train on all layers
        logger.info(f"Training probes for all {L} layers...")

        results = []

        for layer_idx in range(L):
            logger.info(f"\n{'='*70}")
            logger.info(f"Layer {layer_idx}/{L-1}")
            logger.info(f"{'='*70}")

            args.output_model_path = os.path.join(output_dir, f"probe_layer_{layer_idx}.pt")
            attr_checkpoint_dir = None
            if args.attr_enable:
                attr_checkpoint_dir = os.path.join(attr_out_dir, "checkpoints", f"layer_{layer_idx}")

            # Extract single layer from batch
            # Need to modify dataset to return single layer? Or slice in training loop?
            # For simplicity, train on all layers and probe will select via indexing

            model = LayerProbe(
                input_dim=D,
                pooling_type=model_pooling_type
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
                    item = self.base[idx]
                    if len(item) == 3:
                        x, y, eid = item
                        return x[self.layer_idx], y, eid
                    x, y = item
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

            best_val_auc, best_epoch, checkpoints = train_probe(
                model,
                layer_train_loader,
                layer_val_loader,
                device,
                args,
                checkpoint_dir=attr_checkpoint_dir if args.attr_enable else None,
                checkpoint_every=args.attr_checkpoint_every
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

            # Attribution + checkpoint eval per layer
            if args.attr_enable and checkpoints:
                w_star = _flatten_params(model)

                attr_train_loader = DataLoader(
                    layer_train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0
                )

                layer_ood_loader = None
                if ood_dataset is not None:
                    layer_ood_dataset = LayerDataset(ood_dataset, layer_idx)
                    layer_ood_loader = DataLoader(
                        layer_ood_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0
                    )

                run_attribution_for_layer(
                    layer_idx=layer_idx,
                    model=model,
                    checkpoints=checkpoints,
                    w_star=w_star,
                    train_loader=attr_train_loader,
                    val_loader=layer_val_loader,
                    ood_loader=layer_ood_loader,
                    attr_every_n=args.attr_every_n,
                    attr_top_n=args.attr_top_n,
                    attr_out_dir=attr_out_dir,
                    device=device
                )

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

        if args.attr_enable and attr_out_dir:
            summary_rows = [[r['layer'], r['val_auc'], r['val_acc'], r['epoch']] for r in results]
            _write_csv(
                os.path.join(attr_out_dir, "summary.csv"),
                ["layer", "val_auc", "val_acc", "epoch"],
                summary_rows
            )

        # Save best_probe.json for eval_ood.py
        best_probe_info = {
            'best_layer': best_layer['layer'],
            'best_val_auc': best_layer['val_auc'],
            'best_val_acc': best_layer['val_acc'],
            'best_epoch': best_layer['epoch'],
            'probe_path': os.path.join(output_dir, f"probe_layer_{best_layer['layer']}.pt"),
            'pooling': args.pooling,
            'pool_before_batch': args.pool_before_batch,
            'model_pooling_type': model_pooling_type,
            'model': args.model,
            'dataset': args.dataset,
            'suffix_condition': args.suffix_condition if hasattr(args, 'suffix_condition') else None,
            'input_dim': D
        }
        best_probe_path = os.path.join(output_dir, "best_probe.json")
        with open(best_probe_path, 'w') as f:
            json.dump(best_probe_info, f, indent=2)
        logger.info(f"Saved best probe info to {best_probe_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Training Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}\n")

    return 0

if __name__ == "__main__":
    exit(main())
