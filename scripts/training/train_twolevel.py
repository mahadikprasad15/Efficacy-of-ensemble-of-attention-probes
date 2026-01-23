"""
Train two-level attention probes (Variant C from spec).

Trains a unified model that learns:
1. Token-level attention within each layer
2. Layer-level attention across layers
3. Final classification

This is the "cheapest" approach to capture ACT-ViT's idea of adaptive
(layer, token) location selection without the full ViT architecture.

Usage:
    python scripts/train_twolevel.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --held_out_dataset Movies \\
        --shared_token_attn
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

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import TwoLevelAttentionProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_DATASETS = ["HotpotQA", "HotpotQA-WC", "Movies", "TriviaQA", "IMDB"]


class CachedDataset(Dataset):
    def __init__(self, shards: list):
        self.items = []
        for shard in tqdm(shards, desc="Loading Shards"):
            try:
                tensors = load_file(shard)
                manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")

                with open(manifest_path) as f:
                    for line in f:
                        meta = json.loads(line)
                        eid = meta['id']
                        if eid in tensors:
                            self.items.append({
                                "tensor": tensors[eid],  # (L, T, D)
                                "label": meta.get('label', 0)
                            })
            except Exception as e:
                logger.warning(f"Error loading shard {shard}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)


def get_shards(base_dir, model, dataset, split):
    pattern = os.path.join(base_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    return sorted(glob.glob(pattern))


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)  # (B, L, T, D)
            logits = model(x)  # (B, 1)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(y.numpy())

    try:
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, (np.array(preds) > 0.5).astype(int))
    except:
        auc, acc = 0.5, 0.5

    return auc, acc


def train_twolevel(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Select datasets (LODO)
    train_datasets = [d for d in ALL_DATASETS if d != args.held_out_dataset]
    test_dataset = args.held_out_dataset

    logger.info(f"LODO Training: held_out={test_dataset}")
    logger.info(f"Train datasets: {train_datasets}")
    logger.info(f"Config: epochs={args.epochs}, lr={args.lr}, wd={args.wd}, shared_token_attn={args.shared_token_attn}")

    # 2. Load training data (union of all train datasets)
    train_shards = []
    for d in train_datasets:
        shards = get_shards(args.data_dir, args.model, d, args.train_split)
        train_shards.extend(shards)

    if not train_shards:
        logger.error("No training shards found")
        return

    train_ds = CachedDataset(train_shards)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 3. Load validation data for each train dataset
    val_loaders = {}
    for d in train_datasets:
        shards = get_shards(args.data_dir, args.model, d, "validation")
        if shards:
            ds = CachedDataset(shards)
            val_loaders[d] = DataLoader(ds, batch_size=args.batch_size)

    # 4. Load test data (held-out dataset)
    test_shards = get_shards(args.data_dir, args.model, test_dataset, "validation")
    test_loader = None
    if test_shards:
        ds = CachedDataset(test_shards)
        test_loader = DataLoader(ds, batch_size=args.batch_size)

    # 5. Initialize model
    sample, _ = train_ds[0]  # (L, T, D)
    num_layers, _, d_model = sample.shape

    logger.info(f"Model dimensions: L={num_layers}, D={d_model}")

    model = TwoLevelAttentionProbe(
        input_dim=d_model,
        num_layers=num_layers,
        shared_token_attn=args.shared_token_attn
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training loop with early stopping (worst-domain objective)
    best_min_val_auc = -1
    best_model_state = None
    patience_counter = 0

    epoch_pbar = tqdm(range(args.epochs), desc="Training Two-Level Attention", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0

        batch_pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{args.epochs}", leave=False, unit="batch")
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device).unsqueeze(1)  # (B, L, T, D), (B, 1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        batch_pbar.close()
        avg_loss = epoch_loss / len(train_loader)

        # Validation on all train datasets
        val_aucs = {}
        val_pbar = tqdm(val_loaders.items(), desc="  Validating", leave=False, unit="dataset")
        for dataset, loader in val_pbar:
            val_pbar.set_description(f"  Validating {dataset}")
            auc, _ = evaluate(model, loader, device)
            val_aucs[dataset] = auc
        val_pbar.close()

        min_val_auc = min(val_aucs.values()) if val_aucs else 0
        mean_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "min_auc": f"{min_val_auc:.4f}",
            "mean_auc": f"{mean_val_auc:.4f}",
            "best": f"{best_min_val_auc:.4f}"
        })

        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Min Val AUC={min_val_auc:.4f}, Mean Val AUC={mean_val_auc:.4f}")
        logger.info(f"  Per-dataset Val AUCs: {val_aucs}")

        # Early stopping based on worst-domain (min val AUC)
        if min_val_auc > best_min_val_auc:
            best_min_val_auc = min_val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            epoch_pbar.close()
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    epoch_pbar.close()

    # 7. Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model (Min Val AUC={best_min_val_auc:.4f})")

    # 8. Final evaluation on held-out test set
    if test_loader:
        test_auc, test_acc = evaluate(model, test_loader, device)
        logger.info(f"\n=== Final Test Results (OOD: {test_dataset}) ===")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Test Acc: {test_acc:.4f}")

        # Save results
        results = {
            "model": args.model,
            "held_out_dataset": test_dataset,
            "shared_token_attn": args.shared_token_attn,
            "best_min_val_auc": float(best_min_val_auc),
            "test_auc": float(test_auc),
            "test_acc": float(test_acc),
            "val_aucs": {k: float(v) for k, v in val_aucs.items()}
        }

        output_file = os.path.join(args.output_dir, f"twolevel_{test_dataset}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    # 9. Save model weights
    model_path = os.path.join(args.output_dir, f"twolevel_{test_dataset}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train two-level attention probes (Variant C)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--held_out_dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="data/twolevel")
    parser.add_argument("--batch_size", type=int, default=16, help="Smaller batch for memory")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--shared_token_attn", action="store_true", help="Use shared attention query for all layers")
    args = parser.parse_args()

    train_twolevel(args)


if __name__ == "__main__":
    main()
