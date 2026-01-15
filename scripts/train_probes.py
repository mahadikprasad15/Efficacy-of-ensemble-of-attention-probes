"""
Train probes using cached activation shards.
Supports LODO (Leave-One-Dataset-Out) selection.

Usage:
    python scripts/train_probes.py --model meta-llama/Llama-3.2-1B-Instruct --held_out_dataset Movies --pooling mean --layer_mode best
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_DATASETS = ["HotpotQA", "HotpotQA-WC", "Movies", "TriviaQA", "IMDB"]

class CachedDataset(Dataset):
    def __init__(self, shards: list):
        self.items = []
        logger.info(f"Loading {len(shards)} shards...")
        
        for shard in tqdm(shards, desc="Loading Shards"):
            try:
                tensors = load_file(shard)
                manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")
                
                # Load Manifest
                with open(manifest_path) as f:
                     for line in f:
                         meta = json.loads(line)
                         eid = meta['id']
                         # Filter by shard index if needed? 
                         # Usually manifest covers all shards in dir or just this shard?
                         # Our cache script writes ONE manifest for the whole run/split.
                         # And we saved "shard": index in meta.
                         # We match by filename index likely?
                         # Simpler: just match ID.
                         
                         if eid in tensors:
                             label = meta.get('label', 0)
                             self.items.append({
                                 "tensor": tensors[eid], # (L', T', D)
                                 "label": label
                             })
            except Exception as e:
                logger.warning(f"Error loading shard {shard}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # shape (L, T, D)
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)

def get_shards(base_dir, model, dataset, split):
    # data/activations/{model}/{dataset}/{split}/shard_*.safetensors
    pattern = os.path.join(base_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    return sorted(glob.glob(pattern))

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(y.numpy())
            
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = 0.5
    return auc, np.mean(np.array(preds).round() == np.array(targets))

def train_lodo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Select Datasets
    train_datasets = [d for d in ALL_DATASETS if d != args.held_out_dataset]
    val_id_datasets = train_datasets # Use validation split of train datasets as ID Val
    test_ood_dataset = args.held_out_dataset

    logger.info(f"Heterogeneous Training on: {train_datasets}")
    logger.info(f"Target OOD: {test_ood_dataset}")
    logger.info(f"Training Config: {args.epochs} epochs, lr={args.lr}, wd={args.wd}, patience={args.patience}")
    
    # 2. Load Data
    # TRAIN: (dataset, 'train') usually but for this demo we might only have 'validation'. 
    # Spec says: Train 10k, Val 2k. 
    # We will look for --train_split arg, default to 'train'.
    
    train_shards = []
    for d in train_datasets:
        train_shards.extend(get_shards(args.data_dir, args.model, d, args.train_split))
        
    if not train_shards:
        logger.error("No training shards found. Make sure to cache activations first.")
        return

    train_ds = CachedDataset(train_shards)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    # VAL ID: Dict {dataset: loader}
    val_loaders = {}
    for d in val_id_datasets:
        shards = get_shards(args.data_dir, args.model, d, "validation")
        if shards:
            ds = CachedDataset(shards)
            val_loaders[d] = DataLoader(ds, batch_size=args.batch_size)
            
    # TEST OOD
    test_shards = get_shards(args.data_dir, args.model, test_ood_dataset, "validation") # or test
    test_loader = None
    if test_shards:
        ds = CachedDataset(test_shards)
        test_loader = DataLoader(ds, batch_size=args.batch_size)
    
    # 3. Init Probe
    # Get dim from first item
    sample, _ = train_ds[0] # (L, T, D)
    d_model = sample.shape[-1]
    
    probe = LayerProbe(input_dim=d_model, pooling_type=args.pooling).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    logger.info(f"Starting Training: {args.epochs} epochs")
    
    best_min_val_auc = -1
    best_state = None
    
    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            
            # Layer Selection logic? 
            # Current LayerProbe takes (B, T, D) and pools. 
            # But input x is (B, L, T, D).
            # We need to select layers.
            # Simple Pooling Probe: Usually trains on ONE layer.
            # The script should iterate over LAYERS if we are doing Single-Layer probing.
            # OR if we are doing Ensemble, we use all.
            # Spec: "Variant A: Token pooling (per layer)... Train one classifier per layer".
            
            # Currently `train_probes.py` structure implies training ONE model.
            # We should probably loop over layers outside? 
            # Or `LayerProbe` handles (B, T, D) only.
            
            # Let's fix this: We are training ONE probe per layer.
            # So `train_lodo` should run for EACH layer? 
            # Or we simply pick a specific layer via args?
            pass

    # REVISIT: The user wants "Variant A... Train one classifier per layer, then pick layer".
    # So this script should probably Iterate Over Layers and train/eval each.
    # OR accept a --layer_idx arg.
    
    # Let's Implement: Train ALL layers independently (iteratively) and save results.
    
    num_layers = sample.shape[0]
    results = []
    
    for l_idx in range(num_layers):
        logger.info(f"Training Layer {l_idx}...")

        # Init fresh probe for this layer
        probe = LayerProbe(input_dim=d_model, pooling_type=args.pooling).to(device)
        optimizer = optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)

        # Early stopping tracking
        best_val_auc_for_layer = -1
        patience_counter = 0
        best_probe_state = None

        # Train with early stopping
        for epoch in range(args.epochs):
            probe.train()
            epoch_loss = 0
            for x, y in train_loader:
                x_layer = x[:, l_idx, :, :].to(device) # Select layer
                y = y.to(device).unsqueeze(1)

                optimizer.zero_grad()
                logits = probe(x_layer)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation check for early stopping
            probe.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for val_dataset, val_loader in val_loaders.items():
                    for bx, by in val_loader:
                        bx = bx[:, l_idx, :, :].to(device)
                        logits = probe(bx)
                        val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                        val_targets.extend(by.numpy())

            try:
                current_val_auc = roc_auc_score(val_targets, val_preds)
            except:
                current_val_auc = 0.5

            if current_val_auc > best_val_auc_for_layer:
                best_val_auc_for_layer = current_val_auc
                best_probe_state = probe.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info(f"Layer {l_idx}: Early stopping at epoch {epoch+1} (best val AUC: {best_val_auc_for_layer:.4f})")
                break

        # Restore best weights
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)

        # Final Eval
        val_aucs = {}
        for d, loader in val_loaders.items():
            # Create wrapper loader that yields layer specific? 
            # Or just fetch inside loop
            # Memory efficient to reuse loader?
            # We can't easily indexing dataloader. 
            # Just iterating.
            
            # Evaluate helper needs to handle data prep
            # Let's rewrite eval locally
            probe.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in loader:
                    bx = bx[:, l_idx, :, :].to(device)
                    logits = probe(bx)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
                    targets.extend(by.numpy())
            try:
                auc = roc_auc_score(targets, preds)
            except: auc = 0.5
            val_aucs[d] = auc
            
        min_val_auc = min(val_aucs.values()) if val_aucs else 0
        
        # Eval OOD
        ood_auc = 0
        if test_loader:
            probe.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx[:, l_idx, :, :].to(device)
                    logits = probe(bx)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
                    targets.extend(by.numpy())
            try:
                ood_auc = roc_auc_score(targets, preds)
            except: ood_auc = 0.5
            
        logger.info(f"Layer {l_idx}: Min Val AUC={min_val_auc:.4f}, OOD AUC={ood_auc:.4f}")
        
        results.append({
            "layer": l_idx,
            "val_aucs": val_aucs,
            "min_val_auc": min_val_auc,
            "ood_auc": ood_auc
        })
        
        # Save weights for analysis (Ensembles, Cosine Sim, etc require all layers)
        torch.save(probe.state_dict(), os.path.join(args.output_dir, f"probe_layer_{l_idx}.pt"))

        # Save weights if it's the best?
        if min_val_auc >= best_min_val_auc:
            best_min_val_auc = min_val_auc
            # Optional: duplicate as best_ for easy access, or just rely on results.jsonl to find it.
            # Let's keep best_ naming for existing compatibility or update eval_matrix?
            # Creating a copy is safer for now to avoid breaking existing runs if any.
            torch.save(probe.state_dict(), os.path.join(args.output_dir, f"best_probe_layer_{l_idx}.pt"))

    # Save results
    with open(os.path.join(args.output_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Pick Best Layer
    best_layer = max(results, key=lambda x: x['min_val_auc'])
    logger.info(f"Best Layer Selected: {best_layer['layer']} (Min Val {best_layer['min_val_auc']:.4f})")
    logger.info(f"OOD Performance: {best_layer['ood_auc']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train per-layer probes using LODO (Leave-One-Dataset-Out)")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--held_out_dataset", type=str, default="Movies", help="Dataset to hold out for OOD evaluation")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last", "attn"], help="Token pooling method")
    parser.add_argument("--data_dir", type=str, default="data/activations", help="Directory containing cached activations")
    parser.add_argument("--output_dir", type=str, default="data/probes", help="Directory to save trained probes")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--train_split", type=str, default="train", help="Split to use for training (use 'validation' for quick testing)")
    args = parser.parse_args()

    train_lodo(args)

if __name__ == "__main__":
    main()
