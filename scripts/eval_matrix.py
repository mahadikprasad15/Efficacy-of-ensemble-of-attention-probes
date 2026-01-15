"""
Generate 5x5 Generalization Matrix from LODO runs.
Supports accuracy and AUC.

Usage:
    python scripts/eval_matrix.py --model ... --metric accuracy
"""

import argparse
import os
import sys
import json
import torch
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, shards: list):
        self.items = []
        for shard in shards:
            try:
                tensors = load_file(shard)
                manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")
                with open(manifest_path) as f:
                     for line in f:
                         meta = json.loads(line)
                         eid = meta['id']
                         if eid in tensors:
                             label = meta.get('label', 0)
                             self.items.append({
                                 "tensor": tensors[eid],
                                 "label": label
                             })
            except: pass
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)

ALL_DATASETS = ["HotpotQA", "HotpotQA-WC", "Movies", "TriviaQA", "IMDB"]

def get_shards(base_dir, model, dataset, split):
    pattern = os.path.join(base_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    return sorted(glob.glob(pattern))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "accuracy"])
    parser.add_argument("--output_file", type=str, default="generalization_matrix.pdf")
    args = parser.parse_args()

    matrix = pd.DataFrame(index=ALL_DATASETS, columns=ALL_DATASETS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for lodo_target in tqdm(ALL_DATASETS, desc="LODO Experiments"):
        exp_path = os.path.join(args.experiments_dir, f"lodo_{lodo_target}")
        results_path = os.path.join(exp_path, "results.jsonl")
        
        if not os.path.exists(results_path):
            continue
            
        # 1. Determine Best Layer
        best_layer_idx = -1
        best_auc = -1
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                # Use Min Val AUC to pick layer regardless of final metric
                if r['min_val_auc'] > best_auc:
                    best_auc = r['min_val_auc']
                    best_layer_idx = r['layer']
        
        # 2. Load Probe
        # CHANGED: Now looking for probe_layer_X.pt (always saved) or best_probe_layer_X
        # Training script saves BOTH now.
        probe_path = os.path.join(exp_path, f"best_probe_layer_{best_layer_idx}.pt")
        if not os.path.exists(probe_path):
             probe_path = os.path.join(exp_path, f"probe_layer_{best_layer_idx}.pt")
             
        if not os.path.exists(probe_path):
            print(f"Probe weights not found for {lodo_target}")
            continue
            
        # Get dim hack
        input_dim = 2048 # Fallback
        test_shard = get_shards(args.activations_dir, args.model, lodo_target, "validation")
        if test_shard:
             t = load_file(test_shard[0])
             k = list(t.keys())[0]
             input_dim = t[k].shape[-1]
             
        probe = LayerProbe(input_dim, pooling_type=args.pooling).to(device)
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        probe.eval()
        
        # 3. Evaluate on All
        for eval_ds_name in ALL_DATASETS:
            shards = get_shards(args.activations_dir, args.model, eval_ds_name, "validation")
            if not shards:
                continue
                
            ds = CachedDataset(shards)
            loader = DataLoader(ds, batch_size=32)
            
            preds, targets = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x[:, best_layer_idx, :, :].to(device)
                    logits = probe(x)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
                    targets.extend(y.numpy())
            
            try:
                if args.metric == "auc":
                    val = roc_auc_score(targets, preds)
                else:
                    val = accuracy_score(targets, np.array(preds).round())
            except: val = 0.5
            
            matrix.loc[lodo_target, eval_ds_name] = val

    print("Matrix:")
    print(matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, cmap="viridis", vmin=0.5 if args.metric=="auc" else 0, vmax=1.0)
    plt.title(f"Gen Matrix ({args.metric})")
    plt.ylabel("Train Set (LODO Target Out)")
    plt.xlabel("Eval Dataset")
    plt.tight_layout()
    plt.savefig(args.output_file)

if __name__ == "__main__":
    main()
