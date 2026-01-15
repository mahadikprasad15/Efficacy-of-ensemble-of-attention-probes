"""
Representation Cartography.
Reduce dimensionality (PCA) of activations and plot 2D scatter.
Colors:
1. By Label (Correct vs Incorrect)
2. By Dataset (if combining multiple datasets)

Usage:
    python scripts/analysis/analyze_pca.py --model ... --dataset Movies --layer 12
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.features.resample import resample_activations # Likely unneeded if loading resampled

def load_data(data_dir, model, dataset, split="validation", limit=2000):
    pattern = os.path.join(data_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    shards = sorted(glob.glob(pattern))
    activations = []
    labels = []
    ids = []
    
    count = 0
    for shard in tqdm(shards, desc=f"Loading {dataset}"):
        try:
            tensors = load_file(shard)
            manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")
            with open(manifest_path) as f:
                for line in f:
                    meta = json.loads(line)
                    eid = meta['id']
                    if eid in tensors:
                        # (L', T', D)
                        activations.append(tensors[eid].float().numpy())
                        labels.append(meta.get('label', 0))
                        ids.append(eid)
                        count += 1
                        if limit and count >= limit: break
        except: pass
        if limit and count >= limit: break
            
    return np.array(activations), np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--compare_dataset", type=str, default=None, help="Second dataset for domain comparison")
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="plots/pca")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    args = parser.parse_args()

    # 1. Load Data
    X1, y1 = load_data(args.data_dir, args.model, args.dataset)
    X2, y2 = None, None
    if args.compare_dataset:
        X2, y2 = load_data(args.data_dir, args.model, args.compare_dataset)
        
    if len(X1) == 0:
        print("No data.")
        return

    # 2. Extract Features
    # Input is (N, L, T, D)
    # We need (N, D) for the specific layer.
    # Apply pooling first?
    # Usually "Mean of Layer"
    
    def extract(X):
        # Select Layer
        X_layer = X[:, args.layer, :, :] # (N, T, D)
        
        if args.pooling == "mean":
            return np.mean(X_layer, axis=1) # (N, D)
        elif args.pooling == "last":
            return X_layer[:, -1, :]
        return np.mean(X_layer, axis=1) # Default

    feats1 = extract(X1)
    labels = list(y1)
    sources = [args.dataset] * len(y1)
    final_feats = feats1
    
    if X2 is not None:
        feats2 = extract(X2)
        # Concatenate
        final_feats = np.concatenate([feats1, feats2], axis=0)
        labels.extend(list(y2))
        sources.extend([args.compare_dataset] * len(y2))
        
    print(f"Features Shape: {final_feats.shape} (Layer {args.layer})")
    
    # 3. PCA
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(final_feats)
    coords = pca.fit_transform(scaled_feats)
    
    print(f"Explained Variance: {pca.explained_variance_ratio_}")
    
    # 4. Plot
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: By Label (Correctness)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=labels, style=labels, palette="coolwarm", alpha=0.7)
    plt.title(f"PCA Layer {args.layer} - By Correctness")
    plt.savefig(os.path.join(args.output_dir, f"pca_correctness_{args.layer}.pdf"))
    
    # Plot 2: By Dataset (if valid)
    if args.compare_dataset:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=sources, palette="Set2", alpha=0.7)
        plt.title(f"PCA Layer {args.layer} - By Dataset")
        plt.savefig(os.path.join(args.output_dir, f"pca_domain_{args.layer}.pdf"))
        
    print(f"Saved plots to {args.output_dir}")

if __name__ == "__main__":
    main()
