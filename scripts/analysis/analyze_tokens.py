"""
Analyze performance at specific token positions.
Generates a heatmap of AUC vs (Layer, Token Position).
Focuses on first 3 and last 3 tokens as requested.

Usage:
    python scripts/analysis/analyze_tokens.py --model meta-llama/Llama-2-7b --dataset Movies --pooling mean
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
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

def get_activations_and_labels(data_dir, model, dataset, split, limit=1000):
    pattern = os.path.join(data_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    shards = sorted(glob.glob(pattern))
    
    activations = []
    labels = []
    
    count = 0
    for shard in tqdm(shards, desc=f"Loading {dataset}"):
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
                    count += 1
                    if limit and count >= limit:
                        break
        if limit and count >= limit:
            break
            
    return np.array(activations), np.array(labels)

def train_and_eval_token(X, y, layer_idx, token_idx):
    # X: (N, L, T, D)
    # Extract features for specific layer and token
    # Handle negative token_idx
    
    features = X[:, layer_idx, token_idx, :] # (N, D)
    
    # Train/Val split
    # For simplicity, 5-fold CV or simple split?
    # Using simple split 80/20 for diagnostic
    n_train = int(0.8 * len(y))
    X_train, X_val = features[:n_train], features[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Check class balance
    if len(np.unique(y_train)) < 2: return 0.5
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Logistic Regression (Fast)
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_train, y_train)
    
    preds = clf.predict_proba(X_val)[:, 1]
    try:
        return roc_auc_score(y_val, preds)
    except: return 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="plots/tokens")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    # Load Data
    X, y = get_activations_and_labels(args.data_dir, args.model, args.dataset, "validation", args.limit)
    if len(X) == 0:
        print("No data found.")
        return
        
    N, L, T, D = X.shape
    print(f"Data Shape: {X.shape}")
    
    # Define Token Positions of Interest
    # First 3, Last 3
    # If T is small, avoid overlap
    indices = [0, 1, 2, -3, -2, -1]
    labels_pos = ["First", "2nd", "3rd", "3rd-Last", "2nd-Last", "Last"]
    
    heatmap = np.zeros((L, len(indices)))
    
    for l in range(L):
        for i, t_idx in enumerate(indices):
            print(f"Processing split {l} {t_idx}    ", end="\r")
            auc = train_and_eval_token(X, y, l, t_idx)
            heatmap[l, i] = auc
            
    print("\nDone.")
    
    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(8, 10))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels_pos, yticklabels=range(L))
    plt.xlabel("Token Position")
    plt.ylabel("Layer Index")
    plt.title(f"Diagnostic AUC Heatmap: {args.dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"token_heatmap_{args.dataset}.pdf"))
    print(f"Saved plot to {args.output_dir}")

if __name__ == "__main__":
    main()
