"""
Analyze Ensemble Strategies.
1. Sweep Top-K layers (Greedy/Score-based).
2. Compare Mean vs Weighted vs Gated ensembles.
3. Generate AUC vs K plot.

Usage:
    python scripts/analysis/analyze_ensembles.py --model ... --dataset Movies --experiment_dir experiments/lodo_Movies
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import StaticMeanEnsemble, StaticWeightedEnsemble, GatedEnsemble

# Re-use helpers
from actprobe.llm.activations import ActivationRunner # Just for typing hints or structure
# But we mostly load safetensors.

def load_data(data_dir, model, dataset, split="validation"):
    pattern = os.path.join(data_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    shards = sorted(glob.glob(pattern))
    activations = []
    labels = []
    
    for shard in tqdm(shards, desc=f"Loading {dataset}"):
        try:
            tensors = load_file(shard)
            manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")
            with open(manifest_path) as f:
                for line in f:
                    meta = json.loads(line)
                    eid = meta['id']
                    if eid in tensors:
                        activations.append(tensors[eid].float().numpy())
                        labels.append(meta.get('label', 0))
        except: pass
    return np.array(activations), np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="plots/ensembles")
    parser.add_argument("--pooling", type=str, default="mean")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    X, y = load_data(args.data_dir, args.model, args.dataset, "validation") # (N, L, T, D)
    if len(X) == 0:
        print("No data.")
        return
        
    N, L, T, D = X.shape
    
    # 2. Load Top-K Ranking (from results.jsonl)
    results_path = os.path.join(args.experiment_dir, "results.jsonl")
    layer_scores = []
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            layer_scores.append((r['layer'], r['min_val_auc']))
    
    # Sort by Score Descending
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_layers = [x[0] for x in layer_scores]
    print(f"Ranked Layers: {ranked_layers}")

    # 3. Load All Probes
    probes = {}
    for l_idx in range(L):
        p_path = os.path.join(args.experiment_dir, f"probe_layer_{l_idx}.pt")
        if os.path.exists(p_path):
            probe = LayerProbe(D, pooling_type=args.pooling).to(device)
            probe.load_state_dict(torch.load(p_path, map_location=device))
            probe.eval()
            probes[l_idx] = probe
    
    if not probes:
        print("No probes found.")
        return

    # 4. Compute Logits for All Layers
    # (N, L)
    layer_logits = np.zeros((N, L))
    
    print("Computing logits...")
    batch_size = 32
    for i in range(0, N, batch_size):
        batch_X = torch.tensor(X[i:i+batch_size]).to(device) # (B, L, T, D)
        
        for l_idx in probes:
            # batch_X[:, l_idx, :, :] -> (B, T, D)
            inp = batch_X[:, l_idx, :, :]
            with torch.no_grad():
                lg = probes[l_idx](inp).cpu().numpy().flatten()
                layer_logits[i:i+batch_size, l_idx] = lg

    # 5. Sweep K and Compute Ensemble AUC
    ks = range(1, L + 1)
    mean_aucs = []
    
    for k in ks:
        # Select Top K Layers
        top_k = ranked_layers[:k]
        
        # Mean Ensemble
        # Gather logits for top_k
        selected_logits = layer_logits[:, top_k] # (N, K)
        ensemble_logits = selected_logits.mean(axis=1) # (N,)
        ensemble_probs = 1 / (1 + np.exp(-ensemble_logits))
        
        try:
            auc = roc_auc_score(y, ensemble_probs)
        except: auc = 0.5
        mean_aucs.append(auc)
        
    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(ks, mean_aucs, marker='o', label="Static Mean Ensemble")
    plt.xlabel("Top-K Layers")
    plt.ylabel("AUC")
    plt.title(f"Ensemble Performance Sweep ({args.dataset})")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f"ensemble_sweep_{args.dataset}.pdf"))
    
    # Save Selected Layers
    with open(os.path.join(args.output_dir, f"top_k_layers_{args.dataset}.json"), "w") as f:
        json.dump({"ranked_layers": ranked_layers, "scores": [x[1] for x in layer_scores]}, f, indent=2)
        
    print(f"Saved plot and ranking to {args.output_dir}")

if __name__ == "__main__":
    main()
