"""
Analyze Probe Weight Similarity.
Generates a heatmap of Cosine Similarity between layer probes.

Usage:
    python scripts/analysis/analyze_weights.py --experiment_dir experiments/lodo_Movies
"""

import argparse
import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots/weights")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--input_dim", type=int, default=2048, help="Hack if not inferred")
    args = parser.parse_args()

    device = "cpu" # Load safely
    
    # 1. Load All Probes
    weights = []
    indices = []
    
    # Check max layers
    L = 32 # Arbitrary max scan
    for l_idx in range(L):
        p_path = os.path.join(args.experiment_dir, f"probe_layer_{l_idx}.pt")
        if os.path.exists(p_path):
            try:
                # Load state dict
                sd = torch.load(p_path, map_location=device)
                
                # Extract linear.weight
                # LayerProbe.linear is nn.Linear
                # shape (1, D)
                w = sd['linear.weight'].numpy().flatten()
                weights.append(w)
                indices.append(l_idx)
            except Exception as e:
                print(f"Error loading {l_idx}: {e}")
                
    if not weights:
        print("No weights found.")
        return
        
    weights = np.array(weights) # (L_found, D)
    print(f"Loaded {len(weights)} probes.")
    
    # 2. Compute Similarity
    sim_matrix = cosine_similarity(weights)
    
    # 3. Plot
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=False, cmap="coolwarm", xticklabels=indices, yticklabels=indices)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.title(f"Probe Weight Cosine Similarity")
    plt.savefig(os.path.join(args.output_dir, "cosine_sim.pdf"))
    print(f"Saved plot to {args.output_dir}")

if __name__ == "__main__":
    main()
