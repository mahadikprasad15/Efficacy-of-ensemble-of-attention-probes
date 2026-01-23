#!/usr/bin/env python3
"""
Compare Probe Directions Between Datasets
==========================================
Compares probe weight vectors from probes trained on different datasets
to understand if they learn the same or different "deception" directions.

Usage:
    python scripts/compare_probe_directions.py \
        --probes_dir_a data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
        --probes_dir_b data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
        --label_a "Roleplaying" \
        --label_b "InsiderTrading" \
        --output_dir results/probe_comparison
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Compare probe directions between datasets')
parser.add_argument('--probes_dir_a', type=str, required=True,
                    help='Path to first probes directory (contains pooling subdirs)')
parser.add_argument('--probes_dir_b', type=str, required=True,
                    help='Path to second probes directory')
parser.add_argument('--label_a', type=str, default='Dataset A',
                    help='Label for first dataset')
parser.add_argument('--label_b', type=str, default='Dataset B',
                    help='Label for second dataset')
parser.add_argument('--output_dir', type=str, default='results/probe_comparison',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"Comparing probes:")
print(f"  A: {args.probes_dir_a} ({args.label_a})")
print(f"  B: {args.probes_dir_b} ({args.label_b})")

# ============================================================================
# LOAD ALL PROBE WEIGHTS
# ============================================================================
print("\nLoading probe weights...")

pooling_types = ['mean', 'max', 'last', 'attn']
all_weights_a = {}  # pooling -> layer -> weights
all_weights_b = {}

def load_probe_weights(probes_dir, pooling, layer):
    """Load probe weights for a specific layer"""
    probe_path = os.path.join(probes_dir, pooling, f'probe_layer_{layer}.pt')
    if not os.path.exists(probe_path):
        return None
    
    # Get input dimension from the state dict
    state_dict = torch.load(probe_path, map_location='cpu')
    weights = state_dict['classifier.weight'].numpy().flatten()
    bias = state_dict['classifier.bias'].numpy().item()
    return {'weights': weights, 'bias': bias}

# Determine number of layers from available probes
sample_dir = os.path.join(args.probes_dir_a, 'last')
if os.path.exists(sample_dir):
    probe_files = glob.glob(os.path.join(sample_dir, 'probe_layer_*.pt'))
    num_layers = len(probe_files)
    print(f"  Found {num_layers} layers")
else:
    num_layers = 28  # Default

# Load all weights
for pooling in pooling_types:
    all_weights_a[pooling] = {}
    all_weights_b[pooling] = {}
    
    for layer in range(num_layers):
        w_a = load_probe_weights(args.probes_dir_a, pooling, layer)
        w_b = load_probe_weights(args.probes_dir_b, pooling, layer)
        
        if w_a is not None:
            all_weights_a[pooling][layer] = w_a
        if w_b is not None:
            all_weights_b[pooling][layer] = w_b

print(f"  Loaded weights for {len(pooling_types)} pooling types × {num_layers} layers")

# ============================================================================
# COMPUTE COSINE SIMILARITIES
# ============================================================================
print("\nComputing cosine similarities...")

cosine_sims = {}  # pooling -> list of similarities per layer
layer_indices = {}

for pooling in pooling_types:
    sims = []
    layers = []
    
    for layer in range(num_layers):
        if layer in all_weights_a[pooling] and layer in all_weights_b[pooling]:
            w_a = all_weights_a[pooling][layer]['weights']
            w_b = all_weights_b[pooling][layer]['weights']
            
            # Cosine similarity (1 - cosine distance)
            sim = 1 - cosine(w_a, w_b)
            sims.append(sim)
            layers.append(layer)
    
    cosine_sims[pooling] = sims
    layer_indices[pooling] = layers

# Print summary
print("\n" + "="*70)
print("COSINE SIMILARITY SUMMARY")
print("="*70)

for pooling in pooling_types:
    if cosine_sims[pooling]:
        sims = cosine_sims[pooling]
        print(f"\n{pooling.upper()}:")
        print(f"  Mean: {np.mean(sims):.4f}")
        print(f"  Std:  {np.std(sims):.4f}")
        print(f"  Min:  {np.min(sims):.4f} (Layer {layer_indices[pooling][np.argmin(sims)]})")
        print(f"  Max:  {np.max(sims):.4f} (Layer {layer_indices[pooling][np.argmax(sims)]})")

# ============================================================================
# PLOT 1: Layer-wise Cosine Similarity
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = {'mean': '#e74c3c', 'max': '#3498db', 'last': '#27ae60', 'attn': '#9b59b6'}

for i, pooling in enumerate(pooling_types):
    ax = axes[i]
    layers = layer_indices[pooling]
    sims = cosine_sims[pooling]
    
    ax.bar(layers, sims, color=colors[pooling], alpha=0.7, edgecolor='white')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=np.mean(sims), color='black', linestyle='--', alpha=0.5, 
               label=f'Mean: {np.mean(sims):.3f}')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'{pooling.upper()} Pooling\n{args.label_a} vs {args.label_b}')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight layers with high/low similarity
    for j, (layer, sim) in enumerate(zip(layers, sims)):
        if sim > 0.5:
            ax.annotate(f'{sim:.2f}', (layer, sim), ha='center', va='bottom', fontsize=8, color='green')
        elif sim < -0.5:
            ax.annotate(f'{sim:.2f}', (layer, sim), ha='center', va='top', fontsize=8, color='red')

plt.suptitle(f'Probe Direction Similarity: {args.label_a} vs {args.label_b}', fontsize=14, fontweight='bold')
plt.tight_layout()

layerwise_path = os.path.join(args.output_dir, 'cosine_similarity_layerwise.png')
plt.savefig(layerwise_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {layerwise_path}")

# ============================================================================
# PLOT 2: All Poolings Comparison
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 6))

x = np.arange(num_layers)
width = 0.2

for i, pooling in enumerate(pooling_types):
    layers = layer_indices[pooling]
    sims = cosine_sims[pooling]
    # Create full array with NaN for missing layers
    full_sims = np.full(num_layers, np.nan)
    for j, layer in enumerate(layers):
        full_sims[layer] = sims[j]
    
    ax.bar(x + i * width, full_sims, width, label=pooling.upper(), color=colors[pooling], alpha=0.7)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Cosine Similarity')
ax.set_title(f'Probe Direction Similarity Across All Pooling Types\n{args.label_a} vs {args.label_b}')
ax.legend()
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(range(num_layers))
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-1, 1)

comparison_path = os.path.join(args.output_dir, 'cosine_similarity_all_poolings.png')
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {comparison_path}")

# ============================================================================
# PLOT 3: Heatmap of similarities
# ============================================================================
fig3, ax = plt.subplots(figsize=(14, 4))

# Create matrix: pooling × layers
sim_matrix = np.full((len(pooling_types), num_layers), np.nan)
for i, pooling in enumerate(pooling_types):
    for j, layer in enumerate(layer_indices[pooling]):
        sim_matrix[i, layer] = cosine_sims[pooling][j]

im = ax.imshow(sim_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_yticks(range(len(pooling_types)))
ax.set_yticklabels([p.upper() for p in pooling_types])
ax.set_xlabel('Layer')
ax.set_title(f'Cosine Similarity Heatmap: {args.label_a} vs {args.label_b}\n(Red = same direction, Blue = opposite)')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Cosine Similarity')

# Add text annotations
for i in range(len(pooling_types)):
    for j in range(num_layers):
        if not np.isnan(sim_matrix[i, j]):
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center', 
                   fontsize=7, color=color)

heatmap_path = os.path.join(args.output_dir, 'cosine_similarity_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {heatmap_path}")

# ============================================================================
# ADDITIONAL ANALYSIS: Weight Vector Statistics
# ============================================================================
print("\n" + "="*70)
print("WEIGHT VECTOR ANALYSIS")
print("="*70)

# For best performing layer (layer 18, LAST pooling based on earlier results)
best_layer = 18
best_pooling = 'last'

if best_layer in all_weights_a[best_pooling] and best_layer in all_weights_b[best_pooling]:
    w_a = all_weights_a[best_pooling][best_layer]['weights']
    w_b = all_weights_b[best_pooling][best_layer]['weights']
    
    print(f"\nBest Layer Analysis ({best_pooling.upper()} Layer {best_layer}):")
    print(f"  Cosine Similarity: {1 - cosine(w_a, w_b):.4f}")
    print(f"  Angle (degrees): {np.degrees(np.arccos(1 - cosine(w_a, w_b))):.2f}°")
    print(f"  ||w_a||: {np.linalg.norm(w_a):.4f}")
    print(f"  ||w_b||: {np.linalg.norm(w_b):.4f}")
    
    # Correlation of weight values
    corr, p_val = spearmanr(w_a, w_b)
    print(f"  Spearman correlation: {corr:.4f} (p={p_val:.2e})")
    
    # Decomposition
    # Project w_a onto w_b
    proj_a_on_b = np.dot(w_a, w_b) / np.dot(w_b, w_b) * w_b
    component_parallel = np.linalg.norm(proj_a_on_b)
    component_orthogonal = np.linalg.norm(w_a - proj_a_on_b)
    
    print(f"\n  Decomposition of {args.label_a} probe direction:")
    print(f"    Parallel to {args.label_b}: {component_parallel:.4f} ({100*component_parallel/np.linalg.norm(w_a):.1f}%)")
    print(f"    Orthogonal to {args.label_b}: {component_orthogonal:.4f} ({100*component_orthogonal/np.linalg.norm(w_a):.1f}%)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    'config': {
        'probes_dir_a': args.probes_dir_a,
        'probes_dir_b': args.probes_dir_b,
        'label_a': args.label_a,
        'label_b': args.label_b
    },
    'cosine_similarities': {
        pooling: {
            'layers': layer_indices[pooling],
            'similarities': cosine_sims[pooling],
            'mean': float(np.mean(cosine_sims[pooling])) if cosine_sims[pooling] else None,
            'std': float(np.std(cosine_sims[pooling])) if cosine_sims[pooling] else None
        }
        for pooling in pooling_types
    }
}

results_path = os.path.join(args.output_dir, 'probe_comparison.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
print(f"\n✓ Saved: {results_path}")

print("\n" + "="*70)
print("✅ PROBE DIRECTION COMPARISON COMPLETE")
print("="*70)
