#!/usr/bin/env python3
"""
PCA Visualization of Activations Colored by Error Type
=======================================================
Visualizes activation space with PCA, colored by TP/TN/FP/FN,
and shows probe decision direction.

Usage:
    python scripts/visualize_activation_pca.py \
        --ood_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/test \
        --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
        --pooling last \
        --layer 18 \
        --output_dir results_flipped/pca_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='PCA Visualization of Activations by Error Type')
parser.add_argument('--ood_dir', type=str, required=True,
                    help='Path to OOD activations directory')
parser.add_argument('--probes_dir', type=str, required=True,
                    help='Path to probes directory (contains pooling subdirs)')
parser.add_argument('--pooling', type=str, default='last',
                    choices=['mean', 'max', 'last', 'attn'],
                    help='Pooling strategy to analyze')
parser.add_argument('--layer', type=int, default=None,
                    help='Specific layer to use (default: best from layer_results.json)')
parser.add_argument('--output_dir', type=str, default='results/pca_analysis',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"OOD Dir: {args.ood_dir}")
print(f"Probes Dir: {args.probes_dir}")
print(f"Pooling: {args.pooling}")

# ============================================================================
# LOAD OOD ACTIVATIONS
# ============================================================================
print("\nLoading OOD activations...")
manifest_path = os.path.join(args.ood_dir, 'manifest.jsonl')
with open(manifest_path, 'r') as f:
    manifest = [json.loads(line) for line in f]

shards = sorted(glob.glob(os.path.join(args.ood_dir, 'shard_*.safetensors')))
all_tensors = {}
for shard_path in shards:
    all_tensors.update(load_file(shard_path))

samples = []
for entry in manifest:
    eid = entry['id']
    if eid in all_tensors:
        samples.append({
            'id': eid,
            'tensor': all_tensors[eid],
            'label': entry['label']
        })

print(f"✓ Loaded {len(samples)} samples")
label_dist = Counter([s['label'] for s in samples])
print(f"  Labels: {dict(label_dist)}")

# ============================================================================
# LOAD PROBE
# ============================================================================
probe_dir = os.path.join(args.probes_dir, args.pooling)
layer_results_path = os.path.join(probe_dir, 'layer_results.json')

if args.layer is None:
    with open(layer_results_path, 'r') as f:
        layer_results = json.load(f)
    best_layer = max(layer_results, key=lambda x: x['val_auc'])
    layer_idx = best_layer['layer']
    print(f"\n📊 Using best layer: {layer_idx} (Val AUC: {best_layer['val_auc']:.4f})")
else:
    layer_idx = args.layer
    print(f"\n📊 Using specified layer: {layer_idx}")

D = samples[0]['tensor'].shape[-1]
probe_path = os.path.join(probe_dir, f'probe_layer_{layer_idx}.pt')
probe = LayerProbe(input_dim=D, pooling_type=args.pooling).to(device)
probe.load_state_dict(torch.load(probe_path, map_location=device))
probe.eval()
print(f"✓ Loaded probe from {probe_path}")

# Extract probe weight vector
probe_weights = probe.classifier.weight.data.cpu().numpy().flatten()  # Shape: (D,)
probe_bias = probe.classifier.bias.data.cpu().item()
print(f"  Probe weight norm: {np.linalg.norm(probe_weights):.4f}")
print(f"  Probe bias: {probe_bias:.4f}")

# ============================================================================
# POOL ACTIVATIONS AND RUN INFERENCE
# ============================================================================
print("\nPooling activations and running inference...")

pooled_activations = []
categories = []
gold_labels = []
pred_labels = []
confidences = []

for sample in tqdm(samples, desc="Processing"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    x_layer = tensor[:, layer_idx, :, :]  # Shape: (1, T, D)
    
    # Pool the activation
    if args.pooling == 'mean':
        pooled = x_layer.mean(dim=1)  # (1, D)
    elif args.pooling == 'max':
        pooled = x_layer.max(dim=1)[0]  # (1, D)
    elif args.pooling == 'last':
        pooled = x_layer[:, -1, :]  # (1, D) - last token
    elif args.pooling == 'attn':
        # For attn pooling, we need to use the probe's pooling
        pooled = probe.pooling(x_layer.to(device)).cpu()
    
    pooled_np = pooled.numpy().flatten()  # (D,)
    pooled_activations.append(pooled_np)
    
    # Get prediction
    with torch.no_grad():
        logit = probe(x_layer.to(device)).cpu().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
    
    pred = 1 if prob > 0.5 else 0
    gold = sample['label']
    
    # Classify
    if gold == 1 and pred == 1:
        cat = 'TP'
    elif gold == 0 and pred == 0:
        cat = 'TN'
    elif gold == 0 and pred == 1:
        cat = 'FP'
    else:
        cat = 'FN'
    
    categories.append(cat)
    gold_labels.append(gold)
    pred_labels.append(pred)
    confidences.append(prob)

X = np.array(pooled_activations)
print(f"✓ Pooled activations shape: {X.shape}")

# Count categories
cat_counts = Counter(categories)
print(f"  TP: {cat_counts['TP']}, TN: {cat_counts['TN']}, FP: {cat_counts['FP']}, FN: {cat_counts['FN']}")

# ============================================================================
# PCA
# ============================================================================
print("\nRunning PCA...")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Also scale probe weights for visualization
probe_weights_scaled = (probe_weights - scaler.mean_) / scaler.scale_

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.2%}")

# Project probe weights onto PCA space
probe_pca = pca.transform(probe_weights_scaled.reshape(1, -1))[0]
print(f"  Probe direction in PCA: ({probe_pca[0]:.4f}, {probe_pca[1]:.4f})")

# ============================================================================
# PLOT 1: PCA colored by TP/TN/FP/FN
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Colors and markers for each category
cat_styles = {
    'TP': {'color': '#27ae60', 'marker': 'o', 'label': 'TP (Deceptive ✓)'},
    'TN': {'color': '#3498db', 'marker': 's', 'label': 'TN (Honest ✓)'},
    'FP': {'color': '#e74c3c', 'marker': '^', 'label': 'FP (Honest → Deceptive ✗)'},
    'FN': {'color': '#f39c12', 'marker': 'v', 'label': 'FN (Deceptive → Honest ✗)'}
}

# Plot 1: By error category
ax1 = axes[0]
for cat in ['TN', 'TP', 'FN', 'FP']:  # Plot errors on top
    mask = [c == cat for c in categories]
    if sum(mask) > 0:
        X_cat = X_pca[mask]
        style = cat_styles[cat]
        ax1.scatter(X_cat[:, 0], X_cat[:, 1], 
                   c=style['color'], marker=style['marker'],
                   s=80, alpha=0.7, label=f"{style['label']} ({sum(mask)})",
                   edgecolors='white', linewidths=0.5)

# Draw probe direction as arrow
arrow_scale = np.max(np.abs(X_pca)) * 0.8
probe_dir_norm = probe_pca / np.linalg.norm(probe_pca)
ax1.annotate('', xy=(probe_dir_norm[0] * arrow_scale, probe_dir_norm[1] * arrow_scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.annotate('Probe Direction\n(→ Deceptive)', 
            xy=(probe_dir_norm[0] * arrow_scale * 0.7, probe_dir_norm[1] * arrow_scale * 0.7),
            fontsize=10, ha='center')

# Draw perpendicular decision boundary
perp = np.array([-probe_dir_norm[1], probe_dir_norm[0]])
boundary_scale = np.max(np.abs(X_pca))
ax1.plot([perp[0] * -boundary_scale, perp[0] * boundary_scale],
        [perp[1] * -boundary_scale, perp[1] * boundary_scale],
        'k--', alpha=0.3, lw=1, label='Decision Boundary')

ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.set_title(f'Activation PCA: {args.pooling.upper()} Layer {layer_idx}\nColored by Error Category')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: By gold label with confidence as size
ax2 = axes[1]
for gold in [0, 1]:
    mask = [g == gold for g in gold_labels]
    if sum(mask) > 0:
        X_gold = X_pca[mask]
        confs = np.array(confidences)[mask]
        # Size based on confidence (larger = more confident)
        sizes = 30 + 150 * np.abs(confs - 0.5) * 2  # Range 30-180
        color = '#e74c3c' if gold == 1 else '#3498db'
        label = 'Deceptive (Gold)' if gold == 1 else 'Honest (Gold)'
        ax2.scatter(X_gold[:, 0], X_gold[:, 1], 
                   c=color, s=sizes, alpha=0.6, label=f'{label} ({sum(mask)})',
                   edgecolors='white', linewidths=0.5)

# Draw probe direction
ax2.annotate('', xy=(probe_dir_norm[0] * arrow_scale, probe_dir_norm[1] * arrow_scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax2.set_title(f'Activation PCA: Gold Labels\n(Size = Confidence)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
pca_plot_path = os.path.join(args.output_dir, 'activation_pca.png')
plt.savefig(pca_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {pca_plot_path}")

# ============================================================================
# PLOT 2: Confidence distribution by category
# ============================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

cat_order = ['TP', 'TN', 'FP', 'FN']
positions = []
data_to_plot = []

for i, cat in enumerate(cat_order):
    mask = [c == cat for c in categories]
    if sum(mask) > 0:
        data_to_plot.append(np.array(confidences)[mask])
        positions.append(i)

bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)

colors = ['#27ae60', '#3498db', '#e74c3c', '#f39c12']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')
ax.set_xticks(range(len(cat_order)))
ax.set_xticklabels([f'{cat}\n(n={cat_counts[cat]})' for cat in cat_order])
ax.set_ylabel('Prediction Confidence')
ax.set_xlabel('Category')
ax.set_title(f'Confidence Distribution by Error Category\n{args.pooling.upper()} Layer {layer_idx}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

conf_plot_path = os.path.join(args.output_dir, 'confidence_distribution.png')
plt.savefig(conf_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {conf_plot_path}")

# ============================================================================
# PLOT 3: PROBE DIRECTION PROJECTION (Most Important!)
# ============================================================================
print("\nComputing probe direction projections...")

# Project all activations onto probe direction (this is what the classifier actually uses)
# logit = w · x + b, so we compute w · x for each sample
probe_scores = X @ probe_weights  # Raw dot product with probe weights

# Decision threshold in score space: when sigmoid(score + bias) = 0.5, score = -bias
decision_threshold = -probe_bias

print(f"  Decision threshold (score): {decision_threshold:.4f}")
print(f"  Score range: [{probe_scores.min():.4f}, {probe_scores.max():.4f}]")

# Create figure with 2 subplots
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Histogram by gold label
ax_hist = axes[0]
honest_scores = probe_scores[[g == 0 for g in gold_labels]]
deceptive_scores = probe_scores[[g == 1 for g in gold_labels]]

bins = np.linspace(probe_scores.min(), probe_scores.max(), 30)
ax_hist.hist(honest_scores, bins=bins, alpha=0.6, color='#3498db', label=f'Honest ({len(honest_scores)})', edgecolor='white')
ax_hist.hist(deceptive_scores, bins=bins, alpha=0.6, color='#e74c3c', label=f'Deceptive ({len(deceptive_scores)})', edgecolor='white')
ax_hist.axvline(x=decision_threshold, color='black', linestyle='--', lw=2, label=f'Decision Threshold ({decision_threshold:.2f})')
ax_hist.set_xlabel('Probe Score (w · x)')
ax_hist.set_ylabel('Count')
ax_hist.set_title(f'Distribution Along Probe Direction\n{args.pooling.upper()} Layer {layer_idx}')
ax_hist.legend()
ax_hist.grid(True, alpha=0.3, axis='y')

# Add separation metrics
overlap_left = np.sum((honest_scores > decision_threshold))  # FP
overlap_right = np.sum((deceptive_scores < decision_threshold))  # FN
total_correct = len(honest_scores) - overlap_left + len(deceptive_scores) - overlap_right
accuracy = total_correct / len(probe_scores)
ax_hist.text(0.02, 0.98, f'FP: {overlap_left}, FN: {overlap_right}\nAccuracy: {accuracy:.1%}', 
             transform=ax_hist.transAxes, va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right: Strip plot by category
ax_strip = axes[1]
cat_colors_strip = {'TP': '#27ae60', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#f39c12'}
y_positions = {'TN': 0, 'FN': 1, 'TP': 2, 'FP': 3}

for cat in ['TN', 'FN', 'TP', 'FP']:
    mask = [c == cat for c in categories]
    if sum(mask) > 0:
        cat_scores = probe_scores[mask]
        y_jitter = np.random.normal(y_positions[cat], 0.1, len(cat_scores))
        ax_strip.scatter(cat_scores, y_jitter, c=cat_colors_strip[cat], alpha=0.6, s=50, 
                        label=f'{cat} ({sum(mask)})', edgecolors='white', linewidths=0.5)

ax_strip.axvline(x=decision_threshold, color='black', linestyle='--', lw=2, label='Threshold')
ax_strip.set_xlabel('Probe Score (w · x)')
ax_strip.set_yticks([0, 1, 2, 3])
ax_strip.set_yticklabels(['TN (Honest ✓)', 'FN (Deceptive→Honest)', 'TP (Deceptive ✓)', 'FP (Honest→Deceptive)'])
ax_strip.set_title(f'Probe Scores by Error Category\n← Honest | Deceptive →')
ax_strip.legend(loc='upper right')
ax_strip.grid(True, alpha=0.3, axis='x')

# Shade regions
xlim = ax_strip.get_xlim()
ax_strip.axvspan(xlim[0], decision_threshold, alpha=0.1, color='blue', label='Predict Honest')
ax_strip.axvspan(decision_threshold, xlim[1], alpha=0.1, color='red', label='Predict Deceptive')
ax_strip.set_xlim(xlim)

plt.tight_layout()
probe_proj_path = os.path.join(args.output_dir, 'probe_direction_projection.png')
plt.savefig(probe_proj_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {probe_proj_path}")

# Print score statistics
print("\n📊 Probe Score Statistics:")
for cat in ['TP', 'TN', 'FP', 'FN']:
    mask = [c == cat for c in categories]
    if sum(mask) > 0:
        cat_scores = probe_scores[mask]
        print(f"  {cat}: mean={np.mean(cat_scores):.3f}, std={np.std(cat_scores):.3f}, range=[{np.min(cat_scores):.3f}, {np.max(cat_scores):.3f}]")

# Separation quality
honest_mean = np.mean(honest_scores)
deceptive_mean = np.mean(deceptive_scores)
pooled_std = np.sqrt((np.var(honest_scores) * len(honest_scores) + np.var(deceptive_scores) * len(deceptive_scores)) / len(probe_scores))
d_prime = (deceptive_mean - honest_mean) / pooled_std if pooled_std > 0 else 0
print(f"\n  d' (separability): {d_prime:.3f}")
print(f"  Interpretation: {'Good' if d_prime > 1.5 else 'Moderate' if d_prime > 0.8 else 'Poor'} separation")

# ============================================================================
# COMPUTE CLUSTER STATISTICS
# ============================================================================
print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

# Compute centroids for each category
for cat in ['TP', 'TN', 'FP', 'FN']:
    mask = [c == cat for c in categories]
    if sum(mask) > 0:
        centroid = X_pca[mask].mean(axis=0)
        std = X_pca[mask].std(axis=0)
        print(f"{cat} centroid: ({centroid[0]:+.3f}, {centroid[1]:+.3f}) ± ({std[0]:.3f}, {std[1]:.3f})")

# Compute distance from FP/FN to TP/TN centroids
print("\nDistance Analysis:")
tp_mask = [c == 'TP' for c in categories]
tn_mask = [c == 'TN' for c in categories]
fp_mask = [c == 'FP' for c in categories]
fn_mask = [c == 'FN' for c in categories]

if sum(tp_mask) > 0 and sum(tn_mask) > 0:
    tp_centroid = X_pca[tp_mask].mean(axis=0)
    tn_centroid = X_pca[tn_mask].mean(axis=0)
    
    if sum(fp_mask) > 0:
        fp_points = X_pca[fp_mask]
        fp_to_tp = np.mean([np.linalg.norm(p - tp_centroid) for p in fp_points])
        fp_to_tn = np.mean([np.linalg.norm(p - tn_centroid) for p in fp_points])
        print(f"  FP avg distance to TP centroid: {fp_to_tp:.3f}")
        print(f"  FP avg distance to TN centroid: {fp_to_tn:.3f}")
        print(f"  → FP is closer to: {'TP (good)' if fp_to_tp < fp_to_tn else 'TN (expected - they are honest)'}")
    
    if sum(fn_mask) > 0:
        fn_points = X_pca[fn_mask]
        fn_to_tp = np.mean([np.linalg.norm(p - tp_centroid) for p in fn_points])
        fn_to_tn = np.mean([np.linalg.norm(p - tn_centroid) for p in fn_points])
        print(f"  FN avg distance to TP centroid: {fn_to_tp:.3f}")
        print(f"  FN avg distance to TN centroid: {fn_to_tn:.3f}")
        print(f"  → FN is closer to: {'TP (expected - they are deceptive)' if fn_to_tp < fn_to_tn else 'TN (explains error!)'}")

# Save analysis to JSON
analysis = {
    'config': {
        'ood_dir': args.ood_dir,
        'probes_dir': args.probes_dir,
        'pooling': args.pooling,
        'layer': layer_idx
    },
    'pca': {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'probe_direction_pca': probe_pca.tolist()
    },
    'categories': {cat: int(cat_counts[cat]) for cat in cat_order},
    'centroids': {cat: X_pca[[c == cat for c in categories]].mean(axis=0).tolist() 
                  for cat in cat_order if cat_counts[cat] > 0}
}

analysis_path = os.path.join(args.output_dir, 'pca_analysis.json')
with open(analysis_path, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\n✓ Saved: {analysis_path}")

print("\n" + "="*70)
print("✅ PCA ANALYSIS COMPLETE")
print("="*70)
