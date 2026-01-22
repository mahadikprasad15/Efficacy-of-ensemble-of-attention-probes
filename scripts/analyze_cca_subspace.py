#!/usr/bin/env python3
"""
CCA Analysis: Find Shared Subspace Between Domains
===================================================
Uses Canonical Correlation Analysis to find shared dimensions
between Roleplaying and InsiderTrading activations.

Usage:
    python scripts/analyze_cca_subspace.py \
        --activations_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
        --activations_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
        --label_a "Roleplaying" \
        --label_b "InsiderTrading" \
        --layer 18 \
        --pooling last \
        --output_dir results/cca_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from safetensors.torch import load_file
from tqdm import tqdm

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='CCA Analysis for Shared Subspace')
parser.add_argument('--activations_a', type=str, required=True,
                    help='Path to first activations directory')
parser.add_argument('--activations_b', type=str, required=True,
                    help='Path to second activations directory')
parser.add_argument('--label_a', type=str, default='Domain A')
parser.add_argument('--label_b', type=str, default='Domain B')
parser.add_argument('--layer', type=int, default=18,
                    help='Layer to analyze')
parser.add_argument('--pooling', type=str, default='last',
                    choices=['mean', 'max', 'last'],
                    help='Pooling strategy')
parser.add_argument('--n_components', type=int, default=50,
                    help='Number of CCA components')
parser.add_argument('--output_dir', type=str, default='results/cca_analysis')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"CCA Analysis:")
print(f"  A: {args.activations_a} ({args.label_a})")
print(f"  B: {args.activations_b} ({args.label_b})")
print(f"  Layer: {args.layer}, Pooling: {args.pooling}")

# ============================================================================
# LOAD ACTIVATIONS
# ============================================================================
def load_activations(act_dir, layer, pooling):
    """Load and pool activations from a directory"""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations = []
    labels = []
    
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            tensor = all_tensors[eid]  # (L, T, D)
            x_layer = tensor[layer, :, :]  # (T, D)
            
            # Pool
            if pooling == 'mean':
                pooled = x_layer.mean(dim=0)
            elif pooling == 'max':
                pooled = x_layer.max(dim=0)[0]
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            
            activations.append(pooled.numpy())
            labels.append(entry['label'])
    
    return np.array(activations), np.array(labels)

print("\nLoading activations...")
X_a, y_a = load_activations(args.activations_a, args.layer, args.pooling)
X_b, y_b = load_activations(args.activations_b, args.layer, args.pooling)

print(f"  {args.label_a}: {X_a.shape}, labels: {np.bincount(y_a)}")
print(f"  {args.label_b}: {X_b.shape}, labels: {np.bincount(y_b)}")

# ============================================================================
# STANDARDIZE
# ============================================================================
scaler_a = StandardScaler()
scaler_b = StandardScaler()
X_a_scaled = scaler_a.fit_transform(X_a)
X_b_scaled = scaler_b.fit_transform(X_b)

# ============================================================================
# FIT CCA
# ============================================================================
print("\nFitting CCA...")

# CCA requires same number of samples - subsample to match
n_samples = min(len(X_a_scaled), len(X_b_scaled))
# Shuffle and select
np.random.seed(42)
idx_a = np.random.permutation(len(X_a_scaled))[:n_samples]
idx_b = np.random.permutation(len(X_b_scaled))[:n_samples]

X_a_sub = X_a_scaled[idx_a]
X_b_sub = X_b_scaled[idx_b]
y_a_sub = y_a[idx_a]
y_b_sub = y_b[idx_b]

# Fit CCA
n_components = min(args.n_components, n_samples - 1, X_a.shape[1])
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(X_a_sub, X_b_sub)

# Transform all data to shared space
X_a_cca, X_b_cca = cca.transform(X_a_scaled, X_b_scaled)

print(f"  CCA components: {n_components}")
print(f"  Shared space shape: {X_a_cca.shape}")

# ============================================================================
# ANALYZE CORRELATIONS IN CCA SPACE
# ============================================================================
print("\nAnalyzing canonical correlations...")

# Compute correlations for each component (on the subsampled data used for fitting)
X_a_sub_cca, X_b_sub_cca = cca.transform(X_a_sub, X_b_sub)
correlations = []
for i in range(n_components):
    corr = np.corrcoef(X_a_sub_cca[:, i], X_b_sub_cca[:, i])[0, 1]
    correlations.append(corr)

correlations = np.array(correlations)
print(f"  Top 5 canonical correlations: {correlations[:5]}")
print(f"  Mean correlation: {np.mean(correlations):.4f}")

# ============================================================================
# TEST: DO SHARED DIMENSIONS PRESERVE DECEPTION SIGNAL?
# ============================================================================
print("\n" + "="*70)
print("DECEPTION CLASSIFICATION IN SHARED SPACE")
print("="*70)

results = {}

# Test different numbers of CCA components
components_to_test = [5, 10, 20, min(50, n_components)]

for n_comp in components_to_test:
    if n_comp > n_components:
        continue
    
    X_a_proj = X_a_cca[:, :n_comp]
    X_b_proj = X_b_cca[:, :n_comp]
    
    # Train on A, test on A (sanity)
    clf_a = LogisticRegression(max_iter=1000, random_state=42)
    clf_a.fit(X_a_proj, y_a)
    auc_a_on_a = roc_auc_score(y_a, clf_a.predict_proba(X_a_proj)[:, 1])
    
    # Train on A, test on B (cross-domain in shared space)
    try:
        auc_a_on_b = roc_auc_score(y_b, clf_a.predict_proba(X_b_proj)[:, 1])
    except:
        auc_a_on_b = 0.5
    
    # Train on B, test on A (reverse)
    clf_b = LogisticRegression(max_iter=1000, random_state=42)
    clf_b.fit(X_b_proj, y_b)
    try:
        auc_b_on_a = roc_auc_score(y_a, clf_b.predict_proba(X_a_proj)[:, 1])
    except:
        auc_b_on_a = 0.5
    
    auc_b_on_b = roc_auc_score(y_b, clf_b.predict_proba(X_b_proj)[:, 1])
    
    print(f"\n{n_comp} CCA components:")
    print(f"  Train {args.label_a} → Test {args.label_a}: {auc_a_on_a:.4f} (ID)")
    print(f"  Train {args.label_a} → Test {args.label_b}: {auc_a_on_b:.4f} (OOD)")
    print(f"  Train {args.label_b} → Test {args.label_b}: {auc_b_on_b:.4f} (ID)")
    print(f"  Train {args.label_b} → Test {args.label_a}: {auc_b_on_a:.4f} (OOD)")
    
    results[n_comp] = {
        f'{args.label_a}_to_{args.label_a}': auc_a_on_a,
        f'{args.label_a}_to_{args.label_b}': auc_a_on_b,
        f'{args.label_b}_to_{args.label_b}': auc_b_on_b,
        f'{args.label_b}_to_{args.label_a}': auc_b_on_a
    }

# ============================================================================
# COMPARE WITH ORIGINAL SPACE
# ============================================================================
print("\n" + "="*70)
print("COMPARISON WITH ORIGINAL SPACE (NO CCA)")
print("="*70)

# Train on A, test on B in original space
clf_orig_a = LogisticRegression(max_iter=1000, random_state=42)
clf_orig_a.fit(X_a_scaled, y_a)
orig_auc_a_on_a = roc_auc_score(y_a, clf_orig_a.predict_proba(X_a_scaled)[:, 1])
try:
    orig_auc_a_on_b = roc_auc_score(y_b, clf_orig_a.predict_proba(X_b_scaled)[:, 1])
except:
    orig_auc_a_on_b = 0.5

clf_orig_b = LogisticRegression(max_iter=1000, random_state=42)
clf_orig_b.fit(X_b_scaled, y_b)
orig_auc_b_on_b = roc_auc_score(y_b, clf_orig_b.predict_proba(X_b_scaled)[:, 1])
try:
    orig_auc_b_on_a = roc_auc_score(y_a, clf_orig_b.predict_proba(X_a_scaled)[:, 1])
except:
    orig_auc_b_on_a = 0.5

print(f"\nOriginal space ({X_a.shape[1]}D):")
print(f"  Train {args.label_a} → Test {args.label_a}: {orig_auc_a_on_a:.4f} (ID)")
print(f"  Train {args.label_a} → Test {args.label_b}: {orig_auc_a_on_b:.4f} (OOD)")
print(f"  Train {args.label_b} → Test {args.label_b}: {orig_auc_b_on_b:.4f} (ID)")
print(f"  Train {args.label_b} → Test {args.label_a}: {orig_auc_b_on_a:.4f} (OOD)")

results['original'] = {
    f'{args.label_a}_to_{args.label_a}': orig_auc_a_on_a,
    f'{args.label_a}_to_{args.label_b}': orig_auc_a_on_b,
    f'{args.label_b}_to_{args.label_b}': orig_auc_b_on_b,
    f'{args.label_b}_to_{args.label_a}': orig_auc_b_on_a
}

# ============================================================================
# PLOT 1: Canonical Correlations
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Bar chart of canonical correlations
ax1 = axes[0, 0]
ax1.bar(range(len(correlations)), correlations, color='steelblue', alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Weak threshold (0.3)')
ax1.set_xlabel('CCA Component')
ax1.set_ylabel('Canonical Correlation')
ax1.set_title(f'Canonical Correlations: {args.label_a} vs {args.label_b}')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: CCA-projected space visualization
ax2 = axes[0, 1]
# Project to first 2 CCA components
for label, (X, y, color, marker) in [
    (args.label_a, (X_a_cca[:, :2], y_a, 'blue', 'o')),
    (args.label_b, (X_b_cca[:, :2], y_b, 'red', '^'))
]:
    honest = X[y == 0]
    deceptive = X[y == 1]
    ax2.scatter(honest[:, 0], honest[:, 1], c=color, marker=marker, alpha=0.4, 
               label=f'{label} Honest', s=30)
    ax2.scatter(deceptive[:, 0], deceptive[:, 1], c=color, marker=marker, alpha=0.8,
               label=f'{label} Deceptive', s=60, edgecolors='black', linewidths=0.5)

ax2.set_xlabel('CCA Component 1')
ax2.set_ylabel('CCA Component 2')
ax2.set_title('Samples in CCA Space (First 2 Components)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: AUC comparison across CCA components
ax3 = axes[1, 0]
comp_tested = sorted([k for k in results.keys() if k != 'original'])
x_pos = np.arange(len(comp_tested) + 1)
width = 0.2

# OOD transfer from A to B
ood_a_to_b = [results[c][f'{args.label_a}_to_{args.label_b}'] for c in comp_tested]
ood_a_to_b.append(results['original'][f'{args.label_a}_to_{args.label_b}'])

# OOD transfer from B to A
ood_b_to_a = [results[c][f'{args.label_b}_to_{args.label_a}'] for c in comp_tested]
ood_b_to_a.append(results['original'][f'{args.label_b}_to_{args.label_a}'])

ax3.bar(x_pos - width/2, ood_a_to_b, width, label=f'{args.label_a}→{args.label_b}', color='blue', alpha=0.7)
ax3.bar(x_pos + width/2, ood_b_to_a, width, label=f'{args.label_b}→{args.label_a}', color='red', alpha=0.7)
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([str(c) for c in comp_tested] + ['Original'])
ax3.set_xlabel('Number of CCA Components')
ax3.set_ylabel('AUC (OOD Transfer)')
ax3.set_title('Cross-Domain Transfer in CCA Space vs Original')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1)

# Plot 4: Summary table as text
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SUMMARY: CCA Analysis
{'='*50}

Canonical Correlations:
  Mean: {np.mean(correlations):.4f}
  Max:  {np.max(correlations):.4f}
  Components > 0.3: {np.sum(correlations > 0.3)}/{len(correlations)}

Best OOD Transfer in CCA Space:
  {args.label_a}→{args.label_b}: {max(ood_a_to_b[:-1]):.4f} (vs {ood_a_to_b[-1]:.4f} original)
  {args.label_b}→{args.label_a}: {max(ood_b_to_a[:-1]):.4f} (vs {ood_b_to_a[-1]:.4f} original)

Interpretation:
  {'✓ CCA improves OOD transfer' if max(ood_a_to_b[:-1]) > ood_a_to_b[-1] + 0.05 else '✗ CCA does NOT improve OOD transfer'}
  {'✓ Shared structure exists' if np.mean(correlations[:10]) > 0.3 else '✗ No shared structure found'}
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot_path = os.path.join(args.output_dir, 'cca_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output = {
    'config': {
        'activations_a': args.activations_a,
        'activations_b': args.activations_b,
        'label_a': args.label_a,
        'label_b': args.label_b,
        'layer': args.layer,
        'pooling': args.pooling,
        'n_components': n_components
    },
    'canonical_correlations': correlations.tolist(),
    'classification_results': results
}

json_path = os.path.join(args.output_dir, 'cca_results.json')
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
print(f"✓ Saved: {json_path}")

print("\n" + "="*70)
print("✅ CCA ANALYSIS COMPLETE")
print("="*70)
