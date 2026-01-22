#!/usr/bin/env python3
"""
CCA Analysis: Find Shared Subspace Between Domains (Fixed Version)
===================================================================
Uses proper train/test splits:
- Fit CCA on TRAIN data from both domains
- Train classifiers on TRAIN data
- Evaluate on TEST data

Usage:
    python scripts/analyze_cca_subspace_v2.py \
        --train_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
        --test_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
        --train_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
        --test_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
        --label_a "Roleplaying" \
        --label_b "InsiderTrading" \
        --layer 20 \
        --pooling max \
        --output_dir results/cca_analysis_v2
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
parser = argparse.ArgumentParser(description='CCA Analysis with proper train/test splits')
parser.add_argument('--train_a', type=str, required=True, help='Train activations for domain A')
parser.add_argument('--test_a', type=str, required=True, help='Test activations for domain A')
parser.add_argument('--train_b', type=str, required=True, help='Train activations for domain B')
parser.add_argument('--test_b', type=str, required=True, help='Test activations for domain B')
parser.add_argument('--label_a', type=str, default='Domain A')
parser.add_argument('--label_b', type=str, default='Domain B')
parser.add_argument('--layer', type=int, default=20)
parser.add_argument('--pooling', type=str, default='max', choices=['mean', 'max', 'last'])
parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--output_dir', type=str, default='results/cca_analysis_v2')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"CCA Analysis (v2 - proper splits):")
print(f"  Domain A ({args.label_a}):")
print(f"    Train: {args.train_a}")
print(f"    Test:  {args.test_a}")
print(f"  Domain B ({args.label_b}):")
print(f"    Train: {args.train_b}")
print(f"    Test:  {args.test_b}")
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
            tensor = all_tensors[eid]
            x_layer = tensor[layer, :, :]
            
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
X_train_a, y_train_a = load_activations(args.train_a, args.layer, args.pooling)
X_test_a, y_test_a = load_activations(args.test_a, args.layer, args.pooling)
X_train_b, y_train_b = load_activations(args.train_b, args.layer, args.pooling)
X_test_b, y_test_b = load_activations(args.test_b, args.layer, args.pooling)

print(f"  {args.label_a} Train: {X_train_a.shape}, Test: {X_test_a.shape}")
print(f"  {args.label_b} Train: {X_train_b.shape}, Test: {X_test_b.shape}")

# ============================================================================
# STANDARDIZE (fit on train, apply to test)
# ============================================================================
scaler_a = StandardScaler()
scaler_b = StandardScaler()

X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

# ============================================================================
# BASELINE: Train on A, test on B (original space)
# ============================================================================
print("\n" + "="*70)
print("BASELINE: ORIGINAL SPACE (NO CCA)")
print("="*70)

# Train on A
clf_orig_a = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
clf_orig_a.fit(X_train_a_scaled, y_train_a)

# Evaluate
baseline_a_on_a = roc_auc_score(y_test_a, clf_orig_a.predict_proba(X_test_a_scaled)[:, 1])
baseline_a_on_b = roc_auc_score(y_test_b, clf_orig_a.predict_proba(X_test_b_scaled)[:, 1])

print(f"\nTrained on {args.label_a}:")
print(f"  Test on {args.label_a} (ID): {baseline_a_on_a:.4f}")
print(f"  Test on {args.label_b} (OOD): {baseline_a_on_b:.4f}")

# Train on B
clf_orig_b = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
clf_orig_b.fit(X_train_b_scaled, y_train_b)

baseline_b_on_b = roc_auc_score(y_test_b, clf_orig_b.predict_proba(X_test_b_scaled)[:, 1])
baseline_b_on_a = roc_auc_score(y_test_a, clf_orig_b.predict_proba(X_test_a_scaled)[:, 1])

print(f"\nTrained on {args.label_b}:")
print(f"  Test on {args.label_b} (ID): {baseline_b_on_b:.4f}")
print(f"  Test on {args.label_a} (OOD): {baseline_b_on_a:.4f}")

# ============================================================================
# FIT CCA ON TRAIN DATA
# ============================================================================
print("\n" + "="*70)
print("FITTING CCA ON TRAINING DATA")
print("="*70)

# CCA needs paired samples - subsample to match
n_cca_samples = min(len(X_train_a_scaled), len(X_train_b_scaled))
np.random.seed(42)
idx_a = np.random.permutation(len(X_train_a_scaled))[:n_cca_samples]
idx_b = np.random.permutation(len(X_train_b_scaled))[:n_cca_samples]

X_cca_fit_a = X_train_a_scaled[idx_a]
X_cca_fit_b = X_train_b_scaled[idx_b]

n_components = min(args.n_components, n_cca_samples - 1, X_train_a.shape[1])
cca = CCA(n_components=n_components, max_iter=2000)
cca.fit(X_cca_fit_a, X_cca_fit_b)

print(f"  CCA fitted on {n_cca_samples} paired samples")
print(f"  Number of components: {n_components}")

# ============================================================================
# TRANSFORM ALL DATA TO CCA SPACE
# ============================================================================
X_train_a_cca, _ = cca.transform(X_train_a_scaled, X_train_a_scaled)  # First arg matters
X_test_a_cca, _ = cca.transform(X_test_a_scaled, X_test_a_scaled)
_, X_train_b_cca = cca.transform(X_train_b_scaled, X_train_b_scaled)  # Second arg matters
_, X_test_b_cca = cca.transform(X_test_b_scaled, X_test_b_scaled)

# Compute canonical correlations
cca_a, cca_b = cca.transform(X_cca_fit_a, X_cca_fit_b)
correlations = []
for i in range(n_components):
    corr = np.corrcoef(cca_a[:, i], cca_b[:, i])[0, 1]
    correlations.append(corr)
correlations = np.array(correlations)

print(f"  Top 5 canonical correlations: {correlations[:5].round(3)}")

# ============================================================================
# CLASSIFICATION IN CCA SPACE
# ============================================================================
print("\n" + "="*70)
print("CLASSIFICATION IN CCA SPACE")
print("="*70)

results = {'baseline': {
    f'{args.label_a}_to_{args.label_a}': baseline_a_on_a,
    f'{args.label_a}_to_{args.label_b}': baseline_a_on_b,
    f'{args.label_b}_to_{args.label_b}': baseline_b_on_b,
    f'{args.label_b}_to_{args.label_a}': baseline_b_on_a
}}

for n_comp in [5, 10, 20, min(50, n_components)]:
    if n_comp > n_components:
        continue
    
    # Train on A in CCA space
    clf_cca_a = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf_cca_a.fit(X_train_a_cca[:, :n_comp], y_train_a)
    
    cca_a_on_a = roc_auc_score(y_test_a, clf_cca_a.predict_proba(X_test_a_cca[:, :n_comp])[:, 1])
    cca_a_on_b = roc_auc_score(y_test_b, clf_cca_a.predict_proba(X_test_b_cca[:, :n_comp])[:, 1])
    
    # Train on B in CCA space
    clf_cca_b = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf_cca_b.fit(X_train_b_cca[:, :n_comp], y_train_b)
    
    cca_b_on_b = roc_auc_score(y_test_b, clf_cca_b.predict_proba(X_test_b_cca[:, :n_comp])[:, 1])
    cca_b_on_a = roc_auc_score(y_test_a, clf_cca_b.predict_proba(X_test_a_cca[:, :n_comp])[:, 1])
    
    print(f"\n{n_comp} CCA components:")
    print(f"  Train {args.label_a} → Test {args.label_a}: {cca_a_on_a:.4f} (ID)")
    print(f"  Train {args.label_a} → Test {args.label_b}: {cca_a_on_b:.4f} (OOD) [baseline: {baseline_a_on_b:.4f}]")
    print(f"  Train {args.label_b} → Test {args.label_b}: {cca_b_on_b:.4f} (ID)")
    print(f"  Train {args.label_b} → Test {args.label_a}: {cca_b_on_a:.4f} (OOD) [baseline: {baseline_b_on_a:.4f}]")
    
    results[n_comp] = {
        f'{args.label_a}_to_{args.label_a}': cca_a_on_a,
        f'{args.label_a}_to_{args.label_b}': cca_a_on_b,
        f'{args.label_b}_to_{args.label_b}': cca_b_on_b,
        f'{args.label_b}_to_{args.label_a}': cca_b_on_a
    }

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Canonical correlations
ax1 = axes[0, 0]
ax1.bar(range(len(correlations)), correlations, color='steelblue', alpha=0.7)
ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Weak (0.3)')
ax1.set_xlabel('CCA Component')
ax1.set_ylabel('Canonical Correlation')
ax1.set_title(f'Canonical Correlations: {args.label_a} vs {args.label_b}')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Samples in CCA space
ax2 = axes[0, 1]
for label, X, y, color, marker in [
    (args.label_a, X_test_a_cca[:, :2], y_test_a, 'blue', 'o'),
    (args.label_b, X_test_b_cca[:, :2], y_test_b, 'red', '^')
]:
    ax2.scatter(X[y==0, 0], X[y==0, 1], c=color, marker=marker, alpha=0.4, s=30, label=f'{label} Honest')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c=color, marker=marker, alpha=0.8, s=60, 
               edgecolors='black', linewidths=0.5, label=f'{label} Deceptive')
ax2.set_xlabel('CCA 1')
ax2.set_ylabel('CCA 2')
ax2.set_title('Test Samples in CCA Space')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: OOD Transfer comparison
ax3 = axes[1, 0]
comp_tested = sorted([k for k in results.keys() if k != 'baseline'])
x = np.arange(len(comp_tested) + 1)
width = 0.35

ood_a_to_b = [results[c][f'{args.label_a}_to_{args.label_b}'] for c in comp_tested]
ood_a_to_b.append(baseline_a_on_b)
ood_b_to_a = [results[c][f'{args.label_b}_to_{args.label_a}'] for c in comp_tested]
ood_b_to_a.append(baseline_b_on_a)

ax3.bar(x - width/2, ood_a_to_b, width, label=f'{args.label_a}→{args.label_b}', color='blue', alpha=0.7)
ax3.bar(x + width/2, ood_b_to_a, width, label=f'{args.label_b}→{args.label_a}', color='red', alpha=0.7)
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels([str(c) for c in comp_tested] + ['Baseline\n(3072D)'])
ax3.set_xlabel('CCA Components')
ax3.set_ylabel('AUC (OOD Transfer)')
ax3.set_title('Cross-Domain Transfer: CCA vs Baseline')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0.3, 1.0)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
best_cca_a_to_b = max([results[c][f'{args.label_a}_to_{args.label_b}'] for c in comp_tested])
best_cca_b_to_a = max([results[c][f'{args.label_b}_to_{args.label_a}'] for c in comp_tested])

summary = f"""
SUMMARY: CCA Analysis (v2)
{'='*50}

Baseline (Original 3072D space):
  {args.label_a} → {args.label_a}: {baseline_a_on_a:.4f} (ID)
  {args.label_a} → {args.label_b}: {baseline_a_on_b:.4f} (OOD)
  {args.label_b} → {args.label_b}: {baseline_b_on_b:.4f} (ID)
  {args.label_b} → {args.label_a}: {baseline_b_on_a:.4f} (OOD)

Best CCA Space:
  {args.label_a} → {args.label_b}: {best_cca_a_to_b:.4f} (Δ {best_cca_a_to_b - baseline_a_on_b:+.4f})
  {args.label_b} → {args.label_a}: {best_cca_b_to_a:.4f} (Δ {best_cca_b_to_a - baseline_b_on_a:+.4f})

Canonical Correlations:
  Mean: {np.mean(correlations):.4f}
  Top 5: {correlations[:5].round(3)}

Interpretation:
  {'✓ CCA IMPROVES transfer' if best_cca_a_to_b > baseline_a_on_b + 0.02 or best_cca_b_to_a > baseline_b_on_a + 0.02 else '✗ CCA does NOT improve transfer'}
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'cca_analysis_v2.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(args.output_dir, 'cca_analysis_v2.png')}")

# Save JSON
with open(os.path.join(args.output_dir, 'cca_results_v2.json'), 'w') as f:
    json.dump({
        'config': vars(args),
        'correlations': correlations.tolist(),
        'results': results
    }, f, indent=2, default=float)

print("\n" + "="*70)
print("✅ CCA ANALYSIS V2 COMPLETE")
print("="*70)
