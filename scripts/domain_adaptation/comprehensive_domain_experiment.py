#!/usr/bin/env python3
"""
Comprehensive Domain Adaptation Experiment
===========================================
Fair comparison of all approaches on SAME train/test splits.

Compares:
1. Probe trained on Domain A only → tested on A (ID) and B (OOD)
2. Probe trained on Domain B only → tested on B (ID) and A (OOD)
3. Simple probe on Combined (A+B) → tested on both
4. Domain Adversarial on Combined → tested on both
5. Asymmetric: Varying % of Domain B data

Usage:
    python scripts/comprehensive_domain_experiment.py \
        --train_a data/activations/.../Deception-Roleplaying/train \
        --val_a data/activations/.../Deception-Roleplaying/validation \
        --train_b data/activations/.../Deception-InsiderTrading/train \
        --val_b data/activations/.../Deception-InsiderTrading/validation \
        --layer 20 --pooling max \
        --output_dir results/comprehensive_comparison
"""

import os
import sys
import json
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from safetensors.torch import load_file
from tqdm import tqdm

# ============================================================================
# MODELS
# ============================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        return grads * -ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class SimpleProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class DomainAdversarialProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.task_head = nn.Linear(hidden_dim, 1)
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lambda_=1.0):
        feats = self.features(x)
        task = self.task_head(feats).squeeze(-1)
        self.grl.lambda_ = lambda_
        domain = self.domain_head(self.grl(feats)).squeeze(-1)
        return task, domain
    
    def predict(self, x):
        return self.task_head(self.features(x)).squeeze(-1)

# ============================================================================
# UTILITIES
# ============================================================================
def load_activations(act_dir, layer, pooling):
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations, labels = [], []
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

def train_simple(X_train, y_train, X_val, y_val, device, epochs=30, lr=1e-3):
    """Train simple probe"""
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    
    model = SimpleProbe(X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 32):
            batch_x = X_t[perm[i:i+32]].to(device)
            batch_y = y_t[perm[i:i+32]].to(device)
            opt.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    
    return roc_auc_score(y_val, probs), model, mean, std

def train_adversarial(X_a, y_a, X_b, y_b, X_val_a, y_val_a, X_val_b, y_val_b, 
                      device, epochs=30, lr=1e-3, lambda_d=0.5):
    """Train domain adversarial probe on combined data"""
    X_train = np.vstack([X_a, X_b])
    y_train = np.concatenate([y_a, y_b])
    domain = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_a_n = (X_val_a - mean) / std
    X_val_b_n = (X_val_b - mean) / std
    
    model = DomainAdversarialProbe(X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_crit = nn.BCEWithLogitsLoss()
    domain_crit = nn.BCEWithLogitsLoss()
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    d_t = torch.tensor(domain, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        p = epoch / epochs
        lambda_sched = 2 / (1 + np.exp(-10 * p)) - 1
        
        for i in range(0, len(X_t), 32):
            batch_x = X_t[perm[i:i+32]].to(device)
            batch_y = y_t[perm[i:i+32]].to(device)
            batch_d = d_t[perm[i:i+32]].to(device)
            
            opt.zero_grad()
            task_logits, domain_logits = model(batch_x, lambda_=lambda_d * lambda_sched)
            loss = task_crit(task_logits, batch_y) + domain_crit(domain_logits, batch_d)
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        probs_a = torch.sigmoid(model.predict(torch.tensor(X_val_a_n, dtype=torch.float32).to(device))).cpu().numpy()
        probs_b = torch.sigmoid(model.predict(torch.tensor(X_val_b_n, dtype=torch.float32).to(device))).cpu().numpy()
    
    return roc_auc_score(y_val_a, probs_a), roc_auc_score(y_val_b, probs_b), model

# ============================================================================
# MAIN
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--train_a', type=str, required=True)
parser.add_argument('--val_a', type=str, required=True)
parser.add_argument('--train_b', type=str, required=True)
parser.add_argument('--val_b', type=str, required=True)
parser.add_argument('--label_a', type=str, default='Roleplaying')
parser.add_argument('--label_b', type=str, default='InsiderTrading')
parser.add_argument('--layer', type=int, default=20)
parser.add_argument('--pooling', type=str, default='max')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--output_dir', type=str, default='results/comprehensive_comparison')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Comprehensive Domain Experiment | Layer {args.layer}, {args.pooling}")
print(f"Device: {device}")

# Load data
print("\nLoading data...")
X_train_a, y_train_a = load_activations(args.train_a, args.layer, args.pooling)
X_val_a, y_val_a = load_activations(args.val_a, args.layer, args.pooling)
X_train_b, y_train_b = load_activations(args.train_b, args.layer, args.pooling)
X_val_b, y_val_b = load_activations(args.val_b, args.layer, args.pooling)

print(f"  {args.label_a}: Train={X_train_a.shape[0]}, Val={X_val_a.shape[0]}")
print(f"  {args.label_b}: Train={X_train_b.shape[0]}, Val={X_val_b.shape[0]}")

results = {}

# ============================================================================
# EXPERIMENT 1: Single Domain Probes
# ============================================================================
print("\n" + "="*70)
print("1. SINGLE DOMAIN PROBES")
print("="*70)

# Train on A only
auc_a_a, _, mean_a, std_a = train_simple(X_train_a, y_train_a, X_val_a, y_val_a, device, args.epochs)
# Test on B
X_val_b_norm = (X_val_b - mean_a) / std_a
model_a = SimpleProbe(X_train_a.shape[1]).to(device)
# Retrain for OOD test
auc_a_a, model_a, mean_a, std_a = train_simple(X_train_a, y_train_a, X_val_a, y_val_a, device, args.epochs)
model_a.eval()
with torch.no_grad():
    probs_a_on_b = torch.sigmoid(model_a(torch.tensor((X_val_b - mean_a) / std_a, dtype=torch.float32).to(device))).cpu().numpy()
auc_a_b = roc_auc_score(y_val_b, probs_a_on_b)

print(f"  Trained on {args.label_a}:")
print(f"    → {args.label_a} (ID):  {auc_a_a:.4f}")
print(f"    → {args.label_b} (OOD): {auc_a_b:.4f}")

# Train on B only
auc_b_b, model_b, mean_b, std_b = train_simple(X_train_b, y_train_b, X_val_b, y_val_b, device, args.epochs)
model_b.eval()
with torch.no_grad():
    probs_b_on_a = torch.sigmoid(model_b(torch.tensor((X_val_a - mean_b) / std_b, dtype=torch.float32).to(device))).cpu().numpy()
auc_b_a = roc_auc_score(y_val_a, probs_b_on_a)

print(f"  Trained on {args.label_b}:")
print(f"    → {args.label_b} (ID):  {auc_b_b:.4f}")
print(f"    → {args.label_a} (OOD): {auc_b_a:.4f}")

results['single_a'] = {'id': auc_a_a, 'ood': auc_a_b}
results['single_b'] = {'id': auc_b_b, 'ood': auc_b_a}

# ============================================================================
# EXPERIMENT 2: Combined (No Adversarial)
# ============================================================================
print("\n" + "="*70)
print("2. COMBINED TRAINING (NO ADVERSARIAL)")
print("="*70)

X_combined = np.vstack([X_train_a, X_train_b])
y_combined = np.concatenate([y_train_a, y_train_b])

# Train on combined, test on both
auc_comb_a, model_comb, mean_comb, std_comb = train_simple(X_combined, y_combined, X_val_a, y_val_a, device, args.epochs)
model_comb.eval()
with torch.no_grad():
    probs_comb_b = torch.sigmoid(model_comb(torch.tensor((X_val_b - mean_comb) / std_comb, dtype=torch.float32).to(device))).cpu().numpy()
auc_comb_b = roc_auc_score(y_val_b, probs_comb_b)

print(f"  Trained on Combined:")
print(f"    → {args.label_a}: {auc_comb_a:.4f}")
print(f"    → {args.label_b}: {auc_comb_b:.4f}")

results['combined'] = {'a': auc_comb_a, 'b': auc_comb_b}

# ============================================================================
# EXPERIMENT 3: Domain Adversarial
# ============================================================================
print("\n" + "="*70)
print("3. DOMAIN ADVERSARIAL TRAINING")
print("="*70)

auc_da_a, auc_da_b, model_da = train_adversarial(
    X_train_a, y_train_a, X_train_b, y_train_b,
    X_val_a, y_val_a, X_val_b, y_val_b,
    device, args.epochs
)

print(f"  Domain Adversarial:")
print(f"    → {args.label_a}: {auc_da_a:.4f}")
print(f"    → {args.label_b}: {auc_da_b:.4f}")

results['adversarial'] = {'a': auc_da_a, 'b': auc_da_b}

# ============================================================================
# EXPERIMENT 4: Asymmetric Data Sweep
# ============================================================================
print("\n" + "="*70)
print("4. ASYMMETRIC DATA SWEEP (Varying % of Domain B)")
print("="*70)

percentages = [0, 10, 20, 30, 50, 70, 100]
asymmetric_results = []

for pct in percentages:
    n_b = max(1, int(len(X_train_b) * pct / 100))
    X_b_subset = X_train_b[:n_b]
    y_b_subset = y_train_b[:n_b]
    
    if pct == 0:
        # Only A
        auc_a, _, _, _ = train_simple(X_train_a, y_train_a, X_val_a, y_val_a, device, args.epochs)
        auc_b = auc_a_b  # Use pre-computed OOD
    else:
        # Combined with subset of B
        X_comb = np.vstack([X_train_a, X_b_subset])
        y_comb = np.concatenate([y_train_a, y_b_subset])
        auc_a, model_t, mean_t, std_t = train_simple(X_comb, y_comb, X_val_a, y_val_a, device, args.epochs)
        model_t.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model_t(torch.tensor((X_val_b - mean_t) / std_t, dtype=torch.float32).to(device))).cpu().numpy()
        auc_b = roc_auc_score(y_val_b, probs)
    
    asymmetric_results.append({'pct': pct, 'auc_a': auc_a, 'auc_b': auc_b})
    print(f"  {pct:3d}% of {args.label_b}: {args.label_a}={auc_a:.4f}, {args.label_b}={auc_b:.4f}")

results['asymmetric'] = asymmetric_results

# ============================================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON")
print("="*70)

print(f"\n{'Model':<40} | {args.label_a:>12} | {args.label_b:>12}")
print("-"*70)
print(f"{'Probe on ' + args.label_a + ' only':<40} | {auc_a_a:>12.4f} | {auc_a_b:>12.4f} (OOD)")
print(f"{'Probe on ' + args.label_b + ' only':<40} | {auc_b_a:>12.4f} (OOD) | {auc_b_b:>12.4f}")
print(f"{'Combined (no adversarial)':<40} | {auc_comb_a:>12.4f} | {auc_comb_b:>12.4f}")
print(f"{'Domain Adversarial':<40} | {auc_da_a:>12.4f} | {auc_da_b:>12.4f}")
print("-"*70)

# Key comparisons
print("\nKEY INSIGHTS:")
print(f"  OOD improvement (A→B): Combined={auc_comb_b-auc_a_b:+.4f}, DA={auc_da_b-auc_a_b:+.4f}")
print(f"  OOD improvement (B→A): Combined={auc_comb_a-auc_b_a:+.4f}, DA={auc_da_a-auc_b_a:+.4f}")
print(f"  DA vs Combined on B: {auc_da_b-auc_comb_b:+.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Bar comparison
ax1 = axes[0, 0]
x = np.arange(4)
width = 0.35
aucs_a = [auc_a_a, auc_b_a, auc_comb_a, auc_da_a]
aucs_b = [auc_a_b, auc_b_b, auc_comb_b, auc_da_b]
labels = [f'{args.label_a}\nonly', f'{args.label_b}\nonly', 'Combined', 'Domain\nAdversarial']

bars1 = ax1.bar(x - width/2, aucs_a, width, label=f'{args.label_a} Val', color='blue', alpha=0.7)
bars2 = ax1.bar(x + width/2, aucs_b, width, label=f'{args.label_b} Val', color='red', alpha=0.7)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('AUC')
ax1.set_title('Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.set_ylim(0.3, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1 + bars2:
    h = bar.get_height()
    ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

# Plot 2: OOD Focus
ax2 = axes[0, 1]
ood_models = [f'{args.label_a}→{args.label_b}', f'{args.label_b}→{args.label_a}']
ood_single = [auc_a_b, auc_b_a]
ood_combined = [auc_comb_b, auc_comb_a]
ood_da = [auc_da_b, auc_da_a]

x2 = np.arange(2)
w = 0.25
ax2.bar(x2 - w, ood_single, w, label='Single Domain', color='gray')
ax2.bar(x2, ood_combined, w, label='Combined', color='orange')
ax2.bar(x2 + w, ood_da, w, label='Domain Adversarial', color='green')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('AUC')
ax2.set_title('OOD Generalization Comparison')
ax2.set_xticks(x2)
ax2.set_xticklabels(ood_models)
ax2.legend()
ax2.set_ylim(0.3, 1.0)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Asymmetric data sweep
ax3 = axes[1, 0]
pcts = [r['pct'] for r in asymmetric_results]
auc_a_sweep = [r['auc_a'] for r in asymmetric_results]
auc_b_sweep = [r['auc_b'] for r in asymmetric_results]

ax3.plot(pcts, auc_a_sweep, 'b-o', label=f'{args.label_a}', markersize=8)
ax3.plot(pcts, auc_b_sweep, 'r-s', label=f'{args.label_b}', markersize=8)
ax3.axhline(y=auc_da_b, color='green', linestyle='--', label='DA on B')
ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlabel(f'% of {args.label_b} Training Data')
ax3.set_ylabel('AUC')
ax3.set_title(f'Asymmetric Data: How Much {args.label_b} Data Needed?')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.3, 1.0)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
COMPREHENSIVE COMPARISON SUMMARY
{'='*55}

Configuration:
  Layer: {args.layer}, Pooling: {args.pooling}, Epochs: {args.epochs}

PERFORMANCE MATRIX (AUC):
{'-'*55}
{'Model':<35} | {args.label_a[:8]:>8} | {args.label_b[:8]:>8}
{'-'*55}
Trained on {args.label_a} only        | {auc_a_a:>8.4f} | {auc_a_b:>8.4f}*
Trained on {args.label_b} only        | {auc_b_a:>8.4f}* | {auc_b_b:>8.4f}
Combined (no adaptation)    | {auc_comb_a:>8.4f} | {auc_comb_b:>8.4f}
Domain Adversarial          | {auc_da_a:>8.4f} | {auc_da_b:>8.4f}
{'-'*55}
* = OOD (out-of-domain) test

KEY FINDINGS:
• OOD {args.label_a}→{args.label_b}: {auc_a_b:.3f} → DA: {auc_da_b:.3f} (+{auc_da_b-auc_a_b:.3f})
• OOD {args.label_b}→{args.label_a}: {auc_b_a:.3f} → DA: {auc_da_a:.3f} (+{auc_da_a-auc_b_a:.3f})
• DA vs Combined: {auc_da_b-auc_comb_b:+.3f} on {args.label_b}

CONCLUSIONS:
• {'✓ DA improves OOD over single-domain' if max(auc_da_a-auc_b_a, auc_da_b-auc_a_b) > 0.05 else '✗ DA does not help OOD'}
• {'✓ DA > Combined (GRL helps)' if auc_da_b > auc_comb_b + 0.02 else '✗ Combined ≈ DA (just data helps)'}
"""
ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'comprehensive_comparison.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(args.output_dir, 'comprehensive_comparison.png')}")

# Save results
with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)

print("\n" + "="*70)
print("✅ COMPREHENSIVE EXPERIMENT COMPLETE")
print("="*70)
