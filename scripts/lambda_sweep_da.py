#!/usr/bin/env python3
"""
Lambda Sweep for Domain Adversarial Training
=============================================
Finds optimal lambda that balances ID and OOD performance.

Usage:
    python scripts/lambda_sweep_da.py \
        --train_a data/activations/.../Deception-Roleplaying/train \
        --val_a data/activations/.../Deception-Roleplaying/validation \
        --train_b data/activations/.../Deception-InsiderTrading/train \
        --val_b data/activations/.../Deception-InsiderTrading/validation \
        --layer 20 --pooling max \
        --output_dir results/lambda_sweep
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

# ============================================================================
# MODELS (same as before)
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
# DATA LOADING
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

def train_da(X_a, y_a, X_b, y_b, device, lambda_d=0.5, epochs=30, lr=1e-3):
    """Train DA model with specific lambda, return model and normalization params"""
    X_train = np.vstack([X_a, X_b])
    y_train = np.concatenate([y_a, y_b])
    domain = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mean) / std
    
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
    
    return model, mean, std

def evaluate(model, X, y, mean, std, device):
    """Evaluate model on data"""
    model.eval()
    X_n = (X - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(model.predict(torch.tensor(X_n, dtype=torch.float32).to(device))).cpu().numpy()
    return roc_auc_score(y, probs)

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
parser.add_argument('--output_dir', type=str, default='results/lambda_sweep')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Lambda Sweep | Layer {args.layer}, {args.pooling}")
print(f"Device: {device}\n")

# Load data
print("Loading data...")
X_train_a, y_train_a = load_activations(args.train_a, args.layer, args.pooling)
X_val_a, y_val_a = load_activations(args.val_a, args.layer, args.pooling)
X_train_b, y_train_b = load_activations(args.train_b, args.layer, args.pooling)
X_val_b, y_val_b = load_activations(args.val_b, args.layer, args.pooling)

print(f"  {args.label_a}: Train={X_train_a.shape[0]}, Val={X_val_a.shape[0]}")
print(f"  {args.label_b}: Train={X_train_b.shape[0]}, Val={X_val_b.shape[0]}")

# ============================================================================
# LAMBDA SWEEP
# ============================================================================
print("\n" + "="*70)
print("LAMBDA SWEEP")
print("="*70)

lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
results = []

# Also train single-domain baselines for reference
print("\nBaselines:")
# Train on A only
X_temp = np.vstack([X_train_a, X_train_a[:1]])  # Hack to use same function
y_temp = np.concatenate([y_train_a, y_train_a[:1]])
model_a, mean_a, std_a = train_da(X_train_a, y_train_a, X_train_a[:10], y_train_a[:10], device, lambda_d=0, epochs=args.epochs)
baseline_a_a = evaluate(model_a, X_val_a, y_val_a, mean_a, std_a, device)
baseline_a_b = evaluate(model_a, X_val_b, y_val_b, mean_a, std_a, device)
print(f"  Single-{args.label_a}: {args.label_a}={baseline_a_a:.4f}, {args.label_b}={baseline_a_b:.4f}")

for lam in lambdas:
    print(f"\nλ = {lam:.1f}:")
    model, mean, std = train_da(X_train_a, y_train_a, X_train_b, y_train_b, device, lambda_d=lam, epochs=args.epochs)
    
    auc_a = evaluate(model, X_val_a, y_val_a, mean, std, device)
    auc_b = evaluate(model, X_val_b, y_val_b, mean, std, device)
    
    # Harmonic mean (balances both)
    hmean = 2 * auc_a * auc_b / (auc_a + auc_b) if (auc_a + auc_b) > 0 else 0
    
    results.append({
        'lambda': lam,
        'auc_a': auc_a,
        'auc_b': auc_b,
        'harmonic_mean': hmean,
        'sum': auc_a + auc_b
    })
    
    print(f"  {args.label_a}={auc_a:.4f}, {args.label_b}={auc_b:.4f}, H-Mean={hmean:.4f}")

# Find best
best_harmonic = max(results, key=lambda x: x['harmonic_mean'])
best_sum = max(results, key=lambda x: x['sum'])

print("\n" + "="*70)
print("OPTIMAL LAMBDA SELECTION")
print("="*70)
print(f"\nBest by Harmonic Mean: λ={best_harmonic['lambda']}")
print(f"  {args.label_a}={best_harmonic['auc_a']:.4f}, {args.label_b}={best_harmonic['auc_b']:.4f}")
print(f"\nBest by Sum: λ={best_sum['lambda']}")
print(f"  {args.label_a}={best_sum['auc_a']:.4f}, {args.label_b}={best_sum['auc_b']:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: AUC vs Lambda
ax1 = axes[0, 0]
lambdas_plot = [r['lambda'] for r in results]
aucs_a = [r['auc_a'] for r in results]
aucs_b = [r['auc_b'] for r in results]

ax1.plot(lambdas_plot, aucs_a, 'b-o', label=f'{args.label_a}', markersize=10, linewidth=2)
ax1.plot(lambdas_plot, aucs_b, 'r-s', label=f'{args.label_b}', markersize=10, linewidth=2)
ax1.axhline(y=baseline_a_a, color='blue', linestyle='--', alpha=0.5, label=f'Baseline {args.label_a}')
ax1.axhline(y=baseline_a_b, color='red', linestyle='--', alpha=0.5, label=f'Baseline {args.label_b}')
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=best_harmonic['lambda'], color='green', linestyle='-', alpha=0.3, linewidth=10, label=f'Best λ={best_harmonic["lambda"]}')
ax1.set_xlabel('Lambda (λ)', fontsize=12)
ax1.set_ylabel('AUC', fontsize=12)
ax1.set_title('AUC vs Lambda', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.3, 1.05)

# Plot 2: Trade-off curve (Pareto front)
ax2 = axes[0, 1]
ax2.scatter(aucs_a, aucs_b, c=lambdas_plot, cmap='viridis', s=200, edgecolors='black', linewidths=2)
for i, r in enumerate(results):
    ax2.annotate(f'λ={r["lambda"]}', (r['auc_a'], r['auc_b']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)  # Diagonal reference
ax2.scatter([baseline_a_a], [baseline_a_b], c='gray', s=150, marker='X', label='No DA')
ax2.set_xlabel(f'{args.label_a} AUC', fontsize=12)
ax2.set_ylabel(f'{args.label_b} AUC', fontsize=12)
ax2.set_title('ID-OOD Trade-off (Pareto Front)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.3, 1.0)

# Plot 3: Harmonic Mean vs Lambda
ax3 = axes[1, 0]
hmeans = [r['harmonic_mean'] for r in results]
ax3.bar(range(len(lambdas_plot)), hmeans, color='green', alpha=0.7)
ax3.set_xticks(range(len(lambdas_plot)))
ax3.set_xticklabels([f'{l:.1f}' for l in lambdas_plot])
ax3.set_xlabel('Lambda (λ)', fontsize=12)
ax3.set_ylabel('Harmonic Mean of AUCs', fontsize=12)
ax3.set_title('Balanced Performance (Higher = Better on Both)', fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')

# Highlight best
best_idx = lambdas_plot.index(best_harmonic['lambda'])
ax3.bar(best_idx, hmeans[best_idx], color='darkgreen')

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
LAMBDA SWEEP SUMMARY
{'='*55}

Configuration:
  Layer: {args.layer}, Pooling: {args.pooling}
  Epochs: {args.epochs}

Baselines (No DA, trained on {args.label_a} only):
  {args.label_a} (ID):  {baseline_a_a:.4f}
  {args.label_b} (OOD): {baseline_a_b:.4f}

LAMBDA SWEEP RESULTS:
{'-'*55}
{'λ':>6} | {args.label_a[:8]:>8} | {args.label_b[:8]:>8} | H-Mean
{'-'*55}
""" + '\n'.join([f"{r['lambda']:>6.1f} | {r['auc_a']:>8.4f} | {r['auc_b']:>8.4f} | {r['harmonic_mean']:.4f}" for r in results]) + f"""
{'-'*55}

OPTIMAL LAMBDA:
  Best Harmonic Mean: λ = {best_harmonic['lambda']}
    → {args.label_a}: {best_harmonic['auc_a']:.4f}
    → {args.label_b}: {best_harmonic['auc_b']:.4f}
    → H-Mean: {best_harmonic['harmonic_mean']:.4f}

INTERPRETATION:
  λ=0: No domain adaptation (combined training only)
  λ>0: Domain-invariant features encouraged
  Higher λ: More OOD-focused, may sacrifice ID
"""
ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'lambda_sweep.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(args.output_dir, 'lambda_sweep.png')}")

# Save results
with open(os.path.join(args.output_dir, 'lambda_sweep_results.json'), 'w') as f:
    json.dump({
        'config': vars(args),
        'baseline': {'auc_a': baseline_a_a, 'auc_b': baseline_a_b},
        'sweep': results,
        'best_harmonic': best_harmonic,
        'best_sum': best_sum
    }, f, indent=2, default=float)

print("\n" + "="*70)
print("✅ LAMBDA SWEEP COMPLETE")
print("="*70)
