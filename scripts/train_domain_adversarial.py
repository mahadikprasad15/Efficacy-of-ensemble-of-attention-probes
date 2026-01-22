#!/usr/bin/env python3
"""
Domain Adversarial Training for Cross-Domain Generalization
============================================================
Trains a probe with gradient reversal to learn domain-invariant features.

The key idea: Train the probe to classify deception, while CONFUSING a domain
classifier. This forces the probe to learn features that work across domains.

Architecture:
    Activation → [Pooling] → [Hidden] → Probe Head → Deception?
                             ↓
                     [Gradient Reversal]
                             ↓
                     Domain Classifier → Which domain?

Usage:
    python scripts/train_domain_adversarial.py \
        --train_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
        --val_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
        --train_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
        --val_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
        --layer 20 \
        --pooling max \
        --output_dir results/domain_adversarial
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
from sklearn.metrics import roc_auc_score, accuracy_score
from safetensors.torch import load_file
from tqdm import tqdm

# ============================================================================
# GRADIENT REVERSAL LAYER
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
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ============================================================================
# DOMAIN ADVERSARIAL MODEL
# ============================================================================
class DomainAdversarialProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task classifier (deception)
        self.task_classifier = nn.Linear(hidden_dim, 1)
        
        # Domain classifier (with gradient reversal)
        self.gradient_reversal = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lambda_=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction
        task_logits = self.task_classifier(features).squeeze(-1)
        
        # Domain prediction (with gradient reversal)
        self.gradient_reversal.lambda_ = lambda_
        reversed_features = self.gradient_reversal(features)
        domain_logits = self.domain_classifier(reversed_features).squeeze(-1)
        
        return task_logits, domain_logits
    
    def predict_task(self, x):
        features = self.feature_extractor(x)
        return self.task_classifier(features).squeeze(-1)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations(act_dir, layer, pooling):
    """Load and pool activations"""
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

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--train_a', type=str, required=True)
parser.add_argument('--val_a', type=str, required=True)
parser.add_argument('--train_b', type=str, required=True)
parser.add_argument('--val_b', type=str, required=True)
parser.add_argument('--label_a', type=str, default='DomainA')
parser.add_argument('--label_b', type=str, default='DomainB')
parser.add_argument('--layer', type=int, default=20)
parser.add_argument('--pooling', type=str, default='max')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lambda_domain', type=float, default=0.5)
parser.add_argument('--output_dir', type=str, default='results/domain_adversarial')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Domain Adversarial Training")
print(f"  Layer {args.layer}, Pooling: {args.pooling}")
print(f"  Device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
X_train_a, y_train_a = load_activations(args.train_a, args.layer, args.pooling)
X_val_a, y_val_a = load_activations(args.val_a, args.layer, args.pooling)
X_train_b, y_train_b = load_activations(args.train_b, args.layer, args.pooling)
X_val_b, y_val_b = load_activations(args.val_b, args.layer, args.pooling)

print(f"  {args.label_a} - Train: {X_train_a.shape}, Val: {X_val_a.shape}")
print(f"  {args.label_b} - Train: {X_train_b.shape}, Val: {X_val_b.shape}")

# Combine for training
X_train = np.vstack([X_train_a, X_train_b])
y_train = np.concatenate([y_train_a, y_train_b])
domain_train = np.concatenate([np.zeros(len(X_train_a)), np.ones(len(X_train_b))])

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_val_a_norm = (X_val_a - mean) / std
X_val_b_norm = (X_val_b - mean) / std

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
domain_train_t = torch.tensor(domain_train, dtype=torch.float32)

X_val_a_t = torch.tensor(X_val_a_norm, dtype=torch.float32).to(device)
y_val_a_t = torch.tensor(y_val_a, dtype=torch.float32).to(device)
X_val_b_t = torch.tensor(X_val_b_norm, dtype=torch.float32).to(device)
y_val_b_t = torch.tensor(y_val_b, dtype=torch.float32).to(device)

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\nTraining domain-adversarial probe...")

input_dim = X_train.shape[1]
model = DomainAdversarialProbe(input_dim, hidden_dim=args.hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
task_criterion = nn.BCEWithLogitsLoss()
domain_criterion = nn.BCEWithLogitsLoss()

n_samples = len(X_train_t)
history = {'train_task_loss': [], 'train_domain_loss': [], 
           'val_a_auc': [], 'val_b_auc': [], 'domain_acc': []}

for epoch in range(args.epochs):
    model.train()
    
    # Shuffle
    perm = torch.randperm(n_samples)
    X_train_t = X_train_t[perm]
    y_train_t = y_train_t[perm]
    domain_train_t = domain_train_t[perm]
    
    # Progress factor for lambda scheduling (0 → 1)
    p = epoch / args.epochs
    lambda_schedule = 2 / (1 + np.exp(-10 * p)) - 1  # Gradually increase
    
    epoch_task_loss = 0
    epoch_domain_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, args.batch_size):
        batch_x = X_train_t[i:i+args.batch_size].to(device)
        batch_y = y_train_t[i:i+args.batch_size].to(device)
        batch_domain = domain_train_t[i:i+args.batch_size].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        task_logits, domain_logits = model(batch_x, lambda_=args.lambda_domain * lambda_schedule)
        
        # Losses
        task_loss = task_criterion(task_logits, batch_y)
        domain_loss = domain_criterion(domain_logits, batch_domain)
        
        # Combined loss (domain loss gradient is reversed inside the model)
        total_loss = task_loss + domain_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_task_loss += task_loss.item()
        epoch_domain_loss += domain_loss.item()
        n_batches += 1
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Task performance
        logits_a = model.predict_task(X_val_a_t).cpu().numpy()
        logits_b = model.predict_task(X_val_b_t).cpu().numpy()
        
        probs_a = 1 / (1 + np.exp(-logits_a))
        probs_b = 1 / (1 + np.exp(-logits_b))
        
        auc_a = roc_auc_score(y_val_a, probs_a)
        auc_b = roc_auc_score(y_val_b, probs_b)
        
        # Domain classification (on training set - should be ~50% if domain-invariant)
        _, domain_logits_train = model(X_train_t.to(device), lambda_=0)
        domain_preds = (domain_logits_train.cpu().numpy() > 0).astype(int)
        domain_acc = accuracy_score(domain_train, domain_preds)
    
    history['train_task_loss'].append(epoch_task_loss / n_batches)
    history['train_domain_loss'].append(epoch_domain_loss / n_batches)
    history['val_a_auc'].append(auc_a)
    history['val_b_auc'].append(auc_b)
    history['domain_acc'].append(domain_acc)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: Task Loss={epoch_task_loss/n_batches:.4f}, "
              f"Domain Loss={epoch_domain_loss/n_batches:.4f}, "
              f"λ={lambda_schedule:.2f}, "
              f"AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, Domain Acc={domain_acc:.2%}")

# ============================================================================
# BASELINE COMPARISON
# ============================================================================
print("\n" + "="*70)
print("TRAINING BASELINE (NO DOMAIN ADAPTATION)")
print("="*70)

# Train only on domain A, test on both
class SimpleProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Baseline: Train on A only
X_train_a_norm = (X_train_a - X_train_a.mean(axis=0)) / (X_train_a.std(axis=0) + 1e-8)
X_train_a_t = torch.tensor(X_train_a_norm, dtype=torch.float32)
y_train_a_t = torch.tensor(y_train_a, dtype=torch.float32)

baseline_a = SimpleProbe(input_dim, args.hidden_dim).to(device)
optimizer_a = optim.Adam(baseline_a.parameters(), lr=args.lr, weight_decay=1e-4)

for epoch in range(args.epochs):
    baseline_a.train()
    perm = torch.randperm(len(X_train_a_t))
    for i in range(0, len(X_train_a_t), args.batch_size):
        batch_x = X_train_a_t[perm[i:i+args.batch_size]].to(device)
        batch_y = y_train_a_t[perm[i:i+args.batch_size]].to(device)
        optimizer_a.zero_grad()
        loss = task_criterion(baseline_a(batch_x), batch_y)
        loss.backward()
        optimizer_a.step()

baseline_a.eval()
with torch.no_grad():
    X_val_a_baseline = torch.tensor((X_val_a - X_train_a.mean(axis=0)) / (X_train_a.std(axis=0) + 1e-8), 
                                     dtype=torch.float32).to(device)
    X_val_b_baseline = torch.tensor((X_val_b - X_train_a.mean(axis=0)) / (X_train_a.std(axis=0) + 1e-8), 
                                     dtype=torch.float32).to(device)
    
    probs_a_baseline = torch.sigmoid(baseline_a(X_val_a_baseline)).cpu().numpy()
    probs_b_baseline = torch.sigmoid(baseline_a(X_val_b_baseline)).cpu().numpy()
    
    baseline_auc_a = roc_auc_score(y_val_a, probs_a_baseline)
    baseline_auc_b = roc_auc_score(y_val_b, probs_b_baseline)

print(f"\nBaseline (trained on {args.label_a} only):")
print(f"  {args.label_a} (ID):  {baseline_auc_a:.4f}")
print(f"  {args.label_b} (OOD): {baseline_auc_b:.4f}")

print(f"\nDomain Adversarial:")
print(f"  {args.label_a}: {history['val_a_auc'][-1]:.4f}")
print(f"  {args.label_b}: {history['val_b_auc'][-1]:.4f}")

print(f"\nImprovement on OOD ({args.label_b}):")
print(f"  {history['val_b_auc'][-1] - baseline_auc_b:+.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Training losses
ax1 = axes[0, 0]
ax1.plot(history['train_task_loss'], label='Task Loss', color='blue')
ax1.plot(history['train_domain_loss'], label='Domain Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Losses')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation AUC
ax2 = axes[0, 1]
ax2.plot(history['val_a_auc'], label=f'{args.label_a} (ID)', color='blue')
ax2.plot(history['val_b_auc'], label=f'{args.label_b} (OOD)', color='red')
ax2.axhline(y=baseline_auc_a, linestyle='--', color='blue', alpha=0.5, label=f'Baseline {args.label_a}')
ax2.axhline(y=baseline_auc_b, linestyle='--', color='red', alpha=0.5, label=f'Baseline {args.label_b}')
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC Over Training')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.4, 1.0)

# Plot 3: Domain accuracy (should approach 50%)
ax3 = axes[1, 0]
ax3.plot(history['domain_acc'], color='purple')
ax3.axhline(y=0.5, linestyle='--', color='gray', label='Random (domain confusion)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Domain Classification Accuracy')
ax3.set_title('Domain Classifier Accuracy\n(Lower = More Domain-Invariant)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.3, 1.0)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
SUMMARY: Domain Adversarial Training
{'='*50}

Configuration:
  Layer: {args.layer}, Pooling: {args.pooling}
  Hidden: {args.hidden_dim}, λ_domain: {args.lambda_domain}
  Epochs: {args.epochs}

Baseline (Trained on {args.label_a} only):
  {args.label_a} (ID):  {baseline_auc_a:.4f}
  {args.label_b} (OOD): {baseline_auc_b:.4f}

Domain Adversarial (Trained on BOTH):
  {args.label_a}: {history['val_a_auc'][-1]:.4f} (Δ {history['val_a_auc'][-1] - baseline_auc_a:+.4f})
  {args.label_b}: {history['val_b_auc'][-1]:.4f} (Δ {history['val_b_auc'][-1] - baseline_auc_b:+.4f})

Domain Confusion:
  Final domain accuracy: {history['domain_acc'][-1]:.2%}
  (50% = perfect domain-invariance)

Interpretation:
  {'✓ OOD IMPROVED' if history['val_b_auc'][-1] > baseline_auc_b + 0.02 else '✗ OOD did not improve'}
  {'✓ Domain-invariant' if history['domain_acc'][-1] < 0.65 else '✗ Still domain-specific'}
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'domain_adversarial_results.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(args.output_dir, 'domain_adversarial_results.png')}")

# Save model
torch.save(model.state_dict(), os.path.join(args.output_dir, 'domain_adversarial_probe.pt'))

# Save results
with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
    json.dump({
        'config': vars(args),
        'baseline': {'auc_a': baseline_auc_a, 'auc_b': baseline_auc_b},
        'domain_adversarial': {
            'auc_a': history['val_a_auc'][-1],
            'auc_b': history['val_b_auc'][-1],
            'domain_acc': history['domain_acc'][-1]
        },
        'history': history
    }, f, indent=2, default=float)

print("\n" + "="*70)
print("✅ DOMAIN ADVERSARIAL TRAINING COMPLETE")
print("="*70)
