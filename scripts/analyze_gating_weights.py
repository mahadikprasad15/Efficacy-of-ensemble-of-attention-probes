# ============================================================================
# GATING WEIGHT ANALYSIS: ID vs OOD Training
# ============================================================================
# This script:
# 1. Trains gating on ID (validation), extracts weights, evaluates on ID
# 2. Trains gating on OOD (80%), extracts weights, evaluates on OOD (20%)
# 3. Compares weight distributions
# 4. Sweeps OOD training percentage to see when gating becomes effective
# ============================================================================

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import GatedEnsemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Gating Weight Analysis')
parser.add_argument('--id_logits', type=str, 
                    default='results/ensembles/attn/val_logits.npy',
                    help='Path to ID (validation) logits')
parser.add_argument('--id_labels', type=str,
                    default='results/ensembles/attn/val_labels.npy',
                    help='Path to ID labels')
parser.add_argument('--ood_logits', type=str,
                    default='results/ood_evaluation/logits/attn_logits.npy',
                    help='Path to OOD logits')
parser.add_argument('--ood_labels', type=str,
                    default='results/ood_evaluation/logits/labels.npy',
                    help='Path to OOD labels')
parser.add_argument('--probes_dir', type=str,
                    default='data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/attn',
                    help='Path to probe directory (for layer selection)')
parser.add_argument('--k_pct', type=int, default=40,
                    help='K% for top layer selection')
parser.add_argument('--output_dir', type=str,
                    default='results/gating_analysis',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def train_gated_and_get_weights(train_logits, train_labels, test_logits, test_labels,
                                 num_layers, epochs=20, patience=3):
    """Train gating and return weights + AUC."""
    ensemble = GatedEnsemble(input_dim=num_layers, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(ensemble.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_logits_t = torch.tensor(train_logits, dtype=torch.float32)
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(train_logits_t, train_labels_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        ensemble.train()
        epoch_loss = 0
        for batch_logits, batch_labels in loader:
            batch_logits = batch_logits.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            output = ensemble(batch_logits, batch_logits.unsqueeze(-1))
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = ensemble.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        ensemble.load_state_dict(best_state)
    
    # Extract weights for all test samples
    ensemble.eval()
    with torch.no_grad():
        test_logits_t = torch.tensor(test_logits, dtype=torch.float32).to(device)
        weights = ensemble.gate_net(test_logits_t).cpu().numpy()  # (N, K)
        ensemble_logits = ensemble(test_logits_t, test_logits_t.unsqueeze(-1)).cpu().numpy().flatten()
    
    probs = 1 / (1 + np.exp(-ensemble_logits))
    auc = roc_auc_score(test_labels, probs)
    acc = accuracy_score(test_labels, (probs > 0.5).astype(int))
    
    return weights, auc, acc, ensemble

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*60)
print("STEP 1: Load data")
print("="*60)

# Load ID data
if os.path.exists(args.id_logits) and os.path.exists(args.id_labels):
    id_logits = np.load(args.id_logits)
    id_labels = np.load(args.id_labels)
    print(f"✓ ID data: {id_logits.shape}")
else:
    print("⚠️ ID logits not found, will compute from activations if needed")
    id_logits = None

# Load OOD data
ood_logits = np.load(args.ood_logits)
ood_labels = np.load(args.ood_labels)
print(f"✓ OOD data: {ood_logits.shape}")

# Load layer selection
with open(f"{args.probes_dir}/layer_results.json", 'r') as f:
    layer_results = json.load(f)
num_layers = 28
k_layers = max(1, int(num_layers * args.k_pct / 100))
sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
selected_layers = sorted([l['layer'] for l in sorted_layers[:k_layers]])
print(f"K = {args.k_pct}% → {k_layers} layers: {selected_layers}")

# Select only K layers
ood_logits_k = ood_logits[:, selected_layers]
if id_logits is not None:
    id_logits_k = id_logits[:, selected_layers]

# ============================================================================
# STEP 2: TRAIN ON ID, EVALUATE ON ID
# ============================================================================
print("\n" + "="*60)
print("STEP 2: Train gating on ID (validation)")
print("="*60)

if id_logits is not None:
    n_train_id = int(0.8 * len(id_labels))
    id_weights, id_auc, id_acc, _ = train_gated_and_get_weights(
        id_logits_k[:n_train_id], id_labels[:n_train_id],
        id_logits_k[n_train_id:], id_labels[n_train_id:],
        num_layers=k_layers
    )
    print(f"✓ ID-trained gating: AUC={id_auc:.4f}, Acc={id_acc:.4f}")
else:
    id_weights = None
    id_auc = None

# ============================================================================
# STEP 3: TRAIN ON OOD (80%), EVALUATE ON OOD (20%)
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Train gating on OOD (80%)")
print("="*60)

n_train_ood = int(0.8 * len(ood_labels))
ood_weights, ood_auc, ood_acc, _ = train_gated_and_get_weights(
    ood_logits_k[:n_train_ood], ood_labels[:n_train_ood],
    ood_logits_k[n_train_ood:], ood_labels[n_train_ood:],
    num_layers=k_layers
)
print(f"✓ OOD-trained gating: AUC={ood_auc:.4f}, Acc={ood_acc:.4f}")

# ============================================================================
# STEP 4: SWEEP OOD TRAINING PERCENTAGE
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Sweep OOD training percentage")
print("="*60)

train_pcts = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
sweep_results = []

# Fixed test set (last 20%)
test_ood_logits = ood_logits_k[n_train_ood:]
test_ood_labels = ood_labels[n_train_ood:]

for pct in tqdm(train_pcts, desc="Sweeping OOD train %"):
    n_train = max(1, int(pct / 100 * len(ood_labels)))
    train_logits = ood_logits_k[:n_train]
    train_labels = ood_labels[:n_train]
    
    if len(np.unique(train_labels)) < 2:
        # Need both classes
        sweep_results.append({'pct': pct, 'auc': 0.5, 'acc': 0.5})
        continue
    
    _, auc, acc, _ = train_gated_and_get_weights(
        train_logits, train_labels,
        test_ood_logits, test_ood_labels,
        num_layers=k_layers
    )
    sweep_results.append({'pct': pct, 'auc': auc, 'acc': acc})
    print(f"  {pct}%: AUC={auc:.4f}")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Generate visualizations")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Weight distribution comparison
ax1 = axes[0, 0]
if id_weights is not None:
    id_mean = id_weights.mean(axis=0)
    id_std = id_weights.std(axis=0)
    x = np.arange(len(selected_layers))
    ax1.bar(x - 0.2, id_mean, 0.4, yerr=id_std, label=f'ID-trained (AUC={id_auc:.3f})', color='#3498db', capsize=3, alpha=0.8)
    
ood_mean = ood_weights.mean(axis=0)
ood_std = ood_weights.std(axis=0)
ax1.bar(x + 0.2, ood_mean, 0.4, yerr=ood_std, label=f'OOD-trained (AUC={ood_auc:.3f})', color='#e74c3c', capsize=3, alpha=0.8)

ax1.set_xlabel('Layer')
ax1.set_ylabel('Mean Gating Weight')
ax1.set_title('Gating Weights: ID-trained vs OOD-trained')
ax1.set_xticks(x)
ax1.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
ax1.legend()
ax1.axhline(1/k_layers, color='gray', linestyle='--', alpha=0.5, label='Uniform')

# Plot 2: Weight differences
ax2 = axes[0, 1]
if id_weights is not None:
    weight_diff = ood_mean - id_mean
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in weight_diff]
    ax2.bar(x, weight_diff, color=colors)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Weight Difference (OOD - ID)')
    ax2.set_title('How Gating Changes: OOD vs ID')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
else:
    ax2.text(0.5, 0.5, 'ID weights not available', ha='center', va='center', transform=ax2.transAxes)

# Plot 3: OOD train % sweep (line chart)
ax3 = axes[1, 0]
pcts = [r['pct'] for r in sweep_results]
aucs = [r['auc'] for r in sweep_results]
ax3.plot(pcts, aucs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax3.axhline(0.5, color='red', linestyle='--', label='Random', alpha=0.5)
ax3.axhline(0.618, color='blue', linestyle='--', label='Mean Ensemble (no gating)', alpha=0.5)
ax3.set_xlabel('OOD Training %')
ax3.set_ylabel('Test AUC')
ax3.set_title('How Much OOD Data Does Gating Need?')
ax3.set_xticks(pcts)
ax3.legend()
ax3.set_ylim(0.4, 1.0)
ax3.grid(True, alpha=0.3)

# Add annotations for key points
for i, (p, a) in enumerate(zip(pcts, aucs)):
    ax3.annotate(f'{a:.2f}', (p, a), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Plot 4: Per-sample weight heatmaps
ax4 = axes[1, 1]
# Show first 20 samples
n_show = min(20, len(ood_weights))
im = ax4.imshow(ood_weights[:n_show].T, aspect='auto', cmap='YlOrRd')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Layer')
ax4.set_yticks(range(len(selected_layers)))
ax4.set_yticklabels([f'L{l}' for l in selected_layers])
ax4.set_title('Per-Sample Gating Weights (OOD-trained)')
plt.colorbar(im, ax=ax4, label='Weight')

plt.tight_layout()
plot_path = f"{args.output_dir}/gating_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {plot_path}")
plt.show()

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
results = {
    'config': {
        'k_pct': args.k_pct,
        'selected_layers': selected_layers
    },
    'id_trained': {
        'auc': float(id_auc) if id_auc else None,
        'mean_weights': id_mean.tolist() if id_weights is not None else None,
        'std_weights': id_std.tolist() if id_weights is not None else None
    },
    'ood_trained': {
        'auc': float(ood_auc),
        'mean_weights': ood_mean.tolist(),
        'std_weights': ood_std.tolist()
    },
    'ood_sweep': sweep_results
}

results_path = f"{args.output_dir}/gating_analysis_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved: {results_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 GATING ANALYSIS SUMMARY")
print("="*80)
print(f"\n{'Training Source':<20} {'Test AUC':<12}")
print("-" * 35)
if id_auc:
    print(f"{'ID (validation)':<20} {id_auc:.4f}")
print(f"{'OOD (80%)':<20} {ood_auc:.4f}")

print(f"\n📈 OOD Training % Sweep:")
for r in sweep_results:
    bar = "█" * int(r['auc'] * 20)
    print(f"  {r['pct']:3d}%: {r['auc']:.3f} {bar}")

print("\n" + "="*80)
