# ============================================================================
# FIXED WEIGHT vs GATED ENSEMBLE COMPARISON
# ============================================================================
# This script:
# 1. Loads the best gated ensemble (ATTN, K=40%)
# 2. Extracts mean gating weights across ID samples
# 3. Creates a Fixed Weight Ensemble with those mean weights
# 4. Evaluates both on ID (Roleplaying) and OOD (InsiderTrading)
# 5. Generates comparison table
# ============================================================================

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
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
parser = argparse.ArgumentParser(description='Compare Fixed vs Gated Ensemble')
parser.add_argument('--id_activations', type=str, 
                    default='data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation',
                    help='Path to ID (validation) activations')
parser.add_argument('--ood_activations', type=str,
                    default='data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test',
                    help='Path to OOD (test) activations')
parser.add_argument('--probes_dir', type=str,
                    default='data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/attn',
                    help='Path to trained probes')
parser.add_argument('--gated_model', type=str,
                    default='results/ensembles/attn/gated_models_val/gated_k40.pt',
                    help='Path to saved gated ensemble model')
parser.add_argument('--k_pct', type=int, default=40,
                    help='K% for top layer selection')
parser.add_argument('--output_dir', type=str,
                    default='results/fixed_vs_gated',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# HELPER: Load activations
# ============================================================================
def load_activations(activations_dir):
    """Load activations from manifest + shards."""
    manifest_path = f"{activations_dir}/manifest.jsonl"
    if not os.path.exists(manifest_path):
        print(f"⚠️ Manifest not found: {manifest_path}")
        return None, None
    
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(f"{activations_dir}/shard_*.safetensors"))
    all_tensors = {}
    for shard in shards:
        all_tensors.update(load_file(shard))
    
    samples, labels = [], []
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            samples.append(all_tensors[eid])
            labels.append(entry['label'])
    
    print(f"✓ Loaded {len(samples)} samples from {activations_dir}")
    return samples, np.array(labels)

# ============================================================================
# HELPER: Compute FPR at fixed TPR
# ============================================================================
def fpr_at_tpr(labels, probs, target_tpr=0.99):
    """Compute FPR when TPR = target_tpr (default 99%)."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    # Find the threshold where TPR >= target_tpr
    idx = np.where(tpr_arr >= target_tpr)[0]
    if len(idx) > 0:
        return fpr_arr[idx[0]]
    return fpr_arr[-1]

# ============================================================================
# 1. LOAD LAYER RESULTS & SELECT TOP-K
# ============================================================================
print("\n" + "="*60)
print("STEP 1: Load layer configuration")
print("="*60)

with open(f"{args.probes_dir}/layer_results.json", 'r') as f:
    layer_results = json.load(f)

num_layers = 28
k_layers = max(1, int(num_layers * args.k_pct / 100))
sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
selected_layers = sorted([l['layer'] for l in sorted_layers[:k_layers]])

print(f"K = {args.k_pct}% → {k_layers} layers")
print(f"Selected layers: {selected_layers}")

# ============================================================================
# 2. LOAD PROBES FOR ALL SELECTED LAYERS
# ============================================================================
print("\n" + "="*60)
print("STEP 2: Load per-layer probes")
print("="*60)

# Get input dimension from first sample
id_samples, id_labels = load_activations(args.id_activations)
if id_samples is None:
    print("❌ Could not load ID activations")
    exit(1)

D = id_samples[0].shape[-1]
probes = {}
for layer_idx in selected_layers:
    probe_path = f"{args.probes_dir}/probe_layer_{layer_idx}.pt"
    probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval()
    probes[layer_idx] = probe

print(f"✓ Loaded {len(probes)} probes")

# ============================================================================
# 3. LOAD GATED ENSEMBLE
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Load Gated Ensemble")
print("="*60)

gated = GatedEnsemble(input_dim=k_layers, num_layers=k_layers).to(device)
if os.path.exists(args.gated_model):
    gated.load_state_dict(torch.load(args.gated_model, map_location=device))
    print(f"✓ Loaded gated model from {args.gated_model}")
else:
    print(f"⚠️ Gated model not found at {args.gated_model}")
    print("  Will train a new one on ID data")
    # We'll train it below if needed
gated.eval()

# ============================================================================
# 4. EXTRACT GATING WEIGHTS FROM ID DATA
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Extract Gating Weights from ID Samples")
print("="*60)

all_gating_weights = []

for sample in tqdm(id_samples, desc="Extracting gating weights"):
    tensor = torch.tensor(sample).float().unsqueeze(0)
    
    # Get logits from all selected layers
    layer_logits = []
    for layer_idx in selected_layers:
        x_layer = tensor[:, layer_idx, :, :].to(device)
        with torch.no_grad():
            logit = probes[layer_idx](x_layer).cpu().item()
        layer_logits.append(logit)
    
    # Get gating weights
    layer_logits_tensor = torch.tensor(layer_logits, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        gating_weights = gated.gate_net(layer_logits_tensor).cpu().numpy().flatten()
    
    all_gating_weights.append(gating_weights)

all_gating_weights = np.array(all_gating_weights)  # (N, K)
mean_weights = all_gating_weights.mean(axis=0)
std_weights = all_gating_weights.std(axis=0)

print(f"\n📊 Gating Weight Statistics:")
print(f"{'Layer':<10} {'Mean':<10} {'Std':<10}")
print("-" * 30)
for i, layer_idx in enumerate(selected_layers):
    print(f"L{layer_idx:<9} {mean_weights[i]:.4f}     {std_weights[i]:.4f}")

# ============================================================================
# 5. CREATE FIXED WEIGHT ENSEMBLE
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Create Fixed Weight Ensemble")
print("="*60)

# Normalize mean weights to sum to 1
fixed_weights = mean_weights / mean_weights.sum()
print(f"Fixed weights (normalized): {fixed_weights}")

# ============================================================================
# 6. EVALUATE BOTH ENSEMBLES ON ID AND OOD
# ============================================================================
print("\n" + "="*60)
print("STEP 6: Evaluate on ID and OOD")
print("="*60)

def evaluate_ensemble(samples, labels, selected_layers, probes, ensemble_type, weights=None):
    """
    Evaluate ensemble on samples.
    ensemble_type: 'gated', 'fixed', or 'mean'
    """
    all_probs = []
    
    for sample in tqdm(samples, desc=f"Eval {ensemble_type}", leave=False):
        tensor = torch.tensor(sample).float().unsqueeze(0)
        
        # Get logits from all selected layers
        layer_logits = []
        for layer_idx in selected_layers:
            x_layer = tensor[:, layer_idx, :, :].to(device)
            with torch.no_grad():
                logit = probes[layer_idx](x_layer).cpu().item()
            layer_logits.append(logit)
        
        layer_logits = np.array(layer_logits)
        
        if ensemble_type == 'gated':
            # Use gated ensemble
            layer_logits_tensor = torch.tensor(layer_logits, dtype=torch.float32).unsqueeze(0).to(device)
            layer_logits_3d = layer_logits_tensor.unsqueeze(-1)  # (1, K, 1)
            with torch.no_grad():
                ensemble_logit = gated(layer_logits_tensor, layer_logits_3d).cpu().item()
        elif ensemble_type == 'fixed':
            # Use fixed weights
            ensemble_logit = (layer_logits * weights).sum()
        else:  # 'mean'
            ensemble_logit = layer_logits.mean()
        
        prob = 1 / (1 + np.exp(-ensemble_logit))
        all_probs.append(prob)
    
    probs = np.array(all_probs)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))
    fpr_1pct = fpr_at_tpr(labels, probs, target_tpr=0.99)
    
    return auc, acc, fpr_1pct

# Evaluate on ID
print("\n📊 Evaluating on ID (Roleplaying)...")
id_gated_auc, id_gated_acc, id_gated_fpr = evaluate_ensemble(
    id_samples, id_labels, selected_layers, probes, 'gated')
id_fixed_auc, id_fixed_acc, id_fixed_fpr = evaluate_ensemble(
    id_samples, id_labels, selected_layers, probes, 'fixed', fixed_weights)
id_mean_auc, id_mean_acc, id_mean_fpr = evaluate_ensemble(
    id_samples, id_labels, selected_layers, probes, 'mean')

# Load and evaluate on OOD
print("\n📊 Evaluating on OOD (InsiderTrading)...")
ood_samples, ood_labels = load_activations(args.ood_activations)

if ood_samples is not None:
    ood_gated_auc, ood_gated_acc, ood_gated_fpr = evaluate_ensemble(
        ood_samples, ood_labels, selected_layers, probes, 'gated')
    ood_fixed_auc, ood_fixed_acc, ood_fixed_fpr = evaluate_ensemble(
        ood_samples, ood_labels, selected_layers, probes, 'fixed', fixed_weights)
    ood_mean_auc, ood_mean_acc, ood_mean_fpr = evaluate_ensemble(
        ood_samples, ood_labels, selected_layers, probes, 'mean')
else:
    ood_gated_auc = ood_gated_acc = ood_gated_fpr = None
    ood_fixed_auc = ood_fixed_acc = ood_fixed_fpr = None
    ood_mean_auc = ood_mean_acc = ood_mean_fpr = None

# ============================================================================
# 7. GENERATE RESULTS TABLE
# ============================================================================
print("\n" + "="*80)
print("📊 RESULTS: FIXED vs GATED ENSEMBLE COMPARISON")
print("="*80)

header = f"{'Ensemble':<12} | {'ID AUC':<10} | {'ID Acc':<10} | {'OOD AUC':<10} | {'OOD Acc':<10} | {'OOD FPR@99%':<12}"
print(header)
print("-" * len(header))

print(f"{'Gated':<12} | {id_gated_auc:.4f}     | {id_gated_acc:.4f}     | {ood_gated_auc if ood_gated_auc else 'N/A':<10} | {ood_gated_acc if ood_gated_acc else 'N/A':<10} | {ood_gated_fpr if ood_gated_fpr else 'N/A':<12}")
print(f"{'Fixed (Mean)':<12} | {id_fixed_auc:.4f}     | {id_fixed_acc:.4f}     | {ood_fixed_auc if ood_fixed_auc else 'N/A':<10} | {ood_fixed_acc if ood_fixed_acc else 'N/A':<10} | {ood_fixed_fpr if ood_fixed_fpr else 'N/A':<12}")
print(f"{'Uniform':<12} | {id_mean_auc:.4f}     | {id_mean_acc:.4f}     | {ood_mean_auc if ood_mean_auc else 'N/A':<10} | {ood_mean_acc if ood_mean_acc else 'N/A':<10} | {ood_mean_fpr if ood_mean_fpr else 'N/A':<12}")

# Compute delta
if ood_gated_auc and ood_fixed_auc:
    delta = ood_fixed_auc - ood_gated_auc
    print("\n" + "-"*80)
    print(f"📈 OOD AUC Delta (Fixed - Gated): {delta:+.4f}")
    if abs(delta) < 0.02:
        print("   → ✅ HUGE: Fixed weights ≈ Gated! Can simplify to fixed ensemble.")
    else:
        print(f"   → 🔬 Gated is {'better' if delta < 0 else 'worse'} by {abs(delta):.4f}")
        print("      This suggests the ensemble benefits from per-sample adaptation.")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
results = {
    'config': {
        'k_pct': args.k_pct,
        'num_layers': k_layers,
        'selected_layers': selected_layers,
        'fixed_weights': fixed_weights.tolist(),
        'weight_std': std_weights.tolist()
    },
    'id': {
        'gated': {'auc': id_gated_auc, 'acc': id_gated_acc, 'fpr_99': id_gated_fpr},
        'fixed': {'auc': id_fixed_auc, 'acc': id_fixed_acc, 'fpr_99': id_fixed_fpr},
        'mean': {'auc': id_mean_auc, 'acc': id_mean_acc, 'fpr_99': id_mean_fpr}
    },
    'ood': {
        'gated': {'auc': ood_gated_auc, 'acc': ood_gated_acc, 'fpr_99': ood_gated_fpr} if ood_gated_auc else None,
        'fixed': {'auc': ood_fixed_auc, 'acc': ood_fixed_acc, 'fpr_99': ood_fixed_fpr} if ood_fixed_auc else None,
        'mean': {'auc': ood_mean_auc, 'acc': ood_mean_acc, 'fpr_99': ood_mean_fpr} if ood_mean_auc else None
    }
}

results_path = f"{args.output_dir}/fixed_vs_gated_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved results to {results_path}")

# ============================================================================
# 9. PLOT COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart for AUC
ensembles = ['Gated', 'Fixed\n(Mean Weights)', 'Uniform\n(Equal Weights)']
id_aucs = [id_gated_auc, id_fixed_auc, id_mean_auc]
ood_aucs = [ood_gated_auc or 0, ood_fixed_auc or 0, ood_mean_auc or 0]

x = np.arange(len(ensembles))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, id_aucs, width, label='ID (Roleplaying)', color='#3498db')
bars2 = ax.bar(x + width/2, ood_aucs, width, label='OOD (InsiderTrading)', color='#e74c3c')
ax.set_ylabel('AUC')
ax.set_title('AUC Comparison: Fixed vs Gated Ensemble')
ax.set_xticks(x)
ax.set_xticklabels(ensembles)
ax.legend()
ax.set_ylim(0.4, 1.0)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Weight distribution
ax2 = axes[1]
ax2.bar(range(len(selected_layers)), fixed_weights, yerr=std_weights, capsize=3, color='#2ecc71', alpha=0.8)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Mean Weight')
ax2.set_title(f'Fixed Weights (Mean Gating Weights, K={args.k_pct}%)')
ax2.set_xticks(range(len(selected_layers)))
ax2.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
ax2.axhline(1/len(selected_layers), color='gray', linestyle='--', label='Uniform')
ax2.legend()

plt.tight_layout()
plot_path = f"{args.output_dir}/fixed_vs_gated_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved plot to {plot_path}")
plt.show()

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
