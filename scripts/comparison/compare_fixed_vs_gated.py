# ============================================================================
# FIXED WEIGHT vs GATED ENSEMBLE COMPARISON (v2)
# ============================================================================
# Key fix: Uses batched logits extraction like original evaluation
# Also loads saved OOD results as reference if available
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
parser.add_argument('--saved_ood_results', type=str,
                    default='results/ensembles/attn/ensemble_k_sweep_ood.json',
                    help='Path to saved OOD evaluation results (for reference)')
parser.add_argument('--k_pct', type=int, default=40,
                    help='K% for top layer selection')
parser.add_argument('--output_dir', type=str,
                    default='results/fixed_vs_gated',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
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

def fpr_at_tpr(labels, probs, target_tpr=0.99):
    """Compute FPR when TPR = target_tpr."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    idx = np.where(tpr_arr >= target_tpr)[0]
    return fpr_arr[idx[0]] if len(idx) > 0 else fpr_arr[-1]

def extract_logits(samples, selected_layers, probes):
    """Extract logits from all probes for all samples. Returns (N, K) array."""
    all_logits = []
    for sample in tqdm(samples, desc="Extracting logits"):
        if isinstance(sample, torch.Tensor):
            tensor = sample.clone().detach().float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(sample).float().unsqueeze(0)
        
        layer_logits = []
        for layer_idx in selected_layers:
            x_layer = tensor[:, layer_idx, :, :].to(device)
            with torch.no_grad():
                logit = probes[layer_idx](x_layer).cpu().item()
            layer_logits.append(logit)
        
        all_logits.append(layer_logits)
    
    return np.array(all_logits)  # (N, K)

# ============================================================================
# STEP 1: LOAD LAYER CONFIGURATION
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
# STEP 2: LOAD SAVED OOD RESULTS (REFERENCE)
# ============================================================================
print("\n" + "="*60)
print("STEP 2: Load saved OOD results for reference")
print("="*60)

saved_gated_ood_auc = None
if os.path.exists(args.saved_ood_results):
    with open(args.saved_ood_results, 'r') as f:
        saved_results = json.load(f)
    
    # Find K=40% result
    for r in saved_results:
        if r['k_pct'] == args.k_pct:
            saved_gated_ood_auc = r['gated']['auc']
            print(f"✓ Found saved reference: Gated OOD AUC = {saved_gated_ood_auc:.4f}")
            print(f"  (This was the 0.914 AUC from your plots!)")
            break
else:
    print(f"⚠️ No saved OOD results at {args.saved_ood_results}")

# ============================================================================
# STEP 3: LOAD PROBES
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Load per-layer probes")
print("="*60)

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
# STEP 4: LOAD GATED ENSEMBLE
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Load Gated Ensemble")
print("="*60)

gated = GatedEnsemble(input_dim=k_layers, num_layers=k_layers).to(device)
if os.path.exists(args.gated_model):
    gated.load_state_dict(torch.load(args.gated_model, map_location=device))
    print(f"✓ Loaded gated model from {args.gated_model}")
else:
    print(f"⚠️ Gated model not found at {args.gated_model}")
gated.eval()

# ============================================================================
# STEP 5: EXTRACT LOGITS FROM ID DATA
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Extract logits from ID samples")
print("="*60)

id_logits = extract_logits(id_samples, selected_layers, probes)
print(f"ID logits shape: {id_logits.shape}")

# ============================================================================
# STEP 6: EXTRACT GATING WEIGHTS AND COMPUTE MEAN
# ============================================================================
print("\n" + "="*60)
print("STEP 6: Extract gating weights from ID samples")
print("="*60)

with torch.no_grad():
    id_logits_tensor = torch.tensor(id_logits, dtype=torch.float32).to(device)
    all_gating_weights = gated.gate_net(id_logits_tensor).cpu().numpy()

mean_weights = all_gating_weights.mean(axis=0)
std_weights = all_gating_weights.std(axis=0)
fixed_weights = mean_weights / mean_weights.sum()  # Normalize

print(f"\n📊 Gating Weight Statistics:")
print(f"{'Layer':<10} {'Mean':<10} {'Std':<10}")
print("-" * 30)
for i, layer_idx in enumerate(selected_layers):
    print(f"L{layer_idx:<9} {mean_weights[i]:.4f}     {std_weights[i]:.4f}")

print(f"\nFixed weights (normalized): {fixed_weights}")

# ============================================================================
# STEP 7: LOAD OOD DATA AND EXTRACT LOGITS
# ============================================================================
print("\n" + "="*60)
print("STEP 7: Load OOD data and extract logits")
print("="*60)

ood_samples, ood_labels = load_activations(args.ood_activations)
if ood_samples is None:
    print("❌ Could not load OOD activations")
    ood_logits = None
else:
    ood_logits = extract_logits(ood_samples, selected_layers, probes)
    print(f"OOD logits shape: {ood_logits.shape}")

# ============================================================================
# STEP 8: EVALUATE ALL ENSEMBLES
# ============================================================================
print("\n" + "="*60)
print("STEP 8: Evaluate ensembles on ID and OOD")
print("="*60)

def evaluate_all(logits, labels, fixed_weights, gated_model, dataset_name):
    """Evaluate gated, fixed, and mean ensembles."""
    logits_tensor = torch.tensor(logits, dtype=torch.float32).to(device)
    logits_3d = logits_tensor.unsqueeze(-1)  # (N, K, 1)
    
    # Gated ensemble
    with torch.no_grad():
        gated_logits = gated_model(logits_tensor, logits_3d).cpu().numpy().flatten()
    gated_probs = 1 / (1 + np.exp(-gated_logits))
    
    # Fixed weighted
    fixed_logits = (logits * fixed_weights).sum(axis=1)
    fixed_probs = 1 / (1 + np.exp(-fixed_logits))
    
    # Uniform mean
    mean_logits = logits.mean(axis=1)
    mean_probs = 1 / (1 + np.exp(-mean_logits))
    
    results = {}
    for name, probs in [('gated', gated_probs), ('fixed', fixed_probs), ('mean', mean_probs)]:
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, (probs > 0.5).astype(int))
        fpr = fpr_at_tpr(labels, probs, target_tpr=0.99)
        results[name] = {'auc': auc, 'acc': acc, 'fpr_99': fpr}
        print(f"  {name.upper():8} AUC: {auc:.4f}, Acc: {acc:.4f}, FPR@99%: {fpr:.4f}")
    
    return results

print("\n📊 ID (Roleplaying):")
id_results = evaluate_all(id_logits, id_labels, fixed_weights, gated, "ID")

if ood_logits is not None:
    print("\n📊 OOD (InsiderTrading):")
    ood_results = evaluate_all(ood_logits, ood_labels, fixed_weights, gated, "OOD")
else:
    ood_results = None

# ============================================================================
# STEP 9: GENERATE COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("📊 RESULTS: FIXED vs GATED ENSEMBLE COMPARISON")
print("="*80)

print(f"\n{'Ensemble':<15} | {'ID AUC':<10} | {'ID Acc':<10} | {'OOD AUC':<10} | {'OOD Acc':<10}")
print("-" * 65)
for name in ['gated', 'fixed', 'mean']:
    label = 'Gated' if name == 'gated' else 'Fixed (Mean)' if name == 'fixed' else 'Uniform'
    id_auc = id_results[name]['auc']
    id_acc = id_results[name]['acc']
    ood_auc = ood_results[name]['auc'] if ood_results else 'N/A'
    ood_acc = ood_results[name]['acc'] if ood_results else 'N/A'
    print(f"{label:<15} | {id_auc:.4f}     | {id_acc:.4f}     | {ood_auc if isinstance(ood_auc, str) else f'{ood_auc:.4f}':<10} | {ood_acc if isinstance(ood_acc, str) else f'{ood_acc:.4f}':<10}")

# Add reference comparison
if saved_gated_ood_auc:
    print("\n" + "-"*65)
    print(f"⚠️  NOTE: Saved reference Gated OOD AUC = {saved_gated_ood_auc:.4f}")
    if ood_results:
        print(f"   Current computed Gated OOD AUC = {ood_results['gated']['auc']:.4f}")
        if abs(saved_gated_ood_auc - ood_results['gated']['auc']) > 0.1:
            print("   ⚠️  Large discrepancy! The gated model may have been trained differently.")

# Compute delta
if ood_results:
    delta = ood_results['fixed']['auc'] - ood_results['gated']['auc']
    print("\n" + "-"*65)
    print(f"📈 OOD AUC Delta (Fixed - Gated): {delta:+.4f}")
    if abs(delta) < 0.02:
        print("   → ✅ HUGE: Fixed weights ≈ Gated! Can simplify to fixed ensemble.")
    else:
        print(f"   → 🔬 Gated is {'better' if delta < 0 else 'worse'} by {abs(delta):.4f}")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
results = {
    'config': {
        'k_pct': args.k_pct,
        'num_layers': k_layers,
        'selected_layers': selected_layers,
        'fixed_weights': fixed_weights.tolist(),
        'weight_std': std_weights.tolist()
    },
    'id': id_results,
    'ood': ood_results,
    'reference_gated_ood_auc': saved_gated_ood_auc
}

results_path = f"{args.output_dir}/fixed_vs_gated_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved results to {results_path}")

# ============================================================================
# STEP 11: PLOT COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart for AUC
ensembles = ['Gated', 'Fixed\n(Mean Weights)', 'Uniform']
id_aucs = [id_results['gated']['auc'], id_results['fixed']['auc'], id_results['mean']['auc']]
ood_aucs = [ood_results['gated']['auc'], ood_results['fixed']['auc'], ood_results['mean']['auc']] if ood_results else [0, 0, 0]

x = np.arange(len(ensembles))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, id_aucs, width, label='ID (Roleplaying)', color='#3498db')
bars2 = ax.bar(x + width/2, ood_aucs, width, label='OOD (InsiderTrading)', color='#e74c3c')
if saved_gated_ood_auc:
    ax.axhline(saved_gated_ood_auc, color='green', linestyle='--', label=f'Ref Gated OOD ({saved_gated_ood_auc:.3f})', alpha=0.7)
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
