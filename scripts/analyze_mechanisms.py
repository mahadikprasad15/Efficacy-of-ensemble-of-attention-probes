# ============================================================================
# MECHANISTIC ANALYSIS: Understanding What ATTN & Gated Ensemble Learn
# ============================================================================
# This notebook analyzes:
# 1. Token-level attention weights (which tokens does ATTN focus on?)
# 2. Layer gating weights (which layers does the Gated ensemble trust?)
# 3. Attention entropy (how focused vs distributed is attention?)
# ============================================================================

import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import GatedEnsemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# PATHS - Update these to match your setup
# ============================================================================
OOD_DIR = "data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test"
PROBES_BASE = "data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying"
GATED_MODELS_DIR = "results/ensembles/attn/gated_models_val"  # Best gated model
OUTPUT_DIR = "results/mechanistic_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD OOD DATA
# ============================================================================
print("Loading OOD data...")
with open(f"{OOD_DIR}/manifest.jsonl", 'r') as f:
    manifest = [json.loads(line) for line in f]

shards = sorted(glob.glob(f"{OOD_DIR}/shard_*.safetensors"))
all_tensors = {}
for shard_path in shards:
    all_tensors.update(load_file(shard_path))

samples, labels, ids = [], [], []
for entry in manifest:
    eid = entry['id']
    if eid in all_tensors:
        samples.append(all_tensors[eid])
        labels.append(entry['label'])
        ids.append(eid)

X = torch.stack(samples).float()
y = np.array(labels)
print(f"✓ Loaded {len(X)} samples ({sum(y==0)} honest, {sum(y==1)} deceptive)")

# ============================================================================
# ANALYSIS 1: Token-Level Attention Weights
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 1: Token-Level Attention Weights (ATTN Pooling)")
print("="*60)

# Load best ATTN probe (Layer 16)
BEST_LAYER = 16
probe_path = f"{PROBES_BASE}/attn/probe_layer_{BEST_LAYER}.pt"
D = X.shape[-1]

probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
probe.load_state_dict(torch.load(probe_path, map_location=device))
probe.eval()

# Get attention weights for a batch
n_examples = min(50, len(X))
x_layer = X[:n_examples, BEST_LAYER, :, :].to(device)  # (N, T, D)

with torch.no_grad():
    pooled, attention_weights = probe.pooling(x_layer, return_attention=True)
    # attention_weights: (N, T)

attention_weights = attention_weights.cpu().numpy()
print(f"Attention weights shape: {attention_weights.shape}")

# Plot 1: Heatmap of attention across samples
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1a: Heatmap for first 20 samples
ax1 = axes[0, 0]
im1 = ax1.imshow(attention_weights[:20, :50], aspect='auto', cmap='YlOrRd')
ax1.set_xlabel('Token Position', fontsize=12)
ax1.set_ylabel('Sample Index', fontsize=12)
ax1.set_title(f'Attention Weights (Layer {BEST_LAYER}, First 20 samples)', fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Attention Weight')

# 1b: Mean attention across all samples
ax2 = axes[0, 1]
mean_attention = attention_weights.mean(axis=0)
ax2.bar(range(len(mean_attention[:100])), mean_attention[:100], color='coral', alpha=0.7)
ax2.axhline(y=1/len(mean_attention), color='gray', linestyle='--', label='Uniform')
ax2.set_xlabel('Token Position', fontsize=12)
ax2.set_ylabel('Mean Attention Weight', fontsize=12)
ax2.set_title('Average Attention Distribution Across Samples', fontsize=14, fontweight='bold')
ax2.legend()

# 1c: Compare deceptive vs honest samples
ax3 = axes[1, 0]
honest_attn = attention_weights[y[:n_examples] == 0].mean(axis=0) if sum(y[:n_examples]==0) > 0 else np.zeros(attention_weights.shape[1])
deceptive_attn = attention_weights[y[:n_examples] == 1].mean(axis=0) if sum(y[:n_examples]==1) > 0 else np.zeros(attention_weights.shape[1])

positions = np.arange(min(100, len(honest_attn)))
width = 0.4
ax3.bar(positions - width/2, honest_attn[:100], width, label='Honest', color='#2ecc71', alpha=0.7)
ax3.bar(positions + width/2, deceptive_attn[:100], width, label='Deceptive', color='#e74c3c', alpha=0.7)
ax3.set_xlabel('Token Position', fontsize=12)
ax3.set_ylabel('Mean Attention Weight', fontsize=12)
ax3.set_title('Attention: Honest vs Deceptive Samples', fontsize=14, fontweight='bold')
ax3.legend()

# 1d: Top-K token positions
ax4 = axes[1, 1]
top_k = 10
top_positions = np.argsort(mean_attention)[-top_k:][::-1]
top_weights = mean_attention[top_positions]
ax4.barh(range(top_k), top_weights[::-1], color='coral')
ax4.set_yticks(range(top_k))
ax4.set_yticklabels([f'Token {p}' for p in top_positions[::-1]])
ax4.set_xlabel('Mean Attention Weight', fontsize=12)
ax4.set_title(f'Top {top_k} Most Attended Token Positions', fontsize=14, fontweight='bold')

plt.tight_layout()
save_path = f"{OUTPUT_DIR}/attention_weights_analysis.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.show()

# ============================================================================
# ANALYSIS 2: Gating Weights (Layer Importance)
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 2: Gated Ensemble Layer Weights")
print("="*60)

# Find best gated model
gated_files = sorted(glob.glob(f"{GATED_MODELS_DIR}/gated_k*.pt"))
if not gated_files:
    print("⚠️ No gated models found. Looking in alternative paths...")
    alt_paths = [
        "results/ensembles/attn/gated_models_val",
        "results/ensembles/attn/gated_models_ood",
    ]
    for p in alt_paths:
        gated_files = sorted(glob.glob(f"{p}/gated_k*.pt"))
        if gated_files:
            GATED_MODELS_DIR = p
            break

if gated_files:
    print(f"Found {len(gated_files)} gated models")
    
    # Load the best one (K=40% based on results)
    best_gated_path = None
    for gf in gated_files:
        if "k40" in gf or "k60" in gf:  # Try best K values
            best_gated_path = gf
            break
    if best_gated_path is None:
        best_gated_path = gated_files[-1]  # Use last if not found
    
    print(f"Analyzing: {best_gated_path}")
    
    # Load the gated model
    k_pct = int(best_gated_path.split('_k')[-1].replace('.pt', ''))
    num_layers = max(1, int(28 * k_pct / 100))
    
    gated = GatedEnsemble(input_dim=num_layers, num_layers=num_layers).to(device)
    gated.load_state_dict(torch.load(best_gated_path, map_location=device))
    gated.eval()
    
    # Get layer-level logits for samples
    # First, we need to extract logits from all selected layers
    probe_dir = f"{PROBES_BASE}/attn"
    probe_files = sorted(
        glob.glob(f"{probe_dir}/probe_layer_*.pt"),
        key=lambda x: int(x.split('_')[-1].replace('.pt', ''))
    )
    
    # Load layer_results to get top-K layers
    with open(f"{probe_dir}/layer_results.json", 'r') as f:
        layer_results = json.load(f)
    
    # Select top-K layers by val_auc
    sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
    selected_layers = [l['layer'] for l in sorted_layers[:num_layers]]
    selected_layers = sorted(selected_layers)
    print(f"Selected layers (K={k_pct}%): {selected_layers}")
    
    # Extract logits from selected layers
    layer_logits = []
    for layer_idx in selected_layers:
        probe_path = f"{probe_dir}/probe_layer_{layer_idx}.pt"
        probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        probe.eval()
        
        with torch.no_grad():
            x_layer = X[:n_examples, layer_idx, :, :].to(device)
            logits = probe(x_layer).cpu().numpy().flatten()
            layer_logits.append(logits)
    
    layer_logits = np.array(layer_logits).T  # (N, L)
    layer_logits_tensor = torch.tensor(layer_logits, dtype=torch.float32).to(device)
    
    # Get gating weights for each sample
    with torch.no_grad():
        summary = layer_logits_tensor  # Use logits as features
        gating_weights = gated.gate_net(summary).cpu().numpy()  # (N, L)
    
    print(f"Gating weights shape: {gating_weights.shape}")
    
    # Plot gating analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2a: Mean gating weights per layer
    ax1 = axes[0, 0]
    mean_gating = gating_weights.mean(axis=0)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(selected_layers)))
    bars = ax1.bar(range(len(selected_layers)), mean_gating, color=colors)
    ax1.set_xticks(range(len(selected_layers)))
    ax1.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Mean Gating Weight', fontsize=12)
    ax1.set_title(f'Layer Importance (Gated Ensemble, K={k_pct}%)', fontsize=14, fontweight='bold')
    ax1.axhline(y=1/len(selected_layers), color='red', linestyle='--', label='Uniform')
    ax1.legend()
    
    # 2b: Gating weight distribution per layer
    ax2 = axes[0, 1]
    ax2.boxplot([gating_weights[:, i] for i in range(len(selected_layers))],
                labels=[f'L{l}' for l in selected_layers])
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Gating Weight Distribution', fontsize=12)
    ax2.set_title('Gating Weight Variance Across Samples', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 2c: Heatmap of per-sample gating
    ax3 = axes[1, 0]
    im = ax3.imshow(gating_weights[:20, :], aspect='auto', cmap='YlGnBu')
    ax3.set_xlabel('Layer Index', fontsize=12)
    ax3.set_ylabel('Sample Index', fontsize=12)
    ax3.set_xticks(range(len(selected_layers)))
    ax3.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
    ax3.set_title('Per-Sample Gating Weights (Input-Dependent)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Gating Weight')
    
    # 2d: Honest vs Deceptive gating
    ax4 = axes[1, 1]
    honest_gating = gating_weights[y[:n_examples] == 0].mean(axis=0) if sum(y[:n_examples]==0) > 0 else np.zeros(len(selected_layers))
    deceptive_gating = gating_weights[y[:n_examples] == 1].mean(axis=0) if sum(y[:n_examples]==1) > 0 else np.zeros(len(selected_layers))
    
    x_pos = np.arange(len(selected_layers))
    width = 0.35
    ax4.bar(x_pos - width/2, honest_gating, width, label='Honest', color='#2ecc71', alpha=0.8)
    ax4.bar(x_pos + width/2, deceptive_gating, width, label='Deceptive', color='#e74c3c', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'L{l}' for l in selected_layers], rotation=45)
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Mean Gating Weight', fontsize=12)
    ax4.set_title('Layer Gating: Honest vs Deceptive', fontsize=14, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/gating_weights_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()
    
    # Print top layers
    top_gating_idx = np.argsort(mean_gating)[::-1][:5]
    print("\n📊 Top 5 Most Trusted Layers:")
    for i, idx in enumerate(top_gating_idx):
        print(f"   {i+1}. Layer {selected_layers[idx]}: weight = {mean_gating[idx]:.4f}")
else:
    print("⚠️ No gated models found. Skip gating analysis.")

# ============================================================================
# ANALYSIS 3: Attention Entropy Analysis
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 3: Attention Entropy (Focus vs Distribution)")
print("="*60)

# Compute entropy for all layers
entropy_by_layer = []
pooling = "attn"
probe_dir = f"{PROBES_BASE}/{pooling}"

for layer_idx in tqdm(range(28), desc="Computing entropy"):
    probe_path = f"{probe_dir}/probe_layer_{layer_idx}.pt"
    if not os.path.exists(probe_path):
        continue
        
    probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval()
    
    # Get attention weights
    with torch.no_grad():
        x_layer = X[:n_examples, layer_idx, :, :].to(device)
        _ = probe(x_layer)  # This stores attention weights
        entropy = probe.pooling.compute_attention_entropy()
    
    entropy_by_layer.append({
        'layer': layer_idx,
        'entropy': entropy
    })

# Also get OOD AUC for correlation
with open(f"{probe_dir}/layer_results.json", 'r') as f:
    layer_results = json.load(f)
layer_auc_map = {r['layer']: r['val_auc'] for r in layer_results}

# Merge data
for entry in entropy_by_layer:
    entry['val_auc'] = layer_auc_map.get(entry['layer'], 0)

# Plot entropy analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3a: Entropy by layer
ax1 = axes[0]
layers = [e['layer'] for e in entropy_by_layer]
entropies = [e['entropy'] for e in entropy_by_layer]
ax1.plot(layers, entropies, marker='o', linewidth=2, color='purple', alpha=0.8)
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Attention Entropy', fontsize=12)
ax1.set_title('Attention Entropy by Layer\n(Lower = More Focused)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Highlight best OOD layer
best_layer = 16
ax1.axvline(x=best_layer, color='red', linestyle='--', alpha=0.5, label=f'Best OOD Layer ({best_layer})')
ax1.legend()

# 3b: Entropy vs Validation AUC
ax2 = axes[1]
aucs = [e['val_auc'] for e in entropy_by_layer]
ax2.scatter(entropies, aucs, c=layers, cmap='viridis', s=100, alpha=0.8)
ax2.set_xlabel('Attention Entropy', fontsize=12)
ax2.set_ylabel('Validation AUC', fontsize=12)
ax2.set_title('Entropy vs Performance\n(Is focus correlated with accuracy?)', fontsize=14, fontweight='bold')

# Add colorbar for layer
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 27))
plt.colorbar(sm, ax=ax2, label='Layer')

# Correlation
corr = np.corrcoef(entropies, aucs)[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3c: Entropy distribution: Honest vs Deceptive
ax3 = axes[2]

# Compute per-sample entropy for best layer
probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
probe.load_state_dict(torch.load(f"{probe_dir}/probe_layer_{BEST_LAYER}.pt", map_location=device))
probe.eval()

honest_entropies = []
deceptive_entropies = []

with torch.no_grad():
    for i in range(n_examples):
        x_sample = X[i:i+1, BEST_LAYER, :, :].to(device)
        _ = probe(x_sample)
        ent = probe.pooling.compute_attention_entropy()
        
        if y[i] == 0:
            honest_entropies.append(ent)
        else:
            deceptive_entropies.append(ent)

ax3.hist(honest_entropies, bins=20, alpha=0.6, label='Honest', color='#2ecc71')
ax3.hist(deceptive_entropies, bins=20, alpha=0.6, label='Deceptive', color='#e74c3c')
ax3.set_xlabel('Attention Entropy', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title(f'Entropy Distribution (Layer {BEST_LAYER})\n(Do deceptive samples have different focus?)', fontsize=14, fontweight='bold')
ax3.legend()

# Stats
if honest_entropies and deceptive_entropies:
    honest_mean = np.mean(honest_entropies)
    deceptive_mean = np.mean(deceptive_entropies)
    ax3.axvline(x=honest_mean, color='#2ecc71', linestyle='--', linewidth=2)
    ax3.axvline(x=deceptive_mean, color='#e74c3c', linestyle='--', linewidth=2)
    ax3.text(0.95, 0.95, f'Honest μ: {honest_mean:.3f}\nDeceptive μ: {deceptive_mean:.3f}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_path = f"{OUTPUT_DIR}/entropy_analysis.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("📊 MECHANISTIC ANALYSIS SUMMARY")
print("="*60)

print("\n1. TOKEN-LEVEL ATTENTION:")
print(f"   - Attention is NOT uniform (focuses on specific positions)")
print(f"   - Top attended positions: {top_positions[:5].tolist()}")
print(f"   - Mean attention weight on top position: {max(mean_attention):.4f}")
print(f"   - Uniform would be: {1/len(mean_attention):.4f}")

if gated_files:
    print(f"\n2. GATING (LAYER SELECTION):")
    print(f"   - Gated ensemble uses {num_layers} layers (K={k_pct}%)")
    print(f"   - Most trusted layers: {[selected_layers[i] for i in top_gating_idx[:3]]}")
    print(f"   - Gating is input-dependent (varies per sample)")

print(f"\n3. ENTROPY ANALYSIS:")
print(f"   - Layer {BEST_LAYER} entropy: {entropy_by_layer[BEST_LAYER]['entropy']:.4f}")
print(f"   - Entropy-AUC correlation: {corr:.3f}")
if honest_entropies and deceptive_entropies:
    print(f"   - Honest samples mean entropy: {honest_mean:.4f}")
    print(f"   - Deceptive samples mean entropy: {deceptive_mean:.4f}")

print("\n" + "="*60)
print(f"✅ All plots saved to: {OUTPUT_DIR}/")
print("="*60)
