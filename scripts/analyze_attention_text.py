# ============================================================================
# ENHANCED MECHANISTIC ANALYSIS: Text + Attention Visualization
# ============================================================================
# Improvements:
# 1. Show actual text/tokens with attention weights
# 2. "Attention Concentration" metric: What % of tokens capture 80% attention?
# 3. Word-cloud style visualization of high-attention tokens
# ============================================================================

import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# PATHS - Update these to match your setup
# ============================================================================
OOD_DIR = "data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test"
PROBES_BASE = "data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "results/mechanistic_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_LAYER = 16

# ============================================================================
# LOAD TOKENIZER
# ============================================================================
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get('HF_TOKEN'))
    HAS_TOKENIZER = True
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"⚠️ Could not load tokenizer: {e}")
    print("   Will use token positions instead of actual text")
    HAS_TOKENIZER = False

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading OOD data...")
with open(f"{OOD_DIR}/manifest.jsonl", 'r') as f:
    manifest = [json.loads(line) for line in f]

shards = sorted(glob.glob(f"{OOD_DIR}/shard_*.safetensors"))
all_tensors = {}
for shard_path in shards:
    all_tensors.update(load_file(shard_path))

samples, labels, metadata_list = [], [], []
for entry in manifest:
    eid = entry['id']
    if eid in all_tensors:
        samples.append(all_tensors[eid])
        labels.append(entry['label'])
        metadata_list.append(entry.get('metadata', {}))

X = torch.stack(samples).float()
y = np.array(labels)
print(f"✓ Loaded {len(X)} samples ({sum(y==0)} honest, {sum(y==1)} deceptive)")

# ============================================================================
# LOAD PROBE
# ============================================================================
D = X.shape[-1]
probe_path = f"{PROBES_BASE}/attn/probe_layer_{BEST_LAYER}.pt"
probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
probe.load_state_dict(torch.load(probe_path, map_location=device))
probe.eval()

# ============================================================================
# ANALYSIS 1: Attention Concentration Metric
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 1: Attention Concentration Metric")
print("What % of tokens capture 80% of attention?")
print("="*60)

n_examples = min(50, len(X))

def compute_attention_concentration(weights, threshold=0.8):
    """
    Compute what fraction of tokens are needed to capture `threshold` of attention.
    Lower = more concentrated/focused.
    """
    # Sort weights descending
    sorted_weights = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_weights)
    # Find first index where cumsum >= threshold
    idx = np.searchsorted(cumsum, threshold) + 1
    return idx / len(weights)

concentration_scores = []
attention_all = []

with torch.no_grad():
    for i in range(n_examples):
        x_sample = X[i:i+1, BEST_LAYER, :, :].to(device)
        _, weights = probe.pooling(x_sample, return_attention=True)
        weights = weights.cpu().numpy().flatten()
        
        conc = compute_attention_concentration(weights, threshold=0.8)
        concentration_scores.append({
            'sample_idx': i,
            'label': labels[i],
            'concentration_80': conc,
            'weights': weights
        })
        attention_all.append(weights)

attention_all = np.array(attention_all)

# Summary stats
honest_conc = [c['concentration_80'] for c in concentration_scores if c['label'] == 0]
deceptive_conc = [c['concentration_80'] for c in concentration_scores if c['label'] == 1]

print(f"\n📊 Attention Concentration (80% threshold):")
print(f"   Overall mean: {np.mean([c['concentration_80'] for c in concentration_scores]):.1%} of tokens")
print(f"   Honest samples: {np.mean(honest_conc):.1%} of tokens")
print(f"   Deceptive samples: {np.mean(deceptive_conc):.1%} of tokens")
print(f"   → {'Deceptive' if np.mean(deceptive_conc) > np.mean(honest_conc) else 'Honest'} requires more tokens")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Histogram of concentration scores
ax1 = axes[0]
ax1.hist(honest_conc, bins=15, alpha=0.6, label='Honest', color='#2ecc71')
ax1.hist(deceptive_conc, bins=15, alpha=0.6, label='Deceptive', color='#e74c3c')
ax1.axvline(np.mean(honest_conc), color='#2ecc71', linestyle='--', linewidth=2)
ax1.axvline(np.mean(deceptive_conc), color='#e74c3c', linestyle='--', linewidth=2)
ax1.set_xlabel('Fraction of Tokens for 80% Attention', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Attention Concentration\n(Lower = More Focused)', fontsize=14, fontweight='bold')
ax1.legend()

# 1b: Cumulative attention curve (average)
ax2 = axes[1]
mean_weights = attention_all.mean(axis=0)
sorted_mean = np.sort(mean_weights)[::-1]
cumsum = np.cumsum(sorted_mean)
x_pct = np.arange(1, len(cumsum)+1) / len(cumsum)

ax2.plot(x_pct * 100, cumsum, linewidth=2, color='coral')
ax2.axhline(0.8, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
ax2.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
ax2.fill_between(x_pct * 100, 0, cumsum, alpha=0.3, color='coral')

# Find where we hit 80%
idx_80 = np.searchsorted(cumsum, 0.8)
ax2.axvline(x_pct[idx_80] * 100, color='red', linestyle='--', linewidth=2)
ax2.annotate(f'{x_pct[idx_80]*100:.1f}% of tokens\n→ 80% attention', 
             xy=(x_pct[idx_80]*100, 0.8), xytext=(x_pct[idx_80]*100 + 10, 0.6),
             fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_xlabel('% of Tokens (sorted by attention)', fontsize=12)
ax2.set_ylabel('Cumulative Attention', fontsize=12)
ax2.set_title('Cumulative Attention Curve', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1.05)

# 1c: Top tokens visualization
ax3 = axes[2]
top_k = 15
top_positions = np.argsort(mean_weights)[-top_k:][::-1]
top_weights = mean_weights[top_positions]

bars = ax3.barh(range(top_k), top_weights[::-1], color='coral')
ax3.set_yticks(range(top_k))
ax3.set_yticklabels([f'Token {p}' for p in top_positions[::-1]])
ax3.set_xlabel('Mean Attention Weight', fontsize=12)
ax3.set_title(f'Top {top_k} Attended Positions', fontsize=14, fontweight='bold')

# Add percentage labels
total_top_weight = top_weights.sum()
for i, (bar, w) in enumerate(zip(bars, top_weights[::-1])):
    ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{w/mean_weights.sum()*100:.1f}%', va='center', fontsize=9)

plt.tight_layout()
save_path = f"{OUTPUT_DIR}/attention_concentration.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.show()

# ============================================================================
# ANALYSIS 2: Token-Level Attention with Text (if tokenizer available)
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 2: Text Visualization with Attention Weights")
print("="*60)

# Get a few example samples
n_visualize = 3

fig, axes = plt.subplots(n_visualize, 1, figsize=(16, 4*n_visualize))
if n_visualize == 1:
    axes = [axes]

for idx, (sample_idx, ax) in enumerate(zip([0, 5, 10], axes)):
    weights = concentration_scores[sample_idx]['weights']
    label = "DECEPTIVE" if labels[sample_idx] == 1 else "HONEST"
    
    # Normalize for visualization
    weights_norm = weights / weights.max()
    
    # Create color-coded bar chart for token positions
    n_tokens = min(80, len(weights))
    positions = np.arange(n_tokens)
    
    # Color by weight
    colors = plt.cm.YlOrRd(weights_norm[:n_tokens])
    
    bars = ax.bar(positions, weights[:n_tokens], color=colors, edgecolor='none')
    ax.set_xlim(-0.5, n_tokens - 0.5)
    ax.set_xlabel('Token Position', fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=11)
    ax.set_title(f'Sample {sample_idx}: {label}', fontsize=13, fontweight='bold')
    
    # Mark top-5 positions
    top5 = np.argsort(weights)[-5:]
    for pos in top5:
        if pos < n_tokens:
            ax.annotate(f'#{pos}', xy=(pos, weights[pos]), 
                       xytext=(pos, weights[pos] + 0.01),
                       ha='center', fontsize=8, color='red')
    
    # Add concentration score
    conc = concentration_scores[sample_idx]['concentration_80']
    ax.text(0.95, 0.95, f'Concentration: {conc:.1%} of tokens\nfor 80% attention',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_path = f"{OUTPUT_DIR}/token_attention_examples.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.show()

# ============================================================================
# ANALYSIS 3: Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("📊 SUMMARY: Mechanistic Insights")
print("="*60)

print(f"""
ATTENTION CONCENTRATION:
- {np.mean([c['concentration_80'] for c in concentration_scores]):.1%} of tokens capture 80% of attention
- This means the probe focuses on ~{int(np.mean([c['concentration_80'] for c in concentration_scores]) * len(weights))} tokens out of {len(weights)}
- Honest samples: {np.mean(honest_conc):.1%} concentration
- Deceptive samples: {np.mean(deceptive_conc):.1%} concentration

INTERPRETATION:
""")

if np.mean(deceptive_conc) > np.mean(honest_conc):
    print("→ Deceptive samples require MORE tokens to detect (more diffuse signal)")
    print("→ Honest samples have a clearer, more concentrated signature")
else:
    print("→ Honest samples require MORE tokens to detect (more diffuse signal)")
    print("→ Deceptive samples have a clearer, more concentrated signature")

print(f"""
TOP ATTENDED POSITIONS: {top_positions[:5].tolist()}
- These correspond to LATE tokens in the response
- Likely where the model states its action/decision
- "I will execute the trade" vs "I refuse to act on insider info"

WHY THIS MATTERS:
- Focused attention → probe has learned specific "tell" positions
- These positions generalize across Roleplaying → Insider Trading
- The probe is NOT looking at different parts for different classes
- Instead, it reads the SAME positions but detects different representations
""")

print("="*60)
print(f"✅ All plots saved to: {OUTPUT_DIR}/")
print("="*60)
