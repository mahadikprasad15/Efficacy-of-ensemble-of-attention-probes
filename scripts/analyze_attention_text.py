# ============================================================================
# TEXT-BASED ATTENTION VISUALIZATION
# Shows actual words with attention weights highlighted
# 5 honest + 5 deceptive examples
# ============================================================================

import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm
import textwrap

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# PATHS
# ============================================================================
OOD_DIR = "data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test"
APOLLO_DATA = "data/apollo_raw/insider_trading/llama-70b-3.3-generations.json"
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
    HAS_TOKENIZER = False
    # Create a dummy tokenizer for position-based display
    class DummyTokenizer:
        def convert_ids_to_tokens(self, ids):
            return [f"[{i}]" for i in ids]
    tokenizer = DummyTokenizer()

# ============================================================================
# LOAD APOLLO DATA (for original text)
# ============================================================================
print("\nLoading Apollo data for original text...")
apollo_texts = {}
try:
    with open(APOLLO_DATA, 'r') as f:
        apollo_data = json.load(f)
    
    for item in apollo_data:
        # Extract the trade_message (model's response)
        if 'metadata' in item and 'trade_message' in item['metadata']:
            scenario_id = item.get('scenario_id', item.get('id', ''))
            trade_msg = item['metadata']['trade_message']
            made_trade = item['metadata'].get('made_trade', '')
            
            apollo_texts[scenario_id] = {
                'text': trade_msg,
                'made_trade': made_trade
            }
    print(f"✓ Loaded {len(apollo_texts)} Apollo texts")
except Exception as e:
    print(f"⚠️ Could not load Apollo data: {e}")
    apollo_texts = {}

# ============================================================================
# LOAD ACTIVATIONS DATA
# ============================================================================
print("\nLoading OOD activations...")
with open(f"{OOD_DIR}/manifest.jsonl", 'r') as f:
    manifest = [json.loads(line) for line in f]

shards = sorted(glob.glob(f"{OOD_DIR}/shard_*.safetensors"))
all_tensors = {}
for shard_path in shards:
    all_tensors.update(load_file(shard_path))

samples_data = []
for entry in manifest:
    eid = entry['id']
    if eid in all_tensors:
        # Try to get text from Apollo data
        text = ""
        if eid in apollo_texts:
            text = apollo_texts[eid]['text']
        elif 'text' in entry:
            text = entry['text']
        elif 'completion' in entry:
            text = entry['completion']
        
        samples_data.append({
            'id': eid,
            'tensor': all_tensors[eid],
            'label': entry['label'],
            'text': text[:500] if text else f"[Sample {eid}]"  # Truncate long text
        })

print(f"✓ Loaded {len(samples_data)} samples")
print(f"  With text: {sum(1 for s in samples_data if len(s['text']) > 20)}")

# ============================================================================
# LOAD PROBE
# ============================================================================
D = samples_data[0]['tensor'].shape[-1]
probe_path = f"{PROBES_BASE}/attn/probe_layer_{BEST_LAYER}.pt"
probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
probe.load_state_dict(torch.load(probe_path, map_location=device))
probe.eval()

# ============================================================================
# SELECT 5 HONEST + 5 DECEPTIVE
# ============================================================================
honest_samples = [s for s in samples_data if s['label'] == 0][:5]
deceptive_samples = [s for s in samples_data if s['label'] == 1][:5]

if len(honest_samples) < 5 or len(deceptive_samples) < 5:
    print(f"⚠️ Not enough samples: {len(honest_samples)} honest, {len(deceptive_samples)} deceptive")
    # Use whatever we have
    honest_samples = [s for s in samples_data if s['label'] == 0][:5]
    deceptive_samples = [s for s in samples_data if s['label'] == 1][:5]

selected_samples = honest_samples + deceptive_samples
print(f"\n✓ Selected {len(honest_samples)} honest + {len(deceptive_samples)} deceptive samples")

# ============================================================================
# GET ATTENTION WEIGHTS FOR EACH SAMPLE
# ============================================================================
print("\nExtracting attention weights...")

results = []
for sample in tqdm(selected_samples):
    tensor = torch.tensor(sample['tensor']).float().unsqueeze(0)  # (1, L, T, D)
    x_layer = tensor[:, BEST_LAYER, :, :].to(device)  # (1, T, D)
    
    with torch.no_grad():
        _, weights = probe.pooling(x_layer, return_attention=True)
        weights = weights.cpu().numpy().flatten()
    
    # Tokenize the text if we have it
    text = sample['text']
    if HAS_TOKENIZER and text and len(text) > 10:
        tokens = tokenizer.tokenize(text)[:len(weights)]
    else:
        tokens = [f"[{i}]" for i in range(len(weights))]
    
    # Pad tokens to match weights length
    while len(tokens) < len(weights):
        tokens.append("[PAD]")
    
    results.append({
        'id': sample['id'],
        'label': 'HONEST' if sample['label'] == 0 else 'DECEPTIVE',
        'text': text,
        'tokens': tokens[:len(weights)],
        'weights': weights,
        'top_5_idx': np.argsort(weights)[-5:][::-1]
    })

# ============================================================================
# VISUALIZATION: Highlighted Text with Attention
# ============================================================================
print("\nGenerating visualizations...")

def plot_attention_text(result, ax, max_tokens=60):
    """Plot text with attention highlighting."""
    tokens = result['tokens'][:max_tokens]
    weights = result['weights'][:max_tokens]
    
    # Normalize weights for coloring
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Create colormap
    cmap = plt.cm.YlOrRd
    
    # Clear axis
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title with label
    color = '#2ecc71' if result['label'] == 'HONEST' else '#e74c3c'
    ax.set_title(f"{result['label']} | ID: {result['id'][:20]}...", 
                fontsize=12, fontweight='bold', color=color, loc='left')
    
    # Display tokens in wrapped format
    x, y = 0.01, 0.85
    line_height = 0.12
    
    for i, (token, w, w_norm) in enumerate(zip(tokens, weights, weights_norm)):
        # Clean token for display
        token_display = token.replace('▁', ' ').replace('Ġ', ' ').strip()
        if not token_display:
            token_display = '·'
        
        # Background color based on attention
        bg_color = cmap(w_norm)
        
        # Calculate token width
        token_width = len(token_display) * 0.012 + 0.02
        
        # Check if we need to wrap to next line
        if x + token_width > 0.98:
            x = 0.01
            y -= line_height
            if y < 0.1:
                break
        
        # Draw background rectangle
        rect = Rectangle((x, y - 0.04), token_width, 0.08, 
                         facecolor=bg_color, edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Draw token text
        text_color = 'white' if w_norm > 0.6 else 'black'
        ax.text(x + token_width/2, y, token_display, 
               fontsize=8, ha='center', va='center', color=text_color,
               fontfamily='monospace')
        
        x += token_width + 0.005
    
    # Add attention statistics
    concentration = sum(sorted(weights, reverse=True)[:5]) / (sum(weights) + 1e-8)
    # Safe access to top tokens
    valid_top_idx = [i for i in result['top_5_idx'][:5] if i < len(tokens)]
    top_5_tokens = [tokens[i].replace('▁', ' ').replace('Ġ', ' ') for i in valid_top_idx]
    
    stats_text = f"Top 5 tokens get {concentration*100:.1f}% attention"
    ax.text(0.01, 0.02, stats_text, fontsize=8, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Create figure with 10 subplots (5 honest + 5 deceptive)
fig, axes = plt.subplots(10, 1, figsize=(16, 25))
fig.suptitle('Attention Weights on Actual Text\n(Layer 16 ATTN Pooling)', 
             fontsize=16, fontweight='bold', y=0.995)

for i, result in enumerate(results):
    plot_attention_text(result, axes[i])

plt.tight_layout(rect=[0, 0, 1, 0.99])
save_path = f"{OUTPUT_DIR}/attention_on_text.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {save_path}")
plt.show()

# ============================================================================
# PRINT TOP ATTENDED WORDS
# ============================================================================
print("\n" + "="*80)
print("TOP 5 ATTENDED TOKENS PER SAMPLE")
print("="*80)

for result in results:
    label_color = "🟢" if result['label'] == 'HONEST' else "🔴"
    print(f"\n{label_color} {result['label']}")
    print(f"   ID: {result['id'][:30]}...")
    
    top_words = []
    for idx in result['top_5_idx']:
        if idx < len(result['tokens']):
            token = result['tokens'][idx].replace('▁', ' ').replace('Ġ', ' ').strip()
            weight = result['weights'][idx]
            top_words.append(f"'{token}' ({weight:.3f})")
    
    print(f"   Top tokens: {', '.join(top_words)}")

# ============================================================================
# COMPARE: What words do HONEST vs DECEPTIVE attend to?
# ============================================================================
print("\n" + "="*80)
print("HONEST vs DECEPTIVE: TOP ATTENDED WORDS COMPARISON")
print("="*80)

honest_top_words = []
deceptive_top_words = []

for result in results:
    for idx in result['top_5_idx'][:3]:
        if idx < len(result['tokens']):
            word = result['tokens'][idx].replace('▁', ' ').replace('Ġ', ' ').strip().lower()
            if word and len(word) > 1:
                if result['label'] == 'HONEST':
                    honest_top_words.append(word)
                else:
                    deceptive_top_words.append(word)

print(f"\n🟢 HONEST top words: {', '.join(set(honest_top_words))}")
print(f"🔴 DECEPTIVE top words: {', '.join(set(deceptive_top_words))}")

# Find overlapping words
common = set(honest_top_words) & set(deceptive_top_words)
if common:
    print(f"\n⚪ Common to both: {', '.join(common)}")
    print("   → Probe focuses on same semantic positions regardless of class!")

print("\n" + "="*80)
print(f"✅ Visualization saved to: {save_path}")
print("="*80)
