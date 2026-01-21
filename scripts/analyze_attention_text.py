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
    
    # IDs are formatted as "insider_{idx}" based on array index
    for idx, item in enumerate(apollo_data):
        item_id = f"insider_{idx}"
        
        # Get metadata
        metadata = item.get('metadata', {})
        
        # Extract trade_message (the key deceptive/honest content)
        trade_message = metadata.get('trade_message', '')
        
        # If no trade_message, get last assistant message
        if not trade_message:
            transcript = item.get('transcript', [])
            for msg in reversed(transcript):
                if msg.get('role') == 'assistant':
                    trade_message = msg.get('content', '')
                    break
        
        made_trade = metadata.get('made_trade', '')
        
        if trade_message:
            apollo_texts[item_id] = {
                'text': trade_message,
                'made_trade': made_trade
            }
    
    print(f"✓ Loaded {len(apollo_texts)} Apollo texts")
    # Show sample IDs
    sample_ids = list(apollo_texts.keys())[:5]
    print(f"  Sample IDs: {sample_ids}")
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

def create_html_visualization(results, output_path):
    """Create an HTML file with color-coded attention on text."""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
        .sample { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .honest { border-left: 5px solid #2ecc71; }
        .deceptive { border-left: 5px solid #e74c3c; }
        .label { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
        .label.honest { color: #2ecc71; }
        .label.deceptive { color: #e74c3c; }
        .text { font-size: 16px; line-height: 2.2; }
        .token { padding: 3px 6px; margin: 2px; border-radius: 4px; display: inline-block; }
        .stats { margin-top: 15px; font-size: 14px; color: #666; background: #f0f0f0; padding: 10px; border-radius: 5px; }
        h1 { text-align: center; color: #333; }
        .legend { text-align: center; margin-bottom: 20px; }
        .legend span { padding: 5px 15px; margin: 5px; border-radius: 5px; display: inline-block; }
    </style>
</head>
<body>
    <h1>Attention Weights on Text (Layer 16 ATTN Pooling)</h1>
    <div class="legend">
        <span style="background: #ffffcc;">Low Attention</span>
        <span style="background: #ffeda0;">→</span>
        <span style="background: #feb24c;">→</span>
        <span style="background: #f03b20;">High Attention</span>
    </div>
"""
    
    for result in results:
        label_class = 'honest' if result['label'] == 'HONEST' else 'deceptive'
        
        # Build token spans with color
        tokens = result['tokens'][:80]
        weights = result['weights'][:80]
        
        # Normalize weights
        w_min, w_max = weights.min(), weights.max()
        weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)
        
        token_html = ""
        for token, w_norm in zip(tokens, weights_norm):
            # Clean token
            token_display = token.replace('▁', ' ').replace('Ġ', ' ')
            if not token_display.strip():
                continue
            
            # Color gradient: yellow (low) -> orange -> red (high)
            if w_norm < 0.3:
                bg = f"rgba(255, 255, 200, {0.3 + w_norm})"
                text_color = "#333"
            elif w_norm < 0.6:
                bg = f"rgba(254, 178, 76, {0.5 + w_norm*0.5})"
                text_color = "#333"
            else:
                bg = f"rgba(240, 59, 32, {0.6 + w_norm*0.4})"
                text_color = "white"
            
            token_html += f'<span class="token" style="background: {bg}; color: {text_color};">{token_display}</span>'
        
        # Top tokens
        valid_top = [i for i in result['top_5_idx'][:5] if i < len(tokens)]
        top_tokens = ", ".join([f"'{tokens[i].replace('▁', ' ').strip()}'" for i in valid_top if tokens[i].strip()])
        
        concentration = sum(sorted(weights, reverse=True)[:5]) / (sum(weights) + 1e-8)
        
        html += f"""
    <div class="sample {label_class}">
        <div class="label {label_class}">{result['label']} | ID: {result['id']}</div>
        <div class="text">{token_html}</div>
        <div class="stats">
            <strong>Top 5 tokens get {concentration*100:.1f}% attention:</strong> {top_tokens}
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

# Create HTML visualization (much more legible!)
html_path = f"{OUTPUT_DIR}/attention_on_text.html"
create_html_visualization(results, html_path)
print(f"✓ Saved HTML: {html_path}")

# Also create a simple PNG with just the top words per sample
fig, axes = plt.subplots(5, 2, figsize=(16, 12))
fig.suptitle('Top Attended Words per Sample\n(Layer 16 ATTN Pooling)', fontsize=16, fontweight='bold')

for idx, result in enumerate(results):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    # Get top 10 tokens with weights
    valid_idx = [i for i in np.argsort(result['weights'])[::-1][:10] if i < len(result['tokens'])]
    top_tokens = [result['tokens'][i].replace('▁', ' ').replace('Ġ', ' ').strip() for i in valid_idx]
    top_weights = [result['weights'][i] for i in valid_idx]
    
    # Filter empty tokens
    filtered = [(t, w) for t, w in zip(top_tokens, top_weights) if t and len(t) > 1]
    if filtered:
        tokens_f, weights_f = zip(*filtered[:8])
    else:
        tokens_f, weights_f = ['[empty]'], [0]
    
    # Bar chart
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(tokens_f)))
    bars = ax.barh(range(len(tokens_f)), weights_f[::-1], color=colors[::-1])
    ax.set_yticks(range(len(tokens_f)))
    ax.set_yticklabels(tokens_f[::-1], fontsize=10)
    ax.set_xlabel('Attention Weight', fontsize=10)
    
    # Title with label
    label_color = '#2ecc71' if result['label'] == 'HONEST' else '#e74c3c'
    ax.set_title(f"{result['label']}: {result['id'][:15]}...", fontsize=11, fontweight='bold', color=label_color)

plt.tight_layout(rect=[0, 0, 1, 0.96])
png_path = f"{OUTPUT_DIR}/top_attended_words.png"
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved PNG: {png_path}")
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
