# ============================================================================
# LAYER-COLORED ENSEMBLE ATTENTION
# ============================================================================
# Shows which layer(s) contribute attention to each token
# - Each layer gets a distinct color
# - Token border/underline shows which layer focuses on it most
# - Reveals layer agreement vs disagreement
# ============================================================================

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import GatedEnsemble

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Layer Colored Attention Analysis')
parser.add_argument('--ood_dir', type=str, 
                    default='data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test')
parser.add_argument('--apollo_data', type=str,
                    default='data/apollo_raw/insider_trading/llama-70b-3.3-generations.json')
parser.add_argument('--probes_base', type=str,
                    default='data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying')
parser.add_argument('--output_dir', type=str, default='results/mechanistic_analysis')
parser.add_argument('--k_pct', type=int, default=40)
args = parser.parse_args()

OOD_DIR = args.ood_dir
APOLLO_DATA = args.apollo_data
PROBES_BASE = args.probes_base
OUTPUT_DIR = args.output_dir
K_PCT = args.k_pct
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"OOD Dir: {OOD_DIR}")
print(f"Probes Base: {PROBES_BASE}")
print(f"Output Dir: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA (same as before)
# ============================================================================
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get('HF_TOKEN'))
    HAS_TOKENIZER = True
    print("✓ Tokenizer loaded")
except:
    HAS_TOKENIZER = False

print("\nLoading Apollo data...")
apollo_texts = {}
try:
    with open(APOLLO_DATA, 'r') as f:
        apollo_data = json.load(f)
    for idx, item in enumerate(apollo_data):
        item_id = f"insider_{idx}"
        metadata = item.get('metadata', {})
        trade_message = metadata.get('trade_message', '')
        if not trade_message:
            for msg in reversed(item.get('transcript', [])):
                if msg.get('role') == 'assistant':
                    trade_message = msg.get('content', '')
                    break
        if trade_message:
            apollo_texts[item_id] = {'text': trade_message}
    print(f"✓ Loaded {len(apollo_texts)} texts")
except Exception as e:
    print(f"⚠️ Error: {e}")

print("\nLoading activations...")
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
        text = apollo_texts.get(eid, {}).get('text', '')
        samples_data.append({
            'id': eid, 'tensor': all_tensors[eid], 
            'label': entry['label'], 'text': text[:500] if text else f"[{eid}]"
        })
print(f"✓ Loaded {len(samples_data)} samples")

# ============================================================================
# LOAD LAYERS
# ============================================================================
probe_dir = f"{PROBES_BASE}/attn"
with open(f"{probe_dir}/layer_results.json", 'r') as f:
    layer_results = json.load(f)

num_layers = 28
k_layers = max(1, int(num_layers * K_PCT / 100))
sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
selected_layers = sorted([l['layer'] for l in sorted_layers[:k_layers]])
print(f"\n📊 Using {k_layers} layers: {selected_layers}")

# Create color palette for layers
layer_colors = {}
cmap = plt.cm.get_cmap('tab20', len(selected_layers))
for i, layer_idx in enumerate(selected_layers):
    c = cmap(i)
    layer_colors[layer_idx] = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"

D = samples_data[0]['tensor'].shape[-1]
probes = {}
for layer_idx in selected_layers:
    probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(f"{probe_dir}/probe_layer_{layer_idx}.pt", map_location=device))
    probe.eval()
    probes[layer_idx] = probe
print(f"✓ Loaded {len(probes)} probes")

# ============================================================================
# SELECT SAMPLES
# ============================================================================
print("\nClassifying samples...")
classified = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
for sample in tqdm(samples_data, desc="Predicting"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    layer_logits = []
    for layer_idx in selected_layers:
        with torch.no_grad():
            logit = probes[layer_idx](tensor[:, layer_idx, :, :].to(device)).cpu().item()
        layer_logits.append(logit)
    pred = 1 if np.mean(layer_logits) > 0 else 0
    gold = sample['label']
    cat = 'TP' if gold==1 and pred==1 else 'TN' if gold==0 and pred==0 else 'FP' if gold==0 and pred==1 else 'FN'
    sample['prediction'], sample['category'] = pred, cat
    classified[cat].append(sample)

selected_samples = []
for cat in ['TP', 'TN', 'FP', 'FN']:
    selected_samples.extend(classified[cat][:2])
print(f"✓ Selected {len(selected_samples)} samples")

# ============================================================================
# EXTRACT PER-LAYER ATTENTION
# ============================================================================
print("\nExtracting per-layer attention...")

results = []
for sample in tqdm(selected_samples, desc="Extracting"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    
    layer_attentions = {}
    for layer_idx in selected_layers:
        with torch.no_grad():
            _, attn = probes[layer_idx].pooling(tensor[:, layer_idx, :, :].to(device), return_attention=True)
        layer_attentions[layer_idx] = attn.cpu().numpy().flatten()
    
    # Stack attentions: (num_layers, T)
    attn_matrix = np.array([layer_attentions[l] for l in selected_layers])
    
    # For each token, find which layer focuses on it most
    # Normalize per layer first
    attn_normalized = attn_matrix / (attn_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Combined attention
    combined = attn_matrix.mean(axis=0)
    combined = combined / (combined.sum() + 1e-8)
    
    # Tokenize
    text = sample['text']
    if HAS_TOKENIZER and text and len(text) > 10:
        tokens = tokenizer.tokenize(text)[:len(combined)]
    else:
        tokens = [f"[{i}]" for i in range(len(combined))]
    while len(tokens) < len(combined):
        tokens.append("[PAD]")
    
    cat = sample['category']
    results.append({
        'id': sample['id'],
        'category': cat,
        'gold': 'DECEPTIVE' if sample['label'] == 1 else 'HONEST',
        'pred': 'DECEPTIVE' if sample['prediction'] == 1 else 'HONEST',
        'tokens': tokens[:len(combined)],
        'combined_attention': combined,
        'per_layer_attention': {l: layer_attentions[l][:len(combined)] for l in selected_layers},
        'attn_matrix': attn_matrix[:, :len(combined)]
    })

# ============================================================================
# CREATE HTML WITH LAYER-COLORED ATTENTION
# ============================================================================
print("\nGenerating layer-colored visualization...")

# Build layer color legend
layer_legend = " | ".join([f'<span style="background: {layer_colors[l]}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">L{l}</span>' for l in selected_layers])

html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        .sample {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .TP {{ border-left: 5px solid #27ae60; }}
        .TN {{ border-left: 5px solid #3498db; }}
        .FP {{ border-left: 5px solid #e74c3c; }}
        .FN {{ border-left: 5px solid #f39c12; }}
        .label {{ font-weight: bold; font-size: 16px; margin-bottom: 5px; }}
        .label.TP {{ color: #27ae60; }}
        .label.TN {{ color: #3498db; }}
        .label.FP {{ color: #e74c3c; }}
        .label.FN {{ color: #f39c12; }}
        .meta {{ font-size: 12px; color: #666; margin-bottom: 10px; }}
        .text {{ font-size: 14px; line-height: 2.4; margin-top: 10px; }}
        .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; border-bottom: 3px solid transparent; }}
        .stats {{ margin-top: 15px; font-size: 12px; color: #555; background: #f8f8f8; padding: 10px; border-radius: 5px; }}
        h1 {{ text-align: center; color: #333; }}
        h3 {{ color: #555; margin-top: 30px; }}
        .legend {{ text-align: center; margin: 20px 0; padding: 15px; background: white; border-radius: 10px; }}
        .layer-legend {{ text-align: center; margin: 10px 0; }}
        .view-toggle {{ text-align: center; margin: 20px 0; }}
        .view-toggle button {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }}
        .view-toggle .active {{ background: #3498db; color: white; }}
        .view-toggle .inactive {{ background: #ddd; color: #333; }}
    </style>
</head>
<body>
    <h1>🎨 Layer-Colored Attention (K={K_PCT}%)</h1>
    <div class="legend">
        <strong>Border color = dominant layer for that token</strong><br>
        <span style="font-size: 12px;">Background intensity = combined attention weight</span>
    </div>
    <div class="layer-legend">
        <strong>Layer Colors:</strong> {layer_legend}
    </div>
"""

for result in results:
    cat = result['category']
    tokens = result['tokens'][:60]
    combined = result['combined_attention'][:60]
    attn_matrix = result['attn_matrix'][:, :60]  # (L, T)
    
    # Normalize combined for background intensity
    c_min, c_max = combined.min(), combined.max()
    combined_norm = (combined - c_min) / (c_max - c_min + 1e-8)
    
    token_html = ""
    layer_contributions = []
    
    for t_idx, (token, c_norm) in enumerate(zip(tokens, combined_norm)):
        token_display = token.replace('▁', ' ').replace('Ġ', ' ')
        if not token_display.strip():
            continue
        
        # Find which layer has highest attention for this token
        per_token_attn = attn_matrix[:, t_idx]
        dominant_layer_idx = np.argmax(per_token_attn)
        dominant_layer = selected_layers[dominant_layer_idx]
        layer_contributions.append(dominant_layer)
        
        # Background based on combined attention
        if c_norm < 0.3:
            bg = f"rgba(255, 255, 220, {0.4 + c_norm})"
            text_color = "#333"
        elif c_norm < 0.6:
            bg = f"rgba(255, 230, 150, {0.6 + c_norm*0.3})"
            text_color = "#333"
        else:
            bg = f"rgba(255, 200, 100, {0.8})"
            text_color = "#333"
        
        # Border color = dominant layer
        border_color = layer_colors[dominant_layer]
        
        token_html += f'<span class="token" style="background: {bg}; border-bottom-color: {border_color}; color: {text_color};" title="L{dominant_layer}: {per_token_attn[dominant_layer_idx]:.3f}">{token_display}</span>'
    
    # Analyze layer distribution
    layer_counts = {}
    for l in layer_contributions:
        layer_counts[l] = layer_counts.get(l, 0) + 1
    sorted_layer_counts = sorted(layer_counts.items(), key=lambda x: -x[1])[:3]
    layer_dist_str = ", ".join([f"L{l}({c})" for l, c in sorted_layer_counts])
    
    # Check agreement
    unique_layers = len(set(layer_contributions))
    agreement = "HIGH" if unique_layers <= 3 else "MODERATE" if unique_layers <= 5 else "LOW"
    agreement_color = "#27ae60" if agreement == "HIGH" else "#f39c12" if agreement == "MODERATE" else "#e74c3c"
    
    cat_labels = {
        'TP': '✅ TRUE POSITIVE (Deceptive → Correctly Detected)',
        'TN': '✅ TRUE NEGATIVE (Honest → Correctly Detected)',
        'FP': '❌ FALSE POSITIVE (Honest → Wrongly Flagged)',
        'FN': '❌ FALSE NEGATIVE (Deceptive → Escaped Detection)'
    }
    
    html += f"""
    <div class="sample {cat}">
        <div class="label {cat}">{cat_labels[cat]}</div>
        <div class="meta">ID: {result['id']} | Gold: {result['gold']} | Pred: {result['pred']}</div>
        <div class="text">{token_html}</div>
        <div class="stats">
            <strong>Layer agreement:</strong> <span style="color: {agreement_color}; font-weight: bold;">{agreement}</span> ({unique_layers} layers active)<br>
            <strong>Dominant layers:</strong> {layer_dist_str}
        </div>
    </div>
"""

html += """
</body>
</html>
"""

html_path = f"{OUTPUT_DIR}/layer_colored_attention.html"
with open(html_path, 'w') as f:
    f.write(html)
print(f"✓ Saved: {html_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 LAYER AGREEMENT ANALYSIS")
print("="*80)

for result in results:
    attn_matrix = result['attn_matrix']
    tokens = result['tokens'][:60]
    
    # Dominant layer per token
    dominant = [selected_layers[np.argmax(attn_matrix[:, t])] for t in range(len(tokens))]
    unique = len(set(dominant))
    
    cat = result['category']
    print(f"\n{cat} | {result['id']}")
    print(f"   Layers active: {unique} / {len(selected_layers)}")
    
    # Most common dominant layers
    from collections import Counter
    counts = Counter(dominant).most_common(3)
    print(f"   Top layers: {counts}")

print(f"\n✅ Saved: {html_path}")
print("="*80)
