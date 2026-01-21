# ============================================================================
# HYBRID ATTENTION: Combined Heatmap + Layer Contribution Breakdown
# ============================================================================
# Shows:
# 1. Token background = Combined attention (yellow → red heatmap)
# 2. Small stacked bar under each high-attention token = layer contributions
# ============================================================================

import os
import sys
import json
import glob
import torch
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm
import colorsys

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# PATHS & CONFIG
# ============================================================================
OOD_DIR = "data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test"
APOLLO_DATA = "data/apollo_raw/insider_trading/llama-70b-3.3-generations.json"
PROBES_BASE = "data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "results/mechanistic_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_PCT = 40

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get('HF_TOKEN'))
    HAS_TOKENIZER = True
except:
    HAS_TOKENIZER = False
print("✓ Tokenizer loaded" if HAS_TOKENIZER else "⚠️ No tokenizer")

print("\nLoading Apollo data...")
apollo_texts = {}
try:
    with open(APOLLO_DATA, 'r') as f:
        for idx, item in enumerate(json.load(f)):
            item_id = f"insider_{idx}"
            text = item.get('metadata', {}).get('trade_message', '')
            if not text:
                for msg in reversed(item.get('transcript', [])):
                    if msg.get('role') == 'assistant':
                        text = msg.get('content', '')
                        break
            if text:
                apollo_texts[item_id] = text
    print(f"✓ Loaded {len(apollo_texts)} texts")
except Exception as e:
    print(f"⚠️ {e}")

print("\nLoading activations...")
with open(f"{OOD_DIR}/manifest.jsonl", 'r') as f:
    manifest = [json.loads(line) for line in f]
all_tensors = {}
for shard in sorted(glob.glob(f"{OOD_DIR}/shard_*.safetensors")):
    all_tensors.update(load_file(shard))
samples = [{'id': e['id'], 'tensor': all_tensors[e['id']], 'label': e['label'], 
            'text': apollo_texts.get(e['id'], '')[:500]} 
           for e in manifest if e['id'] in all_tensors]
print(f"✓ {len(samples)} samples")

# ============================================================================
# LOAD LAYERS & PROBES
# ============================================================================
with open(f"{PROBES_BASE}/attn/layer_results.json") as f:
    layer_results = json.load(f)
k_layers = max(1, int(28 * K_PCT / 100))
selected_layers = sorted([l['layer'] for l in sorted(layer_results, key=lambda x: -x['val_auc'])[:k_layers]])
print(f"\n📊 K={K_PCT}% → {k_layers} layers: {selected_layers}")

# Generate distinct colors for layers
def get_layer_color(layer_idx, selected_layers):
    idx = selected_layers.index(layer_idx)
    h = idx / len(selected_layers)
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

layer_colors = {l: get_layer_color(l, selected_layers) for l in selected_layers}

D = samples[0]['tensor'].shape[-1]
probes = {}
for l in selected_layers:
    p = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    p.load_state_dict(torch.load(f"{PROBES_BASE}/attn/probe_layer_{l}.pt", map_location=device))
    p.eval()
    probes[l] = p
print(f"✓ Loaded {len(probes)} probes")

# ============================================================================
# CLASSIFY SAMPLES
# ============================================================================
print("\nClassifying...")
classified = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
for s in tqdm(samples, desc="Predicting"):
    t = s['tensor'].clone().detach().float().unsqueeze(0)
    logits = [probes[l](t[:, l, :, :].to(device)).cpu().item() for l in selected_layers]
    pred = 1 if np.mean(logits) > 0 else 0
    gold = s['label']
    cat = 'TP' if gold==1 and pred==1 else 'TN' if gold==0 and pred==0 else 'FP' if gold==0 else 'FN'
    s['pred'], s['cat'] = pred, cat
    classified[cat].append(s)

print("\n📊 Results:", {k: len(v) for k, v in classified.items()})
selected = sum([v[:2] for v in classified.values()], [])
print(f"✓ Selected {len(selected)} samples")

# ============================================================================
# EXTRACT ATTENTION
# ============================================================================
print("\nExtracting attention...")
results = []
for s in tqdm(selected, desc="Extracting"):
    t = s['tensor'].clone().detach().float().unsqueeze(0)
    
    # Get attention from each layer
    layer_attn = {}
    for l in selected_layers:
        with torch.no_grad():
            _, attn = probes[l].pooling(t[:, l, :, :].to(device), return_attention=True)
        layer_attn[l] = attn.cpu().numpy().flatten()
    
    # Stack and compute combined
    attn_matrix = np.array([layer_attn[l] for l in selected_layers])  # (L, T)
    combined = attn_matrix.mean(axis=0)
    combined = combined / (combined.sum() + 1e-8)
    
    # Tokenize
    text = s['text']
    if HAS_TOKENIZER and text:
        tokens = tokenizer.tokenize(text)[:len(combined)]
    else:
        tokens = [f"[{i}]" for i in range(len(combined))]
    while len(tokens) < len(combined):
        tokens.append("[PAD]")
    
    results.append({
        'id': s['id'], 'cat': s['cat'], 
        'gold': 'DECEPTIVE' if s['label']==1 else 'HONEST',
        'pred': 'DECEPTIVE' if s['pred']==1 else 'HONEST',
        'tokens': tokens[:len(combined)],
        'combined': combined,
        'layer_attn': {l: layer_attn[l][:len(combined)] for l in selected_layers},
        'attn_matrix': attn_matrix[:, :len(combined)]
    })

# ============================================================================
# HTML VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

layer_legend = " ".join([f'<span style="background:{layer_colors[l]};color:white;padding:2px 6px;border-radius:3px;font-size:10px;margin:1px;">L{l}</span>' for l in selected_layers])

html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; max-width: 1200px; margin: 0 auto; }}
.sample {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.TP {{ border-left: 5px solid #27ae60; }} .TN {{ border-left: 5px solid #3498db; }}
.FP {{ border-left: 5px solid #e74c3c; }} .FN {{ border-left: 5px solid #f39c12; }}
.label {{ font-weight: bold; font-size: 15px; }}
.label.TP {{ color: #27ae60; }} .label.TN {{ color: #3498db; }}
.label.FP {{ color: #e74c3c; }} .label.FN {{ color: #f39c12; }}
.meta {{ font-size: 12px; color: #666; margin: 5px 0 15px 0; }}
.text {{ line-height: 2.8; }}
.token-wrap {{ display: inline-block; margin: 2px; vertical-align: top; text-align: center; }}
.token {{ padding: 3px 5px; border-radius: 3px; font-size: 13px; display: block; }}
.layer-bar {{ height: 4px; display: flex; margin-top: 2px; border-radius: 2px; overflow: hidden; }}
.layer-seg {{ height: 100%; }}
h1 {{ text-align: center; color: #333; }}
.legend {{ text-align: center; margin: 20px 0; padding: 15px; background: white; border-radius: 10px; }}
.info {{ font-size: 12px; color: #666; margin-top: 10px; }}
</style>
</head>
<body>
<h1>🔥 Hybrid Attention: Heatmap + Layer Contribution</h1>
<div class="legend">
<strong>Background</strong> = Combined attention (yellow→red)<br>
<strong>Bar below</strong> = Which layers contribute to that token<br><br>
<strong>Layers:</strong> {layer_legend}
</div>
"""

cat_labels = {
    'TP': '✅ TRUE POSITIVE (Deceptive Detected)',
    'TN': '✅ TRUE NEGATIVE (Honest Detected)',
    'FP': '❌ FALSE POSITIVE (Honest → Wrongly Flagged)',
    'FN': '❌ FALSE NEGATIVE (Deceptive → Escaped)'
}

for r in results:
    tokens = r['tokens'][:50]
    combined = r['combined'][:50]
    attn_matrix = r['attn_matrix'][:, :50]
    
    c_norm = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    
    token_html = ""
    for t_idx, (tok, cn) in enumerate(zip(tokens, c_norm)):
        tok_display = tok.replace('▁', ' ').replace('Ġ', ' ').strip()
        if not tok_display:
            continue
        
        # Background color based on combined attention
        if cn < 0.3:
            bg = f"rgba(255,255,200,{0.5+cn})"
            tc = "#333"
        elif cn < 0.6:
            bg = f"rgba(255,200,100,{0.6+cn*0.3})"
            tc = "#333"
        else:
            bg = f"rgba(240,80,40,{0.7+cn*0.3})"
            tc = "white"
        
        # Layer contribution bar (only for high-attention tokens)
        bar_html = ""
        if cn > 0.2:  # Only show bar for visible tokens
            layer_contrib = attn_matrix[:, t_idx]
            layer_contrib = layer_contrib / (layer_contrib.sum() + 1e-8)  # Normalize
            
            bar_html = '<div class="layer-bar">'
            for l_idx, l in enumerate(selected_layers):
                width = layer_contrib[l_idx] * 100
                if width > 2:  # Only show visible segments
                    bar_html += f'<div class="layer-seg" style="width:{width:.1f}%;background:{layer_colors[l]}" title="L{l}: {layer_contrib[l_idx]*100:.1f}%"></div>'
            bar_html += '</div>'
        
        token_html += f'''<div class="token-wrap">
            <span class="token" style="background:{bg};color:{tc}">{tok_display}</span>
            {bar_html}
        </div>'''
    
    html += f"""
<div class="sample {r['cat']}">
    <div class="label {r['cat']}">{cat_labels[r['cat']]}</div>
    <div class="meta">ID: {r['id']} | Gold: {r['gold']} | Pred: {r['pred']}</div>
    <div class="text">{token_html}</div>
</div>
"""

html += """
</body>
</html>
"""

html_path = f"{OUTPUT_DIR}/hybrid_attention.html"
with open(html_path, 'w') as f:
    f.write(html)

print(f"\n✅ Saved: {html_path}")
print("="*60)
print("Open in browser to see:")
print("• Token background = combined attention intensity")
print("• Colored bar below = which layers contribute to each token")
print("="*60)
