# ============================================================================
# ENSEMBLE ATTENTION: Aggregate attention from ALL ensemble layers
# ============================================================================
# Instead of just layer 16, this aggregates attention weights from ALL
# layers in the K=40% ensemble, weighted by the gated ensemble's layer weights.
# ============================================================================

import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import GatedEnsemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# PATHS
# ============================================================================
OOD_DIR = "data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test"
APOLLO_DATA = "data/apollo_raw/insider_trading/llama-70b-3.3-generations.json"
PROBES_BASE = "data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying"
GATED_MODELS_DIR = "results/ensembles/attn/gated_models_val"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "results/mechanistic_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_PCT = 40  # Use K=40% ensemble

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

# ============================================================================
# LOAD APOLLO DATA
# ============================================================================
print("\nLoading Apollo data for original text...")
apollo_texts = {}
try:
    with open(APOLLO_DATA, 'r') as f:
        apollo_data = json.load(f)
    
    for idx, item in enumerate(apollo_data):
        item_id = f"insider_{idx}"
        metadata = item.get('metadata', {})
        trade_message = metadata.get('trade_message', '')
        
        if not trade_message:
            transcript = item.get('transcript', [])
            for msg in reversed(transcript):
                if msg.get('role') == 'assistant':
                    trade_message = msg.get('content', '')
                    break
        
        if trade_message:
            apollo_texts[item_id] = {'text': trade_message}
    
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
        text = apollo_texts.get(eid, {}).get('text', '')
        samples_data.append({
            'id': eid,
            'tensor': all_tensors[eid],
            'label': entry['label'],
            'text': text[:500] if text else f"[Sample {eid}]"
        })

print(f"✓ Loaded {len(samples_data)} samples")

# ============================================================================
# LOAD TOP-K LAYERS BASED ON VALIDATION AUC
# ============================================================================
probe_dir = f"{PROBES_BASE}/attn"
with open(f"{probe_dir}/layer_results.json", 'r') as f:
    layer_results = json.load(f)

# Select top K% layers by validation AUC
num_layers = 28
k_layers = max(1, int(num_layers * K_PCT / 100))
sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
selected_layers = sorted([l['layer'] for l in sorted_layers[:k_layers]])

print(f"\n📊 Ensemble Configuration:")
print(f"   K = {K_PCT}% → {k_layers} layers")
print(f"   Selected layers: {selected_layers}")

# ============================================================================
# LOAD PROBES FOR ALL SELECTED LAYERS
# ============================================================================
D = samples_data[0]['tensor'].shape[-1]
probes = {}
for layer_idx in selected_layers:
    probe_path = f"{probe_dir}/probe_layer_{layer_idx}.pt"
    probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval()
    probes[layer_idx] = probe

print(f"✓ Loaded {len(probes)} probes")

# ============================================================================
# (OPTIONAL) LOAD GATED ENSEMBLE FOR LAYER WEIGHTS
# ============================================================================
gated_path = f"{GATED_MODELS_DIR}/gated_k{K_PCT}.pt"
if os.path.exists(gated_path):
    gated = GatedEnsemble(input_dim=k_layers, num_layers=k_layers).to(device)
    gated.load_state_dict(torch.load(gated_path, map_location=device))
    gated.eval()
    USE_GATED_WEIGHTS = True
    print(f"✓ Loaded gated ensemble weights")
else:
    USE_GATED_WEIGHTS = False
    print(f"⚠️ No gated model found at {gated_path}, using uniform weights")

# ============================================================================
# SELECT DIVERSE SAMPLES (TP, TN, FP, FN)
# ============================================================================
print("\nClassifying samples...")

# First, run prediction using ensemble
classified = {'TP': [], 'TN': [], 'FP': [], 'FN': []}

for sample in tqdm(samples_data, desc="Predicting"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    
    # Get logits from all layers
    layer_logits = []
    for layer_idx in selected_layers:
        x_layer = tensor[:, layer_idx, :, :].to(device)
        with torch.no_grad():
            logit = probes[layer_idx](x_layer).cpu().item()
        layer_logits.append(logit)
    
    # Ensemble prediction (mean for simplicity)
    ensemble_logit = np.mean(layer_logits)
    pred = 1 if ensemble_logit > 0 else 0
    
    gold = sample['label']
    if gold == 1 and pred == 1:
        category = 'TP'
    elif gold == 0 and pred == 0:
        category = 'TN'
    elif gold == 0 and pred == 1:
        category = 'FP'
    else:
        category = 'FN'
    
    sample['prediction'] = pred
    sample['category'] = category
    classified[category].append(sample)

print(f"\n📊 Prediction Results (Ensemble K={K_PCT}%):")
for cat in ['TP', 'TN', 'FP', 'FN']:
    print(f"   {cat}: {len(classified[cat])}")

# Select 3 from each category
n_per_cat = 3
selected_samples = []
for cat in ['TP', 'TN', 'FP', 'FN']:
    selected_samples.extend(classified[cat][:n_per_cat])

print(f"\n✓ Selected {len(selected_samples)} samples for visualization")

# ============================================================================
# EXTRACT AND AGGREGATE ATTENTION FROM ALL LAYERS
# ============================================================================
print("\nExtracting ensemble attention...")

results = []
for sample in tqdm(selected_samples, desc="Aggregating attention"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    
    # Collect attention from all layers
    layer_attentions = []
    layer_logits = []
    
    for layer_idx in selected_layers:
        x_layer = tensor[:, layer_idx, :, :].to(device)
        
        with torch.no_grad():
            pooled, attn_weights = probes[layer_idx].pooling(x_layer, return_attention=True)
            logit = probes[layer_idx](x_layer).cpu().item()
        
        layer_attentions.append(attn_weights.cpu().numpy().flatten())
        layer_logits.append(logit)
    
    layer_attentions = np.array(layer_attentions)  # (num_layers, T)
    layer_logits = np.array(layer_logits)
    
    # Get gating weights if available
    if USE_GATED_WEIGHTS:
        with torch.no_grad():
            logit_tensor = torch.tensor(layer_logits, dtype=torch.float32).unsqueeze(0).to(device)
            gating_weights = gated.gate_net(logit_tensor).cpu().numpy().flatten()
    else:
        gating_weights = np.ones(len(selected_layers)) / len(selected_layers)
    
    # Aggregate attention: weighted sum across layers
    # combined_attention[t] = sum(gating_weight[l] * attention[l, t])
    combined_attention = np.zeros(layer_attentions.shape[1])
    for l_idx, (attn, g_weight) in enumerate(zip(layer_attentions, gating_weights)):
        combined_attention += g_weight * attn
    
    # Normalize
    combined_attention = combined_attention / (combined_attention.sum() + 1e-8)
    
    # Tokenize
    text = sample['text']
    if HAS_TOKENIZER and text and len(text) > 10:
        tokens = tokenizer.tokenize(text)[:len(combined_attention)]
    else:
        tokens = [f"[{i}]" for i in range(len(combined_attention))]
    
    while len(tokens) < len(combined_attention):
        tokens.append("[PAD]")
    
    cat = sample['category']
    label_map = {
        'TP': '✅ TRUE POSITIVE (Deceptive → Correctly Detected)',
        'TN': '✅ TRUE NEGATIVE (Honest → Correctly Detected)',
        'FP': '❌ FALSE POSITIVE (Honest → Wrongly Flagged)',
        'FN': '❌ FALSE NEGATIVE (Deceptive → Escaped Detection)'
    }
    
    results.append({
        'id': sample['id'],
        'category': cat,
        'label': label_map[cat],
        'gold': 'DECEPTIVE' if sample['label'] == 1 else 'HONEST',
        'pred': 'DECEPTIVE' if sample['prediction'] == 1 else 'HONEST',
        'text': text,
        'tokens': tokens[:len(combined_attention)],
        'weights': combined_attention,
        'layer_attentions': layer_attentions,
        'gating_weights': gating_weights,
        'top_5_idx': np.argsort(combined_attention)[-5:][::-1]
    })

# ============================================================================
# CREATE HTML VISUALIZATION
# ============================================================================
print("\nGenerating ensemble attention visualization...")

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
        .meta {{ font-size: 13px; color: #666; margin-bottom: 10px; }}
        .text {{ font-size: 16px; line-height: 2.2; }}
        .token {{ padding: 3px 6px; margin: 2px; border-radius: 4px; display: inline-block; }}
        .stats {{ margin-top: 15px; font-size: 14px; color: #666; background: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .layers {{ font-size: 12px; color: #888; margin-top: 5px; }}
        h1 {{ text-align: center; color: #333; }}
        .legend {{ text-align: center; margin-bottom: 30px; padding: 15px; background: white; border-radius: 10px; }}
        .legend span {{ padding: 8px 15px; margin: 5px; border-radius: 5px; display: inline-block; font-weight: bold; }}
        .config {{ text-align: center; color: #666; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>🔍 Ensemble Attention: Aggregated from {k_layers} Layers (K={K_PCT}%)</h1>
    <div class="config">
        <strong>Layers used:</strong> {selected_layers}<br>
        <strong>Weighting:</strong> {'Gated (learned)' if USE_GATED_WEIGHTS else 'Uniform average'}
    </div>
    <div class="legend">
        <span style="background: #d4edda; color: #27ae60;">✅ TP</span>
        <span style="background: #cce5ff; color: #3498db;">✅ TN</span>
        <span style="background: #f8d7da; color: #e74c3c;">❌ FP</span>
        <span style="background: #fff3cd; color: #f39c12;">❌ FN</span>
    </div>
"""

for result in results:
    cat = result['category']
    tokens = result['tokens'][:80]
    weights = result['weights'][:80]
    
    w_min, w_max = weights.min(), weights.max()
    weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)
    
    token_html = ""
    for token, w_norm in zip(tokens, weights_norm):
        token_display = token.replace('▁', ' ').replace('Ġ', ' ')
        if not token_display.strip():
            continue
        
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
    
    valid_top = [i for i in result['top_5_idx'][:5] if i < len(tokens)]
    top_tokens = ", ".join([f"'{tokens[i].replace('▁', ' ').strip()}'" for i in valid_top if tokens[i].strip()])
    
    concentration = sum(sorted(weights, reverse=True)[:5]) / (sum(weights) + 1e-8)
    
    # Show top gating weights
    top_layers_idx = np.argsort(result['gating_weights'])[-3:][::-1]
    top_layers_info = ", ".join([f"L{selected_layers[i]}({result['gating_weights'][i]:.2f})" for i in top_layers_idx])
    
    html += f"""
    <div class="sample {cat}">
        <div class="label {cat}">{result['label']}</div>
        <div class="meta">ID: {result['id']} | Gold: {result['gold']} | Pred: {result['pred']}</div>
        <div class="text">{token_html}</div>
        <div class="stats">
            <strong>Top 5 tokens get {concentration*100:.1f}% attention:</strong> {top_tokens}
        </div>
        <div class="layers">
            <strong>Top layer weights:</strong> {top_layers_info}
        </div>
    </div>
"""

html += """
</body>
</html>
"""

html_path = f"{OUTPUT_DIR}/ensemble_attention.html"
with open(html_path, 'w') as f:
    f.write(html)
print(f"✓ Saved: {html_path}")

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print(f"📊 ENSEMBLE ATTENTION SUMMARY (K={K_PCT}%, {k_layers} layers)")
print("="*80)

for result in results:
    print(f"\n{result['label']}")
    print(f"   ID: {result['id']}")
    top_tokens_str = ", ".join([f"'{result['tokens'][i]}'" for i in result['top_5_idx'][:3] if i < len(result['tokens'])])
    print(f"   Top attended: {top_tokens_str}")
    top_layers_idx = np.argsort(result['gating_weights'])[-3:][::-1]
    print(f"   Top layers: {[selected_layers[i] for i in top_layers_idx]}")

print(f"\n✅ Saved: {html_path}")
print("="*80)
