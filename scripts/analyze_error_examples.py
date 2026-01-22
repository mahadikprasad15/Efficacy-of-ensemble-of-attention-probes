#!/usr/bin/env python3
"""
Error Analysis: TP/TN/FP/FN Analysis with Text and Attention
============================================================
Analyzes which OOD examples the probe gets right/wrong and why.

Usage:
    python scripts/analyze_error_examples.py \
        --ood_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/test \
        --apollo_data data/apollo_raw/roleplaying/dataset.yaml \
        --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
        --pooling last \
        --layer 18 \
        --output_dir results_flipped/error_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from collections import Counter
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))
from actprobe.probes.models import LayerProbe

# Try to load tokenizer for attention visualization
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Error Analysis: TP/TN/FP/FN with text and attention')
parser.add_argument('--ood_dir', type=str, required=True,
                    help='Path to OOD activations directory')
parser.add_argument('--apollo_data', type=str, default=None,
                    help='Path to Apollo raw data (for original text)')
parser.add_argument('--probes_dir', type=str, required=True,
                    help='Path to probes directory (contains pooling subdirs)')
parser.add_argument('--pooling', type=str, default='last',
                    choices=['mean', 'max', 'last', 'attn'],
                    help='Pooling strategy to analyze')
parser.add_argument('--layer', type=int, default=None,
                    help='Specific layer to use (default: best from layer_results.json)')
parser.add_argument('--output_dir', type=str, default='results/error_analysis',
                    help='Output directory')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                    help='Model name for tokenizer')
parser.add_argument('--finance_keywords', type=str, 
                    default='trade,stock,invest,profit,loss,position,recommend,buy,sell,market,portfolio,shares',
                    help='Comma-separated finance keywords to track')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"OOD Dir: {args.ood_dir}")
print(f"Probes Dir: {args.probes_dir}")
print(f"Pooling: {args.pooling}")
print(f"Output Dir: {args.output_dir}")

# Parse finance keywords
FINANCE_KEYWORDS = set(args.finance_keywords.lower().split(','))

# ============================================================================
# LOAD TOKENIZER
# ============================================================================
tokenizer = None
if HAS_TRANSFORMERS:
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.environ.get('HF_TOKEN'))
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠️ Could not load tokenizer: {e}")

# ============================================================================
# LOAD APOLLO DATA (for original text)
# ============================================================================
print("\nLoading text data...")
apollo_texts = {}

if args.apollo_data and os.path.exists(args.apollo_data):
    try:
        if args.apollo_data.endswith('.json'):
            with open(args.apollo_data, 'r') as f:
                data = json.load(f)
            for idx, item in enumerate(data):
                # Try different ID formats
                for id_prefix in ['roleplaying_', 'insider_', '']:
                    item_id = f"{id_prefix}{idx}" if id_prefix else str(idx)
                    metadata = item.get('metadata', {})
                    text = metadata.get('trade_message', '')
                    if not text:
                        for msg in reversed(item.get('transcript', [])):
                            if msg.get('role') == 'assistant':
                                text = msg.get('content', '')
                                break
                    if text:
                        apollo_texts[item_id] = text
        elif args.apollo_data.endswith('.yaml'):
            import yaml
            with open(args.apollo_data, 'r') as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    item_id = f"roleplaying_{idx}"
                    text = item.get('completion', item.get('response', item.get('text', '')))
                    if text:
                        apollo_texts[item_id] = text
        print(f"✓ Loaded {len(apollo_texts)} texts from Apollo data")
    except Exception as e:
        print(f"⚠️ Error loading Apollo data: {e}")

# ============================================================================
# LOAD OOD ACTIVATIONS
# ============================================================================
print("\nLoading OOD activations...")
manifest_path = os.path.join(args.ood_dir, 'manifest.jsonl')
with open(manifest_path, 'r') as f:
    manifest = [json.loads(line) for line in f]

shards = sorted(glob.glob(os.path.join(args.ood_dir, 'shard_*.safetensors')))
all_tensors = {}
for shard_path in shards:
    all_tensors.update(load_file(shard_path))

samples = []
for entry in manifest:
    eid = entry['id']
    if eid in all_tensors:
        # Try to get text - first from manifest's generated_text, then apollo data
        text = entry.get('generated_text', '')
        if not text:
            text = apollo_texts.get(eid, '')
        if not text:
            text = entry.get('text', entry.get('completion', f'[No text for {eid}]'))
        
        # Also get scenario for context
        scenario = entry.get('scenario', '')
        
        samples.append({
            'id': eid,
            'tensor': all_tensors[eid],
            'label': entry['label'],
            'text': text[:1000] if text else '',
            'scenario': scenario[:300] if scenario else ''
        })

print(f"✓ Loaded {len(samples)} samples")
print(f"  With text: {sum(1 for s in samples if len(s['text']) > 20)}")
label_dist = Counter([s['label'] for s in samples])
print(f"  Labels: {dict(label_dist)}")

# ============================================================================
# LOAD PROBE
# ============================================================================
probe_dir = os.path.join(args.probes_dir, args.pooling)
layer_results_path = os.path.join(probe_dir, 'layer_results.json')

# Determine best layer if not specified
if args.layer is None:
    with open(layer_results_path, 'r') as f:
        layer_results = json.load(f)
    best_layer = max(layer_results, key=lambda x: x['val_auc'])
    layer_idx = best_layer['layer']
    print(f"\n📊 Using best layer: {layer_idx} (Val AUC: {best_layer['val_auc']:.4f})")
else:
    layer_idx = args.layer
    print(f"\n📊 Using specified layer: {layer_idx}")

# Load probe
D = samples[0]['tensor'].shape[-1]
probe_path = os.path.join(probe_dir, f'probe_layer_{layer_idx}.pt')
probe = LayerProbe(input_dim=D, pooling_type=args.pooling).to(device)
probe.load_state_dict(torch.load(probe_path, map_location=device))
probe.eval()
print(f"✓ Loaded probe from {probe_path}")

# ============================================================================
# RUN INFERENCE AND CLASSIFY
# ============================================================================
print("\nRunning inference...")

results = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
all_predictions = []

for sample in tqdm(samples, desc="Predicting"):
    tensor = sample['tensor'].clone().detach().float().unsqueeze(0)
    x_layer = tensor[:, layer_idx, :, :].to(device)
    
    with torch.no_grad():
        logit = probe(x_layer).cpu().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        
        # Get attention weights if using attn pooling
        attention = None
        if args.pooling == 'attn':
            _, attn_weights = probe.pooling(x_layer, return_attention=True)
            attention = attn_weights.cpu().numpy().flatten()
    
    pred = 1 if prob > 0.5 else 0
    gold = sample['label']
    
    # Classify
    if gold == 1 and pred == 1:
        category = 'TP'
    elif gold == 0 and pred == 0:
        category = 'TN'
    elif gold == 0 and pred == 1:
        category = 'FP'
    else:
        category = 'FN'
    
    # Tokenize for attention analysis
    tokens = []
    if tokenizer and sample['text']:
        tokens = tokenizer.tokenize(sample['text'])[:64]
    
    # Check for finance keywords
    text_lower = sample['text'].lower()
    found_keywords = [kw for kw in FINANCE_KEYWORDS if kw in text_lower]
    
    result = {
        'id': sample['id'],
        'text': sample['text'],
        'scenario': sample.get('scenario', ''),
        'gold_label': 'DECEPTIVE' if gold == 1 else 'HONEST',
        'pred_label': 'DECEPTIVE' if pred == 1 else 'HONEST',
        'category': category,
        'confidence': prob,
        'logit': logit,
        'tokens': tokens,
        'attention': attention.tolist() if attention is not None else None,
        'finance_keywords': found_keywords,
        'has_finance_keywords': len(found_keywords) > 0
    }
    
    results[category].append(result)
    all_predictions.append(result)

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================
print("\n" + "="*70)
print("CLASSIFICATION RESULTS")
print("="*70)

total = len(samples)
tp, tn, fp, fn = len(results['TP']), len(results['TN']), len(results['FP']), len(results['FN'])

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              HONEST  DECEPTIVE")
print(f"Actual HONEST   {tn:5d}    {fp:5d}")
print(f"     DECEPTIVE  {fn:5d}    {tp:5d}")

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")

# Finance keyword analysis
print(f"\n📊 Finance Keyword Analysis:")
for cat in ['TP', 'TN', 'FP', 'FN']:
    cat_results = results[cat]
    with_kw = sum(1 for r in cat_results if r['has_finance_keywords'])
    pct = 100 * with_kw / len(cat_results) if cat_results else 0
    print(f"  {cat}: {with_kw}/{len(cat_results)} ({pct:.1f}%) contain finance keywords")

# Confidence distribution
print(f"\n📊 Confidence Distribution:")
for cat in ['TP', 'TN', 'FP', 'FN']:
    if results[cat]:
        confs = [r['confidence'] for r in results[cat]]
        print(f"  {cat}: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}, range=[{np.min(confs):.3f}, {np.max(confs):.3f}]")

# ============================================================================
# SAVE JSON RESULTS
# ============================================================================
json_path = os.path.join(args.output_dir, 'error_analysis.json')
summary = {
    'config': {
        'ood_dir': args.ood_dir,
        'probes_dir': args.probes_dir,
        'pooling': args.pooling,
        'layer': layer_idx
    },
    'metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    },
    'examples': {
        'TP': results['TP'][:20],  # Limit to 20 per category
        'TN': results['TN'][:20],
        'FP': results['FP'][:20],
        'FN': results['FN'][:20]
    }
}

with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n✓ Saved JSON: {json_path}")

# ============================================================================
# GENERATE HTML REPORT
# ============================================================================
print("\nGenerating HTML report...")

cat_colors = {
    'TP': ('#27ae60', '#d4edda', '✅ TRUE POSITIVE (Deceptive Correctly Detected)'),
    'TN': ('#3498db', '#cce5ff', '✅ TRUE NEGATIVE (Honest Correctly Detected)'),
    'FP': ('#e74c3c', '#f8d7da', '❌ FALSE POSITIVE (Honest Wrongly Flagged)'),
    'FN': ('#f39c12', '#fff3cd', '❌ FALSE NEGATIVE (Deceptive Escaped Detection)')
}

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error Analysis: {args.pooling.upper()} Layer {layer_idx}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .summary table {{ width: 100%; border-collapse: collapse; }}
        .summary td, .summary th {{ padding: 10px; text-align: center; border: 1px solid #ddd; }}
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
        .tab {{ padding: 15px 25px; border-radius: 10px; cursor: pointer; font-weight: bold; border: none; font-size: 14px; }}
        .tab.active {{ box-shadow: 0 2px 10px rgba(0,0,0,0.2); }}
        .content {{ display: none; }}
        .content.active {{ display: block; }}
        .sample {{ background: white; padding: 20px; margin: 15px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid; }}
        .sample-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .label {{ font-weight: bold; font-size: 14px; }}
        .confidence {{ font-size: 13px; padding: 5px 10px; border-radius: 5px; }}
        .text {{ font-size: 14px; line-height: 1.8; background: #f8f8f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .keywords {{ font-size: 12px; color: #666; margin-top: 10px; }}
        .keyword {{ background: #fff3cd; padding: 2px 6px; border-radius: 3px; margin: 2px; display: inline-block; }}
        .meta {{ font-size: 12px; color: #888; }}
        .filter-bar {{ background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; }}
        .filter-bar input {{ padding: 8px; width: 300px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🔍 Error Analysis: {args.pooling.upper()} Pooling, Layer {layer_idx}</h1>
    
    <div class="summary">
        <h3>📊 Summary</h3>
        <table>
            <tr>
                <th></th><th>Predicted HONEST</th><th>Predicted DECEPTIVE</th>
            </tr>
            <tr>
                <th>Actual HONEST</th>
                <td style="background: {cat_colors['TN'][1]}">TN: {tn}</td>
                <td style="background: {cat_colors['FP'][1]}">FP: {fp}</td>
            </tr>
            <tr>
                <th>Actual DECEPTIVE</th>
                <td style="background: {cat_colors['FN'][1]}">FN: {fn}</td>
                <td style="background: {cat_colors['TP'][1]}">TP: {tp}</td>
            </tr>
        </table>
        <p style="text-align:center; margin-top:15px;">
            <strong>Accuracy:</strong> {accuracy:.2%} | 
            <strong>Precision:</strong> {precision:.2%} | 
            <strong>Recall:</strong> {recall:.2%} | 
            <strong>F1:</strong> {f1:.2%}
        </p>
    </div>
    
    <div class="tabs">
"""

# Add tabs
for cat in ['TP', 'TN', 'FP', 'FN']:
    color, bg, label = cat_colors[cat]
    count = len(results[cat])
    active = 'active' if cat == 'FN' else ''  # Start with FN (most interesting)
    html += f'<button class="tab {active}" onclick="showTab(\'{cat}\')" id="tab-{cat}" style="background: {bg}; color: {color};">{cat}: {count}</button>\n'

html += '</div>\n<div class="filter-bar"><input type="text" id="search" placeholder="Search text..." onkeyup="filterSamples()"></div>\n'

# Add content for each category
for cat in ['TP', 'TN', 'FP', 'FN']:
    color, bg, label = cat_colors[cat]
    active = 'active' if cat == 'FN' else ''
    
    html += f'<div class="content {active}" id="content-{cat}">\n'
    html += f'<h2 style="color: {color}">{label} ({len(results[cat])} samples)</h2>\n'
    
    for i, r in enumerate(results[cat][:50]):  # Limit to 50 per category
        conf_color = '#27ae60' if r['confidence'] > 0.7 or r['confidence'] < 0.3 else '#f39c12'
        keywords_html = ''.join([f'<span class="keyword">{kw}</span>' for kw in r['finance_keywords']])
        # Escape HTML in text
        import html as html_module
        safe_text = html_module.escape(r['text'][:500]) if r['text'] else '[No text available]'
        safe_scenario = html_module.escape(r.get('scenario', '')[:300]) if r.get('scenario') else ''
        
        html += f'''
        <div class="sample" style="border-left-color: {color};" data-text="{r['text'].lower()[:200].replace('"', '')}">
            <div class="sample-header">
                <span class="label" style="color: {color};">#{i+1} | {r['id']}</span>
                <span class="confidence" style="background: {conf_color}20; color: {conf_color};">
                    Confidence: {r['confidence']:.3f}
                </span>
            </div>
            <div class="meta">Gold: {r['gold_label']} | Pred: {r['pred_label']} | Logit: {r['logit']:.3f}</div>
            {f'<div class="scenario" style="background:#e8f4f8; padding:10px; border-radius:5px; margin:5px 0; font-size:12px;"><strong>📋 Scenario:</strong> {safe_scenario}</div>' if safe_scenario else ''}
            <div class="text" style="font-size:14px; background:#f8f8f8; padding:15px; border-radius:5px; margin:10px 0;"><strong>💬 Response:</strong><br>{safe_text}{'...' if len(r.get('text', '')) > 500 else ''}</div>
            <div class="keywords">
                {'<strong>Finance keywords found:</strong> ' + keywords_html if r['finance_keywords'] else '<em>No finance keywords detected</em>'}
            </div>
        </div>
        '''
    
    if len(results[cat]) > 50:
        html += f'<p style="text-align:center; color:#888;">Showing 50 of {len(results[cat])} samples. See JSON for full data.</p>'
    
    html += '</div>\n'

html += """
<script>
function showTab(cat) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + cat).classList.add('active');
    document.getElementById('content-' + cat).classList.add('active');
}

function filterSamples() {
    const query = document.getElementById('search').value.toLowerCase();
    document.querySelectorAll('.sample').forEach(s => {
        const text = s.getAttribute('data-text') || '';
        s.style.display = text.includes(query) ? 'block' : 'none';
    });
}
</script>
</body>
</html>
"""

html_path = os.path.join(args.output_dir, 'error_analysis.html')
with open(html_path, 'w') as f:
    f.write(html)
print(f"✓ Saved HTML: {html_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ ERROR ANALYSIS COMPLETE")
print("="*70)
print(f"HTML Report: {html_path}")
print(f"JSON Data:   {json_path}")
print("="*70)
