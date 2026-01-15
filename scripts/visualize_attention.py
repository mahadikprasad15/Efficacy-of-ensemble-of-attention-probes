"""
Visualize Attention Weights for 'Learned Attention Pooling'.
Requires a trained probe with 'attn' pooling.

Usage:
    python scripts/visualize_attention.py --model gpt2 --probe_path experiments/lodo_Movies/best_probe_layer_X.pt --text "Example text here"
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.llm.activations import ActivationRunner
from actprobe.probes.models import LayerProbe
from actprobe.probes.pooling import LearnedAttentionPooling

def visualize_attention(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print(f"Loading Model {args.model}...")
    runner = ActivationRunner(args.model, dtype=torch.float32) # float32 for vis stability
    
    # 2. Get Activations
    # No resampling here? The probe was trained on resampled (target_T) data usually?
    # If the probe expects T_prime=64, we MUST resample or the dimensions won't match if pooling relies on T?
    # Wait, LearnedAttentionPooling learns 'query' of dim (D, 1). It applies to any T.
    # Scores = (B, T, D) @ (D, 1) -> (B, T, 1).
    # So it works on any sequence length T! 
    # That is the beauty of attention pooling.
    
    # So we can run on full resolution activations!
    prompts = [args.text]
    activations_list = runner.get_activations(prompts) # List of (L, T, D)
    
    # 3. Load Probe
    # Need to know dimensions and layer index.
    # Inspect probe_path filename for layer index?
    # Or arg.
    
    # Attempt to parse layer from filename
    # best_probe_layer_12.pt
    try:
        if args.layer is None:
            base = os.path.basename(args.probe_path)
            # best_probe_layer_(\d+).pt
            import re
            m = re.search(r"layer_(\d+)", base)
            layer_idx = int(m.group(1))
            print(f"Inferred Layer Index: {layer_idx}")
        else:
            layer_idx = args.layer
    except:
        print("Could not infer layer index. Please provide --layer.")
        return

    # Check D
    act = activations_list[0] # (L, T, D)
    D = act.shape[-1]
    
    print(f"Loading Probe from {args.probe_path} (D={D})...")
    probe = LayerProbe(input_dim=D, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(args.probe_path, map_location=device))
    probe.eval()
    
    # 4. Forward Pass & Extract Weights
    target_act = act[layer_idx].unsqueeze(0).to(device) # (1, T, D)
    
    # Hook into pooling to get weights? 
    # Or just manually compute using probe.pooling.query
    # AttnPooling: scores = x @ query
    
    q = probe.pooling.query # (D, 1)
    scores = torch.matmul(target_act, q) # (1, T, 1)
    weights = F.softmax(scores, dim=1).squeeze().detach().cpu().numpy() # (T,)
    
    # 5. Visualize
    tokens = runner.tokenizer.tokenize(args.text)
    # Check length mismatch due to special tokens?
    # Activations usually include bos?
    # runner.get_activations handles full sequence (prompt+gen).
    
    # If using GPT2, tokens might map well. 
    # Let's align.
    
    # Debug lengths
    print(f"Tokens: {len(tokens)}, Weights: {len(weights)}")
    
    # Simple truncate or pad visualization
    L_vis = min(len(tokens), len(weights))
    
    plt.figure(figsize=(15, 2))
    sns.heatmap(weights[:L_vis].reshape(1, -1), xticklabels=tokens[:L_vis], yticklabels=False, cmap="Reds", annot=True, fmt=".2f")
    plt.title(f"Attention Weights (Layer {layer_idx})")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved visualization to {args.output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output", type=str, default="attention_heatmap.pdf")
    args = parser.parse_args()

    visualize_attention(args)

if __name__ == "__main__":
    main()
