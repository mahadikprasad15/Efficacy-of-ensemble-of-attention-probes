"""
Analyze attention entropy for learned attention pooling models.

Loads a trained probe with learned token attention and computes:
- Mean entropy of attention weights per sample
- Distribution of entropies across dataset

Low entropy = attention is concentrated on few tokens (what we want if dynamic selection helps).

Usage:
    python scripts/analyze_attention_entropy.py \\
        --probe_path data/probes/attn_Movies/probe_layer_8.pt \\
        --data_dir data/activations \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset Movies
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from tqdm import tqdm
import logging

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_activations(data_dir, model, dataset, split):
    """Load cached activations."""
    pattern = os.path.join(data_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    shards = sorted(glob.glob(pattern))

    tensors = []
    for shard in tqdm(shards, desc="Loading"):
        try:
            shard_data = load_file(shard)
            manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")

            with open(manifest_path) as f:
                for line in f:
                    meta = json.loads(line)
                    eid = meta['id']
                    if eid in shard_data:
                        tensors.append(shard_data[eid].float())
        except Exception as e:
            logger.warning(f"Error loading {shard}: {e}")

    return torch.stack(tensors, dim=0) if tensors else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--layer", type=int, required=True, help="Which layer this probe was trained on")
    parser.add_argument("--output_dir", type=str, default="data/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    logger.info(f"Loading activations: {args.dataset}")
    X = load_activations(args.data_dir, args.model, args.dataset, args.split)

    if X is None or len(X) == 0:
        logger.error("No data loaded")
        return

    logger.info(f"Loaded {X.shape[0]} samples")

    # Load probe
    logger.info(f"Loading probe from {args.probe_path}")
    sample = X[0]  # (L, T, D)
    d_model = sample.shape[-1]

    probe = LayerProbe(input_dim=d_model, pooling_type="attn").to(device)
    probe.load_state_dict(torch.load(args.probe_path, map_location=device))
    probe.eval()

    # Extract attention entropies
    entropies = []

    with torch.no_grad():
        for i in tqdm(range(len(X)), desc="Computing Entropies"):
            x = X[i, args.layer, :, :].unsqueeze(0).to(device)  # (1, T, D)

            # Forward pass (activates attention)
            _ = probe(x)

            # Get entropy
            entropy = probe.pooling.compute_attention_entropy()
            if entropy is not None:
                entropies.append(entropy)

    entropies = np.array(entropies)

    # Summary stats
    logger.info(f"\n=== Attention Entropy Analysis ===")
    logger.info(f"Mean Entropy: {entropies.mean():.4f}")
    logger.info(f"Std Entropy: {entropies.std():.4f}")
    logger.info(f"Min Entropy: {entropies.min():.4f}")
    logger.info(f"Max Entropy: {entropies.max():.4f}")
    logger.info(f"Median Entropy: {np.median(entropies):.4f}")

    # Max possible entropy for T tokens: log(T)
    T = X.shape[2]
    max_entropy = np.log(T)
    logger.info(f"\nMax possible entropy (uniform over {T} tokens): {max_entropy:.4f}")
    logger.info(f"Mean normalized entropy: {entropies.mean() / max_entropy:.2%}")

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(entropies.mean(), color='red', linestyle='--', label=f'Mean: {entropies.mean():.2f}')
    plt.axvline(max_entropy, color='green', linestyle='--', label=f'Max (uniform): {max_entropy:.2f}')
    plt.xlabel('Attention Entropy')
    plt.ylabel('Frequency')
    plt.title(f'Attention Entropy Distribution: {args.dataset} (Layer {args.layer})')
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(args.output_dir, f"entropy_dist_{args.dataset}_layer{args.layer}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")

    # Save results
    results = {
        "dataset": args.dataset,
        "layer": args.layer,
        "num_samples": len(entropies),
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "median_entropy": float(np.median(entropies)),
        "max_entropy_theoretical": float(max_entropy),
        "normalized_mean_entropy": float(entropies.mean() / max_entropy)
    }

    results_path = os.path.join(args.output_dir, f"entropy_{args.dataset}_layer{args.layer}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
