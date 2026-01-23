"""
Train token-level heatmap probes (Baseline 0 from spec).

For each (layer, token) position, train a logistic regression probe
and compute validation AUC. This produces a (L' x T') heatmap visualizing
where predictive signal is concentrated.

Matches ACT-ViT Figure 2 methodology.

Usage:
    python scripts/train_heatmap_probes.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset HotpotQA \\
        --split validation \\
        --output_dir data/heatmaps
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import logging

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_activations(data_dir, model, dataset, split):
    """Load all cached activation shards for a dataset."""
    pattern = os.path.join(data_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    shards = sorted(glob.glob(pattern))

    if not shards:
        logger.error(f"No shards found: {pattern}")
        return None, None

    all_tensors = []
    all_labels = []

    logger.info(f"Loading {len(shards)} shards...")

    for shard in tqdm(shards, desc="Loading"):
        try:
            tensors = load_file(shard)
            manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")

            with open(manifest_path) as f:
                for line in f:
                    meta = json.loads(line)
                    eid = meta['id']
                    if eid in tensors:
                        all_tensors.append(tensors[eid].float())  # (L, T, D)
                        all_labels.append(meta.get('label', 0))
        except Exception as e:
            logger.warning(f"Error loading {shard}: {e}")

    if not all_tensors:
        return None, None

    # Stack into (N, L, T, D)
    X = torch.stack(all_tensors, dim=0)
    y = np.array(all_labels)

    logger.info(f"Loaded {X.shape[0]} samples with shape (N, L={X.shape[1]}, T={X.shape[2]}, D={X.shape[3]})")

    return X, y


def train_heatmap_probes(X_train, y_train, X_val, y_val, C=1.0):
    """
    Train logistic regression at each (layer, token) position.

    Returns:
        auc_heatmap: (L, T) array of validation AUCs
    """
    N, L, T, D = X_train.shape
    auc_heatmap = np.zeros((L, T))

    logger.info(f"Training probes for {L} layers x {T} tokens = {L*T} positions...")

    # Progress bar for overall progress
    pbar = tqdm(total=L * T, desc="Training Heatmap Probes", unit="probe")

    for l in range(L):
        for t in range(T):
            # Update progress with current position
            pbar.set_postfix({"layer": f"{l}/{L}", "token": f"{t}/{T}"})

            # Extract features at (layer=l, token=t)
            X_train_lt = X_train[:, l, t, :].numpy()  # (N, D)
            X_val_lt = X_val[:, l, t, :].numpy()

            # Skip if no variance
            if X_train_lt.std() < 1e-6:
                auc_heatmap[l, t] = 0.5
                pbar.update(1)
                continue

            try:
                # Train logistic regression
                clf = LogisticRegression(C=C, max_iter=500, solver='lbfgs', random_state=42)
                clf.fit(X_train_lt, y_train)

                # Predict on validation
                y_pred_proba = clf.predict_proba(X_val_lt)[:, 1]

                # Compute AUC
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_heatmap[l, t] = auc

            except Exception as e:
                logger.warning(f"Failed at layer={l}, token={t}: {e}")
                auc_heatmap[l, t] = 0.5

            pbar.update(1)

    pbar.close()
    return auc_heatmap


def plot_heatmap(auc_heatmap, output_path, title="Token-Layer AUC Heatmap"):
    """Plot and save the AUC heatmap."""
    L, T = auc_heatmap.shape

    plt.figure(figsize=(12, 6))
    sns.heatmap(auc_heatmap, cmap='RdYlGn', center=0.5, vmin=0.4, vmax=0.9,
                xticklabels=5, yticklabels=2, cbar_kws={'label': 'Validation AUC'})
    plt.xlabel('Token Position')
    plt.ylabel('Layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train token-level heatmap probes (Baseline 0)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="data/heatmaps")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogReg")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    X, y = load_activations(args.data_dir, args.model, args.dataset, args.split)

    if X is None:
        logger.error("Failed to load data")
        return

    # Split into train/val (80/20 stratified)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Train heatmap probes
    auc_heatmap = train_heatmap_probes(X_train, y_train, X_val, y_val, C=args.C)

    # Save results
    model_clean = args.model.replace("/", "_")
    output_prefix = os.path.join(args.output_dir, f"{model_clean}_{args.dataset}_{args.split}")

    # Save heatmap as numpy array
    np.save(f"{output_prefix}_heatmap.npy", auc_heatmap)

    # Plot
    title = f"Token-Layer AUC Heatmap: {args.model} on {args.dataset}"
    plot_heatmap(auc_heatmap, f"{output_prefix}_heatmap.png", title=title)

    # Summary stats
    logger.info(f"AUC Heatmap Stats:")
    logger.info(f"  Mean: {auc_heatmap.mean():.4f}")
    logger.info(f"  Max: {auc_heatmap.max():.4f} at layer={auc_heatmap.argmax()//auc_heatmap.shape[1]}, token={auc_heatmap.argmax()%auc_heatmap.shape[1]}")
    logger.info(f"  Min: {auc_heatmap.min():.4f}")

    # Find best positions (top 10)
    flat_indices = np.argsort(auc_heatmap.flatten())[::-1][:10]
    L, T = auc_heatmap.shape
    best_positions = [(idx // T, idx % T, auc_heatmap.flat[idx]) for idx in flat_indices]

    logger.info("Top 10 (layer, token) positions:")
    for l, t, auc in best_positions:
        logger.info(f"  Layer {l}, Token {t}: AUC={auc:.4f}")


if __name__ == "__main__":
    main()
