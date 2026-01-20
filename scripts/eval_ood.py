"""
Evaluate best probe on out-of-distribution (OOD) datasets.

This script:
    1. Loads best probe from analysis
    2. Evaluates on OOD test sets (different splits or datasets)
    3. Generates confusion matrix and metrics
    4. Saves results for comparison

Usage:
    # Evaluate on test split of same dataset
    python scripts/eval_ood.py \
        --best_probe_json data/probes/.../mean/best_probe.json \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --eval_dataset Deception-Roleplaying \
        --eval_split test

    # Evaluate on different dataset (if you have multiple cached)
    python scripts/eval_ood.py \
        --best_probe_json data/probes/.../mean/best_probe.json \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --eval_dataset Deception-InsiderTrading \
        --eval_split test
"""

import argparse
import json
import os
import sys
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Dataset Loader (same as training)
# ============================================================================

class CachedDeceptionDataset(Dataset):
    """Load cached activations from safetensors shards + manifest"""

    def __init__(self, activations_dir: str):
        self.items = []

        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))

        if not shards:
            raise FileNotFoundError(f"No shards found in {activations_dir}")

        logger.info(f"Loading {len(shards)} shard(s) from {activations_dir}...")

        # Load manifest
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                meta = json.loads(line)
                manifest[meta['id']] = meta

        # Load tensors
        for shard_path in shards:
            try:
                tensors = load_file(shard_path)
                for eid, tensor in tensors.items():
                    if eid not in manifest:
                        continue

                    meta = manifest[eid]
                    label = meta.get('label', -1)

                    if label == -1:
                        continue  # Skip unknown labels

                    self.items.append({
                        "id": eid,
                        "tensor": tensor,
                        "label": label
                    })
            except Exception as e:
                logger.error(f"Error loading {shard_path}: {e}")

        logger.info(f"✓ Loaded {len(self.items)} examples")

        # Log distribution
        labels = [item['label'] for item in self.items]
        honest = sum(1 for l in labels if l == 0)
        deceptive = sum(1 for l in labels if l == 1)
        logger.info(f"  • Honest: {honest} ({100*honest/len(labels):.1f}%)")
        logger.info(f"  • Deceptive: {deceptive} ({100*deceptive/len(labels):.1f}%)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_probe(model, dataloader, device):
    """
    Evaluate probe and return comprehensive metrics.

    Returns:
        dict with all metrics
    """
    model.eval()
    preds = []
    targets = []
    probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(prob)
            preds.extend((prob > 0.5).astype(int))
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    probs = np.array(probs)

    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(targets, probs),
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0)
    }

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    metrics['confusion_matrix'] = cm.tolist()

    # True negatives, false positives, false negatives, true positives
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Specificity (true negative rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics, preds, targets, probs

def print_evaluation_report(metrics: dict, dataset_name: str, split: str):
    """Print comprehensive evaluation report"""

    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {dataset_name} ({split})")
    print("=" * 70)

    print("\nOVERALL METRICS")
    print("-" * 70)
    print(f"AUROC:       {metrics['auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    if 'specificity' in metrics:
        print(f"Specificity: {metrics['specificity']:.4f}")

    print("\nCONFUSION MATRIX")
    print("-" * 70)
    print("                Predicted")
    print("              Honest  Deceptive")
    print(f"Actual Honest   {metrics['true_negatives']:5d}    {metrics['false_positives']:5d}")
    print(f"     Deceptive  {metrics['false_negatives']:5d}    {metrics['true_positives']:5d}")

    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best probe on OOD datasets"
    )

    parser.add_argument(
        "--best_probe_json",
        type=str,
        required=True,
        help="Path to best_probe.json from analyze_probes.py"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (must match cached activations)"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Dataset to evaluate on (e.g., Deception-Roleplaying)"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Split to evaluate on (train/validation/test)"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="data/activations",
        help="Base directory for cached activations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as probe dir)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}\n")

    # ========================================================================
    # 1. Load best probe info
    # ========================================================================

    logger.info(f"Loading best probe info from {args.best_probe_json}...")

    with open(args.best_probe_json, 'r') as f:
        best_info = json.load(f)

    logger.info(f"✓ Best probe: Layer {best_info['layer']} (Val AUC: {best_info['val_auc']:.4f})")
    logger.info(f"  Probe path: {best_info['probe_path']}\n")

    # ========================================================================
    # 2. Load evaluation dataset
    # ========================================================================

    model_dir = args.model.replace("/", "_")
    eval_dir = os.path.join(
        args.activations_dir,
        model_dir,
        args.eval_dataset,
        args.eval_split
    )

    logger.info(f"Loading evaluation data...")
    logger.info(f"  Dataset: {args.eval_dataset}")
    logger.info(f"  Split: {args.eval_split}")
    logger.info(f"  Path: {eval_dir}\n")

    if not os.path.exists(eval_dir):
        logger.error(f"Evaluation data not found: {eval_dir}")
        logger.error(f"Run cache_deception_activations.py first to create this split.")
        return 1

    # Load dataset
    eval_dataset = CachedDeceptionDataset(eval_dir)

    # Extract specific layer
    class LayerDataset(Dataset):
        def __init__(self, base_dataset, layer_idx):
            self.base = base_dataset
            self.layer_idx = layer_idx

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y = self.base[idx]
            return x[self.layer_idx], y

    layer_dataset = LayerDataset(eval_dataset, best_info['layer'])
    eval_loader = DataLoader(
        layer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================================
    # 3. Load probe model
    # ========================================================================

    logger.info(f"Loading probe model...")

    # Get tensor shape from first batch
    sample_x, _ = next(iter(eval_loader))
    T, D = sample_x.shape[1], sample_x.shape[2]

    logger.info(f"  Tensor shape: ({T} tokens, {D} dim)")

    # Infer pooling from probe directory
    pooling = os.path.basename(os.path.dirname(best_info['probe_path']))
    logger.info(f"  Pooling: {pooling}")

    # Create model
    model = LayerProbe(
        input_dim=D,
        pooling=pooling,
        num_tokens=T
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(best_info['probe_path'], map_location=device))
    logger.info(f"✓ Loaded probe weights\n")

    # ========================================================================
    # 4. Evaluate
    # ========================================================================

    logger.info(f"Evaluating...")
    metrics, preds, targets, probs = evaluate_probe(model, eval_loader, device)

    # Print report
    print_evaluation_report(metrics, args.eval_dataset, args.eval_split)

    # ========================================================================
    # 5. Save results
    # ========================================================================

    output_dir = args.output_dir if args.output_dir else os.path.dirname(best_info['probe_path'])
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    results_file = os.path.join(
        output_dir,
        f"eval_{args.eval_dataset}_{args.eval_split}.json"
    )

    results = {
        "probe_layer": best_info['layer'],
        "eval_dataset": args.eval_dataset,
        "eval_split": args.eval_split,
        "num_examples": len(eval_dataset),
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()}
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Saved results to {results_file}")

    # ========================================================================
    # 6. Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Dataset: {args.eval_dataset} ({args.eval_split})")
    print(f"Probe: Layer {best_info['layer']}")
    print(f"AUROC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Results saved to: {results_file}")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    exit(main())
