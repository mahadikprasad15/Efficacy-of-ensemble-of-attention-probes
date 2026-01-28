"""
Evaluate best probe on out-of-distribution (OOD) datasets.

Supports both:
    - Standard activations: (L, T, D)
    - Prompted-probing activations: (L, D)

Usage:
    # Standard evaluation
    python scripts/eval_ood.py \
        --best_probe_json data/probes/.../mean/best_probe.json \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --eval_dataset Deception-InsiderTrading \
        --eval_split test

    # Prompted-probing evaluation
    python scripts/eval_ood.py \
        --best_probe_json data/probes/.../suffix_deception_yesno/.../none/best_probe.json \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --eval_dataset Deception-InsiderTrading \
        --suffix_condition suffix_deception_yesno \
        --activations_dir data/prompted_activations \
        --eval_split validation
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
        
        # Detect input format from tensor shape
        if self.items:
            sample_shape = self.items[0]['tensor'].shape
            if len(sample_shape) == 2:
                self.input_format = 'final_token'  # (L, D)
            elif len(sample_shape) == 3:
                self.input_format = 'pooled'  # (L, T, D)
            else:
                raise ValueError(f"Unexpected tensor shape: {sample_shape}")

        # Log distribution
        labels = [item['label'] for item in self.items]
        honest = sum(1 for l in labels if l == 0)
        deceptive = sum(1 for l in labels if l == 1)
        logger.info(f"  • Honest: {honest} ({100*honest/len(labels):.1f}%)")
        logger.info(f"  • Deceptive: {deceptive} ({100*deceptive/len(labels):.1f}%)")
        logger.info(f"  • Input format: {self.input_format}")

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
    parser.add_argument(
        "--suffix_condition",
        type=str,
        default=None,
        help="Suffix condition subdirectory (e.g., suffix_deception_yesno) for prompted activations"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}\n")

    # ========================================================================
    # 1. Load best probe info & Identify all probes
    # ========================================================================

    logger.info(f"Loading best probe info from {args.best_probe_json}...")

    with open(args.best_probe_json, 'r') as f:
        best_info = json.load(f)

    # Infer probe directory
    probe_dir = os.path.dirname(args.best_probe_json)
    logger.info(f"Probe directory: {probe_dir}")

    # Find all probe checkpoints
    probe_files = glob.glob(os.path.join(probe_dir, "probe_layer_*.pt"))
    if not probe_files:
        logger.error(f"No probe files found in {probe_dir}")
        return 1
    
    # Parse layers
    probes_by_layer = {}
    for p in probe_files:
        try:
            # Extract layer index from filename "probe_layer_X.pt"
            fname = os.path.basename(p)
            layer_idx = int(fname.replace("probe_layer_", "").replace(".pt", ""))
            probes_by_layer[layer_idx] = p
        except ValueError:
            continue
    
    sorted_layers = sorted(probes_by_layer.keys())
    logger.info(f"Found {len(sorted_layers)} trained probes (Layers {min(sorted_layers)} to {max(sorted_layers)})")

    # Handle backward compatibility keys
    best_layer_key = 'best_layer' if 'best_layer' in best_info else 'layer'
    best_auc_key = 'best_val_auc' if 'best_val_auc' in best_info else 'val_auc'
    
    # ========================================================================
    # 2. Load evaluation dataset
    # ========================================================================

    model_dir = args.model.replace("/", "_")
    
    # Build path - handle prompted activations with suffix_condition
    if args.suffix_condition:
        eval_dir = os.path.join(
            args.activations_dir,
            model_dir,
            args.suffix_condition,
            args.eval_dataset,
            args.eval_split
        )
    else:
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

    # Helper class for layer extraction
    class LayerDataset(Dataset):
        def __init__(self, base_dataset, layer_idx):
            self.base = base_dataset
            self.layer_idx = layer_idx

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y = self.base[idx]
            return x[self.layer_idx], y

    # ========================================================================
    # 3. Evaluate All Layers
    # ========================================================================

    # Infer pooling from directory name (usually the parent folder name)
    pooling = os.path.basename(probe_dir)
    logger.info(f"Inferring pooling from directory: {pooling}")

    # Determine dimensions from dataset
    input_format = eval_dataset.input_format
    
    # Peek at first item to get D
    sample_x = eval_dataset.items[0]['tensor']
    D = sample_x.shape[-1]
    
    # Override pooling for final_token format
    if input_format == 'final_token' and pooling not in ['none', 'last']:
        logger.warning(f"Input is final_token format but probe dir is {pooling}. Overriding to 'none'.")
        pooling = 'none'

    logger.info(f"Starting evaluation of {len(sorted_layers)} layers...")
    
    layer_results = []
    
    for layer_idx in sorted_layers:
        logger.info(f"Evaluating Layer {layer_idx}...")
        
        # specific loader/dataset for this layer
        layer_ds = LayerDataset(eval_dataset, layer_idx)
        layer_loader = DataLoader(layer_ds, batch_size=args.batch_size, shuffle=False)
        
        # Load probe
        probe_path = probes_by_layer[layer_idx]
        model = LayerProbe(input_dim=D, pooling_type=pooling).to(device)
        model.load_state_dict(torch.load(probe_path, map_location=device))
        
        # Eval
        metrics, _, _, _ = evaluate_probe(model, layer_loader, device)
        
        layer_results.append({
            "layer": layer_idx,
            "auc": metrics['auc'],
            "accuracy": metrics['accuracy']
        })
        
        # Log brief result
        print(f"  L{layer_idx}: AUC={metrics['auc']:.4f} Acc={metrics['accuracy']:.4f}")

    # ========================================================================
    # 4. Analyze & Plot
    # ========================================================================
    
    # Find best OOD layer
    best_ood = max(layer_results, key=lambda x: x['auc'])
    logger.info(f"\nBest OOD Performance: Layer {best_ood['layer']} (AUC: {best_ood['auc']:.4f})")

    # Plot
    try:
        plot_ood_performance(layer_results, args.eval_dataset, probe_dir)
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

    # ========================================================================
    # 5. Save results
    # ========================================================================

    output_dir = args.output_dir if args.output_dir else probe_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(
        output_dir,
        f"eval_ood_{args.eval_dataset}_{args.eval_split}.json"
    )

    full_results = {
        "eval_dataset": args.eval_dataset,
        "eval_split": args.eval_split,
        "best_ood_layer": best_ood['layer'],
        "best_ood_auc": best_ood['auc'],
        "best_ood_acc": best_ood['accuracy'],
        "layer_results": layer_results
    }

    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"✓ Saved results to {results_file}")

    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Best OOD Layer: {best_ood['layer']}")
    print(f"AUROC: {best_ood['auc']:.4f}")
    print(f"Accuracy: {best_ood['accuracy']:.4f}")
    print("=" * 70)

    return 0

def plot_ood_performance(results, dataset_name, output_dir):
    """Generate AUC/Accuracy vs Layer plot"""
    import matplotlib.pyplot as plt
    
    layers = [r['layer'] for r in results]
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot AUC
    plt.plot(layers, aucs, 'o-', label='AUC', color='#1f77b4', linewidth=2)
    # Plot Accuracy
    plt.plot(layers, accs, 's--', label='Accuracy', color='#2ca02c', linewidth=2, alpha=0.7)
    
    # Highlight best AUC
    best_idx = np.argmax(aucs)
    plt.plot(layers[best_idx], aucs[best_idx], 'r*', markersize=15, label=f'Best (L{layers[best_idx]})')
    
    plt.title(f"OOD Performance on {dataset_name}", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation for best score
    plt.annotate(
        f"Best AUC: {aucs[best_idx]:.4f}\nLayer {layers[best_idx]}",
        xy=(layers[best_idx], aucs[best_idx]),
        xytext=(0, 10), textcoords='offset points',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
    )
    
    save_path = os.path.join(output_dir, f"ood_metrics_{dataset_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    exit(main())
