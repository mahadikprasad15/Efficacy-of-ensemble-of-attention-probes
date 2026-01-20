"""
Evaluate ALL pooling strategies on OOD dataset.

This script:
1. Loads trained probes for all 4 pooling strategies (mean, max, last, attn)
2. Evaluates each probe on OOD dataset activations
3. Generates per-layer AUC/accuracy for all pooling strategies
4. Creates comprehensive comparison charts
5. Saves results for ensemble evaluation

Usage:
    # Basic usage
    python scripts/evaluate_ood_all_pooling.py \
        --probes_base_dir /content/drive/MyDrive/probes \
        --ood_activations_dir /content/drive/MyDrive/activations/insider_trading/test \
        --output_dir /content/drive/MyDrive/results/ood_evaluation

    # Manual paths for each pooling strategy
    python scripts/evaluate_ood_all_pooling.py \
        --mean_probes_dir /path/to/mean \
        --max_probes_dir /path/to/max \
        --last_probes_dir /path/to/last \
        --attn_probes_dir /path/to/attn \
        --ood_activations_dir /path/to/ood/activations \
        --output_dir results/ood

Expected input structure:
    probes/
    ├── mean/
    │   ├── probe_layer_0.pt
    │   ├── probe_layer_1.pt
    │   └── ...
    ├── max/
    ├── last/
    └── attn/

    ood_activations/
    ├── shard_0.safetensors
    ├── shard_1.safetensors
    └── manifest.jsonl

Output:
    output_dir/
    ├── ood_results_all_pooling.json       # Per-layer results
    ├── ood_layerwise_comparison.png       # Comparison plot
    ├── ood_best_probes_summary.txt        # Text summary
    └── logits/                            # Saved logits for ensemble evaluation
        ├── mean_logits.npy
        ├── max_logits.npy
        ├── last_logits.npy
        └── attn_logits.npy
"""

import argparse
import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

# Color scheme
POOLING_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'attn': '#06A77D'
}

POOLING_ORDER = ['mean', 'max', 'last', 'attn']


class CachedOODDataset(Dataset):
    """Load cached OOD activations from safetensors shards."""

    def __init__(self, activations_dir: str):
        """
        Args:
            activations_dir: Directory with shard_*.safetensors and manifest.jsonl
        """
        self.activations_dir = activations_dir

        # Load all shards
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))

        if not shards:
            raise ValueError(f"No shard files found in {activations_dir}")

        print(f"Loading {len(shards)} OOD shard(s)...")

        self.samples = []
        for shard_path in tqdm(shards, desc="Loading shards"):
            shard_data = load_file(shard_path)

            for key in shard_data.keys():
                if key.endswith("_activations"):
                    label_key = key.replace("_activations", "_label")

                    if label_key not in shard_data:
                        continue

                    activations = shard_data[key]
                    label = shard_data[label_key].item()

                    self.samples.append((activations, label))

        print(f"✓ Loaded {len(self.samples)} OOD samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def find_probe_files(probes_dir: str) -> List[str]:
    """Find all probe_layer_*.pt files in directory."""
    pattern = os.path.join(probes_dir, "probe_layer_*.pt")
    probe_files = sorted(glob.glob(pattern))
    return probe_files


def evaluate_probe_layer(
    probe: LayerProbe,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a single layer probe on OOD data.

    Returns:
        (auc, accuracy, predictions, labels)
    """
    probe.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Extract layer
            x_layer = x[:, layer_idx, :, :]  # (B, T, D)

            # Forward pass
            logits = probe(x_layer)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(logits.cpu().numpy().flatten())

    # Calculate metrics
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5

    accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    return auc, accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_all_layers(
    probes_dir: str,
    ood_dataloader: DataLoader,
    pooling_type: str,
    device: torch.device,
    save_logits: bool = True,
    logits_save_dir: str = None
) -> Dict:
    """
    Evaluate all layer probes for a pooling strategy on OOD data.

    Returns:
        {
            'pooling': str,
            'layers': List[int],
            'aucs': List[float],
            'accuracies': List[float],
            'best_layer': int,
            'best_auc': float,
            'logits': np.ndarray (optional)  # (N, L) for ensemble
        }
    """
    probe_files = find_probe_files(probes_dir)

    if not probe_files:
        print(f"⚠️  No probe files found in {probes_dir}")
        return None

    print(f"\nEvaluating {pooling_type.upper()} ({len(probe_files)} layers)...")

    # Get dimensions from first sample
    sample_x, _ = next(iter(ood_dataloader))
    _, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]

    results = []
    all_layer_logits = []  # For ensemble evaluation

    for probe_file in tqdm(probe_files, desc=f"{pooling_type.upper()}"):
        # Extract layer index
        layer_idx = int(probe_file.split('_')[-1].replace('.pt', ''))

        # Load probe
        probe = LayerProbe(input_dim=D, pooling_type=pooling_type).to(device)
        probe.load_state_dict(torch.load(probe_file, map_location=device))

        # Evaluate
        auc, accuracy, preds, labels = evaluate_probe_layer(
            probe, ood_dataloader, layer_idx, device
        )

        results.append({
            'layer': layer_idx,
            'auc': float(auc),
            'accuracy': float(accuracy)
        })

        # Store logits for ensemble (use raw predictions before sigmoid)
        # Recompute to get logits
        probe.eval()
        layer_logits = []
        with torch.no_grad():
            for x, _ in ood_dataloader:
                x = x.to(device)
                x_layer = x[:, layer_idx, :, :]
                logits = probe(x_layer).cpu().numpy().flatten()
                layer_logits.extend(logits)
        all_layer_logits.append(np.array(layer_logits))

    # Find best layer
    best = max(results, key=lambda x: x['auc'])

    output = {
        'pooling': pooling_type,
        'layers': [r['layer'] for r in results],
        'aucs': [r['auc'] for r in results],
        'accuracies': [r['accuracy'] for r in results],
        'best_layer': best['layer'],
        'best_auc': best['auc'],
        'best_accuracy': best['accuracy']
    }

    # Save logits for ensemble evaluation
    if save_logits and logits_save_dir:
        os.makedirs(logits_save_dir, exist_ok=True)
        logits_array = np.array(all_layer_logits).T  # (N, L)
        logits_path = os.path.join(logits_save_dir, f"{pooling_type}_logits.npy")
        np.save(logits_path, logits_array)
        print(f"  ✓ Saved logits: {logits_path} {logits_array.shape}")
        output['logits_path'] = logits_path

    return output


def plot_ood_comparison(
    all_results: Dict[str, Dict],
    save_path: str,
    ood_dataset_name: str = "OOD"
):
    """
    Create comprehensive comparison plot for OOD evaluation.

    Args:
        all_results: Dict mapping pooling -> results dict
        save_path: Path to save plot
        ood_dataset_name: Name of OOD dataset for title
    """
    fig, (ax_auc, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))

    overall_best_auc = 0
    overall_best_info = None

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        res = all_results[pooling]
        layers = res['layers']
        aucs = res['aucs']
        accs = res['accuracies']
        color = POOLING_COLORS.get(pooling, '#666666')

        # Plot AUC
        ax_auc.plot(layers, aucs, marker='o', linewidth=2.5, markersize=6,
                   color=color, label=pooling.upper(), alpha=0.85)

        # Mark best layer
        best_idx = res['layers'].index(res['best_layer'])
        ax_auc.scatter([res['best_layer']], [res['aucs'][best_idx]],
                      color=color, s=200, zorder=5, edgecolors='black',
                      linewidths=2.5, marker='*')

        # Track overall best
        if res['best_auc'] > overall_best_auc:
            overall_best_auc = res['best_auc']
            overall_best_info = (pooling, res['best_layer'], res['best_auc'])

        # Plot Accuracy
        ax_acc.plot(layers, accs, marker='s', linewidth=2.5, markersize=6,
                   color=color, label=pooling.upper(), alpha=0.85)

        best_acc_idx = res['layers'].index(res['best_layer'])
        ax_acc.scatter([res['best_layer']], [res['accuracies'][best_acc_idx]],
                      color=color, s=200, zorder=5, edgecolors='black',
                      linewidths=2.5, marker='*')

    # Highlight overall best on AUC plot
    if overall_best_info:
        pooling, layer, auc = overall_best_info
        color = POOLING_COLORS.get(pooling, '#666666')

        ax_auc.annotate(
            f'BEST: {pooling.upper()}\nLayer {layer}\nAUC: {auc:.3f}',
            xy=(layer, auc),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.3,
                     edgecolor='black', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color='black', lw=2),
            fontsize=11,
            fontweight='bold'
        )

    # Style AUC
    ax_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax_auc.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong (0.7)')
    ax_auc.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax_auc.set_ylabel('OOD AUC', fontsize=13, fontweight='bold')
    ax_auc.set_title(f'OOD Evaluation: {ood_dataset_name}\nAUC per Layer', fontsize=14, fontweight='bold')
    ax_auc.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_auc.grid(True, alpha=0.3, linestyle='--')
    ax_auc.set_ylim(0.45, 1.0)

    # Style Accuracy
    ax_acc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax_acc.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax_acc.set_ylabel('OOD Accuracy', fontsize=13, fontweight='bold')
    ax_acc.set_title(f'OOD Evaluation: {ood_dataset_name}\nAccuracy per Layer', fontsize=14, fontweight='bold')
    ax_acc.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    ax_acc.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved OOD comparison plot: {save_path}")
    plt.close()


def generate_summary(all_results: Dict[str, Dict]) -> str:
    """Generate text summary of OOD evaluation."""
    lines = []
    lines.append("=" * 90)
    lines.append("OOD EVALUATION - ALL POOLING STRATEGIES")
    lines.append("=" * 90)
    lines.append("")

    # Header
    lines.append(f"{'Pooling':<10} {'Best Layer':<12} {'OOD AUC':<12} {'OOD Acc':<12} {'Precision':<12} {'Recall':<12}")
    lines.append("-" * 90)

    overall_best = None
    overall_best_auc = 0

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        res = all_results[pooling]

        auc_str = f"{res['best_auc']:.4f}"
        acc_str = f"{res['best_accuracy']:.4f}"

        marker = " ⭐" if res['best_auc'] > overall_best_auc else ""
        if res['best_auc'] > overall_best_auc:
            overall_best_auc = res['best_auc']
            overall_best = pooling

        lines.append(
            f"{pooling.upper():<10} {res['best_layer']:<12} {auc_str:<12} "
            f"{acc_str:<12} {'N/A':<12} {'N/A':<12}{marker}"
        )

    lines.append("=" * 90)
    lines.append(f"⭐ Best overall: {overall_best.upper()} (OOD AUC: {overall_best_auc:.4f})")
    lines.append("=" * 90)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all pooling strategies on OOD dataset")

    # Auto-discovery
    parser.add_argument("--probes_base_dir", type=str, help="Base directory containing pooling subdirectories")

    # Manual paths
    parser.add_argument("--mean_probes_dir", type=str, help="Path to mean pooling probes")
    parser.add_argument("--max_probes_dir", type=str, help="Path to max pooling probes")
    parser.add_argument("--last_probes_dir", type=str, help="Path to last pooling probes")
    parser.add_argument("--attn_probes_dir", type=str, help="Path to attn pooling probes")

    # OOD data
    parser.add_argument("--ood_activations_dir", type=str, required=True,
                       help="Directory with OOD activations (shard_*.safetensors)")
    parser.add_argument("--ood_dataset_name", type=str, default="OOD",
                       help="Name of OOD dataset (for plots)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/ood_evaluation",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("OOD EVALUATION - ALL POOLING STRATEGIES")
    print("=" * 90)
    print()

    # Find probe directories
    probe_dirs = {}
    if args.probes_base_dir:
        for pooling in POOLING_ORDER:
            candidate = os.path.join(args.probes_base_dir, pooling)
            if os.path.exists(candidate):
                probe_dirs[pooling] = candidate
                print(f"✓ Found {pooling}: {candidate}")
    else:
        if args.mean_probes_dir:
            probe_dirs['mean'] = args.mean_probes_dir
        if args.max_probes_dir:
            probe_dirs['max'] = args.max_probes_dir
        if args.last_probes_dir:
            probe_dirs['last'] = args.last_probes_dir
        if args.attn_probes_dir:
            probe_dirs['attn'] = args.attn_probes_dir

    if not probe_dirs:
        print("❌ No probe directories found!")
        return 1

    print(f"\n✓ Found {len(probe_dirs)} pooling strategies to evaluate")
    print()

    # Load OOD data
    print(f"Loading OOD activations from: {args.ood_activations_dir}")
    ood_dataset = CachedOODDataset(args.ood_activations_dir)
    ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print()

    # Evaluate each pooling strategy
    all_results = {}
    logits_dir = os.path.join(args.output_dir, "logits")

    for pooling, probes_dir in probe_dirs.items():
        results = evaluate_all_layers(
            probes_dir=probes_dir,
            ood_dataloader=ood_dataloader,
            pooling_type=pooling,
            device=device,
            save_logits=True,
            logits_save_dir=logits_dir
        )

        if results:
            all_results[pooling] = results

    if not all_results:
        print("\n❌ No results generated!")
        return 1

    print(f"\n✓ Successfully evaluated {len(all_results)} pooling strategies")
    print()

    # Generate summary
    summary = generate_summary(all_results)
    print(summary)
    print()

    # Save results
    results_path = os.path.join(args.output_dir, "ood_results_all_pooling.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved results: {results_path}")

    summary_path = os.path.join(args.output_dir, "ood_best_probes_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Saved summary: {summary_path}")

    # Generate plot
    plot_path = os.path.join(args.output_dir, "ood_layerwise_comparison.png")
    plot_ood_comparison(all_results, plot_path, args.ood_dataset_name)

    print()
    print("=" * 90)
    print("✓ OOD EVALUATION COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {args.output_dir}")
    print(f"Logits saved to: {logits_dir} (for ensemble evaluation)")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    exit(main())
