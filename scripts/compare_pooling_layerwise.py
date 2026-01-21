"""
Compare all pooling strategies (mean, max, last, attn) layer-by-layer.

Creates comprehensive comparison plots showing:
    - Validation AUC/Accuracy per layer for all pooling strategies
    - Test AUC/Accuracy per layer for all pooling strategies (if available)
    - Highlights best layer for each pooling strategy
    - Highlights overall best layer across all strategies

Usage:
    # Basic usage - automatic discovery
    python scripts/compare_pooling_layerwise.py \
        --probes_base_dir /content/drive/MyDrive/probes \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying

    # With test evaluation
    python scripts/compare_pooling_layerwise.py \
        --probes_base_dir /content/drive/MyDrive/probes \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --test_activations_dir /content/drive/MyDrive/activations/test \
        --evaluate_test

    # Manual paths for each pooling strategy
    python scripts/compare_pooling_layerwise.py \
        --mean_dir /path/to/mean \
        --max_dir /path/to/max \
        --last_dir /path/to/last \
        --attn_dir /path/to/attn \
        --output_dir results/comparisons
"""

import argparse
import json
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe

# Color scheme for pooling strategies
POOLING_COLORS = {
    'mean': '#2E86AB',  # Blue
    'max': '#A23B72',   # Purple
    'last': '#F18F01',  # Orange
    'attn': '#06A77D'   # Green
}

POOLING_ORDER = ['mean', 'max', 'last', 'attn']

def find_pooling_dirs(base_dir: str, model: str, dataset: str) -> Dict[str, str]:
    """
    Find directories for each pooling strategy.

    Tries multiple path patterns:
        1. {base_dir}/{model}/{dataset}/{pooling}/
        2. {base_dir}/{pooling}/
        3. {base_dir}/{dataset}/{pooling}/

    Args:
        base_dir: Base directory containing probes
        model: Model name (e.g., meta-llama/Llama-3.2-3B-Instruct)
        dataset: Dataset name (e.g., Deception-Roleplaying)

    Returns:
        Dict mapping pooling strategy to directory path
    """
    pooling_dirs = {}

    # Normalize model name for path
    model_normalized = model.replace('/', '_').replace('-', '_')

    for pooling in POOLING_ORDER:
        # Try different patterns
        patterns = [
            os.path.join(base_dir, model_normalized, dataset, pooling),
            os.path.join(base_dir, model, dataset, pooling),
            os.path.join(base_dir, pooling),
            os.path.join(base_dir, dataset, pooling),
        ]

        for pattern in patterns:
            if os.path.exists(pattern):
                results_file = os.path.join(pattern, "layer_results.json")
                if os.path.exists(results_file):
                    pooling_dirs[pooling] = pattern
                    print(f"✓ Found {pooling}: {pattern}")
                    break

    return pooling_dirs

def load_layer_results(probes_dir: str) -> Optional[List[Dict]]:
    """Load layer_results.json from probes directory."""
    results_path = os.path.join(probes_dir, "layer_results.json")

    if not os.path.exists(results_path):
        print(f"⚠️  Results file not found: {results_path}")
        return None

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"⚠️  Error loading {results_path}: {e}")
        return None

def evaluate_test_metrics(
    probes_dir: str,
    test_loader: DataLoader,
    device: torch.device,
    pooling_type: str
) -> List[Dict]:
    """
    Evaluate all layer probes on test data.

    Args:
        probes_dir: Directory containing probe_layer_*.pt files
        test_loader: DataLoader for test data
        device: Torch device
        pooling_type: Pooling strategy used for these probes

    Returns:
        List of dicts with test metrics per layer
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Find all probe files
    # Find all probe files and sort numerically (not string sort!)
    probe_files = sorted(
        glob.glob(os.path.join(probes_dir, "probe_layer_*.pt")),
        key=lambda x: int(x.split('_')[-1].replace('.pt', ''))
    )

    if not probe_files:
        print(f"⚠️  No probe files found in {probes_dir}")
        return []

    print(f"Evaluating {len(probe_files)} layer probes on test data...")

    # Get dimensions from test data
    sample_x, _ = next(iter(test_loader))
    _, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]

    test_results = []

    for probe_file in tqdm(probe_files, desc="Testing layers"):
        # Extract layer index from filename
        layer_idx = int(probe_file.split('_')[-1].replace('.pt', ''))

        # Load probe
        probe = LayerProbe(input_dim=D, pooling_type=pooling_type).to(device)
        probe.load_state_dict(torch.load(probe_file, map_location=device))
        probe.eval()

        # Evaluate on test set
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                # Extract layer
                x_layer = x[:, layer_idx, :, :]  # (B, T, D)

                # Forward pass
                logits = probe(x_layer)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                all_preds.extend(probs)
                all_targets.extend(y.cpu().numpy())

        # Calculate metrics
        try:
            test_auc = roc_auc_score(all_targets, all_preds)
        except:
            test_auc = 0.5

        test_acc = accuracy_score(all_targets, (np.array(all_preds) > 0.5).astype(int))

        test_results.append({
            'layer': layer_idx,
            'test_auc': test_auc,
            'test_acc': test_acc
        })

    return test_results

def merge_results(
    val_results: List[Dict],
    test_results: Optional[List[Dict]] = None
) -> List[Dict]:
    """Merge validation and test results by layer."""
    merged = []

    # Create lookup for test results
    test_lookup = {}
    if test_results:
        test_lookup = {r['layer']: r for r in test_results}

    for val_res in val_results:
        layer_idx = val_res['layer']
        merged_res = val_res.copy()

        # Add test metrics if available
        if layer_idx in test_lookup:
            merged_res['test_auc'] = test_lookup[layer_idx]['test_auc']
            merged_res['test_acc'] = test_lookup[layer_idx]['test_acc']

        merged.append(merged_res)

    return merged

def plot_layerwise_comparison(
    all_results: Dict[str, List[Dict]],
    save_path: str = None,
    show_test: bool = False,
    model: str = "",
    dataset: str = ""
):
    """
    Plot comprehensive layer-wise comparison for all pooling strategies.

    Args:
        all_results: Dict mapping pooling strategy to list of per-layer results
        save_path: Optional path to save plot
        show_test: Whether to include test metrics subplot
        model: Model name for title
        dataset: Dataset name for title
    """
    if not all_results:
        print("❌ No results to plot!")
        return

    # Check if we have test data
    has_test = any('test_auc' in all_results[p][0] for p in all_results if all_results[p])
    show_test = show_test and has_test

    # Check if we have accuracy data
    has_accuracy = any('val_acc' in all_results[p][0] for p in all_results if all_results[p])

    # Determine subplot layout
    if show_test and has_accuracy:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_val_auc = axes[0, 0]
        ax_val_acc = axes[0, 1]
        ax_test_auc = axes[1, 0]
        ax_test_acc = axes[1, 1]
    elif show_test:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_val_auc = axes[0]
        ax_test_auc = axes[1]
        ax_val_acc = None
        ax_test_acc = None
    elif has_accuracy:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_val_auc = axes[0]
        ax_val_acc = axes[1]
        ax_test_auc = None
        ax_test_acc = None
    else:
        fig, ax_val_auc = plt.subplots(figsize=(12, 6))
        ax_val_acc = None
        ax_test_auc = None
        ax_test_acc = None

    # Track overall best
    overall_best_auc = 0
    overall_best_info = None

    # Plot each pooling strategy
    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        results = all_results[pooling]
        layers = [r['layer'] for r in results]
        color = POOLING_COLORS.get(pooling, '#666666')

        # === Validation AUC ===
        val_aucs = [r['val_auc'] for r in results]
        ax_val_auc.plot(layers, val_aucs, marker='o', linewidth=2.5, markersize=6,
                       color=color, label=pooling.upper(), alpha=0.85)

        # Mark best layer for this pooling
        best_val = max(results, key=lambda x: x['val_auc'])
        ax_val_auc.scatter([best_val['layer']], [best_val['val_auc']],
                          color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                          marker='*')

        # Track overall best
        if best_val['val_auc'] > overall_best_auc:
            overall_best_auc = best_val['val_auc']
            overall_best_info = (pooling, best_val['layer'], best_val['val_auc'])

        # === Validation Accuracy ===
        if ax_val_acc is not None and 'val_acc' in results[0]:
            val_accs = [r.get('val_acc', 0.5) for r in results]
            ax_val_acc.plot(layers, val_accs, marker='s', linewidth=2.5, markersize=6,
                           color=color, label=pooling.upper(), alpha=0.85)

            best_acc = max(results, key=lambda x: x.get('val_acc', 0))
            ax_val_acc.scatter([best_acc['layer']], [best_acc.get('val_acc', 0.5)],
                              color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                              marker='*')

        # === Test AUC ===
        if ax_test_auc is not None and 'test_auc' in results[0]:
            test_aucs = [r.get('test_auc', 0.5) for r in results]
            ax_test_auc.plot(layers, test_aucs, marker='o', linewidth=2.5, markersize=6,
                            color=color, label=pooling.upper(), alpha=0.85)

            best_test = max(results, key=lambda x: x.get('test_auc', 0))
            ax_test_auc.scatter([best_test['layer']], [best_test.get('test_auc', 0.5)],
                               color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                               marker='*')

        # === Test Accuracy ===
        if ax_test_acc is not None and 'test_acc' in results[0]:
            test_accs = [r.get('test_acc', 0.5) for r in results]
            ax_test_acc.plot(layers, test_accs, marker='s', linewidth=2.5, markersize=6,
                            color=color, label=pooling.upper(), alpha=0.85)

            best_test_acc = max(results, key=lambda x: x.get('test_acc', 0))
            ax_test_acc.scatter([best_test_acc['layer']], [best_test_acc.get('test_acc', 0.5)],
                               color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                               marker='*')

    # Highlight overall best on validation AUC plot
    if overall_best_info:
        pooling, layer, auc = overall_best_info
        color = POOLING_COLORS.get(pooling, '#666666')

        # Add prominent annotation
        ax_val_auc.annotate(
            f'BEST: {pooling.upper()}\nLayer {layer}\nAUC: {auc:.3f}',
            xy=(layer, auc),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.3, edgecolor='black', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=2),
            fontsize=11,
            fontweight='bold',
            ha='left'
        )

    # Style Validation AUC
    ax_val_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax_val_auc.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong (0.7)')
    ax_val_auc.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax_val_auc.set_ylabel('Validation AUC', fontsize=13, fontweight='bold')
    title = f'Validation AUC - Layerwise Comparison'
    if dataset:
        title += f'\n{dataset}'
    if model:
        title += f' | {model.split("/")[-1]}'
    ax_val_auc.set_title(title, fontsize=14, fontweight='bold')
    ax_val_auc.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_val_auc.grid(True, alpha=0.3, linestyle='--')
    ax_val_auc.set_ylim(0.45, 1.0)

    # Style Validation Accuracy
    if ax_val_acc is not None:
        ax_val_acc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
        ax_val_acc.set_xlabel('Layer', fontsize=13, fontweight='bold')
        ax_val_acc.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
        ax_val_acc.set_title('Validation Accuracy - Layerwise Comparison', fontsize=14, fontweight='bold')
        ax_val_acc.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_val_acc.grid(True, alpha=0.3, linestyle='--')
        ax_val_acc.set_ylim(0.45, 1.0)

    # Style Test AUC
    if ax_test_auc is not None:
        ax_test_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
        ax_test_auc.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong (0.7)')
        ax_test_auc.set_xlabel('Layer', fontsize=13, fontweight='bold')
        ax_test_auc.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
        ax_test_auc.set_title('Test AUC - Layerwise Comparison', fontsize=14, fontweight='bold')
        ax_test_auc.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_test_auc.grid(True, alpha=0.3, linestyle='--')
        ax_test_auc.set_ylim(0.45, 1.0)

    # Style Test Accuracy
    if ax_test_acc is not None:
        ax_test_acc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
        ax_test_acc.set_xlabel('Layer', fontsize=13, fontweight='bold')
        ax_test_acc.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax_test_acc.set_title('Test Accuracy - Layerwise Comparison', fontsize=14, fontweight='bold')
        ax_test_acc.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_test_acc.grid(True, alpha=0.3, linestyle='--')
        ax_test_acc.set_ylim(0.45, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved layerwise comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()

def generate_summary_table(all_results: Dict[str, List[Dict]]) -> str:
    """Generate summary table comparing all pooling strategies."""
    lines = []
    lines.append("=" * 90)
    lines.append("POOLING STRATEGY COMPARISON - SUMMARY")
    lines.append("=" * 90)
    lines.append("")

    # Header
    lines.append(f"{'Pooling':<10} {'Best Layer':<12} {'Val AUC':<12} {'Val Acc':<12} {'Test AUC':<12} {'Test Acc':<12}")
    lines.append("-" * 90)

    overall_best = None
    overall_best_auc = 0

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        results = all_results[pooling]
        best = max(results, key=lambda x: x['val_auc'])

        val_auc = best['val_auc']
        val_acc = best.get('val_acc', 0)
        test_auc = best.get('test_auc', 0)
        test_acc = best.get('test_acc', 0)

        # Track overall best
        if val_auc > overall_best_auc:
            overall_best_auc = val_auc
            overall_best = pooling

        val_auc_str = f"{val_auc:.4f}"
        val_acc_str = f"{val_acc:.4f}" if val_acc > 0 else "N/A"
        test_auc_str = f"{test_auc:.4f}" if test_auc > 0 else "N/A"
        test_acc_str = f"{test_acc:.4f}" if test_acc > 0 else "N/A"

        marker = " ⭐" if pooling == overall_best else ""

        lines.append(
            f"{pooling.upper():<10} {best['layer']:<12} {val_auc_str:<12} "
            f"{val_acc_str:<12} {test_auc_str:<12} {test_acc_str:<12}{marker}"
        )

    lines.append("=" * 90)
    lines.append(f"⭐ Best overall: {overall_best.upper()} (Val AUC: {overall_best_auc:.4f})")
    lines.append("=" * 90)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Compare all pooling strategies layer-by-layer"
    )

    # Automatic discovery
    parser.add_argument(
        "--probes_base_dir",
        type=str,
        help="Base directory containing probes (will auto-discover pooling subdirs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., Deception-Roleplaying)"
    )

    # Manual paths
    parser.add_argument("--mean_dir", type=str, help="Path to mean pooling results")
    parser.add_argument("--max_dir", type=str, help="Path to max pooling results")
    parser.add_argument("--last_dir", type=str, help="Path to last pooling results")
    parser.add_argument("--attn_dir", type=str, help="Path to attn pooling results")

    # Test evaluation
    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        help="Evaluate probes on test data"
    )
    parser.add_argument(
        "--test_activations_dir",
        type=str,
        help="Directory containing test activations (if evaluating test)"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparisons",
        help="Output directory for plots and reports"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("POOLING STRATEGY LAYERWISE COMPARISON")
    print("=" * 90)
    print()

    # Find pooling directories
    pooling_dirs = {}

    if args.probes_base_dir and args.model and args.dataset:
        print(f"Auto-discovering pooling directories in: {args.probes_base_dir}")
        pooling_dirs = find_pooling_dirs(args.probes_base_dir, args.model, args.dataset)
    else:
        # Use manual paths
        if args.mean_dir:
            pooling_dirs['mean'] = args.mean_dir
        if args.max_dir:
            pooling_dirs['max'] = args.max_dir
        if args.last_dir:
            pooling_dirs['last'] = args.last_dir
        if args.attn_dir:
            pooling_dirs['attn'] = args.attn_dir

    if not pooling_dirs:
        print("❌ No pooling directories found!")
        print("   Provide either:")
        print("   1. --probes_base_dir, --model, --dataset for auto-discovery")
        print("   2. Manual paths: --mean_dir, --max_dir, --last_dir, --attn_dir")
        return 1

    print(f"\nFound {len(pooling_dirs)} pooling strategies")
    print()

    # Load results for each pooling strategy
    all_results = {}

    for pooling, probes_dir in pooling_dirs.items():
        print(f"Loading {pooling} results from: {probes_dir}")
        results = load_layer_results(probes_dir)

        if results:
            print(f"  ✓ Loaded {len(results)} layer results")
            all_results[pooling] = results
        else:
            print(f"  ✗ Failed to load results")

    if not all_results:
        print("\n❌ No results loaded!")
        return 1

    print(f"\n✓ Successfully loaded results for {len(all_results)} pooling strategies")
    print()

    # Evaluate test metrics if requested
    if args.evaluate_test:
        if not args.test_activations_dir:
            print("⚠️  --test_activations_dir required for test evaluation")
        else:
            print("Evaluating test metrics...")
            print("(This functionality requires test data loader - not yet implemented)")
            print("For now, showing validation metrics only")

    # Generate summary table
    summary = generate_summary_table(all_results)
    print(summary)
    print()

    # Save summary
    summary_path = os.path.join(args.output_dir, "pooling_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Saved summary to {summary_path}")
    print()

    # Generate plots
    print("Generating layerwise comparison plots...")

    plot_path = os.path.join(args.output_dir, "layerwise_pooling_comparison.png")
    plot_layerwise_comparison(
        all_results,
        save_path=plot_path,
        show_test=args.evaluate_test,
        model=args.model or "",
        dataset=args.dataset or ""
    )

    print()
    print("=" * 90)
    print("✓ COMPARISON COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 90)

    return 0

if __name__ == "__main__":
    exit(main())
