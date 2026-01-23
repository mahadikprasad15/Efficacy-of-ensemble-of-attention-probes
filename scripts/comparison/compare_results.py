"""
Compare results across multiple experiments.

This script aggregates and compares results from:
    - Different pooling strategies (mean, max, last, attn)
    - Different datasets (deception vs hallucination)
    - Different models (1B vs 3B)
    - OOD evaluations

Creates comparison tables and plots.

Usage:
    python scripts/compare_results.py \
        --experiments_dir data/probes \
        --output_dir results/comparisons

    python scripts/compare_results.py \
        --experiments_config experiments.json
"""

import argparse
import json
import os
import glob
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_all_experiments(base_dir: str) -> List[Dict]:
    """
    Recursively find all experiment results.

    Looks for:
        - layer_results.json (training results)
        - eval_*.json (OOD evaluation results)
        - best_probe.json (best probe info)

    Returns list of experiment dicts with metadata.
    """
    experiments = []

    # Find all layer_results.json files
    pattern = os.path.join(base_dir, "**", "layer_results.json")
    for results_file in glob.glob(pattern, recursive=True):
        # Parse path to extract metadata
        # Expected: data/probes/{model}/{dataset}/{pooling}/layer_results.json
        parts = results_file.split(os.sep)

        try:
            probes_idx = parts.index('probes')
            model = parts[probes_idx + 1]
            dataset = parts[probes_idx + 2]
            pooling = parts[probes_idx + 3]

            # Load results
            with open(results_file, 'r') as f:
                layer_results = json.load(f)

            # Find best layer
            best = max(layer_results, key=lambda x: x['val_auc'])

            # Check for OOD evaluations
            ood_evals = []
            eval_pattern = os.path.join(os.path.dirname(results_file), "eval_*.json")
            for eval_file in glob.glob(eval_pattern):
                with open(eval_file, 'r') as f:
                    ood_evals.append(json.load(f))

            experiments.append({
                'model': model,
                'dataset': dataset,
                'pooling': pooling,
                'best_layer': best['layer'],
                'val_auc': best['val_auc'],
                'results_file': results_file,
                'ood_evaluations': ood_evals
            })

        except (ValueError, IndexError, KeyError) as e:
            print(f"⚠️  Skipping {results_file}: {e}")
            continue

    return experiments

def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """Create pandas DataFrame for easy comparison"""

    rows = []
    for exp in experiments:
        row = {
            'Model': exp['model'],
            'Dataset': exp['dataset'],
            'Pooling': exp['pooling'],
            'Best Layer': exp['best_layer'],
            'Val AUC': exp['val_auc']
        }

        # Add OOD results if available
        for ood_eval in exp.get('ood_evaluations', []):
            key = f"{ood_eval['eval_dataset']} ({ood_eval['eval_split']}) AUC"
            row[key] = ood_eval['metrics']['auc']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by Val AUC descending
    df = df.sort_values('Val AUC', ascending=False)

    return df

def plot_pooling_comparison(experiments: List[Dict], save_path: str = None):
    """
    Plot comparison of pooling strategies.

    Creates bar plot showing Val AUC for each pooling method.
    """
    # Group by pooling
    pooling_results = {}
    for exp in experiments:
        pooling = exp['pooling']
        if pooling not in pooling_results:
            pooling_results[pooling] = []
        pooling_results[pooling].append(exp['val_auc'])

    # Calculate means and stds
    pooling_methods = list(pooling_results.keys())
    means = [np.mean(pooling_results[p]) for p in pooling_methods]
    stds = [np.std(pooling_results[p]) if len(pooling_results[p]) > 1 else 0
            for p in pooling_methods]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pooling_methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#2E86AB', '#A23B72', '#F18F01', '#06A77D'])

    # Styling
    ax.set_xlabel('Pooling Strategy', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Pooling Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pooling_methods)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random Chance', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_dataset_comparison(experiments: List[Dict], save_path: str = None):
    """
    Plot comparison across datasets (if multiple exist).
    """
    # Group by dataset
    dataset_results = {}
    for exp in experiments:
        dataset = exp['dataset']
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        dataset_results[dataset].append(exp['val_auc'])

    if len(dataset_results) < 2:
        print("⚠️  Only one dataset found, skipping dataset comparison plot")
        return

    # Calculate means
    datasets = list(dataset_results.keys())
    means = [np.mean(dataset_results[d]) for d in datasets]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    bars = ax.bar(x, means, alpha=0.7, color='#2E86AB')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Mean Validation AUC', fontsize=12)
    ax.set_title('Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_layerwise_comparison(experiments: List[Dict], save_path: str = None):
    """
    Plot layer-wise AUC and accuracy trends for all pooling strategies in one chart.
    
    Creates a combined plot showing how validation metrics change across layers
    for each pooling strategy (mean, max, last, attn).
    
    Args:
        experiments: List of experiment dicts from find_all_experiments()
        save_path: Optional path to save plot
    """
    # Group experiments by model/dataset to find comparable ones
    grouped = {}
    for exp in experiments:
        key = (exp['model'], exp['dataset'])
        if key not in grouped:
            grouped[key] = {}
        pooling = exp['pooling']
        grouped[key][pooling] = exp['results_file']
    
    # Process each model/dataset group
    for (model, dataset), pooling_files in grouped.items():
        if len(pooling_files) < 2:
            print(f"⚠️  Only one pooling strategy for {model}/{dataset}, need ≥2 for comparison")
            continue
        
        print(f"\n📊 Creating layerwise comparison for {model}/{dataset}")
        
        # Load all pooling results
        all_results = {}
        has_accuracy = False
        
        for pooling, results_file in pooling_files.items():
            try:
                with open(results_file, 'r') as f:
                    layer_results = json.load(f)
                all_results[pooling] = layer_results
                # Check if accuracy data exists
                if layer_results and 'val_acc' in layer_results[0]:
                    has_accuracy = True
            except Exception as e:
                print(f"⚠️  Error loading {results_file}: {e}")
                continue
        
        if len(all_results) < 2:
            print(f"⚠️  Could not load enough pooling results for comparison")
            continue
        
        # Colors for pooling strategies
        colors = {
            'mean': '#2E86AB',  # Blue
            'max': '#A23B72',   # Purple
            'last': '#F18F01',  # Orange
            'attn': '#06A77D'   # Green
        }
        
        # Create figure - 2 subplots if we have accuracy, 1 otherwise
        if has_accuracy:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            ax_auc = axes[0]
            ax_acc = axes[1]
        else:
            fig, ax_auc = plt.subplots(figsize=(14, 6))
            ax_acc = None
        
        # Track overall best across all pooling strategies
        overall_best_auc = 0
        overall_best_info = None

        # Plot each pooling strategy
        for pooling, layer_results in all_results.items():
            layers = [r['layer'] for r in layer_results]
            aucs = [r['val_auc'] for r in layer_results]
            color = colors.get(pooling, '#666666')

            # Plot AUC line
            ax_auc.plot(layers, aucs, marker='o', linewidth=2.5, markersize=6,
                       color=color, label=f'{pooling.upper()}', alpha=0.85)

            # Mark best layer for this pooling
            best = max(layer_results, key=lambda x: x['val_auc'])
            ax_auc.scatter([best['layer']], [best['val_auc']],
                          color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                          marker='*')

            # Track overall best
            if best['val_auc'] > overall_best_auc:
                overall_best_auc = best['val_auc']
                overall_best_info = (pooling, best['layer'], best['val_auc'])

            # Plot accuracy if available
            if ax_acc is not None and 'val_acc' in layer_results[0]:
                accs = [r.get('val_acc', 0.5) for r in layer_results]
                ax_acc.plot(layers, accs, marker='s', linewidth=2.5, markersize=6,
                           color=color, label=f'{pooling.upper()}', alpha=0.85)

                # Mark best accuracy layer
                best_acc = max(layer_results, key=lambda x: x.get('val_acc', 0))
                ax_acc.scatter([best_acc['layer']], [best_acc.get('val_acc', 0.5)],
                              color=color, s=200, zorder=5, edgecolors='black', linewidths=2.5,
                              marker='*')

        # Highlight overall best on validation AUC plot
        if overall_best_info:
            pooling, layer, auc = overall_best_info
            color = colors.get(pooling, '#666666')

            # Add prominent annotation for overall best
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
                fontweight='bold',
                ha='left'
            )
        
        # Style AUC subplot
        ax_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random Chance')
        ax_auc.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong Signal (0.7)')
        ax_auc.set_ylabel('Validation AUC', fontsize=13, fontweight='bold')
        ax_auc.set_title(f'Layerwise Validation AUC Comparison\n{dataset} | {model}',
                        fontsize=14, fontweight='bold')
        ax_auc.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_auc.grid(True, alpha=0.3, linestyle='--')
        ax_auc.set_ylim(0.45, 1.0)

        # Style accuracy subplot if present
        if ax_acc is not None:
            ax_acc.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random Chance')
            ax_acc.set_xlabel('Layer', fontsize=13, fontweight='bold')
            ax_acc.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
            ax_acc.set_title('Layerwise Validation Accuracy Comparison', fontsize=14, fontweight='bold')
            ax_acc.legend(loc='best', fontsize=11, framealpha=0.9)
            ax_acc.grid(True, alpha=0.3, linestyle='--')
            ax_acc.set_ylim(0.45, 1.0)
        else:
            ax_auc.set_xlabel('Layer', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            # Create unique filename for each model/dataset combo
            model_short = model.replace('/', '_').replace('-', '_')
            dataset_short = dataset.replace('-', '_')
            actual_path = save_path.replace('.png', f'_{model_short}_{dataset_short}.png')
            plt.savefig(actual_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved layerwise comparison to {actual_path}")
        else:
            plt.show()
        
        plt.close()

def create_ood_matrix(experiments: List[Dict]) -> pd.DataFrame:
    """
    Create matrix of train → test AUCs for OOD generalization.

    Rows: Training dataset
    Columns: Test dataset
    """
    # Collect all unique datasets
    train_datasets = set()
    test_datasets = set()

    for exp in experiments:
        train_datasets.add(exp['dataset'])
        for ood_eval in exp.get('ood_evaluations', []):
            test_datasets.add(ood_eval['eval_dataset'])

    if not test_datasets:
        print("⚠️  No OOD evaluations found")
        return None

    # Create matrix
    matrix = pd.DataFrame(index=sorted(train_datasets), columns=sorted(test_datasets))

    # Fill matrix
    for exp in experiments:
        train_ds = exp['dataset']
        for ood_eval in exp.get('ood_evaluations', []):
            test_ds = ood_eval['eval_dataset']
            auc = ood_eval['metrics']['auc']
            matrix.loc[train_ds, test_ds] = auc

    return matrix

def generate_summary_report(experiments: List[Dict], df: pd.DataFrame, output_path: str = None):
    """Generate comprehensive summary report"""

    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT COMPARISON SUMMARY")
    report.append("=" * 80)
    report.append(f"Total Experiments: {len(experiments)}")
    report.append("")

    # Best overall
    best = max(experiments, key=lambda x: x['val_auc'])
    report.append("BEST OVERALL RESULT")
    report.append("-" * 80)
    report.append(f"Model: {best['model']}")
    report.append(f"Dataset: {best['dataset']}")
    report.append(f"Pooling: {best['pooling']}")
    report.append(f"Layer: {best['best_layer']}")
    report.append(f"Val AUC: {best['val_auc']:.4f}")
    report.append("")

    # Pooling comparison
    pooling_summary = df.groupby('Pooling')['Val AUC'].agg(['mean', 'std', 'count'])
    report.append("POOLING STRATEGY SUMMARY")
    report.append("-" * 80)
    report.append(pooling_summary.to_string())
    report.append("")

    # Dataset comparison (if multiple)
    if len(df['Dataset'].unique()) > 1:
        dataset_summary = df.groupby('Dataset')['Val AUC'].agg(['mean', 'std', 'count'])
        report.append("DATASET SUMMARY")
        report.append("-" * 80)
        report.append(dataset_summary.to_string())
        report.append("")

    # Full table
    report.append("ALL EXPERIMENTS")
    report.append("-" * 80)
    report.append(df.to_string(index=False))
    report.append("")

    report.append("=" * 80)

    # Print and optionally save
    report_text = "\n".join(report)
    print(report_text)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Saved summary to {output_path}")

    return report_text

def main():
    parser = argparse.ArgumentParser(
        description="Compare results across experiments"
    )

    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="data/probes",
        help="Base directory containing all experiments"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparisons",
        help="Output directory for comparison plots and reports"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save comparison table as CSV"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(f"Searching for experiments in: {args.experiments_dir}\n")

    # Find all experiments
    experiments = find_all_experiments(args.experiments_dir)

    if not experiments:
        print("❌ No experiments found!")
        print(f"   Make sure you have run train_deception_probes.py")
        return 1

    print(f"✓ Found {len(experiments)} experiment(s)\n")

    # Create comparison table
    df = create_comparison_table(experiments)

    # Generate summary
    report_path = os.path.join(args.output_dir, "comparison_summary.txt")
    generate_summary_report(experiments, df, report_path)

    # Save CSV
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "comparison_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to {csv_path}")

    # Generate plots
    print("\nGenerating plots...")

    pooling_plot = os.path.join(args.output_dir, "pooling_comparison.png")
    plot_pooling_comparison(experiments, pooling_plot)

    dataset_plot = os.path.join(args.output_dir, "dataset_comparison.png")
    plot_dataset_comparison(experiments, dataset_plot)

    # Layerwise comparison across pooling strategies
    layerwise_plot = os.path.join(args.output_dir, "layerwise_comparison.png")
    plot_layerwise_comparison(experiments, layerwise_plot)

    # OOD matrix
    ood_matrix = create_ood_matrix(experiments)
    if ood_matrix is not None:
        ood_csv = os.path.join(args.output_dir, "ood_matrix.csv")
        ood_matrix.to_csv(ood_csv)
        print(f"✓ Saved OOD matrix to {ood_csv}")

        print("\nOOD Generalization Matrix:")
        print(ood_matrix)

    print("\n" + "=" * 80)
    print("✓ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    exit(main())
