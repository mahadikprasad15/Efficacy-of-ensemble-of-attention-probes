"""
Analyze trained probes to find best layer and visualize results.

This script:
    1. Loads per-layer probe results
    2. Finds best probe based on validation AUC
    3. Plots per-layer AUC and accuracy trends
    4. Saves analysis report and best probe info

Usage:
    python scripts/analyze_probes.py \
        --probes_dir data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean

    python scripts/analyze_probes.py \
        --probes_dir data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean \
        --save_plots
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def load_results(probes_dir: str) -> List[Dict]:
    """Load layer_results.json from probes directory"""
    results_path = os.path.join(probes_dir, "layer_results.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            f"Run train_deception_probes.py first to generate results."
        )

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results

def find_best_probe(results: List[Dict], metric: str = 'val_auc') -> Dict:
    """
    Find best probe based on specified metric.

    Args:
        results: List of per-layer results
        metric: Metric to optimize ('val_auc' or 'epoch')

    Returns:
        Dict with best layer info
    """
    best = max(results, key=lambda x: x[metric])
    return best

def plot_per_layer_metrics(results: List[Dict], save_path: str = None):
    """
    Plot AUC and other metrics per layer.

    Args:
        results: List of per-layer results
        save_path: Optional path to save plot
    """
    layers = [r['layer'] for r in results]
    aucs = [r['val_auc'] for r in results]

    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot AUC per layer
    ax.plot(layers, aucs, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random Chance', alpha=0.5)

    # Mark best layer
    best_layer = max(results, key=lambda x: x['val_auc'])
    ax.scatter([best_layer['layer']], [best_layer['val_auc']],
               color='orange', s=200, zorder=5,
               label=f"Best: Layer {best_layer['layer']} (AUC={best_layer['val_auc']:.3f})")

    # Styling
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Deception Detection Performance by Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add horizontal range indicator for strong performance
    if any(auc >= 0.7 for auc in aucs):
        ax.axhline(y=0.7, color='green', linestyle=':', label='Strong Signal (≥0.7)', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def generate_report(results: List[Dict], probes_dir: str, output_path: str = None):
    """
    Generate analysis report with statistics.

    Args:
        results: List of per-layer results
        probes_dir: Directory containing probe files
        output_path: Optional path to save report
    """
    report = []
    report.append("=" * 70)
    report.append("PROBE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Probes Directory: {probes_dir}")
    report.append(f"Total Layers: {len(results)}")
    report.append("")

    # Best probe
    best = find_best_probe(results, 'val_auc')
    report.append("BEST PROBE")
    report.append("-" * 70)
    report.append(f"Layer: {best['layer']}")
    report.append(f"Validation AUC: {best['val_auc']:.4f}")
    report.append(f"Training Epoch: {best['epoch']}")
    report.append(f"Probe Path: {os.path.join(probes_dir, f'probe_layer_{best[\"layer\"]}.pt')}")
    report.append("")

    # Statistics
    aucs = [r['val_auc'] for r in results]
    report.append("STATISTICS")
    report.append("-" * 70)
    report.append(f"Mean AUC: {np.mean(aucs):.4f}")
    report.append(f"Std AUC: {np.std(aucs):.4f}")
    report.append(f"Min AUC: {np.min(aucs):.4f} (Layer {results[np.argmin(aucs)]['layer']})")
    report.append(f"Max AUC: {np.max(aucs):.4f} (Layer {results[np.argmax(aucs)]['layer']})")
    report.append("")

    # Top 5 layers
    top_5 = sorted(results, key=lambda x: x['val_auc'], reverse=True)[:5]
    report.append("TOP 5 LAYERS")
    report.append("-" * 70)
    for i, layer_result in enumerate(top_5, 1):
        report.append(
            f"{i}. Layer {layer_result['layer']:2d} | "
            f"AUC: {layer_result['val_auc']:.4f} | "
            f"Epoch: {layer_result['epoch']:2d}"
        )
    report.append("")

    # Performance categories
    strong = [r for r in results if r['val_auc'] >= 0.7]
    moderate = [r for r in results if 0.6 <= r['val_auc'] < 0.7]
    weak = [r for r in results if 0.5 <= r['val_auc'] < 0.6]
    no_signal = [r for r in results if r['val_auc'] < 0.5]

    report.append("PERFORMANCE BREAKDOWN")
    report.append("-" * 70)
    report.append(f"Strong (AUC ≥ 0.7):   {len(strong):2d} layers ({100*len(strong)/len(results):5.1f}%)")
    report.append(f"Moderate (0.6-0.7):  {len(moderate):2d} layers ({100*len(moderate)/len(results):5.1f}%)")
    report.append(f"Weak (0.5-0.6):      {len(weak):2d} layers ({100*len(weak)/len(results):5.1f}%)")
    report.append(f"No signal (< 0.5):   {len(no_signal):2d} layers ({100*len(no_signal)/len(results):5.1f}%)")
    report.append("")

    # Layer range analysis
    report.append("LAYER RANGE ANALYSIS")
    report.append("-" * 70)
    total_layers = len(results)
    early = results[:total_layers//3]
    middle = results[total_layers//3:2*total_layers//3]
    late = results[2*total_layers//3:]

    report.append(f"Early layers (0-{total_layers//3-1}):   Mean AUC = {np.mean([r['val_auc'] for r in early]):.4f}")
    report.append(f"Middle layers ({total_layers//3}-{2*total_layers//3-1}): Mean AUC = {np.mean([r['val_auc'] for r in middle]):.4f}")
    report.append(f"Late layers ({2*total_layers//3}-{total_layers-1}):  Mean AUC = {np.mean([r['val_auc'] for r in late]):.4f}")
    report.append("")

    report.append("=" * 70)

    # Print report
    report_text = "\n".join(report)
    print(report_text)

    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Saved report to {output_path}")

    return report_text

def save_best_probe_info(best: Dict, probes_dir: str, output_path: str):
    """
    Save best probe information to JSON for easy loading later.

    Args:
        best: Best probe dict
        probes_dir: Directory containing probes
        output_path: Path to save JSON
    """
    best_info = {
        "layer": best['layer'],
        "val_auc": best['val_auc'],
        "epoch": best['epoch'],
        "probe_path": os.path.join(probes_dir, f"probe_layer_{best['layer']}.pt"),
        "probes_dir": probes_dir
    }

    with open(output_path, 'w') as f:
        json.dump(best_info, f, indent=2)

    print(f"✓ Saved best probe info to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze trained probes and find best layer"
    )

    parser.add_argument(
        "--probes_dir",
        type=str,
        required=True,
        help="Directory containing trained probes and results (e.g., data/probes/.../mean/)"
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots to file instead of displaying"
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save analysis report to file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and reports (default: same as probes_dir)"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.probes_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PROBE ANALYSIS")
    print("=" * 70)
    print(f"Loading results from: {args.probes_dir}\n")

    # Load results
    try:
        results = load_results(args.probes_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    print(f"✓ Loaded results for {len(results)} layers\n")

    # Generate report
    report_path = os.path.join(output_dir, "analysis_report.txt") if args.save_report else None
    generate_report(results, args.probes_dir, report_path)

    # Plot results
    plot_path = os.path.join(output_dir, "per_layer_analysis.png") if args.save_plots else None
    plot_per_layer_metrics(results, plot_path)

    # Save best probe info
    best = find_best_probe(results, 'val_auc')
    best_info_path = os.path.join(output_dir, "best_probe.json")
    save_best_probe_info(best, args.probes_dir, best_info_path)

    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"Best probe: Layer {best['layer']} (AUC: {best['val_auc']:.4f})")
    print(f"Best probe saved to: {best_info_path}")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    exit(main())
