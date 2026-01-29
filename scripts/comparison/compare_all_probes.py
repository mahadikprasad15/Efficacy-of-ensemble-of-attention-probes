"""
Compare ALL Probe Types for OOD Performance.

Aggregates OOD results from various probe types and creates beautiful visualizations.

Expected result locations (Google Drive):
    - results/ood_evaluation/ood_results_all_pooling.json  (vanilla probes)
    - results/combined_all_pooling/ood_results_all_pooling.json  (combined probes)
    - results/per_token_ood/{dataset}/ood_summary.json  (per-token probes)
    - results/invariant_core_analysis/invariant_core_summary.json  (invariant core)
    - results/probes_layer_agnostic/{model}/{dataset}/{pooling}/results.json  (layer-agnostic)

Usage:
    python scripts/comparison/compare_all_probes.py \
        --results_base /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results \
        --dataset Deception-Roleplaying \
        --output_dir results/all_probes_comparison
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ProbeResult:
    """Standardized probe result for comparison."""
    probe_type: str
    sub_type: str
    layer: int
    ood_auc: float
    source_file: str = ""


# ============================================================================
# Color Scheme - Professional & Soft
# ============================================================================

PROBE_COLORS = {
    "vanilla": "#4A90D9",
    "layer_agnostic": "#E97451",
    "prompted": "#7CB342",
    "per_token": "#AB47BC",
    "combined": "#26A69A",
    "invariant_core": "#EC407A",
    "domain_adversarial": "#FF7043",
}

POOLING_COLORS = {
    "mean": "#4A90D9",
    "max": "#E97451",
    "last": "#7CB342",
    "attn": "#AB47BC",
    "vote": "#26A69A",
}


# ============================================================================
# Discovery Functions - Find OOD Results
# ============================================================================

def discover_ood_results(results_base: str, dataset: str = "Deception-Roleplaying") -> Dict[str, List[str]]:
    """Discover all OOD result files in the results directory."""
    discovered = defaultdict(list)
    
    if not os.path.exists(results_base):
        print(f"⚠ Results base not found: {results_base}")
        return discovered
    
    print(f"🔍 Searching for OOD results in: {results_base}")
    print()
    
    # Pattern 1: ood_evaluation/ood_results_*.json (vanilla probes OOD)
    for f in glob.glob(os.path.join(results_base, "ood_evaluation", "ood_results*.json")):
        discovered["vanilla_ood"].append(f)
        print(f"  ✓ Found vanilla OOD: {os.path.basename(f)}")
    
    # Pattern 2: combined_all_pooling/ood_results*.json (combined probes OOD)
    for f in glob.glob(os.path.join(results_base, "combined_all_pooling", "ood_results*.json")):
        discovered["combined_ood"].append(f)
        print(f"  ✓ Found combined OOD: {os.path.basename(f)}")
    
    # Pattern 3: per_token_ood/{dataset}/ood_summary.json
    pattern3 = os.path.join(results_base, "per_token_ood", dataset, "ood_summary.json")
    if os.path.exists(pattern3):
        discovered["per_token_ood"].append(pattern3)
        print(f"  ✓ Found per-token OOD: ood_summary.json")
    
    # Pattern 4: invariant_core_analysis/invariant_core_summary.json
    pattern4 = os.path.join(results_base, "invariant_core_analysis", "invariant_core_summary.json")
    if os.path.exists(pattern4):
        discovered["invariant_core"].append(pattern4)
        print(f"  ✓ Found invariant core: invariant_core_summary.json")
    
    # Pattern 5: probes_layer_agnostic/**/results.json or layer_results.json
    for f in glob.glob(os.path.join(results_base, "probes_layer_agnostic", "**", "results.json"), recursive=True):
        discovered["layer_agnostic"].append(f)
    for f in glob.glob(os.path.join(results_base, "probes_layer_agnostic", "**", "layer_results.json"), recursive=True):
        discovered["layer_agnostic"].append(f)
    if discovered["layer_agnostic"]:
        print(f"  ✓ Found layer-agnostic: {len(discovered['layer_agnostic'])} files")
    
    # Pattern 6: domain_adversarial/results.json
    pattern6 = os.path.join(results_base, "domain_adversarial", "results.json")
    if os.path.exists(pattern6):
        discovered["domain_adversarial"].append(pattern6)
        print(f"  ✓ Found domain adversarial: results.json")
    
    print()
    return discovered


# ============================================================================
# Result Loaders - Parse Different Formats
# ============================================================================

def load_ood_all_pooling(filepath: str, probe_type: str = "vanilla") -> List[ProbeResult]:
    """
    Load ood_results_all_pooling.json format.
    
    Format:
    {
        "mean": {
            "pooling": "mean",
            "layers": [0, 1, 2, ...],
            "aucs": [0.5, 0.6, ...],
            "best_layer": 27,
            "best_auc": 0.817
        },
        "max": {...}
    }
    """
    results = []
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        for pooling in ["mean", "max", "last", "attn"]:
            if pooling not in data:
                continue
            
            pooling_data = data[pooling]
            
            # Check structure
            if not isinstance(pooling_data, dict):
                continue
            
            # Format with layers + aucs arrays
            if "layers" in pooling_data and "aucs" in pooling_data:
                layers = pooling_data["layers"]
                aucs = pooling_data["aucs"]
                
                for layer, auc in zip(layers, aucs):
                    results.append(ProbeResult(
                        probe_type=probe_type,
                        sub_type=pooling,
                        layer=int(layer),
                        ood_auc=float(auc),
                        source_file=filepath
                    ))
            
            # Format with just best_layer + best_auc (summary)
            elif "best_layer" in pooling_data and "best_auc" in pooling_data:
                results.append(ProbeResult(
                    probe_type=probe_type,
                    sub_type=pooling,
                    layer=int(pooling_data["best_layer"]),
                    ood_auc=float(pooling_data["best_auc"]),
                    source_file=filepath
                ))
        
        print(f"    ✓ Loaded {len(results)} results from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    ✗ Failed to load {filepath}: {e}")
    
    return results


def load_ood_summary(filepath: str) -> List[ProbeResult]:
    """
    Load ood_summary.json format (from per-token probes).
    
    Format:
    {
        "aggregations": {
            "mean": {"best_layer": 20, "best_auc": 0.7},
            "max": {"best_layer": 18, "best_auc": 0.65}
        },
        "overall_best": {"aggregation": "last", "layer": 20, "auc": 0.706}
    }
    """
    results = []
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        # Parse aggregations
        aggregations = data.get("aggregations", {})
        for agg, metrics in aggregations.items():
            if isinstance(metrics, dict) and "best_layer" in metrics:
                results.append(ProbeResult(
                    probe_type="per_token",
                    sub_type=agg,
                    layer=int(metrics["best_layer"]),
                    ood_auc=float(metrics.get("best_auc", metrics.get("auc", 0.5))),
                    source_file=filepath
                ))
        
        # If no aggregations found, try overall_best
        if not results and "overall_best" in data:
            best = data["overall_best"]
            results.append(ProbeResult(
                probe_type="per_token",
                sub_type=best.get("aggregation", "unknown"),
                layer=int(best.get("layer", 0)),
                ood_auc=float(best.get("auc", 0.5)),
                source_file=filepath
            ))
        
        print(f"    ✓ Loaded {len(results)} results from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    ✗ Failed to load {filepath}: {e}")
    
    return results


def load_invariant_core(filepath: str) -> List[ProbeResult]:
    """
    Load invariant_core_summary.json format.
    
    Format:
    {
        "config": {"layer": 24, "pooling": "mean"},
        "evaluation": {
            "Residual (r)": {"domain_a": 0.8, "domain_b": 0.7},
            ...
        },
        "conclusion": {"best_component": "Residual (r)", "best_avg_auc": 0.75}
    }
    """
    results = []
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        config = data.get("config", {})
        layer = config.get("layer", 20)
        
        # Get evaluation results
        evaluation = data.get("evaluation", {})
        for component, metrics in evaluation.items():
            if isinstance(metrics, dict):
                # Average across domains
                auc_a = metrics.get("domain_a", 0.5)
                auc_b = metrics.get("domain_b", 0.5)
                avg_auc = (auc_a + auc_b) / 2
                
                # Clean up component name
                clean_name = component.replace(" ", "_").replace("(", "").replace(")", "").lower()
                
                results.append(ProbeResult(
                    probe_type="invariant_core",
                    sub_type=clean_name,
                    layer=int(layer),
                    ood_auc=float(avg_auc),
                    source_file=filepath
                ))
        
        # If no evaluation found, use conclusion
        if not results and "conclusion" in data:
            conclusion = data["conclusion"]
            results.append(ProbeResult(
                probe_type="invariant_core",
                sub_type=conclusion.get("best_component", "residual"),
                layer=int(layer),
                ood_auc=float(conclusion.get("best_avg_auc", 0.5)),
                source_file=filepath
            ))
        
        print(f"    ✓ Loaded {len(results)} results from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    ✗ Failed to load {filepath}: {e}")
    
    return results


def load_layer_agnostic_results(filepath: str) -> List[ProbeResult]:
    """
    Load results.json from layer-agnostic training.
    
    Format:
    {
        "ood_per_layer": {
            "0": {"auc": 0.5, "acc": 0.5},
            "1": {"auc": 0.55, "acc": 0.52},
            ...
        }
    }
    """
    results = []
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        # Extract pooling from path
        path_parts = filepath.split(os.sep)
        pooling = "mean"
        for p in path_parts:
            if p in ["mean", "max", "last", "attn"]:
                pooling = p
                break
        
        # Try ood_per_layer
        ood_per_layer = data.get("ood_per_layer", {})
        for layer_str, metrics in ood_per_layer.items():
            try:
                layer = int(layer_str)
                auc = metrics.get("auc", metrics.get("ood_auc", 0.5))
                results.append(ProbeResult(
                    probe_type="layer_agnostic",
                    sub_type=pooling,
                    layer=layer,
                    ood_auc=float(auc),
                    source_file=filepath
                ))
            except ValueError:
                continue
        
        # Try direct layers/aucs format
        if not results and "layers" in data and "aucs" in data:
            for layer, auc in zip(data["layers"], data["aucs"]):
                results.append(ProbeResult(
                    probe_type="layer_agnostic",
                    sub_type=pooling,
                    layer=int(layer),
                    ood_auc=float(auc),
                    source_file=filepath
                ))
        
        print(f"    ✓ Loaded {len(results)} results from {os.path.basename(filepath)} ({pooling})")
    except Exception as e:
        print(f"    ✗ Failed to load {filepath}: {e}")
    
    return results


def load_all_results(discovered: Dict[str, List[str]]) -> List[ProbeResult]:
    """Load all discovered OOD results."""
    all_results = []
    
    print("Loading results...")
    
    # Vanilla OOD
    for f in discovered.get("vanilla_ood", []):
        all_results.extend(load_ood_all_pooling(f, "vanilla"))
    
    # Combined OOD
    for f in discovered.get("combined_ood", []):
        all_results.extend(load_ood_all_pooling(f, "combined"))
    
    # Per-token OOD
    for f in discovered.get("per_token_ood", []):
        all_results.extend(load_ood_summary(f))
    
    # Invariant core
    for f in discovered.get("invariant_core", []):
        all_results.extend(load_invariant_core(f))
    
    # Layer-agnostic
    for f in discovered.get("layer_agnostic", []):
        all_results.extend(load_layer_agnostic_results(f))
    
    # Domain adversarial (uses similar format to vanilla)
    for f in discovered.get("domain_adversarial", []):
        all_results.extend(load_ood_all_pooling(f, "domain_adversarial"))
    
    return all_results


# ============================================================================
# Beautiful Visualizations
# ============================================================================

def setup_style():
    """Apply beautiful, clean matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def get_probe_color(probe_type: str, sub_type: str = None) -> str:
    """Get color for a probe type/subtype."""
    if probe_type in PROBE_COLORS:
        return PROBE_COLORS[probe_type]
    if sub_type in POOLING_COLORS:
        return POOLING_COLORS[sub_type]
    return "#888888"


def plot_beautiful_bars(all_results: List[ProbeResult], output_path: str, title: str = ""):
    """Create a beautiful, modern horizontal bar chart."""
    setup_style()
    
    # Find best layer for each probe type/subtype
    best_by_probe = {}
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        if key not in best_by_probe or r.ood_auc > best_by_probe[key].ood_auc:
            best_by_probe[key] = r
    
    if not best_by_probe:
        print("No data for bar chart!")
        return
    
    # Sort by OOD AUC (best at top)
    sorted_probes = sorted(best_by_probe.items(), key=lambda x: x[1].ood_auc, reverse=True)
    
    n = len(sorted_probes)
    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.55)))
    
    labels = []
    aucs = []
    colors = []
    layers = []
    
    for label, result in sorted_probes:
        probe_type = label.split("/")[0]
        sub_type = label.split("/")[1] if "/" in label else ""
        
        display_label = f"{probe_type.replace('_', ' ').title()} / {sub_type}"
        labels.append(display_label)
        aucs.append(result.ood_auc)
        layers.append(result.layer)
        colors.append(get_probe_color(probe_type, sub_type))
    
    y_pos = np.arange(n)
    
    # Draw bars
    bars = ax.barh(y_pos, aucs, height=0.7, color=colors, edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add score annotations
    for i, (bar, layer, auc) in enumerate(zip(bars, layers, aucs)):
        # Score + layer annotation
        ax.text(auc + 0.01, y_pos[i], f" L{layer}: {auc:.3f}",
               va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')
    
    # Reference lines
    ax.axvline(x=0.5, color='#EF5350', linestyle='--', alpha=0.7, linewidth=2, label='Random (0.5)')
    ax.axvline(x=0.7, color='#66BB6A', linestyle='--', alpha=0.7, linewidth=2, label='Good (0.7)')
    ax.axvline(x=0.8, color='#42A5F5', linestyle='--', alpha=0.7, linewidth=2, label='Strong (0.8)')
    
    # Styling
    ax.set_xlim(0.4, 1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Best OOD AUC", fontsize=12, fontweight='bold')
    ax.set_title(title or "Best OOD Performance per Probe Type", fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()  # Best at top
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    
    # Clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_layerwise_overlay(all_results: List[ProbeResult], output_path: str, title: str = ""):
    """Create a layerwise overlay plot showing all probe types."""
    setup_style()
    
    # Group by (probe_type, sub_type)
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        grouped[key].append(r)
    
    if not grouped:
        print("No data for overlay plot!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    best_overall = None
    
    for label, results in grouped.items():
        # Skip if only one point (summary only)
        if len(results) <= 1:
            continue
        
        results.sort(key=lambda x: x.layer)
        layer_list = [r.layer for r in results]
        auc_list = [r.ood_auc for r in results]
        
        probe_type = label.split("/")[0]
        sub_type = label.split("/")[1] if "/" in label else ""
        color = get_probe_color(probe_type, sub_type)
        
        display_label = f"{probe_type.replace('_', ' ').title()} ({sub_type})"
        ax.plot(layer_list, auc_list, marker='o', linewidth=2.5, markersize=5,
               label=display_label, color=color, alpha=0.85)
        
        # Mark best
        best_idx = np.argmax(auc_list)
        best_auc = auc_list[best_idx]
        best_layer = layer_list[best_idx]
        
        ax.scatter([best_layer], [best_auc], s=120, zorder=5,
                  edgecolors='white', linewidths=2, marker='o', color=color)
        
        # Track overall best
        if best_overall is None or best_auc > best_overall[1]:
            best_overall = (best_layer, best_auc, label)
    
    # Annotate best
    if best_overall:
        ax.annotate(
            f'Best: {best_overall[2].replace("_", " ")}\nLayer {best_overall[0]}: {best_overall[1]:.3f}',
            xy=(best_overall[0], best_overall[1]),
            xytext=(15, 15),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', 
                     edgecolor='#FBC02D', alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#FBC02D', lw=2)
        )
    
    # Reference line
    ax.axhline(y=0.5, color='#EF5350', linestyle='--', alpha=0.6, linewidth=1.5, label='Random')
    
    # Styling
    ax.set_xlabel("Layer", fontsize=12, fontweight='bold')
    ax.set_ylabel("OOD AUC", fontsize=12, fontweight='bold')
    ax.set_title(title or "OOD AUC by Layer - All Probe Types", fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.95)
    ax.set_ylim(0.35, 1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_report(all_results: List[ProbeResult], output_path: str):
    """Generate text summary report."""
    # Find best per probe type
    best_by_probe = {}
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        if key not in best_by_probe or r.ood_auc > best_by_probe[key].ood_auc:
            best_by_probe[key] = r
    
    sorted_probes = sorted(best_by_probe.items(), key=lambda x: x[1].ood_auc, reverse=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("ALL PROBE TYPES - OOD PERFORMANCE COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Probe Type':<35} {'Best Layer':<12} {'OOD AUC':<10}")
    lines.append("-" * 80)
    
    for rank, (label, result) in enumerate(sorted_probes, 1):
        marker = " ⭐" if rank == 1 else ""
        lines.append(f"{rank:<6} {label:<35} {result.layer:<12} {result.ood_auc:<10.4f}{marker}")
    
    lines.append("=" * 80)
    
    if sorted_probes:
        best = sorted_probes[0]
        lines.append(f"🏆 BEST OVERALL: {best[0]}")
        lines.append(f"   Layer {best[1].layer}, OOD AUC: {best[1].ood_auc:.4f}")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare all probe types for OOD performance")
    
    parser.add_argument("--results_base", type=str, required=True,
                       help="Base directory containing all results")
    parser.add_argument("--dataset", type=str, default="Deception-Roleplaying",
                       help="Dataset name for per-token results")
    parser.add_argument("--output_dir", type=str, default="results/all_probes_comparison",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ALL PROBE TYPES COMPARISON")
    print("=" * 80)
    print()
    
    # Step 1: Discover all OOD result files
    discovered = discover_ood_results(args.results_base, args.dataset)
    
    total_files = sum(len(v) for v in discovered.values())
    if total_files == 0:
        print("❌ No OOD result files found!")
        print("\nExpected locations:")
        print(f"  - {args.results_base}/ood_evaluation/ood_results_*.json")
        print(f"  - {args.results_base}/combined_all_pooling/ood_results_*.json")
        print(f"  - {args.results_base}/per_token_ood/{args.dataset}/ood_summary.json")
        print(f"  - {args.results_base}/invariant_core_analysis/invariant_core_summary.json")
        return 1
    
    print(f"📁 Found {total_files} result file(s)")
    print()
    
    # Step 2: Load all results
    all_results = load_all_results(discovered)
    
    if not all_results:
        print("❌ No results could be parsed!")
        return 1
    
    print(f"\n✓ Loaded {len(all_results)} total probe results")
    print()
    
    # Step 3: Generate visualizations
    print("Generating visualizations...")
    print()
    
    # Bar chart
    plot_beautiful_bars(
        all_results,
        os.path.join(args.output_dir, "ood_comparison_bars.png"),
        title=f"Best OOD Performance per Probe Type\n{args.dataset}"
    )
    
    # Layerwise overlay
    plot_layerwise_overlay(
        all_results,
        os.path.join(args.output_dir, "ood_layerwise_overlay.png"),
        title=f"OOD AUC by Layer - All Probe Types\n{args.dataset}"
    )
    
    # Text report
    generate_report(
        all_results,
        os.path.join(args.output_dir, "ood_summary_report.txt")
    )
    
    print()
    print("=" * 80)
    print("✓ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
