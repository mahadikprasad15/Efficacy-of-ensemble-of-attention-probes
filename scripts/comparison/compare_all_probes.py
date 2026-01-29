"""
Compare ALL Probe Types Across Layers for OOD Performance.

This script aggregates and visualizes OOD performance for:
    - Vanilla probes (per-layer, different pooling)
    - Prompted probes (per-suffix, different pooling)
    - Per-token probes
    - Combined probes (trained on multiple domains)
    - Invariant core (residual from probe direction)
    - Layer-agnostic probes

Visualization Strategies:
    1. Heatmap: Probe types (Y) × Layers (X), color = OOD AUC
    2. Summary bar chart: Best OOD layer + AUC per probe type
    3. Line overlay: Group by category for direct comparison

Usage:
    python scripts/comparison/compare_all_probes.py \
        --results_config results_config.json \
        --output_dir results/all_probes_comparison

    # Or with base directories
    python scripts/comparison/compare_all_probes.py \
        --probes_base /content/drive/MyDrive/probes \
        --layer_agnostic_base /content/drive/MyDrive/probes_layer_agnostic \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --output_dir results/all_probes_comparison
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ProbeResult:
    """Standardized probe result for comparison."""
    probe_type: str          # e.g., "vanilla", "prompted", "layer_agnostic"
    sub_type: str            # e.g., "mean", "suffix_deception_yesno"
    layer: int               # Layer index (0-27)
    ood_auc: float           # OOD AUC score
    id_auc: float = 0.0      # In-distribution AUC (if available)
    extra_info: Dict = field(default_factory=dict)


@dataclass
class ProbeCategory:
    """Category of probes for grouping in visualization."""
    name: str
    results: List[ProbeResult]
    color: str


# ============================================================================
# Color Scheme
# ============================================================================

CATEGORY_COLORS = {
    # Vanilla probes
    "vanilla_mean": "#2E86AB",
    "vanilla_max": "#A23B72", 
    "vanilla_last": "#F18F01",
    "vanilla_attn": "#06A77D",
    
    # Layer-agnostic probes
    "layer_agnostic_mean": "#5B8C5A",
    "layer_agnostic_attn": "#3D5A80",
    
    # Prompted probes
    "prompted_yesno": "#E07A5F",
    "prompted_fabricated": "#81B29A",
    "prompted_inconsistency": "#F2CC8F",
    "prompted_strategic_ab": "#3D405B",
    
    # Per-token probes
    "per_token": "#9B5DE5",
    
    # Combined probes
    "combined": "#00BBF9",
    
    # Invariant core
    "invariant_core": "#F15BB5",
}

# ============================================================================
# Result Loaders
# ============================================================================

def load_vanilla_probes(probes_dir: str, model: str, dataset: str, results_base: str = None) -> List[ProbeResult]:
    """
    Load vanilla probe results (layer_results.json format).
    
    Expected structures:
        OLD: {probes_dir}/{model}/{dataset}/{pooling}/layer_results.json
        NEW: {results_base}/probes/{model}/{dataset}/{pooling}/layer_results.json
    """
    results = []
    model_dir = model.replace("/", "_")
    
    for pooling in ["mean", "max", "last", "attn"]:
        results_path = None
        
        # Try multiple path patterns (new structure first)
        search_paths = []
        
        # New structure: results in separate results/ directory
        if results_base:
            search_paths.append(os.path.join(results_base, "probes", model_dir, dataset, pooling, "layer_results.json"))
            search_paths.append(os.path.join(results_base, model_dir, dataset, pooling, "layer_results.json"))
        
        # Old structure: results alongside probes
        search_paths.extend([
            os.path.join(probes_dir, model_dir, dataset, pooling, "layer_results.json"),
            os.path.join(probes_dir, dataset, pooling, "layer_results.json"),
            os.path.join(probes_dir, pooling, "layer_results.json"),
        ])
        
        for path in search_paths:
            if os.path.exists(path):
                results_path = path
                break
        
        if not results_path:
            continue
        
        try:
            with open(results_path) as f:
                layer_data = json.load(f)
            
            for layer_result in layer_data:
                results.append(ProbeResult(
                    probe_type="vanilla",
                    sub_type=pooling,
                    layer=layer_result["layer"],
                    ood_auc=layer_result.get("test_auc", layer_result.get("val_auc", 0.5)),
                    id_auc=layer_result.get("val_auc", 0.5)
                ))
            print(f"  ✓ Loaded vanilla/{pooling}: {len(layer_data)} layers from {results_path}")
        except Exception as e:
            print(f"  ✗ Failed to load vanilla/{pooling}: {e}")
    
    return results



def load_layer_agnostic_probes(probes_dir: str, model: str, dataset: str, results_base: str = None) -> List[ProbeResult]:
    """
    Load layer-agnostic probe results (results.json format).
    
    Expected structures:
        OLD: {probes_dir}/{model}/{dataset}/{pooling}/results.json
        NEW: {results_base}/probes_layer_agnostic/{model}/{dataset}/{pooling}/results.json
    """
    results = []
    model_dir = model.replace("/", "_")
    
    for pooling in ["mean", "max", "last", "attn"]:
        results_path = None
        
        # Try multiple path patterns
        search_paths = []
        
        # New structure
        if results_base:
            search_paths.append(os.path.join(results_base, "probes_layer_agnostic", model_dir, dataset, pooling, "results.json"))
            search_paths.append(os.path.join(results_base, model_dir, dataset, pooling, "results.json"))
        
        # Old/direct structure
        search_paths.extend([
            os.path.join(probes_dir, model_dir, dataset, pooling, "results.json"),
            os.path.join(probes_dir, dataset, pooling, "results.json"),
            os.path.join(probes_dir, pooling, "results.json"),
        ])
        
        for path in search_paths:
            if os.path.exists(path):
                results_path = path
                break
        
        if not results_path:
            continue
        
        try:
            with open(results_path) as f:
                data = json.load(f)
            
            ood_per_layer = data.get("ood_per_layer", {})
            id_per_layer = data.get("id_per_layer", {})
            
            for layer_str, metrics in ood_per_layer.items():
                layer = int(layer_str)
                results.append(ProbeResult(
                    probe_type="layer_agnostic",
                    sub_type=pooling,
                    layer=layer,
                    ood_auc=metrics.get("auc", 0.5),
                    id_auc=id_per_layer.get(layer_str, {}).get("auc", 0.5)
                ))
            
            print(f"  ✓ Loaded layer_agnostic/{pooling}: {len(ood_per_layer)} layers from {results_path}")
        except Exception as e:
            print(f"  ✗ Failed to load layer_agnostic/{pooling}: {e}")
    
    return results


def load_prompted_probes(probes_dir: str, model: str, dataset: str) -> List[ProbeResult]:
    """
    Load prompted probe results.
    
    Expected structure:
        {probes_dir}/{model}/{dataset}/{suffix}/layer_results.json
    """
    results = []
    model_dir = model.replace("/", "_")
    base_path = os.path.join(probes_dir, model_dir, dataset)
    
    if not os.path.exists(base_path):
        return results
    
    suffixes = [d for d in os.listdir(base_path) if d.startswith("suffix_")]
    
    for suffix in suffixes:
        results_path = os.path.join(base_path, suffix, "layer_results.json")
        
        if not os.path.exists(results_path):
            continue
        
        try:
            with open(results_path) as f:
                layer_data = json.load(f)
            
            for layer_result in layer_data:
                results.append(ProbeResult(
                    probe_type="prompted",
                    sub_type=suffix,
                    layer=layer_result["layer"],
                    ood_auc=layer_result.get("test_auc", layer_result.get("val_auc", 0.5)),
                    id_auc=layer_result.get("val_auc", 0.5)
                ))
            print(f"  ✓ Loaded prompted/{suffix}: {len(layer_data)} layers")
        except Exception as e:
            print(f"  ✗ Failed to load prompted/{suffix}: {e}")
    
    return results


def load_per_token_probes(probes_dir: str, model: str, dataset: str) -> List[ProbeResult]:
    """
    Load per-token probe results.
    """
    results = []
    model_dir = model.replace("/", "_")
    results_path = os.path.join(probes_dir, model_dir, dataset, "ood_summary.json")
    
    if not os.path.exists(results_path):
        # Try layer_results.json
        results_path = os.path.join(probes_dir, model_dir, dataset, "layer_results.json")
    
    if not os.path.exists(results_path):
        return results
    
    try:
        with open(results_path) as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            for layer_result in data:
                results.append(ProbeResult(
                    probe_type="per_token",
                    sub_type="per_token",
                    layer=layer_result["layer"],
                    ood_auc=layer_result.get("test_auc", layer_result.get("val_auc", 0.5)),
                    id_auc=layer_result.get("val_auc", 0.5)
                ))
        elif isinstance(data, dict) and "per_layer_results" in data:
            for layer_str, metrics in data["per_layer_results"].items():
                results.append(ProbeResult(
                    probe_type="per_token",
                    sub_type="per_token",
                    layer=int(layer_str),
                    ood_auc=metrics.get("ood_auc", metrics.get("auc", 0.5)),
                    id_auc=metrics.get("id_auc", 0.5)
                ))
        
        print(f"  ✓ Loaded per_token: {len(results)} layers")
    except Exception as e:
        print(f"  ✗ Failed to load per_token: {e}")
    
    return results


def load_combined_probes(probes_dir: str, model: str) -> List[ProbeResult]:
    """
    Load combined probe results (trained on multiple domains).
    """
    results = []
    model_dir = model.replace("/", "_")
    
    # Look for combined dataset directories
    base_path = os.path.join(probes_dir, model_dir)
    if not os.path.exists(base_path):
        return results
    
    combined_dirs = [d for d in os.listdir(base_path) if "combined" in d.lower() or "+" in d]
    
    for combined_dir in combined_dirs:
        for pooling in ["mean", "max", "last", "attn"]:
            results_path = os.path.join(base_path, combined_dir, pooling, "layer_results.json")
            
            if not os.path.exists(results_path):
                continue
            
            try:
                with open(results_path) as f:
                    layer_data = json.load(f)
                
                for layer_result in layer_data:
                    results.append(ProbeResult(
                        probe_type="combined",
                        sub_type=f"{combined_dir}_{pooling}",
                        layer=layer_result["layer"],
                        ood_auc=layer_result.get("test_auc", layer_result.get("val_auc", 0.5)),
                        id_auc=layer_result.get("val_auc", 0.5)
                    ))
                print(f"  ✓ Loaded combined/{combined_dir}/{pooling}: {len(layer_data)} layers")
            except Exception as e:
                print(f"  ✗ Failed to load combined/{combined_dir}/{pooling}: {e}")
    
    return results


def load_invariant_core_results(results_dir: str) -> List[ProbeResult]:
    """
    Load invariant core (residual) analysis results.
    """
    results = []
    results_path = os.path.join(results_dir, "invariant_core_results.json")
    
    if not os.path.exists(results_path):
        return results
    
    try:
        with open(results_path) as f:
            data = json.load(f)
        
        for layer_str, metrics in data.get("per_layer", {}).items():
            results.append(ProbeResult(
                probe_type="invariant_core",
                sub_type="residual",
                layer=int(layer_str),
                ood_auc=metrics.get("ood_auc", 0.5),
                id_auc=metrics.get("id_auc", 0.5)
            ))
        
        print(f"  ✓ Loaded invariant_core: {len(results)} layers")
    except Exception as e:
        print(f"  ✗ Failed to load invariant_core: {e}")
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_heatmap(all_results: List[ProbeResult], output_path: str, title: str = ""):
    """
    Create heatmap: Probe types (Y) × Layers (X), color = OOD AUC.
    """
    # Group results by (probe_type, sub_type)
    grouped = defaultdict(dict)
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        grouped[key][r.layer] = r.ood_auc
    
    if not grouped:
        print("No data for heatmap!")
        return
    
    # Create matrix
    probe_labels = sorted(grouped.keys())
    n_layers = max(max(layers.keys()) for layers in grouped.values()) + 1
    
    matrix = np.zeros((len(probe_labels), n_layers))
    matrix[:] = np.nan  # Use nan for missing values
    
    for i, probe_label in enumerate(probe_labels):
        for layer, auc in grouped[probe_label].items():
            matrix[i, layer] = auc
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, max(6, len(probe_labels) * 0.5)))
    
    # Custom colormap: red (bad) -> yellow (medium) -> green (good)
    cmap = plt.cm.RdYlGn
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.5, vmax=1.0)
    
    # Axis labels
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers), fontsize=8)
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels(probe_labels, fontsize=10)
    
    ax.set_xlabel("Layer", fontsize=12, fontweight='bold')
    ax.set_ylabel("Probe Type", fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("OOD AUC", fontsize=11)
    
    # Title
    ax.set_title(title or "OOD AUC Heatmap: All Probe Types × Layers", fontsize=14, fontweight='bold')
    
    # Add text annotations for high/low values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                if val >= 0.8 or val <= 0.55:
                    color = 'white' if val >= 0.75 or val <= 0.55 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           fontsize=6, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved heatmap: {output_path}")


def plot_summary_bars(all_results: List[ProbeResult], output_path: str, title: str = ""):
    """
    Bar chart showing best OOD AUC per probe type/sub_type.
    """
    # Find best layer for each probe type
    best_by_probe = {}
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        if key not in best_by_probe or r.ood_auc > best_by_probe[key].ood_auc:
            best_by_probe[key] = r
    
    if not best_by_probe:
        print("No data for summary bars!")
        return
    
    # Sort by OOD AUC
    sorted_probes = sorted(best_by_probe.items(), key=lambda x: x[1].ood_auc, reverse=True)
    
    labels = [p[0] for p in sorted_probes]
    aucs = [p[1].ood_auc for p in sorted_probes]
    layers = [p[1].layer for p in sorted_probes]
    
    # Create colors based on probe type
    colors = []
    for label in labels:
        probe_type = label.split("/")[0]
        sub_type = label.split("/")[1]
        color_key = f"{probe_type}_{sub_type}"
        colors.append(CATEGORY_COLORS.get(color_key, "#666666"))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.4)))
    
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add layer annotations
    for i, (bar, layer, auc) in enumerate(zip(bars, layers, aucs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'L{layer} ({auc:.3f})', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Best OOD AUC", fontsize=12, fontweight='bold')
    ax.set_title(title or "Best OOD Performance per Probe Type", fontsize=14, fontweight='bold')
    
    # Reference lines
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.axvline(x=0.7, color='green', linestyle=':', alpha=0.5, label='Good (0.7)')
    ax.axvline(x=0.8, color='blue', linestyle=':', alpha=0.5, label='Strong (0.8)')
    
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc='lower right')
    ax.invert_yaxis()  # Top is best
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved summary bars: {output_path}")


def plot_layerwise_by_category(all_results: List[ProbeResult], output_path: str, title: str = ""):
    """
    Line plots grouped by probe category, showing layer profiles.
    """
    # Group by probe_type
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r.probe_type].append(r)
    
    n_types = len(by_type)
    if n_types == 0:
        print("No data for layerwise plot!")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5), sharey=True)
    if n_types == 1:
        axes = [axes]
    
    for ax, (probe_type, results) in zip(axes, by_type.items()):
        # Group by sub_type
        by_subtype = defaultdict(list)
        for r in results:
            by_subtype[r.sub_type].append(r)
        
        for sub_type, sub_results in by_subtype.items():
            # Sort by layer
            sub_results.sort(key=lambda x: x.layer)
            layers = [r.layer for r in sub_results]
            aucs = [r.ood_auc for r in sub_results]
            
            color_key = f"{probe_type}_{sub_type}"
            color = CATEGORY_COLORS.get(color_key, None)
            
            ax.plot(layers, aucs, marker='o', linewidth=2, markersize=4,
                   label=sub_type, color=color, alpha=0.8)
            
            # Mark best
            best_idx = np.argmax(aucs)
            ax.scatter([layers[best_idx]], [aucs[best_idx]], s=100, zorder=5,
                      edgecolors='black', linewidths=2, marker='*', color=color)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
        ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.4)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("OOD AUC" if ax == axes[0] else "", fontsize=11)
        ax.set_title(probe_type.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.0)
    
    fig.suptitle(title or "OOD AUC by Layer - Grouped by Probe Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved layerwise by category: {output_path}")


def plot_overlay_comparison(all_results: List[ProbeResult], output_path: str, 
                           probe_types: List[str] = None, title: str = ""):
    """
    Single plot overlaying selected probe types for direct comparison.
    """
    # Filter to selected types
    if probe_types:
        results = [r for r in all_results if r.probe_type in probe_types]
    else:
        results = all_results
    
    # Group by (probe_type, sub_type)
    grouped = defaultdict(list)
    for r in results:
        key = f"{r.probe_type}/{r.sub_type}"
        grouped[key].append(r)
    
    if not grouped:
        print("No data for overlay plot!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for label, group_results in grouped.items():
        group_results.sort(key=lambda x: x.layer)
        layers = [r.layer for r in group_results]
        aucs = [r.ood_auc for r in group_results]
        
        probe_type = label.split("/")[0]
        sub_type = label.split("/")[1]
        color_key = f"{probe_type}_{sub_type}"
        color = CATEGORY_COLORS.get(color_key, None)
        
        ax.plot(layers, aucs, marker='o', linewidth=2, markersize=5,
               label=label, color=color, alpha=0.8)
        
        # Mark best
        best_idx = np.argmax(aucs)
        ax.scatter([layers[best_idx]], [aucs[best_idx]], s=150, zorder=5,
                  edgecolors='black', linewidths=2, marker='*', color=color)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random')
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("Layer", fontsize=12, fontweight='bold')
    ax.set_ylabel("OOD AUC", fontsize=12, fontweight='bold')
    ax.set_title(title or "OOD AUC Comparison - All Probe Types", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overlay comparison: {output_path}")


def generate_summary_report(all_results: List[ProbeResult], output_path: str):
    """
    Generate text summary of all probe comparisons.
    """
    # Find best per probe type
    best_by_type = {}
    for r in all_results:
        key = f"{r.probe_type}/{r.sub_type}"
        if key not in best_by_type or r.ood_auc > best_by_type[key].ood_auc:
            best_by_type[key] = r
    
    # Sort by OOD AUC
    sorted_probes = sorted(best_by_type.items(), key=lambda x: x[1].ood_auc, reverse=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("ALL PROBE TYPES COMPARISON - OOD PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Probe Type':<35} {'Best Layer':<12} {'OOD AUC':<10} {'ID AUC':<10}")
    lines.append("-" * 80)
    
    for rank, (label, result) in enumerate(sorted_probes, 1):
        lines.append(
            f"{rank:<6} {label:<35} {result.layer:<12} "
            f"{result.ood_auc:<10.4f} {result.id_auc:<10.4f}"
        )
    
    lines.append("=" * 80)
    
    # Best overall
    if sorted_probes:
        best = sorted_probes[0]
        lines.append(f"🏆 BEST OVERALL: {best[0]}")
        lines.append(f"   Layer {best[1].layer}, OOD AUC: {best[1].ood_auc:.4f}")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Saved summary report: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare all probe types for OOD performance")
    
    # Base directories
    parser.add_argument("--probes_base", type=str, help="Base directory for vanilla probes")
    parser.add_argument("--layer_agnostic_base", type=str, help="Base directory for layer-agnostic probes")
    parser.add_argument("--prompted_base", type=str, help="Base directory for prompted probes")
    parser.add_argument("--per_token_base", type=str, help="Base directory for per-token probes")
    parser.add_argument("--combined_base", type=str, help="Base directory for combined probes")
    parser.add_argument("--invariant_core_results", type=str, help="Path to invariant core results")
    
    parser.add_argument("--results_base", type=str, help="Base directory for results (new structure: results/probes_*/...)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="Deception-Roleplaying")
    parser.add_argument("--output_dir", type=str, default="results/all_probes_comparison")
    
    # Or load from config
    parser.add_argument("--results_config", type=str, help="JSON config with result paths")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ALL PROBE TYPES COMPARISON")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Load from config if provided
    if args.results_config and os.path.exists(args.results_config):
        with open(args.results_config) as f:
            config = json.load(f)
        
        if "vanilla" in config:
            all_results.extend(load_vanilla_probes(config["vanilla"], args.model, args.dataset))
        if "layer_agnostic" in config:
            all_results.extend(load_layer_agnostic_probes(config["layer_agnostic"], args.model, args.dataset))
        if "prompted" in config:
            all_results.extend(load_prompted_probes(config["prompted"], args.model, args.dataset))
        if "per_token" in config:
            all_results.extend(load_per_token_probes(config["per_token"], args.model, args.dataset))
        if "combined" in config:
            all_results.extend(load_combined_probes(config["combined"], args.model))
        if "invariant_core" in config:
            all_results.extend(load_invariant_core_results(config["invariant_core"]))
    
    else:
        # Load from individual base directories
        results_base = args.results_base
        
        if args.probes_base:
            print("Loading vanilla probes...")
            all_results.extend(load_vanilla_probes(args.probes_base, args.model, args.dataset, results_base))
        
        if args.layer_agnostic_base:
            print("Loading layer-agnostic probes...")
            all_results.extend(load_layer_agnostic_probes(args.layer_agnostic_base, args.model, args.dataset, results_base))
        
        if args.prompted_base:
            print("Loading prompted probes...")
            all_results.extend(load_prompted_probes(args.prompted_base, args.model, args.dataset))
        
        if args.per_token_base:
            print("Loading per-token probes...")
            all_results.extend(load_per_token_probes(args.per_token_base, args.model, args.dataset))
        
        if args.combined_base:
            print("Loading combined probes...")
            all_results.extend(load_combined_probes(args.combined_base, args.model))
        
        if args.invariant_core_results:
            print("Loading invariant core results...")
            all_results.extend(load_invariant_core_results(args.invariant_core_results))
    
    if not all_results:
        print("\n❌ No results loaded! Check paths and try again.")
        print("\nExample usage (new structure with separate results dir):")
        print("  python scripts/comparison/compare_all_probes.py \\")
        print("      --probes_base /content/drive/MyDrive/data/probes \\")
        print("      --layer_agnostic_base /content/drive/MyDrive/data/probes_layer_agnostic \\")
        print("      --results_base /content/drive/MyDrive/results \\")
        print("      --model meta-llama/Llama-3.2-3B-Instruct")
        print("\nExample usage (old structure with results in probe dirs):")
        print("  python scripts/comparison/compare_all_probes.py \\")
        print("      --probes_base /content/drive/MyDrive/probes \\")
        print("      --model meta-llama/Llama-3.2-3B-Instruct")
        return 1
    
    print(f"\n✓ Loaded {len(all_results)} total probe results")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    # 1. Heatmap
    plot_heatmap(
        all_results,
        os.path.join(args.output_dir, "heatmap_all_probes.png"),
        title=f"OOD AUC: All Probe Types × Layers\n{args.dataset}"
    )
    
    # 2. Summary bars
    plot_summary_bars(
        all_results,
        os.path.join(args.output_dir, "summary_bars.png"),
        title=f"Best OOD Performance per Probe Type\n{args.dataset}"
    )
    
    # 3. Layerwise by category
    plot_layerwise_by_category(
        all_results,
        os.path.join(args.output_dir, "layerwise_by_category.png"),
        title=f"OOD AUC by Layer - {args.dataset}"
    )
    
    # 4. Overlay comparison
    plot_overlay_comparison(
        all_results,
        os.path.join(args.output_dir, "overlay_all.png"),
        title=f"OOD AUC Comparison - All Probe Types\n{args.dataset}"
    )
    
    # 5. Summary report
    generate_summary_report(
        all_results,
        os.path.join(args.output_dir, "summary_report.txt")
    )
    
    print()
    print("=" * 80)
    print("✓ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
