"""
Generate plots from experimental results.
1. Layer-wise AUC curve (Single OR Compare Pooling Types).
2. Layer Selection Histogram.

Usage:
    python scripts/make_plots.py --mode layer_curve --input experiments/lodo_Movies/results.jsonl --output plots/auc.pdf
    python scripts/make_plots.py --mode comparison --inputs experiments/lodo_Movies_mean/results.jsonl experiments/lodo_Movies_max/results.jsonl --labels Mean Max --output comparison.pdf
"""

import argparse
import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(fpath):
    data = []
    with open(fpath) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data).sort_values("layer")

def plot_layer_auc(results_file, output_path):
    df = load_results(results_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['min_val_auc'], label="Min ID Val AUC", marker='o')
    if 'ood_auc' in df.columns:
        plt.plot(df['layer'], df['ood_auc'], label="OOD AUC", marker='x', linestyle='--')
    plt.xlabel("Layer Index")
    plt.ylabel("AUC")
    plt.title("Probe Performance per Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def plot_comparison(input_files, labels, output_path):
    plt.figure(figsize=(10, 6))
    
    for fpath, label in zip(input_files, labels):
        try:
            df = load_results(fpath)
            # Plot OOD AUC if available, else Min Val
            # Ideally compare OOD performance
            metric = 'ood_auc' if 'ood_auc' in df.columns else 'min_val_auc'
            plt.plot(df['layer'], df[metric], label=f"{label} ({metric})", marker='o', alpha=0.8)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    plt.xlabel("Layer Index")
    plt.ylabel("AUC")
    plt.title("Pooling Strategy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Saved comparison to {output_path}")

def plot_selection_histogram(experiments_dir, output_path):
    patterns = os.path.join(experiments_dir, "lodo_*", "results.jsonl")
    files = glob.glob(patterns)
    selected_layers = []
    
    for fpath in files:
        best_layer = -1
        best_auc = -1
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if r['min_val_auc'] > best_auc:
                    best_auc = r['min_val_auc']
                    best_layer = r['layer']
        if best_layer != -1:
            selected_layers.append(best_layer)

    if not selected_layers:
        print("No results found.")
        return

    plt.figure(figsize=(8, 6))
    sns.histplot(selected_layers, bins=range(min(selected_layers), max(selected_layers)+2), kde=False)
    plt.xlabel("Layer Index")
    plt.ylabel("Count")
    plt.title("Selected Layers across LODO Splits")
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["layer_curve", "comparison", "histogram"], required=True)
    parser.add_argument("--input", type=str)
    parser.add_argument("--inputs", nargs="+", help="Multiple input files for comparison")
    parser.add_argument("--labels", nargs="+", help="Labels for comparison")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "layer_curve":
        plot_layer_auc(args.input, args.output)
    elif args.mode == "comparison":
        plot_comparison(args.inputs, args.labels, args.output)
    elif args.mode == "histogram":
        plot_selection_histogram(args.input, args.output)

if __name__ == "__main__":
    main()
