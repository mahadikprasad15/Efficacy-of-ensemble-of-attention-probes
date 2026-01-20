"""
Comprehensive ensemble evaluation with K-sweeps and strategy comparison.

This script:
1. Loads saved logits from per-layer probes (validation + OOD)
2. Sweeps through different K% values (top-K layers selection)
3. Evaluates 3 ensemble strategies: Mean, Weighted, Gated
4. Generates comprehensive comparison plots
5. Saves results for each pooling strategy

Key Features:
- Works with pre-computed logits (no need to reload probes)
- Evaluates on both validation and OOD datasets
- Trains GatedEnsemble for each K% value
- Creates per-strategy and cross-strategy comparison charts
- Identifies optimal K% for each ensemble type

Usage:
    # Evaluate on validation data
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling mean \
        --val_activations_dir /content/drive/MyDrive/activations/validation \
        --probes_dir /content/drive/MyDrive/probes/mean \
        --output_dir /content/drive/MyDrive/results/ensembles/mean

    # Evaluate on OOD data (requires prior OOD evaluation)
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling mean \
        --ood_logits_path /path/to/ood_results/logits/mean_logits.npy \
        --ood_labels_path /path/to/ood_labels.npy \
        --output_dir results/ensembles/mean_ood \
        --eval_mode ood

    # Full evaluation (validation + OOD)
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling mean \
        --val_activations_dir /path/to/val \
        --probes_dir /path/to/probes/mean \
        --ood_logits_path /path/to/ood_logits.npy \
        --ood_labels_path /path/to/ood_labels.npy \
        --output_dir results/ensembles/mean \
        --k_values 10,20,30,40,50,60,70,80,90

Expected directory structure:
    val_activations_dir/
    ├── shard_0.safetensors
    └── manifest.jsonl

    probes_dir/
    ├── probe_layer_0.pt
    ├── probe_layer_1.pt
    └── layer_results.json

Output:
    output_dir/
    ├── ensemble_k_sweep_results.json      # K-sweep results
    ├── ensemble_comparison_val.png        # Validation comparison
    ├── ensemble_comparison_ood.png        # OOD comparison (if available)
    ├── gated_models/                      # Trained gated ensembles
    │   ├── gated_k10.pt
    │   ├── gated_k20.pt
    │   └── ...
    └── summary.txt                        # Text summary
"""

import argparse
import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import StaticMeanEnsemble, StaticWeightedEnsemble, GatedEnsemble

# Constants
ENSEMBLE_COLORS = {
    'mean': '#3498db',      # Blue
    'weighted': '#e74c3c',  # Red
    'gated': '#2ecc71'      # Green
}


class CachedDataset(Dataset):
    """Load cached activations."""
    def __init__(self, activations_dir: str):
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))

        if not shards:
            raise ValueError(f"No shards found in {activations_dir}")

        print(f"Loading {len(shards)} shard(s)...")
        self.samples = []

        for shard_path in tqdm(shards, desc="Loading"):
            shard_data = load_file(shard_path)
            for key in shard_data.keys():
                if key.endswith("_activations"):
                    label_key = key.replace("_activations", "_label")
                    if label_key in shard_data:
                        self.samples.append((shard_data[key], shard_data[label_key].item()))

        print(f"✓ Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_logits_from_probes(
    probes_dir: str,
    dataloader: DataLoader,
    pooling_type: str,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-layer logits from all probes.

    Returns:
        logits: (N, L) array
        labels: (N,) array
    """
    probe_files = sorted(glob.glob(os.path.join(probes_dir, "probe_layer_*.pt")))

    if not probe_files:
        raise ValueError(f"No probe files in {probes_dir}")

    # Get dimensions
    sample_x, _ = next(iter(dataloader))
    _, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]

    print(f"Extracting logits from {len(probe_files)} layers...")

    all_layer_logits = []
    labels = None

    for probe_file in tqdm(probe_files, desc="Extracting logits"):
        layer_idx = int(probe_file.split('_')[-1].replace('.pt', ''))

        probe = LayerProbe(input_dim=D, pooling_type=pooling_type).to(device)
        probe.load_state_dict(torch.load(probe_file, map_location=device))
        probe.eval()

        layer_logits = []
        layer_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                x_layer = x[:, layer_idx, :, :]
                logits = probe(x_layer).cpu().numpy().flatten()
                layer_logits.extend(logits)
                layer_labels.extend(y.cpu().numpy())

        all_layer_logits.append(np.array(layer_logits))

        if labels is None:
            labels = np.array(layer_labels)

    logits_array = np.array(all_layer_logits).T  # (N, L)
    print(f"✓ Extracted logits: {logits_array.shape}")

    return logits_array, labels


def load_layer_results(probes_dir: str) -> Optional[Dict]:
    """Load layer_results.json for AUC-based layer selection."""
    results_path = os.path.join(probes_dir, "layer_results.json")

    if not os.path.exists(results_path):
        print(f"⚠️  layer_results.json not found in {probes_dir}")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def select_top_k_layers(layer_results: List[Dict], k_pct: int) -> List[int]:
    """Select top K% layers by validation AUC."""
    sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
    k_layers = max(1, int(len(sorted_layers) * k_pct / 100))
    selected = sorted_layers[:k_layers]
    return sorted([l['layer'] for l in selected])


def evaluate_static_mean(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Evaluate StaticMeanEnsemble."""
    ensemble = StaticMeanEnsemble()
    logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)

    with torch.no_grad():
        ensemble_logits = ensemble(logits_tensor).numpy().flatten()

    probs = 1 / (1 + np.exp(-ensemble_logits))  # Sigmoid
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))

    return auc, acc


def evaluate_static_weighted(
    logits: np.ndarray,
    labels: np.ndarray,
    layer_results: List[Dict],
    selected_layers: List[int]
) -> Tuple[float, float]:
    """Evaluate StaticWeightedEnsemble."""
    # Get AUCs for selected layers
    layer_auc_map = {r['layer']: r['val_auc'] for r in layer_results}
    aucs = np.array([layer_auc_map[l] for l in selected_layers])

    # Normalize to weights
    weights = torch.tensor(aucs / aucs.sum(), dtype=torch.float32)

    ensemble = StaticWeightedEnsemble(weights=weights)
    logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)

    with torch.no_grad():
        ensemble_logits = ensemble(logits_tensor).numpy().flatten()

    probs = 1 / (1 + np.exp(-ensemble_logits))
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))

    return auc, acc


def train_and_evaluate_gated(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    num_layers: int,
    device: torch.device,
    epochs: int = 20,
    patience: int = 3,
    save_path: Optional[str] = None
) -> Tuple[float, float]:
    """Train and evaluate GatedEnsemble."""
    ensemble = GatedEnsemble(input_dim=num_layers, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(ensemble.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare data
    train_logits_tensor = torch.tensor(train_logits, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(train_logits_tensor, train_labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        ensemble.train()
        epoch_loss = 0

        for batch_logits, batch_labels in loader:
            batch_logits = batch_logits.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            # Use logits as both features and logits
            output = ensemble(batch_logits, batch_logits.unsqueeze(-1))
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if save_path:
                torch.save(ensemble.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on test
    ensemble.eval()
    with torch.no_grad():
        test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32).to(device)
        ensemble_logits = ensemble(test_logits_tensor, test_logits_tensor.unsqueeze(-1)).cpu().numpy().flatten()

    probs = 1 / (1 + np.exp(-ensemble_logits))
    auc = roc_auc_score(test_labels, probs)
    acc = accuracy_score(test_labels, (probs > 0.5).astype(int))

    return auc, acc


def evaluate_k_sweep(
    logits: np.ndarray,
    labels: np.ndarray,
    layer_results: List[Dict],
    k_values: List[int],
    device: torch.device,
    gated_save_dir: Optional[str] = None,
    eval_name: str = "validation"
) -> List[Dict]:
    """
    Sweep through K% values and evaluate all ensemble strategies.

    Args:
        logits: (N, L) logits from all layers
        labels: (N,) labels
        layer_results: List of per-layer results for selection
        k_values: List of K% values to try
        device: torch device
        gated_save_dir: Directory to save trained gated models
        eval_name: Name of evaluation set

    Returns:
        List of dicts with results for each K%
    """
    if gated_save_dir:
        os.makedirs(gated_save_dir, exist_ok=True)

    results = []

    for k_pct in tqdm(k_values, desc=f"K% sweep ({eval_name})"):
        # Select top-K layers
        selected_layers = select_top_k_layers(layer_results, k_pct)
        logits_k = logits[:, selected_layers]

        # StaticMean
        auc_mean, acc_mean = evaluate_static_mean(logits_k, labels)

        # StaticWeighted
        auc_weighted, acc_weighted = evaluate_static_weighted(
            logits_k, labels, layer_results, selected_layers
        )

        # Gated (train on 80% of data, test on 20%)
        n_train = int(0.8 * len(labels))
        train_logits, test_logits = logits_k[:n_train], logits_k[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]

        gated_save_path = None
        if gated_save_dir:
            gated_save_path = os.path.join(gated_save_dir, f"gated_k{k_pct}.pt")

        auc_gated, acc_gated = train_and_evaluate_gated(
            train_logits, train_labels,
            test_logits, test_labels,
            num_layers=len(selected_layers),
            device=device,
            save_path=gated_save_path
        )

        results.append({
            'k_pct': k_pct,
            'num_layers': len(selected_layers),
            'selected_layers': selected_layers,
            'mean': {'auc': float(auc_mean), 'acc': float(acc_mean)},
            'weighted': {'auc': float(auc_weighted), 'acc': float(acc_weighted)},
            'gated': {'auc': float(auc_gated), 'acc': float(acc_gated)}
        })

    return results


def plot_ensemble_comparison(
    results: List[Dict],
    save_path: str,
    pooling_name: str,
    eval_name: str = "Validation"
):
    """Plot ensemble strategy comparison across K% values."""
    k_values = [r['k_pct'] for r in results]

    mean_aucs = [r['mean']['auc'] for r in results]
    weighted_aucs = [r['weighted']['auc'] for r in results]
    gated_aucs = [r['gated']['auc'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # AUC plot
    ax1.plot(k_values, mean_aucs, marker='o', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['mean'], label='Mean', alpha=0.85)
    ax1.plot(k_values, weighted_aucs, marker='s', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['weighted'], label='Weighted', alpha=0.85)
    ax1.plot(k_values, gated_aucs, marker='^', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['gated'], label='Gated', alpha=0.85)

    # Mark best for each strategy
    best_mean_idx = np.argmax(mean_aucs)
    best_weighted_idx = np.argmax(weighted_aucs)
    best_gated_idx = np.argmax(gated_aucs)

    ax1.scatter([k_values[best_mean_idx]], [mean_aucs[best_mean_idx]],
               color=ENSEMBLE_COLORS['mean'], s=250, zorder=5, edgecolors='black',
               linewidths=2.5, marker='*')
    ax1.scatter([k_values[best_weighted_idx]], [weighted_aucs[best_weighted_idx]],
               color=ENSEMBLE_COLORS['weighted'], s=250, zorder=5, edgecolors='black',
               linewidths=2.5, marker='*')
    ax1.scatter([k_values[best_gated_idx]], [gated_aucs[best_gated_idx]],
               color=ENSEMBLE_COLORS['gated'], s=250, zorder=5, edgecolors='black',
               linewidths=2.5, marker='*')

    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax1.set_xlabel('Top-K% Layers', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'{eval_name} AUC', fontsize=13, fontweight='bold')
    ax1.set_title(f'Ensemble Comparison: {pooling_name.upper()}\n{eval_name} AUC vs K%',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.45, 1.0)

    # Accuracy plot
    mean_accs = [r['mean']['acc'] for r in results]
    weighted_accs = [r['weighted']['acc'] for r in results]
    gated_accs = [r['gated']['acc'] for r in results]

    ax2.plot(k_values, mean_accs, marker='o', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['mean'], label='Mean', alpha=0.85)
    ax2.plot(k_values, weighted_accs, marker='s', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['weighted'], label='Weighted', alpha=0.85)
    ax2.plot(k_values, gated_accs, marker='^', linewidth=2.5, markersize=8,
            color=ENSEMBLE_COLORS['gated'], label='Gated', alpha=0.85)

    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax2.set_xlabel('Top-K% Layers', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'{eval_name} Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Ensemble Comparison: {pooling_name.upper()}\n{eval_name} Accuracy vs K%',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ensemble comparison: {save_path}")
    plt.close()


def generate_summary(results: List[Dict], pooling_name: str, eval_name: str) -> str:
    """Generate text summary of ensemble evaluation."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"ENSEMBLE EVALUATION: {pooling_name.upper()} - {eval_name}")
    lines.append("=" * 90)
    lines.append("")

    # Find best for each strategy
    best_mean = max(results, key=lambda x: x['mean']['auc'])
    best_weighted = max(results, key=lambda x: x['weighted']['auc'])
    best_gated = max(results, key=lambda x: x['gated']['auc'])

    lines.append("Best Results per Strategy:")
    lines.append("-" * 90)
    lines.append(f"Mean Ensemble:     K={best_mean['k_pct']:3d}% ({best_mean['num_layers']:2d} layers) | AUC={best_mean['mean']['auc']:.4f} | Acc={best_mean['mean']['acc']:.4f}")
    lines.append(f"Weighted Ensemble: K={best_weighted['k_pct']:3d}% ({best_weighted['num_layers']:2d} layers) | AUC={best_weighted['weighted']['auc']:.4f} | Acc={best_weighted['weighted']['acc']:.4f}")
    lines.append(f"Gated Ensemble:    K={best_gated['k_pct']:3d}% ({best_gated['num_layers']:2d} layers) | AUC={best_gated['gated']['auc']:.4f} | Acc={best_gated['gated']['acc']:.4f}")
    lines.append("=" * 90)

    # Overall best
    all_best = [
        ('Mean', best_mean['k_pct'], best_mean['mean']['auc']),
        ('Weighted', best_weighted['k_pct'], best_weighted['weighted']['auc']),
        ('Gated', best_gated['k_pct'], best_gated['gated']['auc'])
    ]
    overall = max(all_best, key=lambda x: x[2])
    lines.append(f"⭐ Best Overall: {overall[0]} @ K={overall[1]}% | AUC={overall[2]:.4f}")
    lines.append("=" * 90)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ensemble evaluation with K-sweeps")

    # Data sources
    parser.add_argument("--pooling", type=str, required=True, choices=['mean', 'max', 'last', 'attn'])
    parser.add_argument("--val_activations_dir", type=str, help="Validation activations directory")
    parser.add_argument("--probes_dir", type=str, help="Probes directory")
    parser.add_argument("--ood_logits_path", type=str, help="Pre-computed OOD logits (.npy)")
    parser.add_argument("--ood_labels_path", type=str, help="OOD labels (.npy)")

    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="validation",
                       choices=['validation', 'ood', 'both'],
                       help="Which dataset to evaluate on")

    # K% sweep
    parser.add_argument("--k_values", type=str, default="10,20,30,40,50,60,70,80,90",
                       help="Comma-separated K% values")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/ensembles")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gating_epochs", type=int, default=20)

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(',')]

    print("=" * 90)
    print(f"ENSEMBLE EVALUATION: {args.pooling.upper()}")
    print("=" * 90)
    print(f"K% values: {k_values}")
    print(f"Evaluation mode: {args.eval_mode}")
    print()

    # Load layer results for layer selection
    if not args.probes_dir:
        print("❌ --probes_dir required for layer selection")
        return 1

    layer_results = load_layer_results(args.probes_dir)
    if not layer_results:
        return 1

    # Check if ensemble evaluation already exists (skip if so)
    val_results_path = os.path.join(args.output_dir, "ensemble_k_sweep_validation.json")
    ood_results_path = os.path.join(args.output_dir, "ensemble_k_sweep_ood.json")

    skip_validation = False
    skip_ood = False

    if args.eval_mode in ['validation', 'both'] and os.path.exists(val_results_path):
        print(f"⚠️  Validation ensemble results already exist: {val_results_path}")
        print("   Skipping validation evaluation.")
        skip_validation = True

    if args.eval_mode in ['ood', 'both'] and os.path.exists(ood_results_path):
        print(f"⚠️  OOD ensemble results already exist: {ood_results_path}")
        print("   Skipping OOD evaluation.")
        skip_ood = True

    if (args.eval_mode == 'validation' and skip_validation) or \
       (args.eval_mode == 'ood' and skip_ood) or \
       (args.eval_mode == 'both' and skip_validation and skip_ood):
        print("=" * 90)
        print("✓ ALL REQUESTED EVALUATIONS ALREADY EXIST")
        print("=" * 90)
        print("To re-run, delete the results files and run again.")
        print("=" * 90)
        return 0

    # Validation evaluation
    if args.eval_mode in ['validation', 'both'] and not skip_validation:
        if not args.val_activations_dir:
            print("❌ --val_activations_dir required for validation evaluation")
            return 1

        print("Loading validation data...")
        val_dataset = CachedDataset(args.val_activations_dir)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("Extracting validation logits...")
        val_logits, val_labels = extract_logits_from_probes(
            args.probes_dir, val_dataloader, args.pooling, device
        )

        print("\nRunning K% sweep on validation...")
        val_results = evaluate_k_sweep(
            val_logits, val_labels, layer_results, k_values, device,
            gated_save_dir=os.path.join(args.output_dir, "gated_models_val"),
            eval_name="Validation"
        )

        # Save results
        val_results_path = os.path.join(args.output_dir, "ensemble_k_sweep_validation.json")
        with open(val_results_path, 'w') as f:
            json.dump(val_results, f, indent=2)
        print(f"\n✓ Saved validation results: {val_results_path}")

        # Generate plot
        val_plot_path = os.path.join(args.output_dir, "ensemble_comparison_validation.png")
        plot_ensemble_comparison(val_results, val_plot_path, args.pooling, "Validation")

        # Generate summary
        val_summary = generate_summary(val_results, args.pooling, "Validation")
        print("\n" + val_summary)

        val_summary_path = os.path.join(args.output_dir, "summary_validation.txt")
        with open(val_summary_path, 'w') as f:
            f.write(val_summary)

    # OOD evaluation
    if args.eval_mode in ['ood', 'both'] and not skip_ood:
        if not args.ood_logits_path or not args.ood_labels_path:
            print("❌ --ood_logits_path and --ood_labels_path required for OOD evaluation")
            return 1

        print("\nLoading OOD logits...")
        ood_logits = np.load(args.ood_logits_path)
        ood_labels = np.load(args.ood_labels_path)
        print(f"✓ Loaded OOD data: {ood_logits.shape}")

        print("\nRunning K% sweep on OOD...")
        ood_results = evaluate_k_sweep(
            ood_logits, ood_labels, layer_results, k_values, device,
            gated_save_dir=os.path.join(args.output_dir, "gated_models_ood"),
            eval_name="OOD"
        )

        # Save results
        ood_results_path = os.path.join(args.output_dir, "ensemble_k_sweep_ood.json")
        with open(ood_results_path, 'w') as f:
            json.dump(ood_results, f, indent=2)
        print(f"\n✓ Saved OOD results: {ood_results_path}")

        # Generate plot
        ood_plot_path = os.path.join(args.output_dir, "ensemble_comparison_ood.png")
        plot_ensemble_comparison(ood_results, ood_plot_path, args.pooling, "OOD")

        # Generate summary
        ood_summary = generate_summary(ood_results, args.pooling, "OOD")
        print("\n" + ood_summary)

        ood_summary_path = os.path.join(args.output_dir, "summary_ood.txt")
        with open(ood_summary_path, 'w') as f:
            f.write(ood_summary)

    print("\n" + "=" * 90)
    print("✓ ENSEMBLE EVALUATION COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    exit(main())
