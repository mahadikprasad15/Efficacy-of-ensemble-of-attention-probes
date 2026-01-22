"""
PATCHED VERSION with manifest.jsonl fix

This is a temporary patched version of evaluate_ood_all_pooling.py
that reads labels from manifest.jsonl correctly.

If the main script still shows "Loaded 0 OOD samples", use this version:

    !python scripts/evaluate_ood_all_pooling_PATCHED.py \
        --ood_activations data/activations/... \
        --probes_base data/probes_flipped/... \
        --output_dir results_flipped/ood_evaluation
"""

# [Copy the entire fixed script here]
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
    """Load cached OOD activations from safetensors shards + manifest.jsonl."""

    def __init__(self, activations_dir: str):
        """
        Args:
            activations_dir: Directory with shard_*.safetensors and manifest.jsonl
        """
        self.activations_dir = activations_dir

        # Load manifest to get labels
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise ValueError(f"No manifest.jsonl found in {activations_dir}")

        # Build label map: {id: label}
        label_map = {}
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                label_map[entry['id']] = entry['label']

        print(f"✓ Loaded {len(label_map)} labels from manifest.jsonl")

        # Load all shards
        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))

        if not shards:
            raise ValueError(f"No shard files found in {activations_dir}")

        print(f"Loading {len(shards)} OOD shard(s)...")

        self.samples = []
        skipped = 0
        for shard_path in tqdm(shards, desc="Loading shards"):
            shard_data = load_file(shard_path)

            for key in shard_data.keys():
                # Check if this sample has a label in manifest
                if key in label_map:
                    activations = shard_data[key]
                    label = label_map[key]

                    # Skip samples with unknown labels (-1)
                    if label == -1:
                        skipped += 1
                        continue

                    self.samples.append((activations, label))

        if skipped > 0:
            print(f"⚠️  Skipped {skipped} samples with unknown labels")
        print(f"✓ Loaded {len(self.samples)} OOD samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def find_probe_files(probes_dir: str) -> List[str]:
    """Find all probe_layer_*.pt files in directory, sorted numerically."""
    pattern = os.path.join(probes_dir, "probe_layer_*.pt")
    probe_files = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split('_')[-1].replace('.pt', ''))
    )
    return probe_files


def evaluate_probe_layer(
    probe: LayerProbe,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate a single layer probe on OOD data."""
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
    """Evaluate all layer probes for a pooling strategy on OOD data."""
    probe_files = find_probe_files(probes_dir)

    if not probe_files:
        print(f"⚠️  No probe files found in {probes_dir}")
        return None

    print(f"\nEvaluating {pooling_type.upper()} ({len(probe_files)} layers)...")

    # Get dimensions from first sample
    if len(ood_dataloader.dataset) == 0:
        print(f"❌ Dataset is empty!")
        return None

    sample_x, _ = next(iter(ood_dataloader))
    _, T, D = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]

    results = []
    all_layer_logits = []

    for probe_file in tqdm(probe_files, desc=f"{pooling_type.upper()}"):
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

        # Store logits for ensemble
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
    best_idx = max(range(len(results)), key=lambda i: results[i]['auc'])
    best_layer = results[best_idx]['layer']
    best_auc = results[best_idx]['auc']

    # Save logits
    if save_logits and logits_save_dir:
        logits_array = np.array(all_layer_logits).T  # (N, L)
        logits_path = os.path.join(logits_save_dir, f"{pooling_type}_logits.npy")
        np.save(logits_path, logits_array)
        print(f"✓ Saved logits: {logits_path} (shape: {logits_array.shape})")

    return {
        'pooling': pooling_type,
        'layers': [r['layer'] for r in results],
        'aucs': [r['auc'] for r in results],
        'accuracies': [r['accuracy'] for r in results],
        'best_layer': best_layer,
        'best_auc': best_auc,
        'results': results
    }


def plot_ood_comparison(all_results: Dict, output_path: str, dataset_name: str = "OOD"):
    """Create comparison plot for all pooling strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for pooling in POOLING_ORDER:
        if pooling not in all_results:
            continue

        data = all_results[pooling]
        layers = data['layers']
        aucs = data['aucs']
        best_layer = data['best_layer']
        best_auc = data['best_auc']

        color = POOLING_COLORS[pooling]
        ax.plot(layers, aucs, marker='o', label=f"{pooling.upper()}",
                color=color, linewidth=2, markersize=6)

        # Mark best layer
        ax.scatter([best_layer], [best_auc], color=color, s=200,
                  marker='*', edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name} Evaluation - All Pooling Strategies',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def generate_summary(all_results: Dict) -> str:
    """Generate text summary of results."""
    lines = []
    lines.append("=" * 90)
    lines.append("OOD EVALUATION SUMMARY - ALL POOLING STRATEGIES")
    lines.append("=" * 90)
    lines.append("")

    for pooling in POOLING_ORDER:
        if pooling not in all_results:
            continue

        data = all_results[pooling]
        lines.append(f"{pooling.upper()} Pooling:")
        lines.append(f"  Best Layer: L{data['best_layer']}")
        lines.append(f"  Best AUC:   {data['best_auc']:.4f}")
        lines.append(f"  Layers:     {min(data['layers'])}-{max(data['layers'])}")
        lines.append("")

    lines.append("=" * 90)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all pooling strategies on OOD")

    # Probe directories
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probes_base_dir", "--probes_base", dest="probes_base_dir", type=str,
                      help="Base directory with mean/max/last/attn subdirectories")
    group.add_argument("--mean_probes_dir", type=str, help="MEAN probes directory")

    parser.add_argument("--max_probes_dir", type=str, help="MAX probes directory")
    parser.add_argument("--last_probes_dir", type=str, help="LAST probes directory")
    parser.add_argument("--attn_probes_dir", type=str, help="ATTN probes directory")

    # OOD data
    parser.add_argument("--ood_activations_dir", "--ood_activations", dest="ood_activations_dir",
                       type=str, required=True, help="OOD activations directory")
    parser.add_argument("--ood_dataset_name", type=str, default="OOD",
                       help="Name for plot title")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/ood_evaluation",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("OOD EVALUATION - ALL POOLING STRATEGIES (PATCHED VERSION)")
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

    if len(ood_dataset) == 0:
        print("❌ No samples loaded! Check your manifest.jsonl for valid labels (0 or 1)")
        return 1

    ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Extract and save labels
    ood_labels = np.array([label for _, label in ood_dataset])
    print()

    # Evaluate each pooling strategy
    all_results = {}
    logits_dir = os.path.join(args.output_dir, "logits")
    os.makedirs(logits_dir, exist_ok=True)

    # Save labels
    labels_path = os.path.join(logits_dir, "labels.npy")
    np.save(labels_path, ood_labels)
    print(f"✓ Saved OOD labels: {labels_path} ({len(ood_labels)} samples)\n")

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
