"""
Train ensemble probes (Variant B from spec).

Given per-layer probes trained by train_probes.py, this script:
1. Loads all layer probes
2. Extracts per-layer logits on validation data
3. Trains ensemble variants:
   - StaticMeanEnsemble: uniform averaging
   - StaticWeightedEnsemble: AUC-weighted averaging
   - GatedEnsemble: learned input-conditioned weighting
4. Sweeps over top-K% layer selection
5. Evaluates on held-out OOD test set

Usage:
    python scripts/train_ensembles.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --held_out_dataset Movies \\
        --pooling mean \\
        --probe_dir data/probes/mean_Movies \\
        --k_pct_list 25,30,40,50,60,70
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.probes.models import LayerProbe
from actprobe.probes.ensemble import StaticMeanEnsemble, StaticWeightedEnsemble, GatedEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_DATASETS = ["HotpotQA", "HotpotQA-WC", "Movies", "TriviaQA", "IMDB"]


class CachedDataset(Dataset):
    def __init__(self, shards: list):
        self.items = []
        for shard in tqdm(shards, desc="Loading Shards"):
            try:
                tensors = load_file(shard)
                manifest_path = os.path.join(os.path.dirname(shard), "manifest.jsonl")

                with open(manifest_path) as f:
                    for line in f:
                        meta = json.loads(line)
                        eid = meta['id']
                        if eid in tensors:
                            self.items.append({
                                "tensor": tensors[eid],  # (L, T, D)
                                "label": meta.get('label', 0)
                            })
            except Exception as e:
                logger.warning(f"Error loading shard {shard}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item['tensor'].float(), torch.tensor(item['label'], dtype=torch.float32)


def get_shards(base_dir, model, dataset, split):
    pattern = os.path.join(base_dir, model.replace("/", "_"), dataset, split, "shard_*.safetensors")
    return sorted(glob.glob(pattern))


def load_layer_probes(probe_dir, num_layers, input_dim, pooling_type, device):
    """Load all trained per-layer probes."""
    probes = []
    for l in range(num_layers):
        probe_path = os.path.join(probe_dir, f"probe_layer_{l}.pt")
        if not os.path.exists(probe_path):
            logger.error(f"Missing probe for layer {l}: {probe_path}")
            return None

        probe = LayerProbe(input_dim=input_dim, pooling_type=pooling_type).to(device)
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        probe.eval()
        probes.append(probe)

    return probes


def extract_layer_logits(probes, dataloader, device):
    """
    Extract per-layer logits for all samples.

    Returns:
        logits: (N, L) array of logits
        labels: (N,) array of labels
    """
    all_logits = []
    all_labels = []

    num_layers = len(probes)

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting Logits"):
            batch_logits = []
            for l_idx in range(num_layers):
                x_layer = x[:, l_idx, :, :].to(device)  # (B, T, D)
                logits = probes[l_idx](x_layer)  # (B, 1)
                batch_logits.append(logits.cpu())

            # Stack to (B, L)
            batch_logits = torch.cat(batch_logits, dim=1)
            all_logits.append(batch_logits)
            all_labels.append(y)

    logits = torch.cat(all_logits, dim=0).numpy()  # (N, L)
    labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

    return logits, labels


def select_top_k_layers(val_logits, val_labels, k_pct, selection_mode="worst_domain"):
    """
    Select top-K% layers based on validation performance.

    Args:
        val_logits: dict {dataset: (N, L) logits}
        val_labels: dict {dataset: (N,) labels}
        k_pct: percentage of layers to keep (e.g., 40 for 40%)
        selection_mode: "worst_domain" (min AUC) or "mean_domain" (mean AUC)

    Returns:
        selected_indices: list of layer indices
        layer_aucs: dict {dataset: per-layer AUCs}
    """
    # Compute per-layer AUC for each dataset
    layer_aucs = {}
    num_layers = None

    for dataset, logits in val_logits.items():
        labels = val_labels[dataset]
        if num_layers is None:
            num_layers = logits.shape[1]

        aucs = []
        for l in range(logits.shape[1]):
            try:
                auc = roc_auc_score(labels, logits[:, l])
            except:
                auc = 0.5
            aucs.append(auc)

        layer_aucs[dataset] = np.array(aucs)

    # Aggregate across datasets
    if selection_mode == "worst_domain":
        # Min AUC across datasets
        aggregate_aucs = np.min(np.stack(list(layer_aucs.values())), axis=0)
    elif selection_mode == "mean_domain":
        # Mean AUC across datasets
        aggregate_aucs = np.mean(np.stack(list(layer_aucs.values())), axis=0)
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    # Select top K%
    k_count = max(1, int(num_layers * k_pct / 100))
    selected_indices = np.argsort(aggregate_aucs)[-k_count:]

    logger.info(f"Selected top {k_pct}% ({k_count}) layers: {selected_indices}")
    logger.info(f"  Aggregate AUCs: {aggregate_aucs[selected_indices]}")

    return selected_indices.tolist(), layer_aucs


def evaluate_ensemble(ensemble, test_logits, test_labels, ensemble_type):
    """Evaluate ensemble model."""
    if ensemble_type in ["static_mean", "static_weighted"]:
        # These ensembles don't need input features, just logits
        probs = ensemble(torch.tensor(test_logits, dtype=torch.float32))
        probs = torch.sigmoid(probs).numpy()
    else:
        # Gated ensemble needs input (features or logits as proxy)
        # For simplicity, use logits as input
        with torch.no_grad():
            probs = ensemble(torch.tensor(test_logits, dtype=torch.float32))
            probs = torch.sigmoid(probs).numpy()

    auc = roc_auc_score(test_labels, probs)
    acc = accuracy_score(test_labels, (probs > 0.5).astype(int))

    return auc, acc


def train_ensembles(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Identify train datasets (LODO)
    train_datasets = [d for d in ALL_DATASETS if d != args.held_out_dataset]
    test_dataset = args.held_out_dataset

    logger.info(f"Training ensembles for LODO: held_out={test_dataset}")
    logger.info(f"Train datasets: {train_datasets}")

    # 2. Load validation data for train datasets
    val_loaders = {}
    for d in train_datasets:
        shards = get_shards(args.data_dir, args.model, d, "validation")
        if shards:
            ds = CachedDataset(shards)
            val_loaders[d] = DataLoader(ds, batch_size=args.batch_size)

    # 3. Load test data for held-out dataset
    test_shards = get_shards(args.data_dir, args.model, test_dataset, "validation")
    test_loader = None
    if test_shards:
        ds = CachedDataset(test_shards)
        test_loader = DataLoader(ds, batch_size=args.batch_size)

    # 4. Get dimensions from first sample
    sample, _ = val_loaders[train_datasets[0]].dataset[0]
    num_layers, _, d_model = sample.shape[0], sample.shape[1], sample.shape[2]

    logger.info(f"Model dimensions: L={num_layers}, D={d_model}")

    # 5. Load per-layer probes
    logger.info(f"Loading layer probes from {args.probe_dir}")
    probes = load_layer_probes(args.probe_dir, num_layers, d_model, args.pooling, device)

    if probes is None:
        logger.error("Failed to load layer probes")
        return

    # 6. Extract logits on validation sets
    logger.info("Extracting validation logits...")
    val_logits = {}
    val_labels = {}

    for dataset, loader in val_loaders.items():
        logits, labels = extract_layer_logits(probes, loader, device)
        val_logits[dataset] = logits
        val_labels[dataset] = labels
        logger.info(f"  {dataset}: {logits.shape[0]} samples")

    # 7. Extract logits on test set
    logger.info("Extracting test logits...")
    test_logits, test_labels = extract_layer_logits(probes, test_loader, device)
    logger.info(f"  {test_dataset}: {test_logits.shape[0]} samples")

    # 8. Sweep over K% values
    k_pct_list = [int(k) for k in args.k_pct_list.split(",")]
    results = []

    k_pbar = tqdm(k_pct_list, desc="Evaluating K% values", unit="K%")
    for k_pct in k_pbar:
        k_pbar.set_postfix({"K": f"{k_pct}%"})

        # Select top-K layers
        selected_layers, layer_aucs = select_top_k_layers(
            val_logits, val_labels, k_pct, selection_mode=args.selection_mode
        )

        # Slice logits to selected layers
        val_logits_k = {d: logits[:, selected_layers] for d, logits in val_logits.items()}
        test_logits_k = test_logits[:, selected_layers]

        # --- Static Mean Ensemble ---
        ensemble_mean = StaticMeanEnsemble()  # FIX: No parameters needed
        auc_mean, acc_mean = evaluate_ensemble(ensemble_mean, test_logits_k, test_labels, "static_mean")
        logger.info(f"  StaticMean: AUC={auc_mean:.4f}, Acc={acc_mean:.4f}")

        # --- Static Weighted Ensemble ---
        # Compute weights from validation AUCs (aggregate across train datasets)
        val_aucs_per_layer = np.mean([layer_aucs[d][selected_layers] for d in train_datasets], axis=0)
        weights = torch.tensor(val_aucs_per_layer / val_aucs_per_layer.sum(), dtype=torch.float32)
        ensemble_weighted = StaticWeightedEnsemble(weights=weights)
        auc_weighted, acc_weighted = evaluate_ensemble(ensemble_weighted, test_logits_k, test_labels, "static_weighted")
        logger.info(f"  StaticWeighted: AUC={auc_weighted:.4f}, Acc={acc_weighted:.4f}")

        # --- Gated Ensemble (needs training) ---
        # Concatenate validation data from all train datasets
        val_logits_concat = np.vstack([val_logits_k[d] for d in train_datasets])
        val_labels_concat = np.hstack([val_labels[d] for d in train_datasets])

        # Train gated ensemble
        # FIX: Remove hidden_dim parameter (not in __init__)
        ensemble_gated = GatedEnsemble(input_dim=len(selected_layers), num_layers=len(selected_layers)).to(device)
        optimizer = optim.AdamW(ensemble_gated.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Convert to tensors
        val_logits_tensor = torch.tensor(val_logits_concat, dtype=torch.float32)
        val_labels_tensor = torch.tensor(val_labels_concat, dtype=torch.float32).unsqueeze(1)

        # Mini training loop
        dataset = torch.utils.data.TensorDataset(val_logits_tensor, val_labels_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        gated_pbar = tqdm(range(args.gating_epochs), desc=f"    Training Gated (K={k_pct}%)", leave=False, unit="epoch")
        for epoch in gated_pbar:
            ensemble_gated.train()
            epoch_loss = 0

            for batch_logits, batch_labels in loader:
                batch_logits = batch_logits.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                # FIX: GatedEnsemble.forward expects (layer_features, layer_logits)
                # Use logits as features (simplified approach - logits contain layer info)
                output = ensemble_gated(batch_logits, batch_logits)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            gated_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "best": f"{best_val_loss:.4f}"})

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                gated_pbar.close()
                logger.info(f"    Gated: Early stopping at epoch {epoch+1}")
                break

        gated_pbar.close()

        # Evaluate gated ensemble
        ensemble_gated.eval()
        with torch.no_grad():
            test_logits_tensor = torch.tensor(test_logits_k, dtype=torch.float32).to(device)
            # FIX: GatedEnsemble needs both features and logits
            probs_gated = torch.sigmoid(ensemble_gated(test_logits_tensor, test_logits_tensor)).cpu().numpy()

        auc_gated = roc_auc_score(test_labels, probs_gated)
        acc_gated = accuracy_score(test_labels, (probs_gated > 0.5).astype(int))
        logger.info(f"  GatedEnsemble: AUC={auc_gated:.4f}, Acc={acc_gated:.4f}")

        # Save results
        results.append({
            "k_pct": k_pct,
            "selected_layers": selected_layers,
            "static_mean": {"auc": float(auc_mean), "acc": float(acc_mean)},
            "static_weighted": {"auc": float(auc_weighted), "acc": float(acc_weighted)},
            "gated": {"auc": float(auc_gated), "acc": float(acc_gated)},
            "layer_aucs": {d: layer_aucs[d][selected_layers].tolist() for d in train_datasets}
        })

        # Update K% progress bar
        k_pbar.set_postfix({
            "mean": f"{auc_mean:.3f}",
            "weighted": f"{auc_weighted:.3f}",
            "gated": f"{auc_gated:.3f}"
        })

    k_pbar.close()

    # 9. Save results
    output_file = os.path.join(args.output_dir, f"ensemble_results_{args.held_out_dataset}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    logger.info("\n=== Summary ===")
    for r in results:
        logger.info(f"K={r['k_pct']}%: Mean={r['static_mean']['auc']:.4f}, Weighted={r['static_weighted']['auc']:.4f}, Gated={r['gated']['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ensemble probes (Variant B)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--held_out_dataset", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--probe_dir", type=str, required=True, help="Directory with trained per-layer probes")
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="data/ensembles")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_pct_list", type=str, default="25,30,40,50,60,70", help="Comma-separated K% values")
    parser.add_argument("--selection_mode", type=str, default="worst_domain", choices=["worst_domain", "mean_domain"])
    parser.add_argument("--gating_epochs", type=int, default=20, help="Max epochs for training gated ensemble")
    args = parser.parse_args()

    train_ensembles(args)


if __name__ == "__main__":
    main()
