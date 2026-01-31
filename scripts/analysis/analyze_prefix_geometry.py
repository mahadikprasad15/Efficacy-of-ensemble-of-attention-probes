#!/usr/bin/env python3
"""
Probe Geometry Analysis with Soft Prefix.

Analyzes how a learned soft prefix changes activation geometry relative to 
probe decision boundaries for ID (Roleplaying train) and OOD (InsiderTrading test).

Implements 4 analysis steps:
    Step A: Probe-axis geometry (logits, histograms, AUROC, Cohen's d)
    Step B: Domain/class shift (Δ_class, Δ_domain before/after prefix)
    Step C: Delta + alignment (w·δ distributions, cos(θ) alignment)  
    Step D: PCA/2D visuals (with/without prefix panels, decision boundary)

Usage:
    python scripts/analysis/analyze_prefix_geometry.py \
        --no_prefix_id_dir <path_to_id_activations_without_prefix> \
        --no_prefix_ood_dir <path_to_ood_activations_without_prefix> \
        --with_prefix_id_dir <path_to_id_activations_with_prefix> \
        --with_prefix_ood_dir <path_to_ood_activations_with_prefix> \
        --probes_dir <path_to_probes_with_last_pooling> \
        --probe_type vanilla \
        --output_dir results/prefix_geometry_analysis
"""

import argparse
import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

try:
    from actprobe.probes.models import LayerProbe
    HAS_LAYERPROBE = True
except ImportError:
    HAS_LAYERPROBE = False
    print("Warning: Could not import LayerProbe")


# ============================================================================
# Color Scheme
# ============================================================================

COLORS = {
    'id_honest': '#4A90D9',      # Blue
    'id_deceptive': '#E74C3C',   # Red  
    'ood_honest': '#27AE60',     # Green
    'ood_deceptive': '#9B59B6',  # Purple
}

PREFIX_STYLES = {
    'no_prefix': {'linestyle': '--', 'alpha': 0.6},
    'with_prefix': {'linestyle': '-', 'alpha': 0.9},
}


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class ActivationData:
    """Loaded activations with metadata."""
    activations: np.ndarray  # (N, L, D) - samples, layers, hidden dim
    labels: np.ndarray       # (N,) - 0=honest, 1=deceptive
    ids: List[str]
    
    @property
    def honest_mask(self) -> np.ndarray:
        return self.labels == 0
    
    @property
    def deceptive_mask(self) -> np.ndarray:
        return self.labels == 1


def load_activations_from_dir(activations_dir: str, pool_to_last: bool = True) -> ActivationData:
    """
    Load activations from safetensors shards and manifest.
    
    Handles both formats:
        - Per-token: (L, T, D) -> pools to (L, D) using last token
        - Already pooled: (L, D) -> uses as-is
    
    Args:
        activations_dir: Directory with shard_*.safetensors and manifest.jsonl
        pool_to_last: If True, pool per-token activations to last token
    
    Returns:
        ActivationData with shape (N, L, D)
    """
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise ValueError(f"No manifest.jsonl found in {activations_dir}")
    
    # Load manifest
    id_to_label = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            id_to_label[entry['id']] = entry['label']
    
    print(f"  Manifest: {len(id_to_label)} entries")
    
    # Load shards
    shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
    shards = sorted(glob.glob(shard_pattern))
    
    if not shards:
        raise ValueError(f"No shard files found in {activations_dir}")
    
    all_acts = []
    all_labels = []
    all_ids = []
    skipped = 0
    detected_format = None
    
    for shard_path in shards:
        shard_data = load_file(shard_path)
        
        for sample_id, tensor in shard_data.items():
            if sample_id not in id_to_label:
                skipped += 1
                continue
            
            label = id_to_label[sample_id]
            if label == -1:
                skipped += 1
                continue
            
            arr = tensor.numpy()
            
            # Detect format from first sample
            if detected_format is None:
                if arr.ndim == 2:
                    detected_format = "pooled"
                    print(f"  Format: Already pooled (L, D) = {arr.shape}")
                elif arr.ndim == 3:
                    detected_format = "per_token"
                    print(f"  Format: Per-token (L, T, D) = {arr.shape}")
                else:
                    raise ValueError(f"Unexpected tensor shape: {arr.shape}")
            
            # Handle per-token format: (L, T, D) -> (L, D) via last token
            if detected_format == "per_token" and pool_to_last:
                # arr shape is (L, T, D), take last token: arr[:, -1, :]
                arr = arr[:, -1, :]  # (L, D)
            
            all_acts.append(arr)
            all_labels.append(label)
            all_ids.append(sample_id)
    
    if skipped > 0:
        print(f"  Skipped: {skipped} samples")
    
    activations = np.stack(all_acts, axis=0)  # (N, L, D)
    labels = np.array(all_labels)
    
    print(f"  Final shape: {activations.shape} (N, L, D)")
    print(f"  Labels: {np.sum(labels == 0)} honest, {np.sum(labels == 1)} deceptive")
    
    return ActivationData(activations=activations, labels=labels, ids=all_ids)


def load_probe(probe_path: str, input_dim: int, pooling_type: str = "last") -> torch.nn.Module:
    """Load a trained probe."""
    if HAS_LAYERPROBE:
        probe = LayerProbe(input_dim=input_dim, pooling_type=pooling_type)
    else:
        # Fallback simple probe
        probe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )
    
    state_dict = torch.load(probe_path, map_location='cpu')
    probe.load_state_dict(state_dict, strict=True)
    probe.eval()
    
    return probe


def get_probe_weight(probe: torch.nn.Module) -> np.ndarray:
    """
    Extract the primary classification direction from a probe.
    
    For LayerProbe with pooling, we extract from the classifier.
    For MLP-style probes, we use SVD of first layer.
    """
    # Try to find linear layer
    for name, module in probe.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.detach().cpu().numpy()  # (out, in)
            
            if W.shape[0] == 1:
                # Single output = logistic regression, use weight directly
                return W.flatten()
            else:
                # MLP hidden layer - use top singular vector
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                return Vh[0]  # Top right singular vector
    
    raise ValueError("Could not find linear layer in probe")


# ============================================================================
# Step A: Probe-axis Geometry
# ============================================================================

def compute_logits(activations: np.ndarray, probe_weight: np.ndarray) -> np.ndarray:
    """
    Compute logits = w · x for each sample.
    
    Args:
        activations: (N, D) - activations for a single layer
        probe_weight: (D,) - probe weight vector
        
    Returns:
        logits: (N,)
    """
    # Normalize weight to unit vector for interpretable projections
    w_norm = probe_weight / (np.linalg.norm(probe_weight) + 1e-10)
    return activations @ w_norm


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def step_a_probe_axis_geometry(
    data_no_prefix: Dict[str, ActivationData],  # {'id': ..., 'ood': ...}
    data_with_prefix: Dict[str, ActivationData],
    probe_weight: np.ndarray,
    layer_idx: int,
    output_dir: str
) -> Dict:
    """
    Step A: Compute logits, plot distributions, compute metrics.
    
    Returns metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect logits for all 8 conditions
    conditions = {}
    
    for prefix_state, data_dict in [('no_prefix', data_no_prefix), ('with_prefix', data_with_prefix)]:
        for domain, data in data_dict.items():
            # Extract layer activations
            acts = data.activations[:, layer_idx, :]  # (N, D)
            logits = compute_logits(acts, probe_weight)
            
            # Split by class
            conditions[f'{domain}_honest_{prefix_state}'] = logits[data.honest_mask]
            conditions[f'{domain}_deceptive_{prefix_state}'] = logits[data.deceptive_mask]
    
    # Compute metrics
    metrics = {}
    
    for prefix_state in ['no_prefix', 'with_prefix']:
        for domain in ['id', 'ood']:
            honest = conditions[f'{domain}_honest_{prefix_state}']
            deceptive = conditions[f'{domain}_deceptive_{prefix_state}']
            
            key = f'{domain}_{prefix_state}'
            metrics[f'{key}_mean_gap'] = float(np.mean(deceptive) - np.mean(honest))
            metrics[f'{key}_cohens_d'] = float(compute_cohens_d(deceptive, honest))
            metrics[f'{key}_var_honest'] = float(np.var(honest))
            metrics[f'{key}_var_deceptive'] = float(np.var(deceptive))
            
            # AUROC
            labels = np.concatenate([np.zeros(len(honest)), np.ones(len(deceptive))])
            scores = np.concatenate([honest, deceptive])
            try:
                metrics[f'{key}_auroc'] = float(roc_auc_score(labels, scores))
            except:
                metrics[f'{key}_auroc'] = 0.5
    
    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, prefix_state in enumerate(['no_prefix', 'with_prefix']):
        for col, domain in enumerate(['id', 'ood']):
            ax = axes[row, col]
            
            honest = conditions[f'{domain}_honest_{prefix_state}']
            deceptive = conditions[f'{domain}_deceptive_{prefix_state}']
            
            # Plot histograms with KDE
            ax.hist(honest, bins=30, alpha=0.5, density=True, 
                   color=COLORS[f'{domain}_honest'], label='Honest')
            ax.hist(deceptive, bins=30, alpha=0.5, density=True,
                   color=COLORS[f'{domain}_deceptive'], label='Deceptive')
            
            # KDE overlay
            if len(honest) > 1:
                kde_h = stats.gaussian_kde(honest)
                kde_d = stats.gaussian_kde(deceptive)
                x_range = np.linspace(min(honest.min(), deceptive.min()), 
                                      max(honest.max(), deceptive.max()), 100)
                ax.plot(x_range, kde_h(x_range), color=COLORS[f'{domain}_honest'], lw=2)
                ax.plot(x_range, kde_d(x_range), color=COLORS[f'{domain}_deceptive'], lw=2)
            
            # Metrics annotation
            key = f'{domain}_{prefix_state}'
            auroc = metrics[f'{key}_auroc']
            cohens_d = metrics[f'{key}_cohens_d']
            gap = metrics[f'{key}_mean_gap']
            
            ax.text(0.95, 0.95, f'AUROC: {auroc:.3f}\nCohen\'s d: {cohens_d:.2f}\nGap: {gap:.2f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            title = f'{"ID (Roleplaying)" if domain == "id" else "OOD (InsiderTrading)"}'
            title += f' - {"No Prefix" if prefix_state == "no_prefix" else "With Prefix"}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Probe Logit (w · x)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Step A: Probe-axis Geometry (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_a_logit_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# Step B: Domain Shift vs Class Shift
# ============================================================================

def step_b_shift_analysis(
    data_no_prefix: Dict[str, ActivationData],
    data_with_prefix: Dict[str, ActivationData],
    layer_idx: int,
    output_dir: str
) -> Dict:
    """
    Step B: Analyze domain shift vs class shift.
    
    Δ_class = μ(deceptive) - μ(honest) per domain
    Δ_domain = μ(OOD) - μ(ID) per class
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    
    # Compute mean activations for each condition
    means = {}
    for prefix_state, data_dict in [('no_prefix', data_no_prefix), ('with_prefix', data_with_prefix)]:
        for domain, data in data_dict.items():
            acts = data.activations[:, layer_idx, :]  # (N, D)
            means[f'{domain}_honest_{prefix_state}'] = acts[data.honest_mask].mean(axis=0)
            means[f'{domain}_deceptive_{prefix_state}'] = acts[data.deceptive_mask].mean(axis=0)
    
    # Compute shifts
    shifts = {}
    for prefix_state in ['no_prefix', 'with_prefix']:
        # Class shifts (deceptive - honest)
        shifts[f'delta_class_id_{prefix_state}'] = (
            means[f'id_deceptive_{prefix_state}'] - means[f'id_honest_{prefix_state}']
        )
        shifts[f'delta_class_ood_{prefix_state}'] = (
            means[f'ood_deceptive_{prefix_state}'] - means[f'ood_honest_{prefix_state}']
        )
        
        # Domain shifts (OOD - ID)
        shifts[f'delta_domain_honest_{prefix_state}'] = (
            means[f'ood_honest_{prefix_state}'] - means[f'id_honest_{prefix_state}']
        )
        shifts[f'delta_domain_deceptive_{prefix_state}'] = (
            means[f'ood_deceptive_{prefix_state}'] - means[f'id_deceptive_{prefix_state}']
        )
    
    # Compute norms and angles
    for key, delta in shifts.items():
        metrics[f'{key}_norm'] = float(np.linalg.norm(delta))
    
    # Cosine similarity between ID and OOD class shifts
    for prefix_state in ['no_prefix', 'with_prefix']:
        delta_class_id = shifts[f'delta_class_id_{prefix_state}']
        delta_class_ood = shifts[f'delta_class_ood_{prefix_state}']
        
        cos_sim = np.dot(delta_class_id, delta_class_ood) / (
            np.linalg.norm(delta_class_id) * np.linalg.norm(delta_class_ood) + 1e-10
        )
        metrics[f'cos_sim_class_shifts_{prefix_state}'] = float(cos_sim)
    
    # Plot bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class shift norms
    ax = axes[0]
    labels = ['ID No Prefix', 'ID With Prefix', 'OOD No Prefix', 'OOD With Prefix']
    values = [
        metrics['delta_class_id_no_prefix_norm'],
        metrics['delta_class_id_with_prefix_norm'],
        metrics['delta_class_ood_no_prefix_norm'],
        metrics['delta_class_ood_with_prefix_norm'],
    ]
    colors = [COLORS['id_honest'], COLORS['id_deceptive'], 
              COLORS['ood_honest'], COLORS['ood_deceptive']]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('||Δ_class||', fontsize=12)
    ax.set_title('Class Shift Magnitude (Deceptive - Honest)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    
    # Domain shift norms
    ax = axes[1]
    labels = ['Honest No Prefix', 'Honest With Prefix', 
              'Deceptive No Prefix', 'Deceptive With Prefix']
    values = [
        metrics['delta_domain_honest_no_prefix_norm'],
        metrics['delta_domain_honest_with_prefix_norm'],
        metrics['delta_domain_deceptive_no_prefix_norm'],
        metrics['delta_domain_deceptive_with_prefix_norm'],
    ]
    
    bars = ax.bar(labels, values, color=['#3498DB', '#2980B9', '#E74C3C', '#C0392B'], 
                 alpha=0.7, edgecolor='black')
    ax.set_ylabel('||Δ_domain||', fontsize=12)
    ax.set_title('Domain Shift Magnitude (OOD - ID)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Step B: Domain Shift vs Class Shift (Layer {layer_idx})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_b_shift_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# Step C: Delta + Alignment
# ============================================================================

def step_c_delta_alignment(
    data_no_prefix: Dict[str, ActivationData],
    data_with_prefix: Dict[str, ActivationData],
    probe_weight: np.ndarray,
    layer_idx: int,
    output_dir: str
) -> Dict:
    """
    Step C: Analyze prefix-induced delta and its alignment with probe.
    
    δ = x_with_prefix - x_without_prefix for each sample
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    
    # We need matching samples - use IDs to align
    # For now, compute mean delta since samples may not be matched
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for col, domain in enumerate(['id', 'ood']):
        # Get activations
        acts_no = data_no_prefix[domain].activations[:, layer_idx, :]
        acts_with = data_with_prefix[domain].activations[:, layer_idx, :]
        labels = data_no_prefix[domain].labels
        
        # Mean delta per class
        honest_mask = labels == 0
        deceptive_mask = labels == 1
        
        # Compute per-sample deltas if sizes match, else use mean difference
        if len(acts_no) == len(acts_with):
            delta = acts_with - acts_no
            delta_honest = delta[honest_mask]
            delta_deceptive = delta[deceptive_mask]
        else:
            # Different sample sizes - compute mean delta
            delta_honest = acts_with[data_with_prefix[domain].honest_mask].mean(axis=0, keepdims=True) - \
                          acts_no[honest_mask].mean(axis=0, keepdims=True)
            delta_deceptive = acts_with[data_with_prefix[domain].deceptive_mask].mean(axis=0, keepdims=True) - \
                             acts_no[deceptive_mask].mean(axis=0, keepdims=True)
        
        # Normalize probe weight
        w_norm = probe_weight / (np.linalg.norm(probe_weight) + 1e-10)
        
        # w · δ for each sample
        w_delta_honest = delta_honest @ w_norm
        w_delta_deceptive = delta_deceptive @ w_norm
        
        # Mean delta direction
        mu_delta_honest = delta_honest.mean(axis=0)
        mu_delta_deceptive = delta_deceptive.mean(axis=0)
        
        # Alignment angles
        cos_theta_honest = np.dot(mu_delta_honest, w_norm) / (np.linalg.norm(mu_delta_honest) + 1e-10)
        cos_theta_deceptive = np.dot(mu_delta_deceptive, w_norm) / (np.linalg.norm(mu_delta_deceptive) + 1e-10)
        
        metrics[f'{domain}_cos_theta_honest'] = float(cos_theta_honest)
        metrics[f'{domain}_cos_theta_deceptive'] = float(cos_theta_deceptive)
        metrics[f'{domain}_mean_w_delta_honest'] = float(w_delta_honest.mean())
        metrics[f'{domain}_mean_w_delta_deceptive'] = float(w_delta_deceptive.mean())
        metrics[f'{domain}_delta_norm_honest'] = float(np.linalg.norm(mu_delta_honest))
        metrics[f'{domain}_delta_norm_deceptive'] = float(np.linalg.norm(mu_delta_deceptive))
        
        # Plot w·δ distribution
        ax = axes[0, col]
        if len(w_delta_honest) > 1:
            ax.hist(w_delta_honest, bins=30, alpha=0.6, density=True, 
                   color=COLORS[f'{domain}_honest'], label='Honest')
            ax.hist(w_delta_deceptive, bins=30, alpha=0.6, density=True,
                   color=COLORS[f'{domain}_deceptive'], label='Deceptive')
        else:
            # Single point - show as bar
            ax.bar(['Honest', 'Deceptive'], 
                   [w_delta_honest[0], w_delta_deceptive[0]],
                   color=[COLORS[f'{domain}_honest'], COLORS[f'{domain}_deceptive']])
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('w · δ (prefix shift along probe)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{"ID" if domain == "id" else "OOD"}: w · δ Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot alignment angles
        ax = axes[1, col]
        angles = [cos_theta_honest, cos_theta_deceptive]
        labels_ax = ['Honest', 'Deceptive']
        colors = [COLORS[f'{domain}_honest'], COLORS[f'{domain}_deceptive']]
        
        bars = ax.bar(labels_ax, angles, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('cos(θ) = μ_δ · w / ||μ_δ|| ||w||', fontsize=11)
        ax.set_title(f'{"ID" if domain == "id" else "OOD"}: Alignment of Mean Delta with Probe', fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        
        # Annotate with values
        for bar, angle in zip(bars, angles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{angle:.3f}', ha='center', fontsize=11)
    
    plt.suptitle(f'Step C: Delta + Alignment (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_c_delta_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# Step D: PCA/2D Visuals
# ============================================================================

def step_d_pca_visualization(
    data_no_prefix: Dict[str, ActivationData],
    data_with_prefix: Dict[str, ActivationData],
    probe_weight: np.ndarray,
    layer_idx: int,
    output_dir: str
) -> Dict:
    """
    Step D: PCA visualization with probe decision boundary.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    
    # Combine all activations for PCA
    all_acts = []
    all_meta = []
    
    for prefix_state, data_dict in [('no_prefix', data_no_prefix), ('with_prefix', data_with_prefix)]:
        for domain, data in data_dict.items():
            acts = data.activations[:, layer_idx, :]  # (N, D)
            for i in range(len(acts)):
                all_acts.append(acts[i])
                all_meta.append({
                    'domain': domain,
                    'class': 'honest' if data.labels[i] == 0 else 'deceptive',
                    'prefix': prefix_state
                })
    
    X = np.stack(all_acts, axis=0)  # (N_total, D)
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    metrics['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
    
    # Project probe weight into PC space
    w_pca = pca.transform(probe_weight.reshape(1, -1))[0]
    w_pca_norm = w_pca / (np.linalg.norm(w_pca) + 1e-10)
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax_idx, prefix_state in enumerate(['no_prefix', 'with_prefix']):
        ax = axes[ax_idx]
        
        # Get indices for this prefix state
        for domain in ['id', 'ood']:
            for cls in ['honest', 'deceptive']:
                mask = [(m['domain'] == domain and m['class'] == cls and m['prefix'] == prefix_state) 
                        for m in all_meta]
                mask = np.array(mask)
                
                if mask.sum() > 0:
                    color = COLORS[f'{domain}_{cls}']
                    marker = 'o' if domain == 'id' else 's'
                    label = f'{domain.upper()} {cls.capitalize()}'
                    
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=color, marker=marker, alpha=0.6, s=30, label=label)
        
        # Draw probe direction
        scale = max(np.abs(X_pca).max() * 0.8, 1)
        ax.arrow(0, 0, w_pca_norm[0] * scale, w_pca_norm[1] * scale,
                head_width=scale*0.05, head_length=scale*0.03, 
                fc='black', ec='black', linewidth=2, label='Probe direction')
        
        # Draw decision boundary (perpendicular to probe)
        perp = np.array([-w_pca_norm[1], w_pca_norm[0]])
        ax.plot([-perp[0]*scale*2, perp[0]*scale*2], 
               [-perp[1]*scale*2, perp[1]*scale*2],
               'k--', linewidth=1.5, alpha=0.5, label='Decision boundary')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        title = 'Without Prefix' if prefix_state == 'no_prefix' else 'With Prefix'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Step D: PCA Visualization (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_d_pca_panels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Probe Geometry Analysis with Soft Prefix")
    
    # Activation directories
    parser.add_argument("--no_prefix_id_dir", type=str, required=True,
                       help="ID activations without prefix")
    parser.add_argument("--no_prefix_ood_dir", type=str, required=True,
                       help="OOD activations without prefix")
    parser.add_argument("--with_prefix_id_dir", type=str, default=None,
                       help="ID activations with prefix (optional)")
    parser.add_argument("--with_prefix_ood_dir", type=str, required=True,
                       help="OOD activations with prefix")
    
    # Probes
    parser.add_argument("--probes_dir", type=str, required=True,
                       help="Directory with probe_layer_*.pt files")
    parser.add_argument("--probe_type", type=str, default="vanilla",
                       choices=["vanilla", "layer_agnostic", "per_token"],
                       help="Type of probe for labeling")
    parser.add_argument("--pooling", type=str, default="last",
                       help="Pooling type of the probes")
    
    # Analysis options
    parser.add_argument("--layers", type=str, default="all",
                       help="Layers to analyze: 'all' or comma-separated (e.g., '0,5,10,15,20')")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                       default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/prefix_last_token_results/prefix_geometry_analysis",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROBE GEOMETRY ANALYSIS WITH SOFT PREFIX")
    print("=" * 80)
    
    # Load activations
    print("\n[1/5] Loading activations...")
    
    print("  No-prefix ID:")
    data_no_prefix_id = load_activations_from_dir(args.no_prefix_id_dir)
    
    print("  No-prefix OOD:")
    data_no_prefix_ood = load_activations_from_dir(args.no_prefix_ood_dir)
    
    if args.with_prefix_id_dir:
        print("  With-prefix ID:")
        data_with_prefix_id = load_activations_from_dir(args.with_prefix_id_dir)
    else:
        print("  With-prefix ID: Not provided, will skip ID prefix analysis")
        data_with_prefix_id = data_no_prefix_id  # Use no-prefix as placeholder
    
    print("  With-prefix OOD:")
    data_with_prefix_ood = load_activations_from_dir(args.with_prefix_ood_dir)
    
    # Prepare data dicts
    data_no_prefix = {'id': data_no_prefix_id, 'ood': data_no_prefix_ood}
    data_with_prefix = {'id': data_with_prefix_id, 'ood': data_with_prefix_ood}
    
    # Get hidden dim
    hidden_dim = data_no_prefix_id.activations.shape[2]
    n_layers = data_no_prefix_id.activations.shape[1]
    print(f"\n  Hidden dim: {hidden_dim}, Layers: {n_layers}")
    
    # Determine layers to analyze
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    
    print(f"  Analyzing layers: {layers}")
    
    # Find probe files
    print(f"\n[2/5] Loading probes from {args.probes_dir}...")
    probe_pattern = os.path.join(args.probes_dir, "probe_layer_*.pt")
    probe_files = sorted(glob.glob(probe_pattern), 
                        key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    
    if not probe_files:
        print(f"  ERROR: No probe files found!")
        return 1
    
    print(f"  Found {len(probe_files)} probe files")
    
    # Create output directory
    probe_type_dir = os.path.join(args.output_dir, f"{args.probe_type}_{args.pooling}")
    os.makedirs(probe_type_dir, exist_ok=True)
    
    # Run analysis for each layer
    print(f"\n[3/5] Running analysis for {len(layers)} layers...")
    
    all_layer_metrics = {}
    
    for layer_idx in tqdm(layers, desc="Analyzing layers"):
        # Find probe for this layer
        probe_path = os.path.join(args.probes_dir, f"probe_layer_{layer_idx}.pt")
        if not os.path.exists(probe_path):
            print(f"  Warning: No probe for layer {layer_idx}, skipping")
            continue
        
        # Load probe and get weight
        probe = load_probe(probe_path, hidden_dim, args.pooling)
        probe_weight = get_probe_weight(probe)
        
        # Create layer output directory
        layer_dir = os.path.join(probe_type_dir, f"layer_{layer_idx:02d}")
        
        # Run all steps
        layer_metrics = {}
        
        # Step A
        metrics_a = step_a_probe_axis_geometry(
            data_no_prefix, data_with_prefix, probe_weight, layer_idx, layer_dir
        )
        layer_metrics.update({f'step_a_{k}': v for k, v in metrics_a.items()})
        
        # Step B
        metrics_b = step_b_shift_analysis(
            data_no_prefix, data_with_prefix, layer_idx, layer_dir
        )
        layer_metrics.update({f'step_b_{k}': v for k, v in metrics_b.items()})
        
        # Step C
        metrics_c = step_c_delta_alignment(
            data_no_prefix, data_with_prefix, probe_weight, layer_idx, layer_dir
        )
        layer_metrics.update({f'step_c_{k}': v for k, v in metrics_c.items()})
        
        # Step D
        metrics_d = step_d_pca_visualization(
            data_no_prefix, data_with_prefix, probe_weight, layer_idx, layer_dir
        )
        layer_metrics.update({f'step_d_{k}': v for k, v in metrics_d.items()})
        
        # Save layer metrics
        with open(os.path.join(layer_dir, 'metrics.json'), 'w') as f:
            json.dump(layer_metrics, f, indent=2)
        
        all_layer_metrics[layer_idx] = layer_metrics
    
    # Create cross-layer summary
    print(f"\n[4/5] Creating cross-layer summary...")
    
    summary = {
        'probe_type': args.probe_type,
        'pooling': args.pooling,
        'n_layers': n_layers,
        'layers_analyzed': layers,
        'layer_metrics': all_layer_metrics
    }
    
    summary_path = os.path.join(probe_type_dir, 'cross_layer_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved: {summary_path}")
    
    # Create cross-layer plots
    print(f"\n[5/5] Creating cross-layer visualization...")
    
    # AUROC across layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, prefix_state in enumerate(['no_prefix', 'with_prefix']):
        for col, domain in enumerate(['id', 'ood']):
            ax = axes[row, col]
            
            aurocs = []
            for layer in layers:
                if layer in all_layer_metrics:
                    key = f'step_a_{domain}_{prefix_state}_auroc'
                    aurocs.append(all_layer_metrics[layer].get(key, 0.5))
                else:
                    aurocs.append(0.5)
            
            ax.plot(layers, aurocs, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylabel('AUROC', fontsize=11)
            title = f'{"ID" if domain == "id" else "OOD"} - {"No Prefix" if prefix_state == "no_prefix" else "With Prefix"}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylim(0.4, 1.0)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'AUROC Across Layers ({args.probe_type}_{args.pooling})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(probe_type_dir, 'auroc_across_layers.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output: {probe_type_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
