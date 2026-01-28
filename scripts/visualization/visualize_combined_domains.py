#!/usr/bin/env python3
"""
Visualize Combined Domain Activations with UMAP/PCA
====================================================
Cluster and visualize activations from both domains to understand:
1. Do domains form separate clusters or overlap?
2. Is there a shared "deception" manifold?
3. How does pooling type affect the representation space?

Usage:
    python scripts/visualization/visualize_combined_domains.py \
        --act_a /path/to/Deception-Roleplaying/train \
        --act_b /path/to/Deception-InsiderTrading/train \
        --layer 20 --pooling mean \
        --output_dir results/domain_clustering
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try to import UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")

POOLING_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'attn': '#06A77D'
}


def load_activations(act_dir, layer, pooling):
    """Load and pool activations from a directory.
    
    Supports both:
    - Standard activations: (L, T, D) - requires pooling
    - Prompted-probing activations: (L, D) - no pooling needed
    """
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations, labels, ids = [], [], []
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            tensor = all_tensors[eid]
            
            # Auto-detect format based on tensor shape
            if len(tensor.shape) == 2:
                # Prompted-probing format: (L, D)
                pooled = tensor[layer, :]  # (D,)
            elif len(tensor.shape) == 3:
                # Standard format: (L, T, D)
                x_layer = tensor[layer, :, :]  # (T, D)
                
                if pooling == 'mean':
                    pooled = x_layer.mean(dim=0)
                elif pooling == 'max':
                    pooled = x_layer.max(dim=0)[0]
                elif pooling == 'last':
                    pooled = x_layer[-1, :]
                else:
                    pooled = x_layer.mean(dim=0)  # Default to mean
            else:
                print(f"Unexpected tensor shape for {eid}: {tensor.shape}")
                continue
            
            activations.append(pooled.numpy())
            labels.append(entry['label'])
            ids.append(eid)
    
    return np.array(activations), np.array(labels), ids


def plot_2d_projection(X_2d, domain_labels, deception_labels, title, output_path, method_name):
    """Create 2D scatter plot with domain and deception coloring."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Color by Domain
    ax = axes[0]
    for domain, color, marker in [('A', '#3498db', 'o'), ('B', '#e74c3c', 's')]:
        mask = domain_labels == domain
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, marker=marker, 
                   alpha=0.6, s=50, label=f'Domain {domain}', edgecolors='white', linewidths=0.5)
    ax.set_xlabel(f'{method_name} 1', fontsize=12)
    ax.set_ylabel(f'{method_name} 2', fontsize=12)
    ax.set_title(f'Colored by Domain', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by Deception Label (within each domain)
    ax = axes[1]
    colors = {
        ('A', 0): '#aed6f1',  # Light blue - Domain A, No deception
        ('A', 1): '#2874a6',  # Dark blue - Domain A, Deception
        ('B', 0): '#f5b7b1',  # Light red - Domain B, No deception
        ('B', 1): '#922b21',  # Dark red - Domain B, Deception
    }
    markers = {'A': 'o', 'B': 's'}
    
    for domain in ['A', 'B']:
        for label in [0, 1]:
            mask = (domain_labels == domain) & (deception_labels == label)
            deception_str = 'Deceptive' if label == 1 else 'Truthful'
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[(domain, label)], marker=markers[domain],
                       alpha=0.7, s=50, label=f'{domain}: {deception_str}',
                       edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel(f'{method_name} 1', fontsize=12)
    ax.set_ylabel(f'{method_name} 2', fontsize=12)
    ax.set_title(f'Colored by Domain + Deception Label', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cluster_analysis(X, domain_labels, deception_labels, output_path):
    """Analyze clustering quality and domain separation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. PCA Explained Variance
    ax = axes[0]
    pca = PCA(n_components=min(50, X.shape[1]))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=4)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    n_90 = np.argmax(cumvar >= 0.9) + 1
    ax.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title(f'PCA Variance (90% at {n_90} components)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. K-Means Clustering (find natural clusters)
    ax = axes[1]
    silhouette_scores = []
    k_range = range(2, 8)
    X_pca = PCA(n_components=50).fit_transform(X)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, cluster_labels)
        silhouette_scores.append(score)
    
    ax.plot(list(k_range), silhouette_scores, 'g-o', markersize=8)
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Optimal Cluster Count', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Domain/Class Distribution in Clusters
    ax = axes[2]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    cluster_stats = []
    for c in range(best_k):
        mask = cluster_labels == c
        n_total = mask.sum()
        n_domain_a = ((domain_labels == 'A') & mask).sum()
        n_domain_b = ((domain_labels == 'B') & mask).sum()
        n_deceptive = (deception_labels[mask] == 1).sum()
        cluster_stats.append({
            'cluster': c,
            'total': n_total,
            'domain_a_pct': n_domain_a / n_total * 100,
            'domain_b_pct': n_domain_b / n_total * 100,
            'deceptive_pct': n_deceptive / n_total * 100
        })
    
    x = np.arange(best_k)
    width = 0.25
    ax.bar(x - width, [s['domain_a_pct'] for s in cluster_stats], width, label='Domain A %', color='#3498db')
    ax.bar(x, [s['domain_b_pct'] for s in cluster_stats], width, label='Domain B %', color='#e74c3c')
    ax.bar(x + width, [s['deceptive_pct'] for s in cluster_stats], width, label='Deceptive %', color='#2ecc71')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Cluster Composition', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    return cluster_stats, best_k


def compute_domain_overlap_metrics(X, domain_labels):
    """Compute metrics for domain overlap/separation."""
    X_a = X[domain_labels == 'A']
    X_b = X[domain_labels == 'B']
    
    # Centroid distance
    centroid_a = X_a.mean(axis=0)
    centroid_b = X_b.mean(axis=0)
    centroid_dist = np.linalg.norm(centroid_a - centroid_b)
    
    # Average within-domain variance
    var_a = np.var(X_a, axis=0).mean()
    var_b = np.var(X_b, axis=0).mean()
    avg_var = (var_a + var_b) / 2
    
    # Separation ratio (higher = more separated)
    separation_ratio = centroid_dist / (2 * np.sqrt(avg_var) + 1e-8)
    
    return {
        'centroid_distance': float(centroid_dist),
        'avg_within_variance': float(avg_var),
        'separation_ratio': float(separation_ratio)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_a', type=str, required=True, help='Domain A activations dir')
    parser.add_argument('--act_b', type=str, required=True, help='Domain B activations dir')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'last'])
    parser.add_argument('--output_dir', type=str, default='results/domain_clustering')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("COMBINED DOMAIN ACTIVATION CLUSTERING")
    print("=" * 70)
    print(f"Layer: {args.layer}, Pooling: {args.pooling}")
    print("=" * 70)
    
    # Load activations
    print("\nLoading activations...")
    X_a, y_a, ids_a = load_activations(args.act_a, args.layer, args.pooling)
    X_b, y_b, ids_b = load_activations(args.act_b, args.layer, args.pooling)
    
    print(f"  Domain A ({args.label_a}): {len(X_a)} samples")
    print(f"  Domain B ({args.label_b}): {len(X_b)} samples")
    
    # Combine
    X = np.vstack([X_a, X_b])
    deception_labels = np.concatenate([y_a, y_b])
    domain_labels = np.array(['A'] * len(X_a) + ['B'] * len(X_b))
    
    # Normalize
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    
    print(f"\nCombined: {len(X)} samples, {X.shape[1]} dimensions")
    
    # Compute domain overlap metrics
    print("\n" + "=" * 70)
    print("DOMAIN OVERLAP METRICS")
    print("=" * 70)
    metrics = compute_domain_overlap_metrics(X_norm, domain_labels)
    print(f"  Centroid distance: {metrics['centroid_distance']:.4f}")
    print(f"  Avg within-domain variance: {metrics['avg_within_variance']:.4f}")
    print(f"  Separation ratio: {metrics['separation_ratio']:.4f}")
    
    if metrics['separation_ratio'] < 1.0:
        print("  → Domains are OVERLAPPING (separation < 1)")
    else:
        print("  → Domains are SEPARATED (separation >= 1)")
    
    # PCA
    print("\n" + "=" * 70)
    print("GENERATING PCA VISUALIZATION")
    print("=" * 70)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)
    
    title = f'PCA: Layer {args.layer}, {args.pooling.upper()} Pooling\n' \
            f'{args.label_a} vs {args.label_b}'
    plot_2d_projection(X_pca, domain_labels, deception_labels, title,
                       os.path.join(args.output_dir, f'pca_layer{args.layer}_{args.pooling}.png'),
                       'PC')
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # t-SNE
    print("\n" + "=" * 70)
    print("GENERATING t-SNE VISUALIZATION")
    print("=" * 70)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_norm)
    
    title = f't-SNE: Layer {args.layer}, {args.pooling.upper()} Pooling\n' \
            f'{args.label_a} vs {args.label_b}'
    plot_2d_projection(X_tsne, domain_labels, deception_labels, title,
                       os.path.join(args.output_dir, f'tsne_layer{args.layer}_{args.pooling}.png'),
                       't-SNE')
    
    # UMAP
    if HAS_UMAP:
        print("\n" + "=" * 70)
        print("GENERATING UMAP VISUALIZATION")
        print("=" * 70)
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_norm)
        
        title = f'UMAP: Layer {args.layer}, {args.pooling.upper()} Pooling\n' \
                f'{args.label_a} vs {args.label_b}'
        plot_2d_projection(X_umap, domain_labels, deception_labels, title,
                           os.path.join(args.output_dir, f'umap_layer{args.layer}_{args.pooling}.png'),
                           'UMAP')
    
    # Cluster analysis
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)
    cluster_stats, best_k = plot_cluster_analysis(X_norm, domain_labels, deception_labels,
                           os.path.join(args.output_dir, f'cluster_analysis_layer{args.layer}_{args.pooling}.png'))
    
    print(f"\n  Best number of clusters: {best_k}")
    print("\n  Cluster composition:")
    for stat in cluster_stats:
        print(f"    Cluster {stat['cluster']}: {stat['total']} samples | "
              f"A: {stat['domain_a_pct']:.1f}% | B: {stat['domain_b_pct']:.1f}% | "
              f"Deceptive: {stat['deceptive_pct']:.1f}%")
    
    # Check for mixed clusters (both domains present)
    mixed_clusters = sum(1 for s in cluster_stats if s['domain_a_pct'] > 20 and s['domain_b_pct'] > 20)
    print(f"\n  Mixed clusters (>20% each domain): {mixed_clusters}/{best_k}")
    
    if mixed_clusters > 0:
        print("  → Some shared representation exists between domains!")
    
    # Save summary
    summary = {
        'config': {
            'layer': args.layer,
            'pooling': args.pooling,
            'label_a': args.label_a,
            'label_b': args.label_b
        },
        'sample_counts': {
            'domain_a': len(X_a),
            'domain_b': len(X_b)
        },
        'domain_overlap_metrics': metrics,
        'cluster_analysis': {
            'best_k': best_k,
            'cluster_stats': cluster_stats
        }
    }
    
    with open(os.path.join(args.output_dir, f'clustering_summary_layer{args.layer}_{args.pooling}.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
