#!/usr/bin/env python3
"""
Invariant Core Semantic Analysis (Investigation 1)
===================================================

This script analyzes what the invariant core direction captures:
1. Cosine Similarity Analysis - Verify orthogonality between components
2. Vocabulary Projection - Find tokens aligned with deception signal
3. Activation Patching Analysis - Per-token contributions
4. Cross-Domain Activation Comparison - Scatter visualizations

Usage (Colab):
    python scripts/analysis/analyze_invariant_core_semantics.py \
        --base_data_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data \
        --invariant_probe_dir data/invariant_probes/meta-llama_Llama-3.2-3B-Instruct/layer_16_mean \
        --output_dir data/results/invariant_probe_analysis \
        --model_path meta-llama/Llama-3.2-3B-Instruct

Output:
    data/results/invariant_probe_analysis/
        ├── cosine_similarity_matrix.png
        ├── vocabulary_projection.json
        ├── cross_domain_scatter.png
        └── per_token_contributions.json
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# For vocabulary projection
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Vocabulary projection will be skipped.")


# ============================================================================
# COSINE SIMILARITY ANALYSIS
# ============================================================================
def compute_cosine_similarity_matrix(directions):
    """Compute pairwise cosine similarities between directions."""
    names = list(directions.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            d_i = directions[name_i]
            d_j = directions[name_j]
            norm_i = np.linalg.norm(d_i)
            norm_j = np.linalg.norm(d_j)
            if norm_i > 1e-10 and norm_j > 1e-10:
                matrix[i, j] = np.dot(d_i, d_j) / (norm_i * norm_j)
            else:
                matrix[i, j] = 0
    
    return matrix, names


def plot_cosine_similarity_matrix(matrix, names, output_path):
    """Plot cosine similarity matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    
    # Add values
    for i in range(len(names)):
        for j in range(len(names)):
            text = f'{matrix[i, j]:.3f}'
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Cosine Similarity Between Probe Directions', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# VOCABULARY PROJECTION
# ============================================================================
def project_onto_vocabulary(direction, model_path, tokenizer_path=None, top_k=50):
    """
    Project the invariant direction onto token embeddings.
    Returns top-k tokens most aligned with the direction.
    """
    if not HAS_TRANSFORMERS:
        return {'error': 'transformers not installed'}
    
    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    
    print(f"  Loading model embeddings from {model_path}...")
    try:
        # Load only the embedding layer
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        embeddings = model.get_input_embeddings().weight.float().detach().cpu().numpy()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  Warning: Could not load model embeddings: {e}")
        return {'error': str(e)}
    
    print(f"  Embedding matrix shape: {embeddings.shape}")
    
    # Normalize direction
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Check dimension compatibility
    if embeddings.shape[1] != len(direction):
        print(f"  Warning: Dimension mismatch! Embeddings: {embeddings.shape[1]}, Direction: {len(direction)}")
        return {'error': 'dimension_mismatch', 'embed_dim': embeddings.shape[1], 'dir_dim': len(direction)}
    
    # Project: scores = E @ direction
    scores = embeddings @ direction
    
    # Get top-k positive (aligned with deception)
    top_pos_indices = np.argsort(scores)[-top_k:][::-1]
    # Get top-k negative (aligned with honesty)
    top_neg_indices = np.argsort(scores)[:top_k]
    
    results = {
        'top_deceptive_tokens': [],
        'top_honest_tokens': []
    }
    
    for idx in top_pos_indices:
        token = tokenizer.decode([idx])
        results['top_deceptive_tokens'].append({
            'token': token,
            'token_id': int(idx),
            'score': float(scores[idx])
        })
    
    for idx in top_neg_indices:
        token = tokenizer.decode([idx])
        results['top_honest_tokens'].append({
            'token': token,
            'token_id': int(idx),
            'score': float(scores[idx])
        })
    
    return results


# ============================================================================
# ACTIVATION LOADING
# ============================================================================
def load_activations(act_dir, layer, pooling='mean', num_samples=None, return_raw=False):
    """Load activations from a directory."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        return None, None, None
    
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    if num_samples is not None and num_samples < len(manifest):
        manifest = manifest[:num_samples]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations, labels, raw_activations = [], [], []
    
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            tensor = all_tensors[eid].numpy()
            x_layer = tensor[layer, :, :]  # (T, D)
            
            # Apply pooling
            if pooling == 'mean':
                pooled = x_layer.mean(axis=0)
            elif pooling == 'max':
                pooled = x_layer.max(axis=0)
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            else:
                pooled = x_layer.mean(axis=0)
            
            activations.append(pooled)
            labels.append(entry['label'])
            if return_raw:
                raw_activations.append(x_layer)
    
    if return_raw:
        return np.array(activations), np.array(labels), raw_activations
    return np.array(activations), np.array(labels), None


# ============================================================================
# PER-TOKEN CONTRIBUTION ANALYSIS
# ============================================================================
def analyze_per_token_contributions(raw_activations, labels, direction, num_samples=10):
    """
    Analyze per-token contributions to the residual direction.
    For each token t: contrib_t = x_t · direction
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    results = []
    
    for i, (x_raw, label) in enumerate(zip(raw_activations[:num_samples], labels[:num_samples])):
        # x_raw: (T, D)
        contributions = x_raw @ direction  # (T,)
        
        # Find most important tokens
        top_positive_idx = np.argsort(contributions)[-5:][::-1]
        top_negative_idx = np.argsort(contributions)[:5]
        
        sample_result = {
            'sample_idx': i,
            'label': int(label),
            'total_projection': float(contributions.sum()),
            'mean_projection': float(contributions.mean()),
            'std_projection': float(contributions.std()),
            'top_positive_positions': [int(idx) for idx in top_positive_idx],
            'top_positive_contributions': [float(contributions[idx]) for idx in top_positive_idx],
            'top_negative_positions': [int(idx) for idx in top_negative_idx],
            'top_negative_contributions': [float(contributions[idx]) for idx in top_negative_idx]
        }
        results.append(sample_result)
    
    return results


# ============================================================================
# CROSS-DOMAIN SCATTER PLOT
# ============================================================================
def plot_cross_domain_scatter(X_a, y_a, X_b, y_b, direction, output_path, 
                              label_a='Roleplaying', label_b='InsiderTrading'):
    """
    Create scatter plot of activations projected onto residual direction.
    Shows whether deceptive samples from both domains cluster together.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Project onto direction
    proj_a = X_a @ direction
    proj_b = X_b @ direction
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram comparison
    ax = axes[0]
    
    # Domain A
    ax.hist(proj_a[y_a == 0], bins=30, alpha=0.5, label=f'{label_a} Honest', color='blue')
    ax.hist(proj_a[y_a == 1], bins=30, alpha=0.5, label=f'{label_a} Deceptive', color='red')
    # Domain B
    ax.hist(proj_b[y_b == 0], bins=30, alpha=0.5, label=f'{label_b} Honest', color='cyan', 
            histtype='step', linewidth=2)
    ax.hist(proj_b[y_b == 1], bins=30, alpha=0.5, label=f'{label_b} Deceptive', color='orange',
            histtype='step', linewidth=2)
    
    ax.set_xlabel('Projection onto Invariant Core', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Projections by Domain and Label', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 2D scatter (add random dim for visualization)
    ax = axes[1]
    
    # Create a second dimension (e.g., projection onto random orthogonal direction)
    np.random.seed(42)
    random_dir = np.random.randn(len(direction))
    random_dir = random_dir - np.dot(random_dir, direction) * direction  # Orthogonalize
    random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
    
    proj_a_2 = X_a @ random_dir
    proj_b_2 = X_b @ random_dir
    
    # Plot
    colors = ['blue', 'red', 'cyan', 'orange']
    labels_plot = [f'{label_a} Honest', f'{label_a} Deceptive', 
                   f'{label_b} Honest', f'{label_b} Deceptive']
    
    ax.scatter(proj_a[y_a == 0], proj_a_2[y_a == 0], alpha=0.5, label=labels_plot[0], c='blue', marker='o')
    ax.scatter(proj_a[y_a == 1], proj_a_2[y_a == 1], alpha=0.5, label=labels_plot[1], c='red', marker='o')
    ax.scatter(proj_b[y_b == 0], proj_b_2[y_b == 0], alpha=0.5, label=labels_plot[2], c='cyan', marker='s')
    ax.scatter(proj_b[y_b == 1], proj_b_2[y_b == 1], alpha=0.5, label=labels_plot[3], c='orange', marker='s')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Projection onto Invariant Core', fontsize=12)
    ax.set_ylabel('Orthogonal Dimension', fontsize=12)
    ax.set_title('Cross-Domain Scatter Plot', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    # Return statistics
    return {
        'domain_a': {
            'honest_mean': float(np.mean(proj_a[y_a == 0])),
            'honest_std': float(np.std(proj_a[y_a == 0])),
            'deceptive_mean': float(np.mean(proj_a[y_a == 1])),
            'deceptive_std': float(np.std(proj_a[y_a == 1]))
        },
        'domain_b': {
            'honest_mean': float(np.mean(proj_b[y_b == 0])),
            'honest_std': float(np.std(proj_b[y_b == 0])),
            'deceptive_mean': float(np.mean(proj_b[y_b == 1])),
            'deceptive_std': float(np.std(proj_b[y_b == 1]))
        }
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Invariant Core Semantic Analysis")
    parser.add_argument('--base_data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--invariant_probe_dir', type=str, required=True,
                        help='Directory with saved invariant probe')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='HuggingFace model path for vocabulary projection')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct',
                        help='Model name')
    parser.add_argument('--layer', type=int, default=16, help='Layer')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling type')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for analysis')
    parser.add_argument('--skip_vocab', action='store_true',
                        help='Skip vocabulary projection (requires loading model)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("INVARIANT CORE SEMANTIC ANALYSIS (Investigation 1)")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DIRECTIONS
    # ========================================================================
    print("\n1. Loading invariant core directions...")
    
    # Handle relative or absolute paths
    if os.path.isabs(args.invariant_probe_dir):
        probe_dir = args.invariant_probe_dir
    else:
        probe_dir = os.path.join(args.base_data_dir, args.invariant_probe_dir)
    
    residual_path = os.path.join(probe_dir, 'residual_direction.npy')
    e1_path = os.path.join(probe_dir, 'e1_roleplaying.npy')
    e2_path = os.path.join(probe_dir, 'e2_insider_orth.npy')
    
    for path, name in [(residual_path, 'residual'), (e1_path, 'e1'), (e2_path, 'e2')]:
        if not os.path.exists(path):
            print(f"  ERROR: {name} not found: {path}")
            return 1
    
    residual = np.load(residual_path)
    e1 = np.load(e1_path)
    e2 = np.load(e2_path)
    
    print(f"   Loaded directions with dimension: {len(residual)}")
    
    directions = {
        'Roleplaying (e1)': e1,
        'InsiderTrading Orth (e2)': e2,
        'Invariant Core (residual)': residual
    }
    
    # ========================================================================
    # 2. COSINE SIMILARITY ANALYSIS
    # ========================================================================
    print("\n2. Computing cosine similarity matrix...")
    
    matrix, names = compute_cosine_similarity_matrix(directions)
    
    print("   Cosine Similarity Matrix:")
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            print(f"     cos({name_i[:15]}, {name_j[:15]}) = {matrix[i, j]:.4f}")
    
    plot_cosine_similarity_matrix(
        matrix, names,
        os.path.join(args.output_dir, 'cosine_similarity_matrix.png')
    )
    
    # ========================================================================
    # 3. VOCABULARY PROJECTION
    # ========================================================================
    vocab_results = {}
    if not args.skip_vocab and args.model_path:
        print("\n3. Projecting onto vocabulary...")
        vocab_results = project_onto_vocabulary(
            residual, 
            args.model_path,
            top_k=50
        )
        
        if 'error' not in vocab_results:
            print("\n   Top 10 tokens aligned with deception:")
            for token_info in vocab_results['top_deceptive_tokens'][:10]:
                print(f"     {repr(token_info['token']):20s} score={token_info['score']:.4f}")
            
            print("\n   Top 10 tokens aligned with honesty:")
            for token_info in vocab_results['top_honest_tokens'][:10]:
                print(f"     {repr(token_info['token']):20s} score={token_info['score']:.4f}")
        
        with open(os.path.join(args.output_dir, 'vocabulary_projection.json'), 'w') as f:
            json.dump(vocab_results, f, indent=2)
        print(f"  ✓ Saved: vocabulary_projection.json")
    else:
        print("\n3. Skipping vocabulary projection (--skip_vocab or no --model_path)")
    
    # ========================================================================
    # 4. LOAD ACTIVATIONS FOR CROSS-DOMAIN ANALYSIS
    # ========================================================================
    print("\n4. Loading activations for cross-domain analysis...")
    
    act_a_dir = os.path.join(args.base_data_dir, 'activations', args.model, 
                             'Deception-Roleplaying', 'validation')
    act_b_dir = os.path.join(args.base_data_dir, 'activations', args.model, 
                             'Deception-InsiderTrading', 'validation')
    
    X_a, y_a, raw_a = None, None, None
    X_b, y_b, raw_b = None, None, None
    
    if os.path.exists(act_a_dir):
        X_a, y_a, raw_a = load_activations(act_a_dir, args.layer, args.pooling, 
                                            args.num_samples, return_raw=True)
        print(f"   Loaded {len(X_a)} Roleplaying samples")
    else:
        print(f"   Warning: Roleplaying activations not found: {act_a_dir}")
    
    if os.path.exists(act_b_dir):
        X_b, y_b, raw_b = load_activations(act_b_dir, args.layer, args.pooling, 
                                            args.num_samples, return_raw=True)
        print(f"   Loaded {len(X_b)} InsiderTrading samples")
    else:
        print(f"   Warning: InsiderTrading activations not found: {act_b_dir}")
    
    # ========================================================================
    # 5. PER-TOKEN CONTRIBUTION ANALYSIS
    # ========================================================================
    token_analysis = {}
    if raw_a is not None or raw_b is not None:
        print("\n5. Analyzing per-token contributions...")
        
        if raw_a is not None:
            token_analysis['roleplaying'] = analyze_per_token_contributions(
                raw_a, y_a, residual, num_samples=min(10, len(raw_a))
            )
            print(f"   Analyzed {len(token_analysis['roleplaying'])} Roleplaying samples")
        
        if raw_b is not None:
            token_analysis['insider_trading'] = analyze_per_token_contributions(
                raw_b, y_b, residual, num_samples=min(10, len(raw_b))
            )
            print(f"   Analyzed {len(token_analysis['insider_trading'])} InsiderTrading samples")
        
        with open(os.path.join(args.output_dir, 'per_token_contributions.json'), 'w') as f:
            json.dump(token_analysis, f, indent=2)
        print(f"  ✓ Saved: per_token_contributions.json")
    else:
        print("\n5. Skipping per-token analysis (no raw activations)")
    
    # ========================================================================
    # 6. CROSS-DOMAIN SCATTER PLOT
    # ========================================================================
    scatter_stats = {}
    if X_a is not None and X_b is not None:
        print("\n6. Creating cross-domain scatter plot...")
        
        # Normalize activations jointly
        X_all = np.vstack([X_a, X_b])
        mean, std = X_all.mean(0), X_all.std(0) + 1e-8
        X_a_norm = (X_a - mean) / std
        X_b_norm = (X_b - mean) / std
        
        scatter_stats = plot_cross_domain_scatter(
            X_a_norm, y_a, X_b_norm, y_b, residual,
            os.path.join(args.output_dir, 'cross_domain_scatter.png')
        )
    else:
        print("\n6. Skipping cross-domain scatter (missing activations)")
    
    # ========================================================================
    # 7. SAVE SUMMARY
    # ========================================================================
    print("\n7. Saving analysis summary...")
    
    analysis_summary = {
        'config': {
            'model': args.model,
            'layer': args.layer,
            'pooling': args.pooling,
            'num_samples': args.num_samples
        },
        'cosine_similarity': {
            names[i]: {names[j]: float(matrix[i, j]) for j in range(len(names))}
            for i in range(len(names))
        },
        'orthogonality_verified': {
            'residual_e1': float(np.abs(np.dot(residual, e1) / (np.linalg.norm(residual) * np.linalg.norm(e1) + 1e-10))),
            'residual_e2': float(np.abs(np.dot(residual, e2) / (np.linalg.norm(residual) * np.linalg.norm(e2) + 1e-10))),
            'is_orthogonal_e1': bool(np.abs(np.dot(residual, e1)) < 0.01),
            'is_orthogonal_e2': bool(np.abs(np.dot(residual, e2)) < 0.01)
        },
        'scatter_statistics': scatter_stats,
        'vocabulary_projection': {
            'has_results': 'error' not in vocab_results,
            'top_deceptive_tokens': vocab_results.get('top_deceptive_tokens', [])[:10],
            'top_honest_tokens': vocab_results.get('top_honest_tokens', [])[:10]
        } if vocab_results else {}
    }
    
    with open(os.path.join(args.output_dir, 'semantic_analysis_summary.json'), 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"  ✓ Saved: semantic_analysis_summary.json")
    
    # ========================================================================
    # 8. SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nKey findings:")
    
    # Check orthogonality
    cos_r_e1 = np.abs(np.dot(residual, e1) / (np.linalg.norm(residual) * np.linalg.norm(e1) + 1e-10))
    cos_r_e2 = np.abs(np.dot(residual, e2) / (np.linalg.norm(residual) * np.linalg.norm(e2) + 1e-10))
    
    if cos_r_e1 < 0.01 and cos_r_e2 < 0.01:
        print("  ✅ Orthogonality verified: residual is orthogonal to both e1 and e2")
    else:
        print(f"  ⚠️  Residual not fully orthogonal: cos(r,e1)={cos_r_e1:.4f}, cos(r,e2)={cos_r_e2:.4f}")
    
    if scatter_stats:
        sep_a = scatter_stats['domain_a']['deceptive_mean'] - scatter_stats['domain_a']['honest_mean']
        sep_b = scatter_stats['domain_b']['deceptive_mean'] - scatter_stats['domain_b']['honest_mean']
        print(f"  📊 Separation in Roleplaying: {sep_a:.4f}")
        print(f"  📊 Separation in InsiderTrading: {sep_b:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
