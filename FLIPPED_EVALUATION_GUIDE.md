# Flipped ID/OOD Evaluation Guide

## Overview
You're testing dataset symmetry by flipping the ID and OOD datasets:
- **Original**: ID = Roleplaying, OOD = InsiderTrading
- **Flipped**: ID = InsiderTrading, OOD = Roleplaying

## Current Status
✅ Collected activations for flipped ID dataset (InsiderTrading)
✅ Trained probes for all 4 pooling strategies
❌ Need to evaluate on flipped OOD dataset (Roleplaying)

---

## Problem: 0 Samples Loaded

The error you're seeing:
```
✓ Loaded 0 OOD samples
```

**Root Cause**: The script expects OOD activations in the format:
```
data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation/
├── shard_0.safetensors
├── shard_1.safetensors
└── manifest.jsonl
```

**Solution**: You need to either:
1. Cache activations for Roleplaying dataset first (see PHASE 1 below)
2. OR use existing activations from the correct path

---

## Complete Flipped Evaluation Pipeline

### PHASE 0: Check Your Activation Paths

```bash
# In Colab, check what activations you have:
!ls -la /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/

# Look for both datasets:
!ls -la /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/
!ls -la /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/
```

### PHASE 1: Cache Activations (if needed)

If you don't have Roleplaying activations cached:

```python
# Cache Roleplaying activations (for OOD evaluation in flipped case)
!python scripts/cache_deception_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name Deception-Roleplaying \
    --output_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --split validation \
    --batch_size 4 \
    --max_samples 1000
```

### PHASE 2: OOD Evaluation (All Pooling Strategies)

```python
# Evaluate all 4 pooling strategies on flipped OOD dataset
!python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --probes_base data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/ood_evaluation

# This will create:
# - results_flipped/ood_evaluation/ood_results_all_pooling.json
# - results_flipped/ood_evaluation/ood_layerwise_comparison.png
# - results_flipped/ood_evaluation/logits/*.npy (for ensemble evaluation)
```

### PHASE 3: Ensemble Evaluation (K-Sweep for Each Pooling)

Run comprehensive ensemble evaluation for each pooling strategy:

```python
# MEAN pooling ensembles
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/mean_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type mean \
    --output_dir results_flipped/ensembles/mean

# MAX pooling ensembles
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/max_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type max \
    --output_dir results_flipped/ensembles/max

# LAST pooling ensembles
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/last_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type last \
    --output_dir results_flipped/ensembles/last

# ATTN pooling ensembles
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type attn \
    --output_dir results_flipped/ensembles/attn
```

### PHASE 4: Compare All Pooling Ensembles

```python
# Compare all 4 pooling strategies side-by-side
!python scripts/compare_all_pooling_ensembles.py \
    --mean_results results_flipped/ensembles/mean/ensemble_k_sweep_ood.json \
    --max_results results_flipped/ensembles/max/ensemble_k_sweep_ood.json \
    --last_results results_flipped/ensembles/last/ensemble_k_sweep_ood.json \
    --attn_results results_flipped/ensembles/attn/ensemble_k_sweep_ood.json \
    --output_dir results_flipped/pooling_comparison

# This creates comprehensive comparison plots
```

### PHASE 5: Fixed vs Gated Comparison (Per Pooling)

```python
# Compare fixed vs gated ensembles for MEAN
!python scripts/compare_fixed_vs_gated.py \
    --ensemble_results results_flipped/ensembles/mean/ensemble_k_sweep_ood.json \
    --gated_models_dir results_flipped/ensembles/mean/gated_models_val \
    --ood_logits results_flipped/ood_evaluation/logits/mean_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --output_dir results_flipped/fixed_vs_gated/mean

# Repeat for MAX, LAST, ATTN
!python scripts/compare_fixed_vs_gated.py \
    --ensemble_results results_flipped/ensembles/max/ensemble_k_sweep_ood.json \
    --gated_models_dir results_flipped/ensembles/max/gated_models_val \
    --ood_logits results_flipped/ood_evaluation/logits/max_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --output_dir results_flipped/fixed_vs_gated/max

!python scripts/compare_fixed_vs_gated.py \
    --ensemble_results results_flipped/ensembles/last/ensemble_k_sweep_ood.json \
    --gated_models_dir results_flipped/ensembles/last/gated_models_val \
    --ood_logits results_flipped/ood_evaluation/logits/last_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --output_dir results_flipped/fixed_vs_gated/last

!python scripts/compare_fixed_vs_gated.py \
    --ensemble_results results_flipped/ensembles/attn/ensemble_k_sweep_ood.json \
    --gated_models_dir results_flipped/ensembles/attn/gated_models_val \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --output_dir results_flipped/fixed_vs_gated/attn
```

### PHASE 6: Gating Analysis (Fair Comparison)

```python
# Analyze gating weights with FAIR COMPARISON
# This trains gating on OOD data and compares to single probes
!python scripts/analyze_gating_weights.py \
    --id_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --k_pct 40 \
    --output_dir results_flipped/gating_analysis

# This creates the fair comparison plot with 4 subplots:
# 1. Single probe performance vs layer
# 2. ID-trained ensemble vs OOD-trained ensemble
# 3. Gating weights learned on OOD data
# 4. Performance on OOD test split
```

### PHASE 7: Attention Visualizations

```python
# Analyze attention entropy (where does attention focus?)
!python scripts/analyze_attention_entropy.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/attention_analysis

# Token-level attention visualization
!python scripts/analyze_attention_text.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/attention_text \
    --num_samples 10

# Ensemble attention visualization (per-layer attention)
!python scripts/analyze_ensemble_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/ensemble_attention

# Layer-colored attention (color-coded by layer contribution)
!python scripts/analyze_layer_colored_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/layer_colored_attention

# Hybrid attention analysis (compare pooling strategies)
!python scripts/analyze_hybrid_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_mean data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --probes_attn data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/hybrid_attention
```

### PHASE 8: Mechanistic Analysis

```python
# Deep dive into probe mechanisms
!python scripts/analyze_mechanisms.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/mechanistic_analysis

# Compare probe behavior across layers
!python scripts/analyze_probes.py \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/probe_analysis
```

### PHASE 9: Layerwise Pooling Comparison

```python
# Compare all pooling strategies layer-by-layer
!python scripts/compare_pooling_layerwise.py \
    --mean_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --max_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/max \
    --last_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/last \
    --attn_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --output_dir results_flipped/pooling_layerwise
```

---

## Quick Reference: All Scripts You Have

### Evaluation Scripts
- `evaluate_ood_all_pooling.py` - Evaluate all pooling strategies on OOD
- `evaluate_ensembles_comprehensive.py` - K-sweep for ensembles

### Comparison Scripts
- `compare_all_pooling_ensembles.py` - Compare 4 pooling strategies
- `compare_fixed_vs_gated.py` - Fixed vs gated ensemble comparison
- `compare_pooling_layerwise.py` - Layer-by-layer pooling comparison
- `compare_results.py` - Compare multiple result files

### Analysis Scripts
- `analyze_gating_weights.py` - Gating analysis with fair comparison
- `analyze_probes.py` - Probe performance analysis
- `analyze_mechanisms.py` - Mechanistic analysis

### Attention Visualization Scripts
- `analyze_attention_entropy.py` - Where attention focuses
- `analyze_attention_text.py` - Token-level attention
- `analyze_ensemble_attention.py` - Per-layer ensemble attention
- `analyze_layer_colored_attention.py` - Color-coded layer attention
- `analyze_hybrid_attention.py` - Compare pooling strategies
- `visualize_attention.py` - Basic attention visualization

---

## Expected Output Structure

```
results_flipped/
├── ood_evaluation/
│   ├── ood_results_all_pooling.json
│   ├── ood_layerwise_comparison.png
│   └── logits/
│       ├── mean_logits.npy
│       ├── max_logits.npy
│       ├── last_logits.npy
│       ├── attn_logits.npy
│       └── labels.npy
├── ensembles/
│   ├── mean/ensemble_k_sweep_ood.json
│   ├── max/ensemble_k_sweep_ood.json
│   ├── last/ensemble_k_sweep_ood.json
│   └── attn/ensemble_k_sweep_ood.json
├── pooling_comparison/
│   └── all_pooling_comparison.png
├── fixed_vs_gated/
│   ├── mean/comparison_results.json
│   ├── max/comparison_results.json
│   ├── last/comparison_results.json
│   └── attn/comparison_results.json
├── gating_analysis/
│   ├── gating_analysis.png (4-panel fair comparison)
│   └── gating_analysis_results.json
├── attention_analysis/
├── attention_text/
├── ensemble_attention/
├── layer_colored_attention/
├── hybrid_attention/
├── mechanistic_analysis/
└── probe_analysis/
```

---

## Troubleshooting

### Issue: "Loaded 0 OOD samples"

**Cause**: Activation files don't exist or path is wrong

**Fix**:
1. Check the path exists:
   ```bash
   !ls -la data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation/
   ```
2. If empty, run PHASE 1 to cache activations
3. Check for `shard_*.safetensors` files

### Issue: "No probe files found"

**Cause**: Probes not trained yet

**Fix**: Train probes first using `train_deception_probes.py`

### Issue: Out of memory

**Cause**: Too many samples loaded at once

**Fix**:
- Reduce batch size: `--batch_size 2`
- Reduce max samples: `--max_samples 500`

---

## Comparing Original vs Flipped Results

After running both pipelines, compare:

```python
# Compare OOD performance: Original vs Flipped
!python scripts/compare_results.py \
    --results1 results/ood_evaluation/ood_results_all_pooling.json \
    --results2 results_flipped/ood_evaluation/ood_results_all_pooling.json \
    --labels "Original (ID:Role→OOD:Insider)" "Flipped (ID:Insider→OOD:Role)" \
    --output_dir results_comparison/symmetry_test
```

This will show if the results are symmetric or if there's a domain bias.
