# Colab Commands for Flipped ID/OOD Evaluation

## 🔴 IMMEDIATE FIX: Your Current Issue

The error "Loaded 0 OOD samples" means your activation files don't exist or are in the wrong location.

### Step 1: Diagnose the Problem

```python
# Run the diagnostic script to check your activation files
!python scripts/diagnose_activations.py \
    --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation
```

This will tell you:
- ✅ If the directory exists
- ✅ If shard files are present
- ✅ How many samples are in each shard
- ❌ What's wrong if there are no samples

### Step 2: Check Alternative Paths

Your activations might be in a different split. Try:

```python
# Check what actually exists
!ls -la data/activations/meta-llama_Llama-3.2-3B-Instruct/
!ls -la data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/
!ls -la data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/

# Try other splits
!python scripts/diagnose_activations.py \
    --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train

!python scripts/diagnose_activations.py \
    --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/test
```

### Step 3: Cache Activations (if needed)

If you don't have Roleplaying activations, you need to cache them:

```python
# Cache Roleplaying validation set (for flipped OOD evaluation)
!python scripts/cache_deception_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name Deception-Roleplaying \
    --output_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --split validation \
    --batch_size 4 \
    --max_samples 1000

# Or use test split if you prefer
!python scripts/cache_deception_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name Deception-Roleplaying \
    --output_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --split test \
    --batch_size 4 \
    --max_samples 1000
```

---

## 🎯 COMPLETE EVALUATION PIPELINE

Once your activations are ready, run these commands in order:

### PHASE 1: OOD Evaluation (All Pooling Strategies)

```python
# Evaluate all 4 pooling strategies on flipped OOD dataset
!python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --probes_base data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/ood_evaluation

# 📊 Output:
# - results_flipped/ood_evaluation/ood_results_all_pooling.json
# - results_flipped/ood_evaluation/ood_layerwise_comparison.png
# - results_flipped/ood_evaluation/logits/*.npy
```

### PHASE 2: Ensemble K-Sweep (For Each Pooling Strategy)

```python
# Run K-sweep for MEAN pooling
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/mean_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type mean \
    --output_dir results_flipped/ensembles/mean

# Run K-sweep for MAX pooling
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/max_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type max \
    --output_dir results_flipped/ensembles/max

# Run K-sweep for LAST pooling
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/last_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type last \
    --output_dir results_flipped/ensembles/last

# Run K-sweep for ATTN pooling
!python scripts/evaluate_ensembles_comprehensive.py \
    --id_logits_dir results_flipped/ood_evaluation/logits \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --pooling_type attn \
    --output_dir results_flipped/ensembles/attn

# 📊 Output for each: results_flipped/ensembles/{pooling}/ensemble_k_sweep_ood.json
```

### PHASE 3: Compare All Pooling Strategies

```python
# Create side-by-side comparison of all pooling strategies
!python scripts/compare_all_pooling_ensembles.py \
    --mean_results results_flipped/ensembles/mean/ensemble_k_sweep_ood.json \
    --max_results results_flipped/ensembles/max/ensemble_k_sweep_ood.json \
    --last_results results_flipped/ensembles/last/ensemble_k_sweep_ood.json \
    --attn_results results_flipped/ensembles/attn/ensemble_k_sweep_ood.json \
    --output_dir results_flipped/pooling_comparison

# 📊 Output: results_flipped/pooling_comparison/all_pooling_comparison.png
```

### PHASE 4: Fixed vs Gated Comparison

```python
# Compare fixed ensembles vs gated ensembles for MEAN
!python scripts/compare_fixed_vs_gated.py \
    --ensemble_results results_flipped/ensembles/mean/ensemble_k_sweep_ood.json \
    --gated_models_dir results_flipped/ensembles/mean/gated_models_val \
    --ood_logits results_flipped/ood_evaluation/logits/mean_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --output_dir results_flipped/fixed_vs_gated/mean

# Repeat for other pooling strategies (MAX, LAST, ATTN)
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

### PHASE 5: Gating Analysis (Fair Comparison)

```python
# Analyze gating weights with fair comparison on OOD data
!python scripts/analyze_gating_weights.py \
    --id_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --k_pct 40 \
    --output_dir results_flipped/gating_analysis

# 📊 Output: 4-panel plot showing:
# 1. Single probe performance vs layer
# 2. ID-trained vs OOD-trained ensemble comparison
# 3. Gating weights learned on OOD data
# 4. Performance on OOD test split
```

### PHASE 6: Attention Visualizations

```python
# Attention entropy analysis (where does attention focus?)
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

# Ensemble attention (per-layer attention heatmaps)
!python scripts/analyze_ensemble_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/ensemble_attention

# Layer-colored attention (color by layer contribution)
!python scripts/analyze_layer_colored_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/layer_colored_attention

# Hybrid attention (compare pooling strategies)
!python scripts/analyze_hybrid_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_mean data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --probes_attn data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/hybrid_attention
```

### PHASE 7: Mechanistic Analysis

```python
# Deep dive into probe mechanisms
!python scripts/analyze_mechanisms.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/mechanistic_analysis

# Analyze probe behavior across layers
!python scripts/analyze_probes.py \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/probe_analysis
```

### PHASE 8: Layer-by-Layer Pooling Comparison

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

## 📊 COMPARING ORIGINAL VS FLIPPED RESULTS

After running both original and flipped evaluations, compare them:

```python
# Compare OOD performance: Original vs Flipped
!python scripts/compare_results.py \
    --results1 results/ood_evaluation/ood_results_all_pooling.json \
    --results2 results_flipped/ood_evaluation/ood_results_all_pooling.json \
    --labels "Original (ID:Role→OOD:Insider)" "Flipped (ID:Insider→OOD:Role)" \
    --output_dir results_comparison/symmetry_test

# This will show if results are symmetric or if there's domain bias
```

---

## 🚀 BATCH RUN ALL EVALUATIONS

If you want to run everything at once:

```python
# Create a shell script to run all commands
cat > run_flipped_evaluation.sh << 'EOF'
#!/bin/bash
set -e

echo "PHASE 1: OOD Evaluation"
python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --probes_base data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/ood_evaluation

echo "PHASE 2: Ensemble K-Sweep"
for pooling in mean max last attn; do
    echo "Running K-sweep for $pooling..."
    python scripts/evaluate_ensembles_comprehensive.py \
        --id_logits_dir results_flipped/ood_evaluation/logits \
        --ood_logits results_flipped/ood_evaluation/logits/${pooling}_logits.npy \
        --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
        --pooling_type $pooling \
        --output_dir results_flipped/ensembles/$pooling
done

echo "PHASE 3: Compare All Pooling"
python scripts/compare_all_pooling_ensembles.py \
    --mean_results results_flipped/ensembles/mean/ensemble_k_sweep_ood.json \
    --max_results results_flipped/ensembles/max/ensemble_k_sweep_ood.json \
    --last_results results_flipped/ensembles/last/ensemble_k_sweep_ood.json \
    --attn_results results_flipped/ensembles/attn/ensemble_k_sweep_ood.json \
    --output_dir results_flipped/pooling_comparison

echo "✅ All evaluations complete!"
EOF

chmod +x run_flipped_evaluation.sh
!./run_flipped_evaluation.sh
```

---

## 💡 TIPS & TRICKS

### Tip 1: Check Before Running

Always diagnose first to avoid wasting time:

```python
# Quick check
!python scripts/diagnose_activations.py \
    --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation
```

### Tip 2: Resume from Checkpoints

Most scripts skip if output already exists. To re-run:

```python
# Delete existing results
!rm -rf results_flipped/ood_evaluation

# Then run again
!python scripts/evaluate_ood_all_pooling.py ...
```

### Tip 3: Monitor GPU Memory

```python
# Check GPU usage
!nvidia-smi

# If OOM, reduce batch size
!python scripts/evaluate_ood_all_pooling.py \
    --batch_size 16 \
    ... # other args
```

### Tip 4: Use Relative Paths in Colab

If you're in Colab with mounted Drive:

```python
# Change to your project directory first
%cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes

# Then run commands with relative paths
!python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/... \
    --probes_base data/probes_flipped/... \
    --output_dir results_flipped/ood_evaluation
```

---

## 📋 EXPECTED OUTPUT STRUCTURE

After running all phases:

```
results_flipped/
├── ood_evaluation/
│   ├── ood_results_all_pooling.json          # Per-layer AUC/accuracy
│   ├── ood_layerwise_comparison.png          # 4 pooling strategies plot
│   ├── ood_best_probes_summary.txt           # Text summary
│   └── logits/
│       ├── mean_logits.npy                   # (N, 28) logits
│       ├── max_logits.npy
│       ├── last_logits.npy
│       ├── attn_logits.npy
│       └── labels.npy                        # (N,) ground truth
├── ensembles/
│   ├── mean/
│   │   ├── ensemble_k_sweep_ood.json         # K% vs AUC
│   │   ├── gated_models_val/                 # Trained gating models
│   │   └── k_sweep_results/
│   ├── max/
│   ├── last/
│   └── attn/
├── pooling_comparison/
│   └── all_pooling_comparison.png            # Side-by-side comparison
├── fixed_vs_gated/
│   ├── mean/comparison_results.json
│   ├── max/comparison_results.json
│   ├── last/comparison_results.json
│   └── attn/comparison_results.json
├── gating_analysis/
│   ├── gating_analysis.png                   # 4-panel fair comparison
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

## 🔍 WHAT EACH SCRIPT DOES

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `evaluate_ood_all_pooling.py` | Evaluate all 4 pooling strategies on OOD | Per-layer AUC, comparison plot, logits for ensembles |
| `evaluate_ensembles_comprehensive.py` | K-sweep for ensembles (fixed + gated) | K% vs AUC curves, trained gating models |
| `compare_all_pooling_ensembles.py` | Side-by-side pooling comparison | Multi-panel comparison plot |
| `compare_fixed_vs_gated.py` | Fixed vs gated ensemble comparison | Performance delta, best K% analysis |
| `analyze_gating_weights.py` | Fair comparison: single vs ensemble | 4-panel plot with OOD training analysis |
| `analyze_attention_*.py` | Attention mechanism visualizations | Heatmaps, token-level attention, entropy |
| `analyze_mechanisms.py` | Mechanistic interpretability | PCA, feature importance, layer analysis |
| `compare_pooling_layerwise.py` | Layer-by-layer pooling comparison | Per-layer pooling rankings |

---

## ❓ FAQ

**Q: How long does this take?**
A: Depends on dataset size:
- OOD evaluation (PHASE 1): ~5-10 mins
- Ensemble K-sweep (PHASE 2): ~10-20 mins per pooling
- Visualizations: ~5-10 mins each

**Q: What if I run out of GPU memory?**
A: Reduce `--batch_size` (try 16, 8, or even 4)

**Q: Can I run PHASES in parallel?**
A: No, they're sequential:
1. PHASE 1 creates logits needed for PHASE 2
2. PHASE 2 creates results needed for PHASE 3+

**Q: What if a script fails midway?**
A: Most scripts checkpoint. Delete output_dir and re-run:
```python
!rm -rf results_flipped/ood_evaluation
!python scripts/evaluate_ood_all_pooling.py ...
```

**Q: How do I know if results are symmetric?**
A: Run the comparison script:
```python
!python scripts/compare_results.py \
    --results1 results/ood_evaluation/ood_results_all_pooling.json \
    --results2 results_flipped/ood_evaluation/ood_results_all_pooling.json
```

If AUC scores are similar (within ~0.02), results are symmetric.
If there's a big gap (>0.05), there's domain bias.

---

## ✅ CHECKLIST

Before starting:
- [ ] Activations cached for Roleplaying dataset
- [ ] Probes trained for InsiderTrading dataset (all 4 pooling)
- [ ] Enough GPU memory (check `nvidia-smi`)
- [ ] Google Drive mounted (in Colab)

After PHASE 1:
- [ ] `ood_results_all_pooling.json` exists
- [ ] `logits/*.npy` files created (5 files: mean, max, last, attn, labels)
- [ ] `ood_layerwise_comparison.png` generated

After PHASE 2:
- [ ] 4 pooling directories created in `ensembles/`
- [ ] Each has `ensemble_k_sweep_ood.json`
- [ ] Gated models saved in `gated_models_val/`

Final:
- [ ] All visualizations generated
- [ ] Compare original vs flipped results
- [ ] Document findings in paper/report
