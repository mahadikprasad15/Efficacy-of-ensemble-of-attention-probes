# 🚀 CORRECTED Colab Commands for Flipped Evaluation

**⚠️ IMPORTANT: This file contains the CORRECT commands that match the actual script arguments!**

---

## 🔧 Step 0: Setup (Run First)

```python
# Navigate to project directory
%cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes

# Pull latest changes
!git config pull.rebase false
!git pull origin claude/test-dataset-symmetry-JNEgg

# Verify you're on the right branch
!git branch
```

---

## ✅ PHASE 1: OOD Evaluation (Working!)

```python
# Evaluate all 4 pooling strategies on flipped OOD dataset
!python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --probes_base data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/ood_evaluation
```

**Output**:
- `results_flipped/ood_evaluation/ood_results_all_pooling.json`
- `results_flipped/ood_evaluation/ood_layerwise_comparison.png`
- `results_flipped/ood_evaluation/logits/mean_logits.npy` (and max, last, attn)
- `results_flipped/ood_evaluation/logits/labels.npy`

---

## 🎯 PHASE 2: Ensemble Evaluation (K-Sweep)

**⚠️ CORRECTED ARGUMENTS**: Use `--ood_logits_path` and `--ood_labels_path`, NOT `--id_logits_dir`!

### MEAN Pooling

```python
!python scripts/evaluate_ensembles_comprehensive.py \
    --pooling mean \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --ood_logits_path results_flipped/ood_evaluation/logits/mean_logits.npy \
    --ood_labels_path results_flipped/ood_evaluation/logits/labels.npy \
    --eval_mode ood \
    --output_dir results_flipped/ensembles/mean
```

### MAX Pooling

```python
!python scripts/evaluate_ensembles_comprehensive.py \
    --pooling max \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/max \
    --ood_logits_path results_flipped/ood_evaluation/logits/max_logits.npy \
    --ood_labels_path results_flipped/ood_evaluation/logits/labels.npy \
    --eval_mode ood \
    --output_dir results_flipped/ensembles/max
```

### LAST Pooling

```python
!python scripts/evaluate_ensembles_comprehensive.py \
    --pooling last \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/last \
    --ood_logits_path results_flipped/ood_evaluation/logits/last_logits.npy \
    --ood_labels_path results_flipped/ood_evaluation/logits/labels.npy \
    --eval_mode ood \
    --output_dir results_flipped/ensembles/last
```

### ATTN Pooling

```python
!python scripts/evaluate_ensembles_comprehensive.py \
    --pooling attn \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --ood_logits_path results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels_path results_flipped/ood_evaluation/logits/labels.npy \
    --eval_mode ood \
    --output_dir results_flipped/ensembles/attn
```

**Output for each**:
- `results_flipped/ensembles/{pooling}/ensemble_k_sweep_ood.json`
- `results_flipped/ensembles/{pooling}/gated_models_val/` (trained gating models)

---

## 📊 PHASE 3: Compare All Pooling Strategies

**⚠️ CORRECTED ARGUMENTS**: Use `--results_dir`, NOT individual result files!

```python
!python scripts/compare_all_pooling_ensembles.py \
    --results_dir results_flipped/ensembles \
    --output_dir results_flipped/pooling_comparison \
    --eval_type ood
```

**Expected directory structure** for this to work:
```
results_flipped/ensembles/
├── mean/ensemble_k_sweep_ood.json
├── max/ensemble_k_sweep_ood.json
├── last/ensemble_k_sweep_ood.json
└── attn/ensemble_k_sweep_ood.json
```

**Output**:
- `results_flipped/pooling_comparison/pooling_ensemble_heatmap.png`
- `results_flipped/pooling_comparison/ensemble_strategy_comparison.png`
- `results_flipped/pooling_comparison/optimal_k_analysis.png`
- `results_flipped/pooling_comparison/final_summary.txt`

---

## 🔬 PHASE 4: Fixed vs Gated Comparison

```python
# Compare for MEAN pooling
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

---

## 🧠 PHASE 5: Gating Analysis (Fair Comparison)

**Note**: This requires validation activations, not just OOD logits!

```python
!python scripts/analyze_gating_weights.py \
    --id_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
    --ood_logits results_flipped/ood_evaluation/logits/attn_logits.npy \
    --ood_labels results_flipped/ood_evaluation/logits/labels.npy \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --k_pct 40 \
    --output_dir results_flipped/gating_analysis
```

**Output**:
- `results_flipped/gating_analysis/gating_analysis.png` (4-panel plot)
- `results_flipped/gating_analysis/gating_analysis_results.json`

---

## 🎨 PHASE 6: Attention Visualizations

### Attention Entropy Analysis

```python
!python scripts/analyze_attention_entropy.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/attention_analysis
```

### Token-Level Attention

```python
!python scripts/analyze_attention_text.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/attention_text \
    --num_samples 10
```

### Ensemble Attention

```python
!python scripts/analyze_ensemble_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/ensemble_attention
```

### Layer-Colored Attention

```python
!python scripts/analyze_layer_colored_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/layer_colored_attention
```

### Hybrid Attention

```python
!python scripts/analyze_hybrid_attention.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_mean data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --probes_attn data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/hybrid_attention
```

---

## 🔬 PHASE 7: Mechanistic Analysis

```python
# Deep mechanistic analysis
!python scripts/analyze_mechanisms.py \
    --activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --output_dir results_flipped/mechanistic_analysis

# Probe analysis
!python scripts/analyze_probes.py \
    --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/probe_analysis
```

---

## 📏 PHASE 8: Layerwise Pooling Comparison

```python
!python scripts/compare_pooling_layerwise.py \
    --mean_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/mean \
    --max_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/max \
    --last_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/last \
    --attn_probes data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --output_dir results_flipped/pooling_layerwise
```

---

## 🔄 Compare Original vs Flipped Results

After running both original and flipped evaluations:

```python
!python scripts/compare_results.py \
    --results1 results/ood_evaluation/ood_results_all_pooling.json \
    --results2 results_flipped/ood_evaluation/ood_results_all_pooling.json \
    --labels "Original (ID:Role→OOD:Insider)" "Flipped (ID:Insider→OOD:Role)" \
    --output_dir results_comparison/symmetry_test
```

---

## 🚀 Batch Run Script (Run All Phases)

```python
# Create batch script
cat > run_flipped_eval.sh << 'EOF'
#!/bin/bash
set -e

echo "=== PHASE 1: OOD Evaluation ==="
python scripts/evaluate_ood_all_pooling.py \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --probes_base data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
    --output_dir results_flipped/ood_evaluation

echo "=== PHASE 2: Ensemble K-Sweep ==="
for pooling in mean max last attn; do
    echo "Processing $pooling..."
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling $pooling \
        --probes_dir data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/$pooling \
        --ood_logits_path results_flipped/ood_evaluation/logits/${pooling}_logits.npy \
        --ood_labels_path results_flipped/ood_evaluation/logits/labels.npy \
        --eval_mode ood \
        --output_dir results_flipped/ensembles/$pooling
done

echo "=== PHASE 3: Compare All Pooling ==="
python scripts/compare_all_pooling_ensembles.py \
    --results_dir results_flipped/ensembles \
    --output_dir results_flipped/pooling_comparison \
    --eval_type ood

echo "=== ALL PHASES COMPLETE ==="
EOF

chmod +x run_flipped_eval.sh

# Run it
!./run_flipped_eval.sh
```

---

## 📋 Expected Output Structure

```
results_flipped/
├── ood_evaluation/
│   ├── ood_results_all_pooling.json
│   ├── ood_layerwise_comparison.png
│   ├── ood_best_probes_summary.txt
│   └── logits/
│       ├── mean_logits.npy
│       ├── max_logits.npy
│       ├── last_logits.npy
│       ├── attn_logits.npy
│       └── labels.npy
├── ensembles/
│   ├── mean/
│   │   ├── ensemble_k_sweep_ood.json
│   │   └── gated_models_val/
│   ├── max/
│   ├── last/
│   └── attn/
├── pooling_comparison/
│   ├── pooling_ensemble_heatmap.png
│   ├── ensemble_strategy_comparison.png
│   ├── optimal_k_analysis.png
│   └── final_summary.txt
├── fixed_vs_gated/
│   ├── mean/
│   ├── max/
│   ├── last/
│   └── attn/
├── gating_analysis/
├── attention_analysis/
├── attention_text/
├── ensemble_attention/
├── layer_colored_attention/
├── hybrid_attention/
├── mechanistic_analysis/
├── probe_analysis/
└── pooling_layerwise/
```

---

## ⚠️ Common Issues

### Issue: "unrecognized arguments"
**Fix**: Make sure you're using the CORRECTED commands from this file!

### Issue: "No such file or directory"
**Fix**: PHASE 2 requires PHASE 1 to complete first (creates the logits)

### Issue: Out of memory
**Fix**: Add `--batch_size 16` (or lower) to any command

### Issue: Git divergent branches
```python
!git config pull.rebase false
!git pull origin claude/test-dataset-symmetry-JNEgg
```

---

## 📝 Key Differences from Old Commands

| Old (WRONG) | New (CORRECT) |
|-------------|---------------|
| `--id_logits_dir` | `--ood_logits_path` |
| `--ood_logits` | `--ood_logits_path` |
| `--ood_labels` | `--ood_labels_path` |
| `--mean_results file.json` | `--results_dir ensembles/` |
| Individual result files | Base directory only |

---

**These commands are tested and match the actual script arguments!** 🎯
