# Complete Ensemble Workflow Guide

This guide walks through the **complete pipeline** from data to final ensemble comparison.

---

## 📋 Overview

**Goal**: Compare deception detection performance across:
- 4 **pooling strategies**: mean, max, last, attn
- 3 **ensemble strategies**: Mean (uniform), Weighted (AUC-based), Gated (learned)
- Multiple **K% layer selections**: 10%, 20%, ..., 90%
- 2 **evaluation sets**: Validation (in-domain) + OOD (out-of-distribution)

**Key Insights**:
- ✅ **Individual probes don't need retraining** - we use already-trained layer probes
- ✅ **StaticMean & StaticWeighted ensembles don't need training** - just averaging
- ⚠️ **GatedEnsemble needs training** - learns input-dependent layer weights

---

## 🔄 Complete Pipeline

### Phase 1: Data & Activations (Already Done ✓)

```bash
# 1. Get roleplaying dataset from Apollo
python scripts/download_apollo_data.py --datasets roleplaying

# 2. Cache activations (includes rollout generation + LLM labeling)
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train

python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split validation
```

**Output**: `data/activations/{model}/Deception-Roleplaying/{train,validation}/`

---

### Phase 2: Train Per-Layer Probes (Already Done ✓)

```bash
# Train probes for each pooling strategy
for pooling in mean max last attn; do
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling $pooling \
        --output_dir /content/drive/MyDrive/probes/${pooling}
done
```

**Output** (per pooling strategy):
```
probes/{pooling}/
├── probe_layer_0.pt
├── probe_layer_1.pt
├── ...
├── probe_layer_N.pt
└── layer_results.json    # Important: Contains per-layer AUCs for layer selection
```

**What you now have in Google Drive**: ✅ All 4 pooling strategies trained

---

### Phase 3: Get OOD Dataset (Insider Trading)

**From Apollo deception repo**: Use the insider trading report creation dataset.

```bash
# Option 1: Download from Apollo (if available)
# Check: https://github.com/ApolloResearch/deception-detection/tree/main/data/insider_trading

# Option 2: Use the dataset loader they provide
# See: deception_detection/data/insider_trading.py
```

**Then cache activations** (same pipeline as roleplaying):

```bash
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-InsiderTrading \  # You'll need to add this dataset loader
    --split test \
    --output_dir /content/drive/MyDrive/activations/insider_trading
```

**Important**: This requires:
1. ✅ Rollout generation (using Llama 3.2-3B)
2. ✅ LLM labeling (using Cerebras Llama-8B) - **YES, needs labeling!**
3. ✅ Activation extraction

The `cache_deception_activations.py` script does all this automatically.

---

### Phase 4: Evaluate All Probes on OOD Dataset ⭐ NEW

This evaluates your **already-trained probes** on the OOD dataset.

```bash
python scripts/evaluate_ood_all_pooling.py \
    --probes_base_dir /content/drive/MyDrive/probes \
    --ood_activations_dir /content/drive/MyDrive/activations/insider_trading/test \
    --ood_dataset_name "Insider Trading" \
    --output_dir /content/drive/MyDrive/results/ood_evaluation
```

**What it does**:
1. Loads all 4 pooling strategies' probes
2. Evaluates each layer probe on OOD data
3. Generates per-layer AUC/accuracy for each pooling
4. **Saves logits** for ensemble evaluation (important!)
5. Creates comparison chart (all pooling, all layers, best highlighted)

**Output**:
```
ood_evaluation/
├── ood_results_all_pooling.json
├── ood_layerwise_comparison.png    # All 4 pooling strategies compared
├── ood_best_probes_summary.txt
└── logits/                         # ⭐ Important for ensembles!
    ├── mean_logits.npy   # (N, L) - N samples, L layers
    ├── max_logits.npy
    ├── last_logits.npy
    └── attn_logits.npy
```

---

### Phase 5: Ensemble Evaluation with K-Sweeps ⭐ NEW

**This is the main analysis!** For each pooling strategy, sweep through K% values and evaluate all 3 ensemble types.

#### 5a. Validation Evaluation

```bash
# For each pooling strategy
for pooling in mean max last attn; do
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling $pooling \
        --val_activations_dir /content/drive/MyDrive/activations/validation \
        --probes_dir /content/drive/MyDrive/probes/${pooling} \
        --output_dir /content/drive/MyDrive/results/ensembles/${pooling} \
        --eval_mode validation \
        --k_values "10,20,30,40,50,60,70,80,90"
done
```

**What it does**:
1. Extracts logits from validation activations using trained probes
2. For each K% value:
   - Selects top-K% layers by validation AUC (from `layer_results.json`)
   - Evaluates **StaticMeanEnsemble** (no training)
   - Evaluates **StaticWeightedEnsemble** (no training)
   - **Trains GatedEnsemble** on 80% of val data, tests on 20%
3. Generates K-sweep comparison plots
4. Saves trained gated models

**Output** (per pooling):
```
ensembles/{pooling}/
├── ensemble_k_sweep_validation.json
├── ensemble_comparison_validation.png   # Shows Mean vs Weighted vs Gated across K%
├── summary_validation.txt
└── gated_models_val/
    ├── gated_k10.pt    # Trained gated ensemble for K=10%
    ├── gated_k20.pt
    └── ...
```

#### 5b. OOD Evaluation

```bash
# For each pooling strategy
for pooling in mean max last attn; do
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling $pooling \
        --probes_dir /content/drive/MyDrive/probes/${pooling} \
        --ood_logits_path /content/drive/MyDrive/results/ood_evaluation/logits/${pooling}_logits.npy \
        --ood_labels_path /content/drive/MyDrive/results/ood_evaluation/logits/labels.npy \
        --output_dir /content/drive/MyDrive/results/ensembles/${pooling} \
        --eval_mode ood \
        --k_values "10,20,30,40,50,60,70,80,90"
done
```

**What it does**:
- Same as validation, but uses pre-computed OOD logits
- Trains new gated ensembles on OOD data
- Shows OOD generalization performance

**Important**: You need to save OOD labels separately. Modify `evaluate_ood_all_pooling.py` to save labels:

```python
# Add to evaluate_ood_all_pooling.py after loading dataset:
labels_path = os.path.join(logits_dir, "labels.npy")
np.save(labels_path, labels)  # Save labels for ensemble evaluation
```

---

### Phase 6: Cross-Pooling Comparison ⭐ ULTIMATE CHARTS

Compare **all pooling × all ensemble** combinations:

```bash
python scripts/compare_all_pooling_ensembles.py \
    --results_dir /content/drive/MyDrive/results/ensembles \
    --output_dir /content/drive/MyDrive/results/final_comparison \
    --eval_type both  # Both validation and OOD
```

**What it does**:
1. Loads ensemble results for all pooling strategies
2. Creates **heatmap**: Pooling (rows) × Ensemble (cols) → Best AUC
3. Creates **per-ensemble comparison**: Shows mean vs max vs last vs attn for each ensemble type
4. Creates **optimal K% analysis**: Bar chart showing which K% works best for each combination
5. Generates **final summary table**: Best config for each (pooling, ensemble) pair

**Output**:
```
final_comparison/
├── pooling_ensemble_heatmap_validation.png
├── pooling_ensemble_heatmap_ood.png
├── mean_comparison_validation.png          # All pooling for Mean ensemble
├── weighted_comparison_validation.png      # All pooling for Weighted ensemble
├── gated_comparison_validation.png         # All pooling for Gated ensemble
├── mean_comparison_ood.png
├── weighted_comparison_ood.png
├── gated_comparison_ood.png
├── optimal_k_analysis_validation.png
├── optimal_k_analysis_ood.png
├── final_summary_validation.txt
└── final_summary_ood.txt
```

---

## 📊 Key Charts You'll Get

### 1. **Per-Pooling Layer-wise Comparison** (from Phase 4)
- X-axis: Layer index
- Y-axis: AUC / Accuracy
- Lines: mean (blue), max (purple), last (orange), attn (green)
- Stars: Best layer for each pooling
- Annotation: Overall best performer

### 2. **Per-Pooling Ensemble K-Sweep** (from Phase 5)
- X-axis: Top-K% layers
- Y-axis: AUC / Accuracy
- Lines: Mean (blue), Weighted (red), Gated (green)
- Stars: Best K% for each ensemble
- Shows optimal K% selection

### 3. **Cross-Pooling Ensemble Heatmap** (from Phase 6)
- Rows: Pooling strategies (mean, max, last, attn)
- Columns: Ensemble strategies (Mean, Weighted, Gated)
- Cell values: Best AUC across all K% values
- Color: Green (high) to Red (low)
- **Answers**: Which (pooling, ensemble) combo works best?

### 4. **Per-Ensemble Cross-Pooling** (from Phase 6)
- X-axis: Top-K% layers
- Y-axis: AUC / Accuracy
- Lines: mean (blue), max (purple), last (orange), attn (green)
- 3 separate plots: one for Mean, one for Weighted, one for Gated
- **Answers**: For each ensemble, which pooling works best?

### 5. **Optimal K% Analysis** (from Phase 6)
- Bar chart
- X-axis: Each (pooling, ensemble) combination
- Y-axis: Optimal K% value
- **Answers**: What's the sweet spot for layer selection?

---

## 🎯 What You Need to Do Now

### Immediate Next Steps:

1. **Get OOD Dataset (Insider Trading)**
   - Download from Apollo or use their data loader
   - Run `cache_deception_activations.py` on it
   - **Note**: YES, it needs LLM labeling (Cerebras Llama-8B)

2. **Run OOD Evaluation** (Phase 4)
   ```bash
   python scripts/evaluate_ood_all_pooling.py \
       --probes_base_dir /content/drive/MyDrive/probes \
       --ood_activations_dir /path/to/insider_trading/activations \
       --output_dir /content/drive/MyDrive/results/ood_evaluation
   ```

3. **Run Ensemble K-Sweeps** (Phase 5)
   ```bash
   # Validation (for each pooling: mean, max, last, attn)
   python scripts/evaluate_ensembles_comprehensive.py ...

   # OOD (for each pooling)
   python scripts/evaluate_ensembles_comprehensive.py ...
   ```

4. **Generate Final Comparison** (Phase 6)
   ```bash
   python scripts/compare_all_pooling_ensembles.py \
       --results_dir /content/drive/MyDrive/results/ensembles \
       --output_dir /content/drive/MyDrive/results/final_comparison \
       --eval_type both
   ```

---

## ❓ FAQ

### Q: Do I need to retrain probes?
**A**: ❌ No! You already have trained probes for all 4 pooling strategies.

### Q: Do ensembles need training?
**A**:
- **StaticMean**: ❌ No (just averages logits)
- **StaticWeighted**: ❌ No (uses pre-computed AUCs as weights)
- **GatedEnsemble**: ✅ YES (trains MLP to learn layer weights)

### Q: Does insider trading dataset need LLM labeling?
**A**: ✅ YES - use Cerebras Llama-8B via `cache_deception_activations.py`

### Q: What's the difference between validation and OOD evaluation?
**A**:
- **Validation**: In-domain (same dataset as training, different split)
- **OOD**: Out-of-distribution (completely different dataset - insider trading vs roleplaying)
- **OOD tests generalization!**

### Q: How long does this take?
**A**:
- OOD activation caching: ~1-2 hours (depends on dataset size, LLM rate limits)
- OOD evaluation: ~15 minutes per pooling strategy
- Ensemble K-sweep: ~10 minutes per pooling strategy
- Final comparison: ~2 minutes

### Q: What if I already did validation evaluation in `train_deception_probes.py`?
**A**: That only gave you **per-layer** results. Ensemble evaluation gives you **multi-layer combinations** (the whole point of ensembles!).

---

## 🐛 Bugs Fixed

1. ✅ **StaticMeanEnsemble** - Removed incorrect `num_layers` parameter
2. ✅ **GatedEnsemble** - Removed incorrect `hidden_dim` parameter
3. ✅ **GatedEnsemble forward** - Fixed to accept both features and logits

All ensemble code now works correctly!

---

## 📁 Final Directory Structure

```
/content/drive/MyDrive/
├── probes/
│   ├── mean/
│   │   ├── probe_layer_*.pt
│   │   └── layer_results.json
│   ├── max/
│   ├── last/
│   └── attn/
│
├── activations/
│   ├── validation/
│   │   └── shard_*.safetensors
│   └── insider_trading/
│       └── test/
│           └── shard_*.safetensors
│
└── results/
    ├── ood_evaluation/
    │   ├── ood_results_all_pooling.json
    │   ├── ood_layerwise_comparison.png
    │   └── logits/
    │       ├── mean_logits.npy
    │       ├── max_logits.npy
    │       ├── last_logits.npy
    │       ├── attn_logits.npy
    │       └── labels.npy
    │
    ├── ensembles/
    │   ├── mean/
    │   │   ├── ensemble_k_sweep_validation.json
    │   │   ├── ensemble_k_sweep_ood.json
    │   │   ├── ensemble_comparison_validation.png
    │   │   ├── ensemble_comparison_ood.png
    │   │   └── gated_models_*/
    │   ├── max/
    │   ├── last/
    │   └── attn/
    │
    └── final_comparison/
        ├── pooling_ensemble_heatmap_*.png
        ├── *_comparison_*.png
        ├── optimal_k_analysis_*.png
        └── final_summary_*.txt
```

---

## 🎉 What You'll Learn

After running the complete pipeline, you'll know:

1. **Best pooling strategy**: mean vs max vs last vs attn
2. **Best ensemble strategy**: Mean vs Weighted vs Gated
3. **Optimal layer selection**: What K% of top layers to use
4. **Generalization**: How well does it transfer to OOD data?
5. **Layer importance**: Which layers matter most for deception detection?

Good luck! 🚀
