# Per-Token Probe Training (Apollo Approach)

This guide explains how to train and evaluate probes using the Apollo Research per-token approach.

## Key Difference

| Approach | Method | Effective Training Size |
|----------|--------|------------------------|
| **Pooled** | Pool tokens → 1 vector/sample | N samples |
| **Per-Token** | Flatten tokens → N×T vectors | N × T (e.g., 200×50=10,000) |

---

## 🚀 Colab Commands

### Step 1: Train Per-Token Probes on Single Domain

```python
# Train on Roleplaying (ID)
!python scripts/training/train_per_token_probes.py \
    --train_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --val_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --output_dir data/probes_per_token \
    --model meta-llama_Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying

# Train on InsiderTrading (flipped)
!python scripts/training/train_per_token_probes.py \
    --train_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/train \
    --val_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/validation \
    --output_dir data/probes_per_token_flipped \
    --model meta-llama_Llama-3.2-3B-Instruct \
    --dataset Deception-InsiderTrading-SallyConcat
```

---

### Step 2: Train Combined Per-Token Probes

```python
# Train on combined domains (A+B)
!python scripts/training/train_combined_per_token_probes.py \
    --train_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --val_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --train_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/train \
    --val_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/validation \
    --output_dir data/probes_combined_per_token \
    --model meta-llama_Llama-3.2-3B-Instruct \
    --label_a Roleplaying \
    --label_b InsiderTrading
```

**Output**: 
- Probes: `data/probes_combined_per_token/.../Deception-Combined/`
- Plots: `layerwise_mean.png`, `layerwise_max.png`, `layerwise_last.png`, `aggregation_comparison.png`

---

### Step 3: Evaluate on OOD Dataset (All Aggregations)

```python
# Runs all 4 aggregations (mean/max/last/vote) by default
!python scripts/evaluation/eval_per_token_probes.py \
    --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/validation \
    --output_dir results/per_token_ood
```

---

### Step 4: Invariant Core Analysis

```python
# Is there a domain-invariant deception direction?
!python scripts/analysis/analyze_invariant_core_per_token_probes.py \
    --probes_a data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --probes_b data/probes_per_token_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat \
    --probes_combined data/probes_combined_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Combined \
    --val_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --val_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading-SallyConcat/validation \
    --layer 20 \
    --label_a Roleplaying \
    --label_b InsiderTrading \
    --output_dir results/invariant_core_per_token
```

**Outputs**:
- `decomposition_analysis.png` - Component magnitudes and per-domain AUCs
- `generalization_comparison.png` - Which component generalizes best
- `aggregation_comparison.png` - Results for mean/max/last aggregation
- `invariant_core_summary.json` - Full results

---

### Step 5: Compare Per-Token vs Pooled

```python
!python scripts/comparison/compare_per_token_vs_pooled.py \
    --pooled_results data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean/layer_results.json \
    --per_token_results data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/layer_results.json \
    --output_dir results/per_token_comparison
```

---

## Aggregation Methods

| Method | Description |
|--------|-------------|
| `mean` | Average probability across tokens (default, stable) |
| `max` | Maximum probability (catches deception anywhere) |
| `last` | Last token's probability (matches next-token prediction) |
| `vote` | Majority vote fraction (robust to outliers) |

---

## Output Structure

```
data/
├── probes_per_token/                    # Single domain A training
├── probes_per_token_flipped/            # Single domain B training  
└── probes_combined_per_token/           # Combined A+B training

results/
├── per_token_ood/                       # OOD evaluation
│   ├── mean/, max/, last/, vote/        # Per-aggregation results
│   ├── eval_ood_all_aggregations.png    # Combined plot
│   └── eval_ood_aggregation_bar.png     # Bar chart
└── invariant_core_per_token/            # Direction analysis
    ├── decomposition_analysis.png
    ├── generalization_comparison.png
    └── invariant_core_summary.json
```
