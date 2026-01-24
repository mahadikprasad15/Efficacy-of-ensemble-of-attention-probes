# Per-Token Probe Training (Apollo Approach)

This guide explains how to train and evaluate probes using the Apollo Research per-token approach.

## Key Difference

| Approach | Method | Effective Training Size |
|----------|--------|------------------------|
| **Pooled** | Pool tokens → 1 vector/sample | N samples |
| **Per-Token** | Flatten tokens → N×T vectors | N × T (e.g., 200×50=10,000) |

---

## 🚀 Colab Commands

### Step 1: Train Per-Token Probes on ID Dataset

```python
# Train on Roleplaying (ID)
!python scripts/training/train_per_token_probes.py \
    --train_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --val_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
    --output_dir data/probes_per_token \
    --model meta-llama_Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying
```

---

### Step 2: Evaluate on OOD Dataset (All Aggregations)

```python
# Runs all 4 aggregations (mean/max/last/vote) by default
!python scripts/evaluation/eval_per_token_probes.py \
    --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --output_dir results/per_token_ood
```

**Or run a specific aggregation:**
```python
!python scripts/evaluation/eval_per_token_probes.py \
    --probes_dir ... \
    --ood_activations ... \
    --aggregation last  # Only run 'last' aggregation
```

---

### Step 3: Flipped Training (OOD as ID)

```python
# Train on InsiderTrading (for flipped experiment)
!python scripts/training/train_per_token_probes.py \
    --train_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/train \
    --val_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --output_dir data/probes_per_token_flipped \
    --model meta-llama_Llama-3.2-3B-Instruct \
    --dataset Deception-InsiderTrading
```

---

### Step 4: Compare Per-Token vs Pooled

```python
!python scripts/comparison/compare_per_token_vs_pooled.py \
    --pooled_results data/probes/.../mean/layer_results.json \
    --per_token_results data/probes_per_token/.../layer_results.json \
    --output_dir results/per_token_comparison
```

---

## Aggregation Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| `mean` | Average probability across tokens | Default, most stable |
| `max` | Maximum probability (most confident) | Catches deception anywhere |
| `last` | Last token's probability | Matches next-token prediction |
| `vote` | Majority vote (fraction >50%) | Robust to outliers |

---

## Output Structure

When running with `--aggregation all` (default):

```
results/per_token_ood/
├── mean/
│   ├── eval_results.json
│   └── eval_ood_mean.png
├── max/
│   └── ...
├── last/
│   └── ...
├── vote/
│   └── ...
├── eval_ood_all_aggregations.png   # Combined layerwise plot
└── eval_ood_aggregation_bar.png    # Bar chart comparison
```
