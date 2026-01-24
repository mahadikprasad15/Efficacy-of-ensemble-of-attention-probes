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

**Output**: `data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/`

---

### Step 2: Evaluate on OOD Dataset

```python
# Evaluate on InsiderTrading (OOD)
!python scripts/evaluation/eval_per_token_probes.py \
    --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
    --ood_activations data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
    --output_dir results/per_token_ood
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
# Compare results
!python scripts/comparison/compare_per_token_vs_pooled.py \
    --pooled_results data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean/layer_results.json \
    --per_token_results data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/layer_results.json \
    --output_dir results/per_token_comparison
```

---

## Aggregation Methods

When evaluating per-token probes, token predictions are aggregated to sample level:

| Method | Description | Usage |
|--------|-------------|-------|
| `mean` | Average probability across tokens | Default, stable |
| `max` | Maximum probability (most confident token) | Catches deception anywhere |
| `vote` | Majority vote (>50% tokens) | Robust to outliers |

```python
# Use different aggregation
!python scripts/evaluation/eval_per_token_probes.py \
    --probes_dir ... \
    --ood_activations ... \
    --aggregation max  # or 'vote'
```

---

## Output Structure

```
data/probes_per_token/
└── meta-llama_Llama-3.2-3B-Instruct/
    └── Deception-Roleplaying/
        ├── probe_layer_0.pt
        ├── probe_layer_1.pt
        ├── ...
        ├── probe_layer_27.pt
        ├── norm_layer_0.npz      # Normalization stats
        └── layer_results.json    # Per-layer metrics
```
