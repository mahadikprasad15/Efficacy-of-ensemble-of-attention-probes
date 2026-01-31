# Clean Soft Prefix Pipeline v2 - With Stability Loss

## What Changed

The previous run had terrible validation AUROC (~0.53) because:
1. **No stability anchoring** - the prefix destroyed the probe's learned behavior
2. **OOM on OOD** - 5000+ samples with long transcripts

**Fix**: Added **L_stability** term that anchors honest samples to their **original baseline logits**.

```
L = L_contrast + λ_norm * ||prefix|| + λ_stability * MSE(honest_logits, baseline_honest_logits)
```

---

## Step 1: Pull Latest Code

```bash
cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes
!git pull
```

---

## Step 2: Run the Fixed Pipeline

```python
!python scripts/pipelines/run_soft_prefix_pipeline.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --id_raw_yaml /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/roleplaying/dataset.yaml \
    --ood_raw_json /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/insider_trading/llama-70b-3.3-generations.json \
    --probe_path /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes_layer_agnostic/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/last/probe.pt \
    --layer_idx 15 \
    --steps 500 \
    --lambda_stability 1.0 \
    --ood_limit 300 \
    --output_base /content/drive/MyDrive/checkpoints/soft_prefix_v2/meta-llama_Llama-3.2-3B-Instruct/layer_agnostic_last_layer15
```

---

## Key Changes from v1

| Parameter | v1 | v2 |
|-----------|----|----|
| Loss | `L_contrast + L_norm + μ_hon²` | `L_contrast + L_norm + MSE(hon, baseline)` |
| Baseline | None | Computed before training |
| OOD samples | 5003 (OOM!) | 300 (balanced) |
| Cache batch | 4 | 1 for OOD |
| Steps | 3000 | 500 |

---

## Expected Output

```
Computing baseline logits (no prefix)...
Baseline stats:
  Honest:    mean=-0.5, std=0.3  # These should match your trained probe
  Deceptive: mean=0.8, std=0.4
  Separation: 1.3

Training:
  Step 50: val_auroc=0.85+, L_stab=0.01  # Should MAINTAIN high AUROC!
  Step 100: val_auroc=0.86, L_stab=0.02
  ...

Results:
  ID Val AUROC: ~0.85+ (should stay high!)
  ID Test AUROC: ~0.85
  OOD Test AUROC: ??? (the metric we care about)
```

---

## If Validation AUROC is Still Bad

Try increasing stability weight:
```
--lambda_stability 5.0
```

Or decrease learning rate:
```
--lr 1e-4
```

The goal is: **maintain ID performance while hoping OOD improves**.
