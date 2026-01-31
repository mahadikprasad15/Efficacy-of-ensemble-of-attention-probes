# Probe Geometry Analysis - Colab Commands

## Prerequisites
Sync repo and set up paths:
```bash
cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes
!git pull
```

## Step 1: Generate ID With-Prefix Activations

```python
# Run this in Colab
!python scripts/analysis/generate_prefix_activations_id.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --soft_prefix_ckpt /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/soft_prefix/meta-llama_Llama-3.2-3B-Instruct/layer_agnostic_last_layer15 \
    --data_yaml /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/roleplaying/dataset.yaml \
    --cached_manifest /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train/manifest.jsonl \
    --split train \
    --output_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --batch_size 2
```

## Step 2: Run Geometry Analysis

### Vanilla Last Probes
```python
!python scripts/analysis/analyze_prefix_geometry.py \
    --no_prefix_id_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --no_prefix_ood_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --with_prefix_id_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --with_prefix_ood_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --probes_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes/last \
    --probe_type vanilla \
    --pooling last \
    --layers all \
    --output_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/prefix_last_token_results/prefix_geometry_analysis
```

### Layer-Agnostic Last Probes
```python
!python scripts/analysis/analyze_prefix_geometry.py \
    --no_prefix_id_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --no_prefix_ood_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --with_prefix_id_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --with_prefix_ood_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --probes_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes_layer_agnostic/last \
    --probe_type layer_agnostic \
    --pooling last \
    --layers all \
    --output_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/prefix_last_token_results/prefix_geometry_analysis
```

### Per-Token Last Probes
```python
!python scripts/analysis/analyze_prefix_geometry.py \
    --no_prefix_id_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --no_prefix_ood_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --with_prefix_id_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
    --with_prefix_ood_dir /content/drive/MyDrive/data/soft_prefix_activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/test \
    --probes_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes_per_token/last \
    --probe_type per_token \
    --pooling last \
    --layers all \
    --output_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/prefix_last_token_results/prefix_geometry_analysis
```

## Output Structure

```
results/prefix_last_token_results/prefix_geometry_analysis/
├── vanilla_last/
│   ├── layer_00/
│   │   ├── step_a_logit_histograms.png
│   │   ├── step_b_shift_bars.png
│   │   ├── step_c_delta_alignment.png
│   │   ├── step_d_pca_panels.png
│   │   └── metrics.json
│   ├── layer_15/
│   │   └── ...
│   ├── auroc_across_layers.png
│   └── cross_layer_summary.json
├── layer_agnostic_last/
│   └── ...
└── per_token_last/
    └── ...
```

## Notes

1. **Adjust paths** if your activation directories differ from the expected structure
2. **Check probe directory names** - they might be slightly different (e.g., `probes/last` vs `probes_layer_agnostic/last`)
3. **Run Step 1 first** to generate the missing ID with-prefix activations
4. **GPU recommended** for Step 1 (model inference needed)
