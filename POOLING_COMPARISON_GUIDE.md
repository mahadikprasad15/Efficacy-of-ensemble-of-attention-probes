# Pooling Strategy Comparison Guide

This guide explains how to compare all 4 pooling strategies (mean, max, last, attn) layer-by-layer with comprehensive visualization.

## Overview

After training probes with different pooling strategies, you can use these scripts to:
- **Compare validation AUC/accuracy** across all layers for all pooling strategies on the same chart
- **Highlight the best layer** for each pooling strategy with star markers
- **Identify the overall best** performer across all strategies with annotations
- **Evaluate test metrics** (optional) for comprehensive performance analysis

## Quick Start (Google Colab)

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to repo
```bash
cd /content/Efficacy-of-ensemble-of-attention-probes
```

### 3. Run comparison

**Option A: If your probes are in separate pooling directories:**
```bash
python scripts/compare_pooling_layerwise.py \
    --mean_dir /content/drive/MyDrive/probes/mean \
    --max_dir /content/drive/MyDrive/probes/max \
    --last_dir /content/drive/MyDrive/probes/last \
    --attn_dir /content/drive/MyDrive/probes/attn \
    --output_dir /content/drive/MyDrive/results/comparisons
```

**Option B: If your probes are organized by model/dataset/pooling:**
```bash
python scripts/compare_pooling_layerwise.py \
    --probes_base_dir /content/drive/MyDrive/probes \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --output_dir /content/drive/MyDrive/results/comparisons
```

### 4. View results in Colab
```python
from IPython.display import Image, display

# Display summary
with open('/content/drive/MyDrive/results/comparisons/pooling_comparison_summary.txt', 'r') as f:
    print(f.read())

# Display plot
display(Image('/content/drive/MyDrive/results/comparisons/layerwise_pooling_comparison.png'))
```

## Expected Directory Structure

### Option A: Simple structure (pooling at root)
```
probes/
├── mean/
│   ├── probe_layer_0.pt
│   ├── probe_layer_1.pt
│   ├── ...
│   └── layer_results.json      # Required!
├── max/
│   ├── probe_layer_0.pt
│   └── layer_results.json      # Required!
├── last/
│   └── layer_results.json      # Required!
└── attn/
    └── layer_results.json      # Required!
```

### Option B: Organized structure
```
probes/
└── meta-llama_Llama-3.2-3B-Instruct/
    └── Deception-Roleplaying/
        ├── mean/
        │   └── layer_results.json
        ├── max/
        │   └── layer_results.json
        ├── last/
        │   └── layer_results.json
        └── attn/
            └── layer_results.json
```

## Required Files

Each pooling directory MUST contain:
1. **`layer_results.json`** - Per-layer validation metrics
   ```json
   [
     {
       "layer": 0,
       "val_auc": 0.7234,
       "val_acc": 0.6891,
       "epoch": 5
     },
     ...
   ]
   ```

2. **`probe_layer_*.pt`** files (optional, only needed for test evaluation)

## Output Files

The script generates:

1. **`pooling_comparison_summary.txt`** - Text summary table
   ```
   ==================================================================================
   POOLING STRATEGY COMPARISON - SUMMARY
   ==================================================================================

   Pooling    Best Layer   Val AUC      Val Acc      Test AUC     Test Acc
   ----------------------------------------------------------------------------------
   MEAN       15           0.8234       0.7891       N/A          N/A
   MAX        18           0.8456       0.8012       N/A          N/A          ⭐
   LAST       12           0.7234       0.6891       N/A          N/A
   ATTN       16           0.8123       0.7756       N/A          N/A
   ==================================================================================
   ⭐ Best overall: MAX (Val AUC: 0.8456)
   ==================================================================================
   ```

2. **`layerwise_pooling_comparison.png`** - Visualization with:
   - All 4 pooling strategies on the same chart
   - Different colors for each strategy (blue=mean, purple=max, orange=last, green=attn)
   - Star markers (⭐) for best layer of each pooling strategy
   - Annotated box highlighting the overall best performer
   - Reference lines for random chance (0.5) and strong signal (0.7)
   - Separate subplots for AUC and Accuracy (if available)

## Advanced Usage

### Using the general comparison script
The original `compare_results.py` can also be used for broader comparisons:

```bash
python scripts/compare_results.py \
    --experiments_dir data/probes \
    --output_dir results/comparisons
```

This will:
- Auto-discover all experiments in the directory
- Generate pooling comparison bar charts
- Create layer-wise comparison plots
- Build OOD evaluation matrices (if available)

### Test Evaluation (Coming Soon)
```bash
python scripts/compare_pooling_layerwise.py \
    --mean_dir /path/to/mean \
    --max_dir /path/to/max \
    --last_dir /path/to/last \
    --attn_dir /path/to/attn \
    --evaluate_test \
    --test_activations_dir /path/to/test/activations \
    --output_dir results
```

## Interpreting the Results

### What to look for:
1. **Best pooling strategy**: Which pooling method achieves the highest validation AUC?
2. **Best layers**: Do different pooling strategies peak at different layers?
3. **Consistency**: Is one pooling strategy consistently better across all layers?
4. **Layer trends**: Do later layers perform better? (common in many models)

### Common patterns:
- **Mean pooling**: Often most stable, good baseline
- **Max pooling**: Can capture salient features, may spike at specific layers
- **Last token**: Works well if the model processes sequentially
- **Attention pooling**: Learned weighting, potentially most flexible

## Troubleshooting

### "No pooling directories found!"
- Check your paths are correct
- Ensure `layer_results.json` exists in each pooling directory
- Try using manual paths (`--mean_dir`, etc.) instead of auto-discovery

### "Results file not found"
- Make sure you've run `train_deception_probes.py` for each pooling strategy
- Check that `layer_results.json` was generated (not gitignored)

### Empty plots
- Verify your `layer_results.json` files contain valid data
- Check that validation metrics exist in the JSON

## Example Workflow

```bash
# 1. Train probes for all pooling strategies
for pooling in mean max last attn; do
    python scripts/train_deception_probes.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying \
        --pooling $pooling \
        --output_dir /content/drive/MyDrive/probes/${pooling}
done

# 2. Compare all strategies
python scripts/compare_pooling_layerwise.py \
    --mean_dir /content/drive/MyDrive/probes/mean \
    --max_dir /content/drive/MyDrive/probes/max \
    --last_dir /content/drive/MyDrive/probes/last \
    --attn_dir /content/drive/MyDrive/probes/attn \
    --output_dir /content/drive/MyDrive/results

# 3. View results
cat /content/drive/MyDrive/results/pooling_comparison_summary.txt
```

## Related Scripts

- **`train_deception_probes.py`** - Train probes with different pooling strategies
- **`analyze_probes.py`** - Analyze single pooling strategy in detail
- **`compare_results.py`** - Broader comparison across experiments
- **`eval_ood.py`** - Evaluate best probe on out-of-distribution data

## Questions?

Check the example script: `scripts/compare_pooling_colab_example.py`
