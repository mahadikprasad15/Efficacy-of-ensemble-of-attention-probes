# Deception Detection Pipeline Guide

Complete guide for detecting strategic deception (lying) in LLM generations using activation probes.

## 📋 Overview

This pipeline detects when models are being **deceptive** (strategic lying) vs. **honest** (truthful). This is different from hallucination detection (factual errors).

**Pipeline:**
1. Download Apollo Research deception datasets
2. Load scenarios and create prompts
3. Generate completions using Llama-3.2-3B
4. Label completions using Cerebras Llama-8B (LLM judge)
5. Extract activations from generated tokens
6. Cache activations as (L, T, D) tensors
7. Train probes with different pooling strategies
8. Evaluate deception detection performance

---

## 🚀 Quick Start (Proof of Concept)

### Step 0: Prerequisites

```bash
# Install dependencies (if not already done)
pip install torch transformers safetensors pyyaml requests tqdm scikit-learn

# Set Cerebras API key
export CEREBRAS_API_KEY="your_api_key_here"
```

### Step 1: Download Apollo Dataset

```bash
python scripts/download_apollo_data.py \
    --datasets roleplaying \
    --output_dir data/apollo_raw
```

**Output:**
- `data/apollo_raw/roleplaying/dataset.yaml`
- Contains 300+ deception scenarios

### Step 2: Cache Activations (100 examples for PoC)

```bash
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train \
    --limit 100 \
    --batch_size 4 \
    --L_prime 28 \
    --T_prime 64
```

**Time:** ~15 minutes on free Colab
- Generation: ~5-10 min (3B model)
- Labeling: ~4-5 min (Cerebras API with rate limiting)

**Output:**
- `data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train/`
  - `shard_000.safetensors` (100 tensors)
  - `manifest.jsonl` (100 lines)

### Step 3: Validate Data Quality

```bash
python scripts/validate_deception_data.py \
    --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train
```

**Checks:**
- ✓ Data balance (honest vs deceptive ratio)
- ✓ Tensor shapes (28, 64, 3072)
- ✓ Label distribution
- ✓ Sample examples for manual inspection

### Step 4: Train Probes

```bash
# Mean pooling (recommended to start)
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling mean \
    --batch_size 32 \
    --epochs 50 \
    --patience 5
```

**Output:**
- Trains probes for all 28 layers
- Saves best model for each layer
- Reports validation AUROC
- `data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean/`

**Expected Results:**
- Random chance: 0.5 AUROC
- Good deception detection: 0.6-0.8 AUROC
- Strong signal: 0.8+ AUROC

---

## 📊 Full Pipeline (Complete Dataset)

### Step 1: Cache All Splits

```bash
# Training set
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train

# Validation set
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split validation

# Test set
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split test
```

**Data Splits** (automatic):
- Train: 60% of scenarios
- Validation: 20% of scenarios
- Test: 20% of scenarios

### Step 2: Train with Different Pooling Strategies

```bash
# Mean pooling
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling mean

# Max pooling
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling max

# Last token pooling
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling last

# Attention pooling (learned)
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling attn \
    --epochs 100
```

### Step 3: Analyze Results

```python
import json

# Load per-layer results
with open('data/probes/.../mean/layer_results.json') as f:
    results = json.load(f)

# Find best layer
best = max(results, key=lambda x: x['val_auc'])
print(f"Best layer: {best['layer']} with AUC: {best['val_auc']:.3f}")

# Plot per-layer AUCs
import matplotlib.pyplot as plt
layers = [r['layer'] for r in results]
aucs = [r['val_auc'] for r in results]
plt.plot(layers, aucs)
plt.xlabel('Layer')
plt.ylabel('Validation AUC')
plt.title('Deception Detection by Layer')
plt.show()
```

---

## 🔬 Advanced Usage

### Training on Specific Layer Only

```bash
# Train only on layer 20 (often strong for 3B models)
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling mean \
    --layer 20 \
    --epochs 100
```

### Custom Hyperparameters

```bash
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling attn \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --batch_size 64 \
    --epochs 200 \
    --patience 10
```

### Using Different Models

```bash
# Llama-3.2-1B (faster, less capable)
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Deception-Roleplaying \
    --split train \
    --L_prime 16  # 1B has 16 layers, not 28

# Then train
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling mean
```

---

## 📁 Directory Structure

```
Efficacy-of-ensemble-of-attention-probes/
├── data/
│   ├── apollo_raw/                          # Downloaded datasets
│   │   └── roleplaying/
│   │       └── dataset.yaml
│   │
│   ├── activations/                         # Cached activations
│   │   └── meta-llama_Llama-3.2-3B-Instruct/
│   │       └── Deception-Roleplaying/
│   │           ├── train/
│   │           │   ├── shard_000.safetensors
│   │           │   ├── shard_001.safetensors
│   │           │   └── manifest.jsonl
│   │           ├── validation/
│   │           └── test/
│   │
│   └── probes/                              # Trained probes
│       └── meta-llama_Llama-3.2-3B-Instruct/
│           └── Deception-Roleplaying/
│               ├── mean/
│               │   ├── probe_layer_0.pt
│               │   ├── probe_layer_1.pt
│               │   └── layer_results.json
│               ├── max/
│               ├── last/
│               └── attn/
│
├── scripts/
│   ├── download_apollo_data.py              # Download datasets
│   ├── cache_deception_activations.py       # Generate & cache
│   ├── validate_deception_data.py           # Validate quality
│   └── train_deception_probes.py            # Train probes
│
└── actprobe/src/actprobe/
    └── datasets/
        ├── deception_loaders.py             # Dataset classes
        └── templates.py                     # Prompt templates
```

---

## 🔧 Troubleshooting

### Issue: Cerebras API rate limit errors

**Solution:** Reduce `--requests_per_minute`:
```bash
python scripts/cache_deception_activations.py \
    ... \
    --requests_per_minute 20  # Instead of default 25
```

### Issue: CUDA out of memory during generation

**Solution 1:** Reduce batch size:
```bash
python scripts/cache_deception_activations.py \
    ... \
    --batch_size 2  # Instead of 4
```

**Solution 2:** Use CPU (slower):
```bash
# Modify ActivationRunner to force CPU
# Or use smaller model (1B instead of 3B)
```

### Issue: Imbalanced dataset (too many honest or deceptive)

**Check:**
```bash
python scripts/validate_deception_data.py \
    --activations_dir ...
```

**Solutions:**
- Adjust LLM judge prompt in `cache_deception_activations.py`
- Manually filter manifest.jsonl to balance classes
- Use weighted loss during training

### Issue: Low AUROC (< 0.55)

**Possible causes:**
1. Labeling is unreliable (LLM judge confused)
2. Model too small to exhibit deception (try 3B or 7B)
3. Not enough training data
4. Signal is weak in this model

**Debugging:**
1. Manually inspect 20 examples from manifest
2. Check if labels match your intuition
3. Try different layers (plot per-layer AUCs)
4. Try attention pooling (more flexible)

---

## 📊 Expected Performance

### Proof of Concept (100 examples)

| Pooling | Expected Val AUC | Training Time |
|---------|------------------|---------------|
| Mean    | 0.55 - 0.65      | 2-3 min       |
| Max     | 0.55 - 0.70      | 2-3 min       |
| Last    | 0.50 - 0.60      | 2-3 min       |
| Attn    | 0.60 - 0.75      | 5-10 min      |

### Full Dataset (500+ examples)

| Pooling | Expected Val AUC | Training Time |
|---------|------------------|---------------|
| Mean    | 0.60 - 0.75      | 10-15 min     |
| Max     | 0.65 - 0.80      | 10-15 min     |
| Last    | 0.55 - 0.70      | 10-15 min     |
| Attn    | 0.70 - 0.85      | 20-30 min     |

**Note:** These are rough estimates. Actual performance depends on:
- Model size (1B vs 3B)
- Label quality (LLM judge reliability)
- Dataset difficulty
- Hyperparameters

---

## 🎯 Next Steps After PoC

1. **If PoC works (AUC > 0.6):**
   - Scale to full dataset (500+ examples)
   - Try different models (1B, 7B)
   - Experiment with pooling strategies
   - Add more deception datasets (Insider Trading, etc.)

2. **If PoC is weak (AUC < 0.6):**
   - Manually label 50 examples to validate LLM judge
   - Try different labeling prompts
   - Visualize per-layer AUCs to see if any layer has signal
   - Consider using pre-labeled data (Option A from earlier discussion)

3. **Advanced experiments:**
   - Cross-model transfer (train on 3B, test on 1B)
   - Cross-dataset transfer (train on roleplaying, test on insider trading)
   - Ensemble probes across layers
   - Compare to hallucination detection probes

---

## 📚 Key Differences: Deception vs Hallucination

| Aspect | Hallucination Detection | Deception Detection |
|--------|------------------------|---------------------|
| **Task** | Detect factual errors | Detect strategic lies |
| **Label Source** | Compare to gold answer | LLM judge classification |
| **Data Type** | QA datasets | Dialogue scenarios |
| **Signal** | Knowledge retrieval failure | Intent/strategy signal |
| **Difficulty** | Moderate (ACT-ViT: 0.75 AUC) | Unknown (research question!) |

---

## 💡 Tips for Google Colab

### Mounting Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Change to your repo directory
%cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes
```

### Saving outputs to Drive

All scripts already save to `data/` which will persist in Drive if you run from a Drive-mounted directory.

### Checking GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Monitoring progress

The scripts include progress bars (tqdm) and logging. You'll see:
- Generation progress
- Labeling progress (with label counts)
- Training progress (per epoch)

---

## ❓ FAQ

**Q: Can I use different labeling models?**

A: Yes! Modify `--labeling_model`:
```bash
--labeling_model llama3.1-70b  # More capable, but slower
```

**Q: What if I don't have Cerebras API access?**

A: You can:
1. Use local model for labeling (modify CerebrasGenerator)
2. Use pre-labeled data from Apollo repo (requires different pipeline)
3. Manually label small PoC dataset

**Q: Can I skip resampling?**

A: Currently required for batching. See earlier discussion on resampling trade-offs.

**Q: How do I compare to baseline?**

A: Train probes on hallucination datasets too:
```bash
python scripts/train_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Movies \
    --pooling mean
```

Then compare AUROCs: deception vs hallucination.

---

## 📧 Support

For issues or questions:
1. Check existing GitHub issues
2. Run validation script to diagnose data problems
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

**Good luck with your deception detection experiments! 🚀**
