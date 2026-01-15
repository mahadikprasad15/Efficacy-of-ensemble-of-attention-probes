# Efficacy of Ensemble of Attention Probes

**Testing whether token pooling + layer ensembling captures ACT-ViT's gains without the full ViT architecture.**

This repository implements a systematic ablation study comparing:
- **Baseline**: Static single (layer, token) probes
- **Variant A**: Token pooling (mean/max/last/learned attention)
- **Variant B**: Layer ensembling (static mean/weighted/gated)
- **Variant C**: Two-level attention (token + layer attention jointly)

---

## Experimental Setup

### Datasets (5 environments)
- HotpotQA (no context)
- HotpotQA-WC (with context)
- TriviaQA
- IMDB (sentiment, 1-shot)
- Movies (factual QA)

**Splits**: 10k train / 2k val / 10k test (Movies: 7857 test)

### Models
- Llama-3.2-1B-Instruct
- Llama-3.2-3B-Instruct
- Qwen3 ~1.5B

### Training Protocol
- **LODO (Leave-One-Dataset-Out)**: Train on 4 datasets, test on held-out 5th
- **Worst-domain selection**: Pick hyperparams by `min(val_AUC)` across train datasets
- **Metrics**: AUROC (primary), Accuracy, FPR@95TPR

---

## Pipeline Overview

### Phase 0: Data Preparation
```bash
# Datasets are loaded on-the-fly from HuggingFace
# No separate download needed
```

### Phase 1: Activation Caching
Extract and cache (L', T', D) resampled activation tensors for all datasets.

```bash
python scripts/cache_activations.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Movies \
    --split validation \
    --extractor cerebras \
    --L_prime 16 \
    --T_prime 64 \
    --batch_size 4
```

**Output**: `data/activations/{model}/{dataset}/{split}/shard_*.safetensors` + `manifest.jsonl`

**Key features**:
- Left-padding for batched generation (fixed critical bug)
- Cerebras API for LLM-based answer extraction (fallback to regex)
- Generation length logged to manifest

---

### Phase 2: Baseline Probes

#### 2a. Token Heatmap (Diagnostic)
Trains logistic regression at each (layer, token) position to visualize signal concentration.

```bash
python scripts/train_heatmap_probes.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Movies \
    --output_dir data/heatmaps
```

**Output**: `data/heatmaps/{model}_{dataset}_heatmap.png` + `.npy`

---

### Phase 3: Variant A - Token Pooling

Train per-layer probes with different pooling strategies.

```bash
python scripts/train_probes.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --held_out_dataset Movies \
    --pooling mean \  # Options: mean, max, last, attn
    --epochs 10 \
    --patience 3 \
    --output_dir data/probes/mean_Movies
```

**Output**: `data/probes/{pooling}_{dataset}/probe_layer_{i}.pt` + `results.jsonl`

**Key features**:
- Early stopping (patience=3 epochs)
- Per-layer training with worst-domain (min val AUC) selection
- Weight decay (1e-4) for regularization

**Analyze attention entropy** (for `pooling=attn`):
```bash
python scripts/analyze_attention_entropy.py \
    --probe_path data/probes/attn_Movies/probe_layer_8.pt \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Movies \
    --layer 8
```

---

### Phase 4: Variant B - Ensemble Training

Given trained per-layer probes, train ensemble variants.

```bash
python scripts/train_ensembles.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --held_out_dataset Movies \
    --pooling mean \
    --probe_dir data/probes/mean_Movies \
    --k_pct_list 25,30,40,50,60,70 \
    --selection_mode worst_domain \
    --output_dir data/ensembles
```

**Output**: `data/ensembles/ensemble_results_{dataset}.json`

**Ensemble types**:
1. **StaticMean**: Uniform averaging over top-K% layers
2. **StaticWeighted**: AUC-weighted averaging
3. **GatedEnsemble**: Learned input-conditioned weighting (MLP with dropout=0.2)

**Key features**:
- K% selected per-dataset based on worst-domain AUC
- Gated ensemble trained with early stopping
- Layer selection logged for analysis

---

### Phase 5: Variant C - Two-Level Attention

Train a unified model with token + layer attention.

```bash
python scripts/train_twolevel.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --held_out_dataset Movies \
    --shared_token_attn \
    --epochs 20 \
    --batch_size 16 \
    --output_dir data/twolevel
```

**Output**: `data/twolevel/twolevel_{dataset}.pt` + `.json`

**Architecture**:
1. Token attention within each layer (shared or per-layer)
2. Layer attention across layer summaries
3. Final linear classifier

---

### Phase 6: Evaluation & Analysis

#### Generalization Matrix
```bash
python scripts/eval_matrix.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --probe_dir data/probes/mean_Movies \
    --output_dir data/results
```

**Output**: 5×5 heatmap (train dataset → test dataset AUC)

#### Analysis Scripts
```bash
# Probe weight similarity
python scripts/analysis/analyze_weights.py --probe_dir data/probes/mean_Movies

# PCA visualization
python scripts/analysis/analyze_pca.py --data_dir data/activations

# Token position analysis
python scripts/analysis/analyze_tokens.py --data_dir data/activations

# Ensemble breakdown
python scripts/analysis/analyze_ensembles.py --results_path data/ensembles/ensemble_results_Movies.json
```

---

## Key Implementation Decisions

### 1. **LLM Judge vs Regex** → **Cerebras API (with regex fallback)**
- More reliable for QA answer extraction
- Handles formatting variations (e.g., "Who acted as Neo?" → "Keanu Reeves" vs "The answer is Keanu Reeves")

### 2. **Activation Extraction** → **HuggingFace `output_hidden_states`**
- Extracts post-FFN + residual hidden states (matches ACT-ViT)
- **Critical fix**: Left-padding for batched generation (avoids token alignment issues)

### 3. **K% Selection** → **Per-dataset during LODO**
- During training: Select best K% based on 4 train datasets' min val AUC
- Report both per-dataset K% and global K%=40 baseline

### 4. **Regularization**
- Weight decay: 1e-4 (spec recommendation)
- Dropout: 0.2 in gated ensemble MLP
- Early stopping: patience=3 epochs

---

## Directory Structure

```
.
├── actprobe/src/actprobe/      # Core library
│   ├── datasets/               # Dataset loaders
│   ├── llm/                    # LLM generation + activation extraction
│   ├── probes/                 # Probe models (LayerProbe, TwoLevelAttentionProbe, ensembles)
│   ├── features/               # Activation resampling
│   ├── evaluation/             # Answer extraction + scoring
│   └── utils/                  # Normalization, etc.
├── scripts/
│   ├── cache_activations.py   # Phase 1: Cache (L', T', D) tensors
│   ├── train_heatmap_probes.py # Phase 2: Baseline heatmaps
│   ├── train_probes.py         # Phase 3: Variant A (pooling)
│   ├── train_ensembles.py      # Phase 4: Variant B (ensembles)
│   ├── train_twolevel.py       # Phase 5: Variant C (two-level attention)
│   ├── eval_matrix.py          # Generalization matrix
│   ├── analyze_attention_entropy.py
│   └── analysis/               # PCA, weight similarity, etc.
├── data/
│   ├── activations/            # Cached tensors (gitignored)
│   ├── probes/                 # Trained probe weights
│   ├── ensembles/              # Ensemble results
│   ├── twolevel/               # Two-level attention results
│   ├── heatmaps/               # Token-layer AUC heatmaps
│   └── results/                # Generalization matrices, plots
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Google Drive Integration (for Colab)

All scripts support `--data_dir` and `--output_dir` args:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Run with Drive paths
!python scripts/cache_activations.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset Movies \
    --data_dir /content/drive/MyDrive/actvit/activations \
    --output_dir /content/drive/MyDrive/actvit/activations
```

---

## Expected Results

### If **token-attn + layer-gating ≈ ACT-ViT gains**:
✅ "Most of ACT-ViT's advantage is adaptive location selection, not ViT magic"

### If **OOD barely improves**:
⚠️ "Dynamic pooling learns dataset identity shortcuts; need stronger multi-environment constraints"

### If **cheap 2D conv helps**:
💡 "Local 2D smoothing is useful; signal is spatially coherent but jittery"

---

## Citation

If you use this codebase, please cite the original ACT-ViT and Orgad et al. papers:

```bibtex
@article{actvit2024,
  title={Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT},
  url={https://arxiv.org/abs/2510.00296},
  year={2024}
}

@article{orgad2023llmsknow,
  title={LLMs Know: Detecting hallucinations in LLMs via internal representations},
  url={https://github.com/technion-cs-nlp/LLMsKnow},
  year={2023}
}
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 4 or 2)
- Use `--L_prime 16` (fewer layers) or `--T_prime 32` (fewer tokens)
- For two-level attention, use `--shared_token_attn` (fewer parameters)

### Activation extraction fails
- Check HuggingFace token: `export HF_TOKEN=your_token`
- For gated models (Llama), authenticate: `huggingface-cli login`

### Cerebras API errors
- Falls back to regex automatically
- To force regex: `--extractor regex`
- Check API key: `export CEREBRAS_API_KEY=your_key`

### Cache not found
- Run `cache_activations.py` first for all datasets
- Check paths: `ls data/activations/{model_id}/{dataset}/{split}/`
---
