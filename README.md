# Efficacy of Ensemble of Attention Probes

**Do attention-based pooling and layer ensembling improve probe generalization for deception detection?**

This project investigates whether learned pooling strategies (mean, max, last, **attention**) combined with layer ensembling (static, weighted, **gated**) can create robust probes that generalize from in-distribution to out-of-distribution scenarios.

## Experimental Setup

### Datasets

| Dataset | Type | Source | Use |
|---------|------|--------|-----|
| **Deception-Roleplaying** | AI roleplaying deceptive characters | Custom | In-distribution (train/val) |
| **Deception-InsiderTrading** | AI hiding insider trading decisions | Apollo Research | Out-of-distribution (test) |

### Model
- **Llama-3.2-3B-Instruct** (meta-llama)
- 28 transformer layers for probing

### Labeling
- **Cerebras API** (Llama-3.1-8B) for automated deception classification
- Labels: `honest` (0) vs `deceptive` (1)

---

## Pipeline Overview

### Phase 1: Activation Caching
Extract hidden states from Llama-3.2-3B while processing deceptive/honest scenarios.

```bash
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train \
    --limit 500 \
    --batch_size 4 \
    --hf_token $HF_TOKEN \
    --cerebras_key $CEREBRAS_API_KEY
```

**Output**: `data/activations/{model}/{dataset}/{split}/shard_*.safetensors`

---

### Phase 2: Probe Training (4 Pooling Strategies)

Train per-layer probes with different token pooling methods.

```bash
python scripts/train_probes_all_pooling.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --output_dir data/probes
```

**Pooling strategies:**
| Strategy | Description |
|----------|-------------|
| `mean` | Average all token representations |
| `max` | Max-pool across tokens |
| `last` | Use final token only |
| `attn` | **Learned attention** over tokens (query vector) |

**Output**: `data/probes/{model}/{dataset}/{pooling}/probe_layer_{i}.pt`

---

### Phase 3: OOD Evaluation

Evaluate trained probes on out-of-distribution data (Insider Trading).

```bash
# Cache OOD activations
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-InsiderTrading \
    --split test \
    --limit 200
```

Run OOD evaluation:
```bash
python scripts/evaluate_ood_all_pooling.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --ood_dir data/activations/.../Deception-InsiderTrading/test \
    --probes_dir data/probes/.../Deception-Roleplaying
```

---

### Top-5 Normalized Probes Evaluation

Train and evaluate the Top-5 source→target pairs from a CSV using **train-split normalization stats** (stored under each activation split). This runs normalized training on the source, then evaluates on the target split with the same stats.

```bash
python scripts/pipelines/run_normalized_top5_eval.py \
    --top5_csv /content/drive/MyDrive/.../top5.csv \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt \
    --results_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation \
    --target_split validation
```

**Outputs**:  
`results/ood_evaluation/<model_dir>/normalized_probes_evaluation/from-<source>_to-<target>/metrics.json` and `metrics.csv`, plus `summary.json` and `summary.csv` at the root of `normalized_probes_evaluation/`.

---

### Phase 4: Ensemble Training & Evaluation

Combine multiple layers with different ensemble strategies.

```bash
python scripts/evaluate_ensembles_comprehensive.py \
    --pooling attn \
    --val_activations_dir data/activations/.../validation \
    --probes_dir data/probes/.../attn \
    --output_dir results/ensembles/attn \
    --k_values 10,20,30,40,50,60,70,80,90
```

**Ensemble strategies:**
| Strategy | Description |
|----------|-------------|
| `Mean` | Simple average of layer logits |
| `Weighted` | AUC-weighted average (from validation) |
| `Gated` | **Learned input-dependent weighting** (MLP) |

---

### Phase 5: Mechanistic Analysis

Understand what the attention pooling and gated ensemble learn.

```bash
python scripts/analyze_mechanisms.py
```

**Generates:**
- Token-level attention weight visualization
- Layer gating weight analysis
- Attention entropy (focused vs distributed)
- Honest vs Deceptive comparison

---

## Architecture

### Single-Layer Probe (LayerProbe)
```
Input: (Batch, Tokens, Dim) from layer L
   ↓
Pooling: mean/max/last/attn → (Batch, Dim)
   ↓
Linear: Dim → 1 (binary logit)
   ↓
Output: P(deceptive)
```

### Learned Attention Pooling
```python
# Learns which tokens are relevant for deception detection
query = nn.Parameter(torch.randn(D, 1))   # Learned query vector
scores = x @ query                         # (B, T, 1) 
weights = softmax(scores, dim=1)           # Attention weights
pooled = sum(x * weights, dim=1)           # (B, D)
```

### Gated Ensemble
```python
# Learns input-dependent layer weighting
gate_net = MLP(D → 64 → L)      # Predicts layer weights per sample
weights = softmax(gate_net(x))   # (B, L)
output = sum(layer_logits * weights)  # Weighted combination
```

---

## Directory Structure

```
.
├── actprobe/src/actprobe/      # Core library
│   ├── datasets/               # Deception dataset loaders
│   ├── llm/                    # LLM generation + activation extraction
│   ├── probes/                 # Probe models (LayerProbe, GatedEnsemble)
│   └── evaluation/             # Cerebras-based deception labeling
├── scripts/
│   ├── cache_deception_activations.py  # Cache activations + labels
│   ├── train_probes_all_pooling.py     # Train per-layer probes
│   ├── evaluate_ood_all_pooling.py     # OOD evaluation
│   ├── evaluate_ensembles_comprehensive.py  # Ensemble K-sweep
│   ├── compare_pooling_layerwise.py    # Cross-pooling comparison
│   └── analyze_mechanisms.py           # Mechanistic analysis
├── data/
│   ├── activations/            # Cached tensors (gitignored)
│   ├── probes/                 # Trained probe weights
│   └── apollo_raw/             # Apollo Research insider trading data
├── results/
│   ├── ood_evaluation/         # OOD probe performance
│   ├── ensembles/              # Ensemble K-sweep results
│   └── mechanistic_analysis/   # Attention/gating visualizations
└── README.md
```

---

## Google Colab Usage

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project
%cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes

# 3. Set API keys
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token'
os.environ['CEREBRAS_API_KEY'] = 'your_cerebras_key'

# 4. Run pipeline
!python scripts/cache_deception_activations.py --dataset Deception-Roleplaying ...
!python scripts/train_probes_all_pooling.py ...
!python scripts/evaluate_ensembles_comprehensive.py ...
```

---

Related work:
- [Apollo Research: Scheming Evaluations](https://github.com/apollonetwork/scheming-evals)
- [Burns et al.: Discovering Latent Knowledge](https://arxiv.org/abs/2212.03827)
