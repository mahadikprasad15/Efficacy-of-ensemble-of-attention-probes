# Clean Soft Prefix Pipeline - Colab Commands

## Step 0: Clean Up Old Confusing Directories

```python
# DELETE OLD ACTIVATION DIRECTORIES (they were confusing/broken)
import shutil
import os

old_dirs = [
    "/content/drive/MyDrive/data/soft_prefix_activations",
    "/content/drive/MyDrive/checkpoints/soft_prefix",
]

for d in old_dirs:
    if os.path.exists(d):
        print(f"Deleting: {d}")
        shutil.rmtree(d)
        print("  Deleted!")
    else:
        print(f"Not found: {d}")
```

---

## Step 1: Pull Latest Code

```bash
cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes
!git pull
```

---

## Step 2: Run the Complete Pipeline

```python
!python scripts/pipelines/run_soft_prefix_pipeline.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --id_raw_yaml /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/roleplaying/dataset.yaml \
    --ood_raw_json /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/insider_trading/llama-70b-3.3-generations.json \
    --probe_path /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes_layer_agnostic/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/last/probe.pt \
    --layer_idx 15 \
    --pooling last \
    --prompt_len 16 \
    --steps 3000 \
    --batch_size 8 \
    --lr 1e-3 \
    --val_every 50 \
    --output_base /content/drive/MyDrive/checkpoints/soft_prefix_clean/meta-llama_Llama-3.2-3B-Instruct/layer_agnostic_last_layer15
```

---

## What This Does

### Phase 1: Data Preparation
- Loads raw YAML (NO cached manifests!)
- Creates train/val/test splits (60/20/20 stratified)
- Saves splits to `output_base/splits/id/`

### Phase 2: Training
- Initializes random 16-token prefix
- Trains with contrastive separation loss
- **Validates on ID val every 50 steps**
- Saves best checkpoint based on **validation AUROC**
- Saves to `output_base/prefix/`

### Phase 3: Activation Caching
- Caches ID val activations → `output_base/activations/id_val/`
- Caches ID test activations → `output_base/activations/id_test/`
- Caches OOD test activations → `output_base/activations/ood_test/`

### Phase 4: Evaluation
- Evaluates probe AUROC on ID val, ID test, OOD test
- Saves results to `output_base/results/evaluation_results.json`

---

## Expected Output Structure

```
checkpoints/soft_prefix_clean/meta-llama_Llama-3.2-3B-Instruct/layer_agnostic_last_layer15/
├── prefix/
│   ├── prefix.safetensors      # Trained prefix
│   ├── config.json             # Training config
│   └── training_log.json       # Metrics per step
├── splits/
│   └── id/
│       ├── train.jsonl         # 60% of ID
│       ├── val.jsonl           # 20% of ID
│       └── test.jsonl          # 20% of ID
├── activations/
│   ├── id_val/
│   │   ├── shard_*.safetensors
│   │   └── manifest.jsonl
│   ├── id_test/
│   │   └── ...
│   └── ood_test/
│       └── ...
└── results/
    └── evaluation_results.json
```

---

## Sanity Check Results

After the pipeline completes, you should see:

```
Results:
  ID Val AUROC:  ~0.85+ (should be high - this was validated during training)
  ID Test AUROC: ~0.85  (should be similar to val - confirms no overfitting)
  OOD Test AUROC: ???   (this is what we want to measure)
```

If ID Val/Test AUROC is low (~0.5), something is still wrong!
