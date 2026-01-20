# Skip-Existing Behavior - Quick Reference

All scripts now check for existing results and skip regeneration automatically!

## ✅ What Gets Skipped Automatically

### 1. **Activation Caching** (`cache_deception_activations.py`)
**Checks for:**
- `shard_*.safetensors` files
- `manifest.jsonl`

**Skip Message:**
```
⚠️  Activations already exist in data/activations/...
   Found 10 shard(s) and manifest.jsonl
   Skipping generation to avoid overwriting existing data.
   To regenerate, use --force flag or delete the directory.
```

**Override:** `--force` flag

---

### 2. **Probe Training** (`train_deception_probes.py`)
**Checks for:**
- `layer_results.json` with expected number of layers

**Skip Message:**
```
⚠️  Probes already trained in data/probes/mean/
   Found layer_results.json with 28 layers
   Skipping training to avoid overwriting existing probes.
   To retrain, delete the directory or train specific layer with --layer flag.
```

**Override:** Delete directory OR use `--layer X` to train specific layer

---

### 3. **OOD Evaluation** (`evaluate_ood_all_pooling.py`)
**Checks for:**
- `ood_results_all_pooling.json`
- `logits/*.npy` files

**Skip Message:**
```
⚠️  OOD EVALUATION RESULTS ALREADY EXIST
Found: results/ood_evaluation/ood_results_all_pooling.json
Found: 4 logit files in results/ood_evaluation/logits
Skipping evaluation to avoid overwriting existing results.
To re-evaluate, delete the output directory and run again.
```

**Override:** Delete output directory

---

### 4. **Ensemble Evaluation** (`evaluate_ensembles_comprehensive.py`)
**Checks for:**
- `ensemble_k_sweep_validation.json` (for validation mode)
- `ensemble_k_sweep_ood.json` (for OOD mode)

**Skip Message:**
```
⚠️  Validation ensemble results already exist: .../ensemble_k_sweep_validation.json
   Skipping validation evaluation.
⚠️  OOD ensemble results already exist: .../ensemble_k_sweep_ood.json
   Skipping OOD evaluation.
✓ ALL REQUESTED EVALUATIONS ALREADY EXIST
To re-run, delete the results files and run again.
```

**Override:** Delete result files

---

## 🎯 Your Current Situation

### ✅ **Already Have (Will Skip):**
```bash
# Roleplaying activations
/content/drive/MyDrive/activations/Deception-Roleplaying/train/
/content/drive/MyDrive/activations/Deception-Roleplaying/validation/

# Trained probes for all 4 pooling strategies
/content/drive/MyDrive/probes/mean/layer_results.json
/content/drive/MyDrive/probes/max/layer_results.json
/content/drive/MyDrive/probes/last/layer_results.json
/content/drive/MyDrive/probes/attn/layer_results.json
```

### 🆕 **Need to Generate (Will NOT Skip):**
```bash
# Insider trading OOD activations
/content/drive/MyDrive/activations/insider_trading/test/

# OOD evaluation results
/content/drive/MyDrive/results/ood_evaluation/ood_results_all_pooling.json
/content/drive/MyDrive/results/ood_evaluation/logits/*.npy

# Ensemble K-sweep results (per pooling)
/content/drive/MyDrive/results/ensembles/mean/ensemble_k_sweep_*.json
/content/drive/MyDrive/results/ensembles/max/ensemble_k_sweep_*.json
/content/drive/MyDrive/results/ensembles/last/ensemble_k_sweep_*.json
/content/drive/MyDrive/results/ensembles/attn/ensemble_k_sweep_*.json

# Final comparison
/content/drive/MyDrive/results/final_comparison/*.png
```

---

## 📝 Safe Workflow Commands

All these commands are now safe to run - they'll skip existing work:

```bash
# 1. Cache roleplaying activations (WILL SKIP - already exists)
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train
# Output: ⚠️ Activations already exist... Skipping

# 2. Train probes (WILL SKIP - already exists)
python scripts/train_deception_probes.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --pooling mean \
    --output_dir /content/drive/MyDrive/probes
# Output: ⚠️ Probes already trained... Skipping

# 3. Cache insider trading activations (WILL RUN - doesn't exist yet)
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-InsiderTrading \
    --split test \
    --output_dir /content/drive/MyDrive/activations/insider_trading
# Output: Will generate new activations

# 4. Evaluate all probes on OOD (WILL RUN - doesn't exist yet)
python scripts/evaluate_ood_all_pooling.py \
    --probes_base_dir /content/drive/MyDrive/probes \
    --ood_activations_dir /content/drive/MyDrive/activations/insider_trading/test \
    --output_dir /content/drive/MyDrive/results/ood_evaluation
# Output: Will evaluate and save results

# 5. Run ensemble K-sweeps (WILL RUN - doesn't exist yet)
for pooling in mean max last attn; do
    python scripts/evaluate_ensembles_comprehensive.py \
        --pooling $pooling \
        --val_activations_dir /content/drive/MyDrive/activations/validation \
        --probes_dir /content/drive/MyDrive/probes/${pooling} \
        --output_dir /content/drive/MyDrive/results/ensembles/${pooling} \
        --eval_mode validation
done
# Output: Will run K-sweeps and save results

# 6. Generate final comparison (ALWAYS SAFE - just loads results)
python scripts/compare_all_pooling_ensembles.py \
    --results_dir /content/drive/MyDrive/results/ensembles \
    --output_dir /content/drive/MyDrive/results/final_comparison \
    --eval_type validation
# Output: Creates comparison charts
```

---

## 🔄 When You WANT to Regenerate

### Option 1: Delete Specific Results
```bash
# Delete OOD results to re-evaluate
rm -rf /content/drive/MyDrive/results/ood_evaluation

# Delete ensemble results for one pooling to re-run
rm -rf /content/drive/MyDrive/results/ensembles/mean

# Delete probes to retrain
rm -rf /content/drive/MyDrive/probes/mean
```

### Option 2: Use --force Flag (where available)
```bash
# Force regenerate activations
python scripts/cache_deception_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset Deception-Roleplaying \
    --split train \
    --force
```

---

## 🎯 Bottom Line

**You can now safely run the complete pipeline from start to finish!**

- ✅ Existing work in Google Drive is protected
- ✅ Only NEW work will be generated
- ✅ No wasted computation or API credits
- ✅ Clear messages show what's being skipped

**Next Steps:**
1. Get insider trading activations (step 3 above)
2. Run OOD evaluation (step 4 above)
3. Run ensemble K-sweeps (step 5 above)
4. Generate final comparison (step 6 above)

All committed and pushed! 🎉
