# Activation Oracle PCA Pipeline (Global PC Mode)

This pipeline uses **existing PCA artifacts** to create global activation-vector artifacts and run activation-oracle inference.

Implemented now:
- Experiment 1: combined PC interpretation (global, raw PC sum).
- Experiment 3: per-PC attribution (global, raw PC direction).
- Experiment 2: probe flip mining (orig vs clean comparisons, OOD only).

## Inputs

Saved PCA root example:

`/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying`

Expected PCA files:
- `<saved_pca_root>/<pooling>/pca_artifacts/layer_<L>.npz`

Locked matrix config:
- `configs/activation_oracle/locked_matrix_v1.json`

Question config:
- `configs/activation_oracle/questions_default_v1.json`
- `configs/activation_oracle/questions_exp2_v1.json`

## Step 1: Prepare global vectors + jobs

```bash
python -u scripts/activation_oracle/prepare_oracle_vectors_from_saved_pca.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --saved_pca_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --matrix_preset locked_v1 \
  --experiments exp1_combined,exp3_per_pc \
  --progress_every 25 \
  --output_root artifacts
```

What gets built:
- Exp1: 12 combined vectors (pooling x layer x k).
- Exp3: 80 individual PC vectors (pooling x layer x pc index).

## Step 2: Run activation oracle inference

```bash
python -u scripts/activation_oracle/run_activation_oracle.py \
  --jobs_jsonl artifacts/runs/activation_oracle_pca_global/meta-llama_llama-3.2-1b-instruct/deception-roleplaying/roleplaying_probes/locked_v1/<run_id>/results/jobs/exp1_combined_jobs.jsonl \
  --ao_model_id adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct \
  --placeholder_text " ?" \
  --hook_layer_index 1 \
  --norm_policy hidden_state_match \
  --progress_every 25 \
  --max_new_tokens 96 \
  --do_sample false \
  --output_root artifacts
```

Repeat for per-PC jobs by swapping `exp1_combined_jobs.jsonl` with `exp3_per_pc_jobs.jsonl`.

Both scripts emit live `tqdm` progress bars and periodic log lines to stdout.

## Exp2: Prepare flip-mined jobs (orig vs clean)

```bash
python -u scripts/activation_oracle/prepare_oracle_vectors_exp2.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --saved_pca_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --activations_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-1B-Instruct/Deception-InsiderTrading/test \
  --probes_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --matrix_preset locked_v1 \
  --questions_config configs/activation_oracle/questions_exp2_v1.json \
  --top_n 5 \
  --progress_every 100 \
  --output_root artifacts
```

This creates:
- `results/tables/exp2_top_flips.csv`
- `results/jobs/exp2_compare_jobs.jsonl`

Run AO inference on exp2 jobs with the same `run_activation_oracle.py` command, swapping `--jobs_jsonl`.

## Artifact layout

Run roots are canonical:

`artifacts/runs/<experiment>/<model>/<dataset>/<probe_set>/<variant>/<run_id>/`

Key files:
- `meta/run_manifest.json`
- `meta/status.json`
- `checkpoints/progress.json`
- `results/results.json`
- `logs/run.log`

Prepared-vector run writes:
- `results/vectors/exp1_combined/<pooling>/layer_<L>/k_<k>/vector.npz`
- `results/vectors/exp1_combined/<pooling>/layer_<L>/k_<k>/vector_meta.json`
- `results/vectors/exp3_per_pc/<pooling>/layer_<L>/pc_<idx>/vector.npz`
- `results/vectors/exp3_per_pc/<pooling>/layer_<L>/pc_<idx>/vector_meta.json`
- `results/vectors/vectors_index.csv`
- `results/jobs/exp1_combined_jobs.jsonl`
- `results/jobs/exp3_per_pc_jobs.jsonl`

AO run writes:
- `results/responses.jsonl`
- `results/results.json`

## Injection rule

At each placeholder token position:
- additive edit (not replacement)
- default scaling: `hidden_state_match`
- update: `x <- x + (||x|| / ||v||) * v`

where `x` is AO hidden state at placeholder position and `v` is injected vector.
