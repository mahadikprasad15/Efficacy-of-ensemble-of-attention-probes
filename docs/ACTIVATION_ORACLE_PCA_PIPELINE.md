# Activation Oracle PCA Pipeline

This pipeline uses **existing PCA artifacts** from prior ablations to create durable activation-vector artifacts and run activation-oracle inference.

Implemented now:
- Experiment 1: combined removed-PC interpretation.
- Experiment 3: per-PC attribution.

Deferred:
- Experiment 2 (wrong-to-right OOD flips with orig vs clean comparison), but prerequisite prediction artifacts are persisted.

## Inputs

Saved PCA root example:

`/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying`

Expected PCA files:
- `<saved_pca_root>/<pooling>/pca_artifacts/layer_<L>.npz`

Locked matrix config:
- `configs/activation_oracle/locked_matrix_v1.json`

Question config:
- `configs/activation_oracle/questions_default_v1.json`

## Step 1: Prepare persisted vectors + jobs

```bash
python -u scripts/activation_oracle/prepare_oracle_vectors_from_saved_pca.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --saved_pca_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --eval_split id_val=/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying/validation \
  --eval_split ood_test=/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations/meta-llama_Llama-3.2-1B-Instruct/Deception-InsiderTrading/test \
  --probes_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --matrix_preset locked_v1 \
  --progress_every 100 \
  --job_splits ood_test \
  --experiments exp1_combined,exp3_per_pc \
  --output_root artifacts
```

## Step 2: Run activation oracle inference

```bash
python -u scripts/activation_oracle/run_activation_oracle.py \
  --jobs_jsonl artifacts/runs/activation_oracle_pca/meta-llama_llama-3.2-1b-instruct/deception-roleplaying/roleplaying_probes/locked_v1/<run_id>/results/jobs/exp1_combined_jobs.jsonl \
  --ao_model_id adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct \
  --placeholder_text " ?" \
  --hook_layer_index 1 \
  --norm_policy hidden_state_match \
  --progress_every 25 \
  --max_new_tokens 96 \
  --do_sample false \
  --output_root artifacts
```

Both scripts now emit live `tqdm` progress bars and periodic log lines to stdout.

Repeat for per-PC jobs by swapping `exp1_combined_jobs.jsonl` with `exp3_per_pc_jobs.jsonl`.

## Artifact layout

Run roots are canonical:

`artifacts/runs/<experiment>/<model>/<dataset>/<probe_set>/<variant>/<run_id>/`

Key files:
- `meta/run_manifest.json`
- `meta/status.json`
- `checkpoints/progress.json`
- `results/results.json`
- `logs/run.log`

Prepared-vector run also writes:
- `results/vectors/<split>/<pooling>/layer_<L>/k_<k>/bundle.npz`
- `results/vectors/<split>/<pooling>/layer_<L>/k_<k>/bundle_meta.json`
- `results/tables/predictions_by_sample.parquet` (or csv fallback)
- `results/jobs/exp1_combined_jobs.jsonl`
- `results/jobs/exp3_per_pc_jobs.jsonl`
- `results/jobs/payloads/<job_id>.npz`

AO run writes:
- `results/responses.jsonl`
- `results/results.json`

## bundle.npz schema

- `sample_ids` `(N,)`
- `labels` `(N,)`
- `orig` `(N,D)`
- `removed_sum` `(N,D)`
- `clean` `(N,D)`
- `coeff_topk` `(N,k)`
- `pc_indices` `(k,)`

## Injection rule

At each placeholder token position:
- additive edit (not replacement)
- default scaling: `hidden_state_match`
- update: `x <- x + (||x|| / ||v||) * v`

where `x` is AO hidden state at placeholder position and `v` is injected vector.

## Notes on Experiment 2 (deferred)

`predictions_by_sample` plus persisted `orig/clean/removed_sum/coeff_topk` are sufficient to later:
- select wrong-to-right OOD flips,
- build AO prompts for original vs cleaned and differences,
- run with same AO engine.
