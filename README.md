# Efficacy of Ensemble of Attention Probes

Research code for studying whether pooling strategy, layer selection, and ensemble design improve the transfer of activation probes for deception detection.

The project compares single-layer probes, attention-pooled probes, source-to-target transfer matrices, and gated ensembles across in-distribution and out-of-distribution deception datasets. It is built around cached transformer activations so expensive model passes can be reused across training, evaluation, and analysis runs.

## What This Repository Supports

- caching hidden-state activations for deception datasets,
- training per-layer probes with mean, max, last-token, attention, and no-pooling variants,
- evaluating probes on held-out and out-of-distribution datasets,
- running pairwise source-to-target probe transfer matrices,
- training and evaluating static, weighted, and gated layer ensembles,
- comparing pooling and ensemble strategies across datasets,
- analyzing probe geometry, learned attention, gating behavior, and residual directions,
- preparing plots and summary artifacts for experiment reports.

Most long-running workflows support resume or skip-existing behavior. New experiment outputs should be kept under `artifacts/`, `results/`, or the configured Drive-backed roots rather than mixed into source directories.

## Project Structure

```text
actprobe/src/actprobe/
  datasets/       dataset loaders and typed deception examples
  llm/            generation and activation extraction utilities
  probes/         layer probes, pooling modules, and ensemble models
  evaluation/     answer extraction and scoring helpers
  features/       feature transforms such as dataset fingerprints and PCA projections
scripts/
  data/           dataset preparation and activation caching
  training/       probe and ensemble training entrypoints
  evaluation/     OOD, matrix, and ablation evaluators
  pipelines/      resumable multi-stage experiment wrappers
  analysis/       geometry, attribution, PCA, residual, and plotting analyses
  comparison/     cross-run and cross-pooling comparison utilities
  visualization/  figure-generation helpers
  activation_oracle/ activation-oracle preparation and execution
configs/          JSON configs for structured experiment families
docs/             Colab workflows and experiment runbooks
tests/            unit and integration tests
```

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expose the local package:

```bash
export PYTHONPATH=actprobe/src
```

Set API credentials only through environment variables:

```bash
export HF_TOKEN=...
export CEREBRAS_API_KEY=...
```

## Quick Checks

Run the lightweight pipeline sanity check:

```bash
PYTHONPATH=actprobe/src python verify_pipeline.py
```

Run the test suite:

```bash
PYTHONPATH=actprobe/src pytest -q
```

For large experiments, first run a bounded activation cache or a dry-run pipeline where available.

## Core Workflow

### 1. Cache Activations

Cache activations for a bounded sample before launching full runs:

```bash
PYTHONPATH=actprobe/src python scripts/data/cache_deception_activations.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset Deception-Roleplaying \
  --split train \
  --limit 100 \
  --batch_size 4
```

Cached activations are typically stored under:

```text
data/activations/<model>/<dataset>/<split>/
```

Drive-backed Colab workflows often use:

```text
data/activations_fullprompt/<model>/<dataset>/<split>/
```

### 2. Train Layer Probes

Train per-layer probes for one pooling strategy:

```bash
PYTHONPATH=actprobe/src python scripts/training/train_deception_probes.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset Deception-Roleplaying \
  --pooling mean \
  --resume
```

Supported pooling modes include `mean`, `max`, `last`, `attn`, and `none`.

### 3. Evaluate Transfer

Run a resumable pairwise source-to-target matrix:

```bash
PYTHONPATH=actprobe/src python scripts/pipelines/run_pairwise_eval_matrix.py \
  --activations_root data/activations_fullprompt \
  --probes_root data/probes \
  --results_root results/ood_evaluation \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --run_id example-run \
  --resume \
  --dry_run
```

Remove `--dry_run` after checking the planned work.

### 4. Evaluate Ensembles

Run a source-specific gated ensemble matrix over frozen probe banks:

```bash
PYTHONPATH=actprobe/src python scripts/pipelines/run_probe_gated_ensemble_matrix.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --activations_root data/activations_fullprompt \
  --probes_root data/probes \
  --artifact_root artifacts \
  --run_id example-run \
  --resume
```

For older K-sweep ensemble comparisons, use:

```bash
PYTHONPATH=actprobe/src python scripts/evaluation/evaluate_ensembles_comprehensive.py \
  --pooling mean \
  --val_activations_dir data/activations/.../validation \
  --probes_dir data/probes/.../mean \
  --output_dir results/ensembles/mean \
  --eval_mode validation
```

## Artifacts

This repository uses several artifact roots because the workflows evolved over multiple experiment families:

- `data/activations*/` for cached model activations,
- `data/probes/` for trained probe checkpoints,
- `results/` for evaluation summaries, matrices, plots, and comparison outputs,
- `artifacts/runs/` for newer resumable pipeline manifests, checkpoints, and structured outputs.

New resumable runs should write:

```text
artifacts/runs/<workflow>/<model>/<variant>/<run_id>/
  meta/
  checkpoints/
  results/
  logs/
```

When a script exposes `--resume`, `--dry_run`, `--force_*`, or skip-existing behavior, prefer those controls over deleting partial outputs.

## Documentation

Detailed runbooks and experiment notes live outside the README:

- `ENSEMBLE_WORKFLOW_GUIDE.md`
- `DECEPTION_DETECTION_GUIDE.md`
- `POOLING_COMPARISON_GUIDE.md`
- `PER_TOKEN_GUIDE.md`
- `SKIP_EXISTING_GUIDE.md`
- `docs/COLAB_CLEAN_PIPELINE.md`
- `docs/colab_3b_principled_cross_dataset.md`
- `docs/ACTIVATION_ORACLE_PCA_PIPELINE.md`
- `docs/COLAB_PCA_ABLATION_DEEP_DIVE.md`

Use the README as the project map and the runbooks for exact Colab or long-running experiment commands.

## Development Notes

- Keep generated tensors, checkpoints, and plots out of source directories.
- Prefer bounded runs with `--limit` before full activation caching.
- Prefer fixed `--run_id` values when coordinating resumed runs across Colab, Drive, and local machines.
- Format Python changes with `black actprobe/src scripts`.
- For pipeline changes, include the command run and the output artifact path in the commit or PR notes.
