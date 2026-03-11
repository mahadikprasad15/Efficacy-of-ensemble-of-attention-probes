# 3B Principled Cross-Dataset Colab Cell Set

This is a Colab-first execution sheet for the 3B principled cross-dataset run.

It enforces the intended workflow:

- cache `-full` activations once for every dataset and split
- save raw activations alongside full activations
- derive `-completion` activations by slicing raw activations
- train probes for the 6 source-row datasets only
- run the generic pairwise pipeline with the fixed `run_pairwise_eval_matrix.py`
- build principled matrices
- run principled Mahalanobis probe angles

All commands stream output with `python -u` and print `[phase/unit/total]` progress updates.

## Cell 1: Setup

```python
%cd /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes
```

## Cell 2: Configuration

```python
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes")
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = MODEL.replace("/", "_")

ACTS_ROOT = ROOT / "data" / "activations_fullprompt"
ACTS_RAW_ROOT = ROOT / "data" / "activations_raw"
PROBES_ROOT = ROOT / "data" / "probes"
OOD_ROOT = ROOT / "results" / "ood_evaluation"
ARTIFACT_ROOT = ROOT / "artifacts"
PAIRWISE_PIPELINE_ROOT = ROOT / "results" / "ood_evaluation" / MODEL_DIR / "All_dataset_pairwise_results"
SCORE_MATRIX_ROOT = ROOT / "results" / "pairwise_score_matrix_from_artifacts" / MODEL_DIR
MAHAL_ROOT = ROOT / "results" / "pairwise_mahalanobis_alignment" / MODEL_DIR / "matrix6x7" / "v1"

SOURCE_DATASETS = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
    "Deception-Roleplaying",
]
TARGET_ONLY_DATASET = "Deception-InsiderTrading"
ALL_DATASETS = SOURCE_DATASETS + [TARGET_ONLY_DATASET]

SEGMENTS = ["completion", "full"]
SPLITS = ["train", "validation", "test"]
POOLINGS = ["mean", "max", "last", "attn"]
FIXED_LAYERS = "10,12,15"

L_PRIME = 28
T_PRIME = 64
CACHE_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

# Keep these stable across reconnects if you want true resume behavior.
PAIRWISE_RUN_ID = "20260311T000000Z-3b-principled-pairwise-v1"
PRINCIPLED_RUN_ID = "20260311T000000Z-3b-principled-matrices-v1"
MAHAL_RUN_ID = "20260311T000000Z-3b-principled-mahal-v1"

print("ROOT =", ROOT)
print("MODEL =", MODEL)
print("MODEL_DIR =", MODEL_DIR)
print("ACTS_ROOT =", ACTS_ROOT)
print("ACTS_RAW_ROOT =", ACTS_RAW_ROOT)
print("PROBES_ROOT =", PROBES_ROOT)
print("OOD_ROOT =", OOD_ROOT)
print("ARTIFACT_ROOT =", ARTIFACT_ROOT)
print("PAIRWISE_PIPELINE_ROOT =", PAIRWISE_PIPELINE_ROOT)
print("SCORE_MATRIX_ROOT =", SCORE_MATRIX_ROOT)
print("MAHAL_ROOT =", MAHAL_ROOT)
```

## Cell 3: Streaming Helpers

```python
import json
import subprocess
import sys
import time


def has_manifest_and_shards(path: Path) -> bool:
    manifest = path / "manifest.jsonl"
    shards = sorted(path.glob("shard_*.safetensors"))
    return manifest.exists() and len(shards) > 0


def resolve_cache_dataset_args(dataset: str) -> list[str]:
    if dataset == "Deception-ConvincingGame":
        patterns = ["**/convincing-game__*.jsonl", "**/convincing-game__*.parquet", "**/convincing-game__*.json"]
        base_dataset = "Deception-InstructedDeception"
    elif dataset == "Deception-HarmPressureChoice":
        patterns = ["**/harm-pressure-choice__*.jsonl", "**/harm-pressure-choice__*.parquet", "**/harm-pressure-choice__*.json"]
        base_dataset = "Deception-InstructedDeception"
    else:
        return ["--dataset", dataset]

    raw_root = ROOT / "data" / "apollo_raw"
    matches = []
    for pattern in patterns:
        matches.extend(sorted(raw_root.rglob(pattern.replace("**/", ""))))
    if not matches:
        raise FileNotFoundError(f"Could not find typed-deception source file for {dataset} under {raw_root}")
    dataset_file = matches[0]
    print(f"[cache] {dataset} uses dataset_file={dataset_file}")
    return ["--dataset", base_dataset, "--dataset_file", str(dataset_file)]


def probe_unit_done(dataset: str, segment: str, pooling: str, expected_layers: int = 28) -> bool:
    results_path = (
        PROBES_ROOT
        / MODEL_DIR
        / f"{dataset}_slices"
        / f"{dataset}-{segment}"
        / pooling
        / "layer_results.json"
    )
    if not results_path.exists():
        return False
    try:
        rows = json.loads(results_path.read_text())
    except Exception:
        return False
    return isinstance(rows, list) and len(rows) == expected_layers


def run_stream(label: str, cmd: list[str]) -> None:
    print(f"[start] {label}")
    print("[cmd]", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc} after {elapsed:.1f}s")
    print(f"[done] {label} elapsed={elapsed:.1f}s")
```

## Cell 4: Phase 1, Cache Full + Raw Activations

```python
total_units = len(ALL_DATASETS) * len(SPLITS)
unit_idx = 0

for dataset in ALL_DATASETS:
    for split in SPLITS:
        unit_idx += 1
        full_dir = ACTS_ROOT / MODEL_DIR / f"{dataset}-full" / split
        raw_dir = ACTS_RAW_ROOT / MODEL_DIR / f"{dataset}-full" / split
        label = f"[phase1 {unit_idx}/{total_units}] cache {dataset}-full {split}"

        if has_manifest_and_shards(full_dir) and has_manifest_and_shards(raw_dir):
            print(f"[skip] {label} outputs already exist")
            continue

        cmd = [
            sys.executable,
            "-u",
            "scripts/data/cache_deception_activations.py",
            "--model", MODEL,
            "--split", split,
            "--dataset_output_name", f"{dataset}-full",
            "--include_prompt_tokens",
            "--save_raw",
            "--output_dir", str(ACTS_ROOT),
            "--raw_output_dir", str(ACTS_RAW_ROOT),
            "--L_prime", str(L_PRIME),
            "--T_prime", str(T_PRIME),
            "--batch_size", str(CACHE_BATCH_SIZE),
        ]
        cmd.extend(resolve_cache_dataset_args(dataset))

        if dataset == "Deception-InsiderTrading":
            cmd.append("--use_pregenerated")
        elif dataset == "Deception-Roleplaying":
            cmd.append("--use_gold_completions")

        run_stream(label, cmd)
```

## Cell 5: Phase 2, Slice Completion Activations From Raw

```python
total_units = len(ALL_DATASETS) * len(SPLITS)
unit_idx = 0

for dataset in ALL_DATASETS:
    for split in SPLITS:
        unit_idx += 1
        raw_dir = ACTS_RAW_ROOT / MODEL_DIR / f"{dataset}-full" / split
        completion_dir = ACTS_ROOT / MODEL_DIR / f"{dataset}-completion" / split
        label = f"[phase2 {unit_idx}/{total_units}] slice {dataset}-completion {split}"

        if has_manifest_and_shards(completion_dir):
            print(f"[skip] {label} outputs already exist")
            continue

        cmd = [
            sys.executable,
            "-u",
            "scripts/data/slice_cached_activations.py",
            "--raw_activations_dir", str(raw_dir),
            "--output_dir", str(completion_dir),
            "--slice_type", "completion",
            "--L_prime", str(L_PRIME),
            "--T_prime", str(T_PRIME),
        ]
        run_stream(label, cmd)
```

## Cell 6: Phase 3, Train Probes For Source Datasets

```python
total_units = len(SOURCE_DATASETS) * len(SEGMENTS) * len(POOLINGS)
unit_idx = 0

for dataset in SOURCE_DATASETS:
    for segment in SEGMENTS:
        for pooling in POOLINGS:
            unit_idx += 1
            label = f"[phase3 {unit_idx}/{total_units}] train {dataset}-{segment} {pooling}"

            if probe_unit_done(dataset, segment, pooling):
                print(f"[skip] {label} outputs already exist")
                continue

            cmd = [
                sys.executable,
                "-u",
                "scripts/training/train_deception_probes.py",
                "--model", MODEL,
                "--dataset", f"{dataset}-{segment}",
                "--activations_dir", str(ACTS_ROOT),
                "--pooling", pooling,
                "--output_dir", str(PROBES_ROOT),
                "--output_subdir", f"{dataset}_slices",
                "--output_dataset_name", f"{dataset}-{segment}",
                "--batch_size", str(TRAIN_BATCH_SIZE),
                "--resume",
            ]
            run_stream(label, cmd)
```

## Cell 7: Phase 4, Generic Pairwise Cross-Dataset Evaluation

```python
pairwise_summary = (
    PAIRWISE_PIPELINE_ROOT
    / PAIRWISE_RUN_ID
    / "results"
    / "summary.json"
)

label = "[phase4] pairwise eval matrix"
if pairwise_summary.exists():
    print(f"[skip] {label} summary already exists at {pairwise_summary}")
else:
    cmd = [
        sys.executable,
        "-u",
        "scripts/pipelines/run_pairwise_eval_matrix.py",
        "--activations_root", str(ACTS_ROOT),
        "--probes_root", str(PROBES_ROOT),
        "--results_root", str(OOD_ROOT),
        "--artifact_root", str(ARTIFACT_ROOT),
        "--pipeline_results_root", str(PAIRWISE_PIPELINE_ROOT),
        "--model", MODEL,
        "--run_id", PAIRWISE_RUN_ID,
        "--poolings", ",".join(POOLINGS),
        "--eval_batch_size", str(EVAL_BATCH_SIZE),
        "--resume",
        "--skip_training",
        "--no_tqdm",
        "--progress_every", "5",
    ]
    run_stream(label, cmd)
```

This pipeline now also writes same-dataset diagonal `test` summaries under:

- `results/ood_evaluation/<model_dir>/from-<dataset>/to-<dataset>/pair_summary.json`

That matters for the principled matrix build: the diagonal cell is then filled the same way as every off-diagonal cell, namely by choosing `(pooling, layer)` on source validation and reading the selected config's AUROC from the dataset's own `test` pair artifact.

## Cell 8: Phase 5, Build Principled Matrices

```python
principled_summary = (
    SCORE_MATRIX_ROOT
    / PRINCIPLED_RUN_ID
    / "results"
    / "summary.json"
)

label = "[phase5] principled matrix build"
if principled_summary.exists():
    print(f"[skip] {label} summary already exists at {principled_summary}")
else:
    cmd = [
        sys.executable,
        "-u",
        "scripts/analysis/build_pairwise_score_matrices_from_artifacts.py",
        "--probes_root", str(PROBES_ROOT),
        "--ood_results_root", str(OOD_ROOT),
        "--artifact_root", str(ARTIFACT_ROOT),
        "--output_root", str(SCORE_MATRIX_ROOT),
        "--model", MODEL,
        "--run_id", PRINCIPLED_RUN_ID,
        "--segments", "completion,full",
        "--poolings", ",".join(POOLINGS),
        "--fixed_layers", FIXED_LAYERS,
        "--resume",
    ]
    run_stream(label, cmd)
```

## Cell 9: Phase 6, Principled Mahalanobis Probe Angles

```python
principled_results_dir = (
    SCORE_MATRIX_ROOT
    / PRINCIPLED_RUN_ID
    / "results"
)

mahal_summary = (
    MAHAL_ROOT
    / MAHAL_RUN_ID
    / "results"
    / "summary.json"
)

label = "[phase6] principled mahalanobis"
if mahal_summary.exists():
    print(f"[skip] {label} summary already exists at {mahal_summary}")
else:
    cmd = [
        sys.executable,
        "-u",
        "scripts/analysis/run_principled_probe_angle_sigma_test.py",
        "--principled_results_dir", str(principled_results_dir),
        "--activations_root", str(ACTS_ROOT),
        "--probes_root", str(PROBES_ROOT),
        "--artifact_root", str(ARTIFACT_ROOT),
        "--output_root", str(MAHAL_ROOT),
        "--model", MODEL,
        "--run_id", MAHAL_RUN_ID,
        "--target_split", "test",
        "--target_split_fallback", "validation",
        "--cov_backend", "auto",
        "--covariance_scope", "required",
        "--progress_every", "20",
        "--no_tqdm",
        "--resume",
    ]
    run_stream(label, cmd)
```

## Cell 10: Sanity Checks

```python
assert pairwise_summary.exists(), pairwise_summary
assert principled_summary.exists(), principled_summary
assert mahal_summary.exists(), mahal_summary

print("pairwise summary:", pairwise_summary)
print("principled summary:", principled_summary)
print("mahal summary:", mahal_summary)
```
