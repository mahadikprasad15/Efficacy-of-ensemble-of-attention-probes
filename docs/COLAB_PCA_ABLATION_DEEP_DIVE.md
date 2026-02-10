# Colab PCA Ablation Deep Dive

This guide runs the notebook-first PCA deep dive workflow implemented in:

- `scripts/analysis/pca_ablation_colab_deep_dive.py`
- `notebooks/pca_ablation_deep_dive_colab.py` (tracked notebook-style script)
- `notebooks/pca_ablation_deep_dive_colab.ipynb` (local convenience file; ignored by git)

## What It Produces

Under:

`<input_root>/cross_pooling_analysis_notebook/`

it writes:

- `tables/`
  - `pca_sweep_long.csv` (+ parquet when available)
  - `least_k_ood_gain_per_pooling_layer.csv` (+ parquet when available)
  - ranking tables (`rank_by_k_then_ood_gain.csv`, etc.)
  - full-point ranking tables (`rank_all_points_by_ood_gain_ascending.csv`, `rank_all_points_by_ood_gain_desc.csv`)
  - `ood_gain_summary_by_pooling_k.csv`
  - `delta_consistency_summary.csv`
- `figures/`
  - per-pooling delta heatmaps
  - ID/OOD scatter and Pareto plots
  - best least-k OOD gain bar chart
- `phase2_prep/`
  - `pca_direction_catalog.csv` (+ parquet when available)
  - `pca_layer_inventory.csv`
  - `oracle_input_schema.json`
  - `oracle_input_manifest_template.csv`
- `meta/`
  - `run_manifest.json`
  - `status.json`
- `checkpoints/progress.json`
- `results.json`

## Colab Command

```bash
python scripts/analysis/pca_ablation_colab_deep_dive.py \
  --input_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying \
  --poolings mean,max,last
```

Optional flags:

- `--strict` fail if any requested pooling is missing
- `--force_rebuild` rerun even if prior status is complete
- `--output_root <path>` override default output directory
