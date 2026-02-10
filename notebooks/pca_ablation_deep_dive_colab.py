# %% [markdown]
# # PCA Ablation Deep Dive (Colab)
#
# This notebook-style script aggregates pooling-wise PCA ablation artifacts (`mean`, `max`, `last`),
# builds least-k OOD-gain datasets, ranking tables, plots, and phase-2 PCA direction prep outputs.

# %%
# Mount Drive in Colab
from google.colab import drive

drive.mount("/content/drive")

# %%
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes")
INPUT_ROOT = REPO_ROOT / "results/pca_ablation/meta-llama_Llama-3.2-1B-Instruct/Deception-Roleplaying"
POOLINGS = "mean,max,last"
EFFICIENT_K_CAP = 5
TOP_N = 20
STRICT = False
FORCE_REBUILD = False

assert REPO_ROOT.exists(), f"Repo not found: {REPO_ROOT}"
os.chdir(REPO_ROOT)
print("Working dir:", Path.cwd())
print("Input root :", INPUT_ROOT)

# %%
cmd = [
    "python",
    "scripts/analysis/pca_ablation_colab_deep_dive.py",
    "--input_root",
    str(INPUT_ROOT),
    "--poolings",
    POOLINGS,
    "--efficient_k_cap",
    str(EFFICIENT_K_CAP),
    "--top_n",
    str(TOP_N),
]
if STRICT:
    cmd.append("--strict")
if FORCE_REBUILD:
    cmd.append("--force_rebuild")

print("Running command:")
print(" ".join(cmd))
subprocess.run(cmd, check=True)

# %%
import pandas as pd

OUT_ROOT = INPUT_ROOT / "cross_pooling_analysis_notebook"
TABLES = OUT_ROOT / "tables"
FIGS = OUT_ROOT / "figures"
PHASE2 = OUT_ROOT / "phase2_prep"

display(pd.read_csv(TABLES / "least_k_ood_gain_per_pooling_layer.csv").head(20))
display(pd.read_csv(TABLES / "best_k_per_pooling_layer_by_max_ood_delta.csv").head(20))
display(pd.read_csv(TABLES / "rank_by_k_then_ood_gain.csv").head(20))
display(pd.read_csv(TABLES / "rank_all_points_by_ood_gain_desc.csv").head(20))
display(pd.read_csv(TABLES / f"top{TOP_N}_ood_gain_lowk_leq{EFFICIENT_K_CAP}.csv").head(20))
display(pd.read_csv(TABLES / f"top{TOP_N}_ood_gain_lowk_leq{EFFICIENT_K_CAP}_by_pooling.csv").head(20))
display(pd.read_csv(TABLES / "pareto_k_vs_ood_delta_frontier_only.csv").head(20))
display(pd.read_csv(TABLES / "ood_gain_summary_by_pooling_k.csv").head(20))
display(pd.read_csv(PHASE2 / "pca_direction_catalog.csv").head(20))

# %%
from IPython.display import Image, display

plot_files = [
    FIGS / "delta_ood_heatmap_mean.png",
    FIGS / "delta_ood_heatmap_max.png",
    FIGS / "delta_ood_heatmap_last.png",
    FIGS / "id_vs_ood_delta_scatter.png",
    FIGS / "id_vs_ood_delta_pareto.png",
    FIGS / "pareto_k_vs_ood_delta.png",
    FIGS / "best_leastk_ood_gain_by_pooling.png",
]

for p in plot_files:
    if p.exists():
        print(p.name)
        display(Image(filename=str(p)))
    else:
        print("Missing:", p)

# %% [markdown]
# ## Phase-2 outputs prepared
# - `phase2_prep/pca_direction_catalog.csv`: PCA direction inventory across poolings/layers/PCs.
# - `phase2_prep/oracle_input_schema.json`: contract for future activation-oracle input table.
# - `phase2_prep/oracle_input_manifest_template.csv`: empty template to populate with per-sample vectors.
