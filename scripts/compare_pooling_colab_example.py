"""
Google Colab example for comparing pooling strategies layer-by-layer.

This script shows how to use compare_pooling_layerwise.py in Google Colab
with probes stored in Google Drive.

Setup in Colab:
    1. Mount Google Drive
    2. Clone repo and install dependencies
    3. Run this script

Example directory structure in Google Drive:
    /content/drive/MyDrive/probes/
    ├── mean/
    │   ├── probe_layer_0.pt
    │   ├── probe_layer_1.pt
    │   ├── ...
    │   └── layer_results.json
    ├── max/
    │   ├── probe_layer_0.pt
    │   └── layer_results.json
    ├── last/
    │   └── layer_results.json
    └── attn/
        └── layer_results.json
"""

# ============================================================================
# STEP 1: Mount Google Drive (run in Colab)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# STEP 2: Setup paths
# ============================================================================
import os
import sys

# Navigate to repo
repo_path = "/content/Efficacy-of-ensemble-of-attention-probes"
os.chdir(repo_path)

# Add to path
sys.path.append(os.path.join(repo_path, 'actprobe', 'src'))

# ============================================================================
# STEP 3: Define your paths
# ============================================================================

# Where your probes are stored in Google Drive
PROBES_BASE_DIR = "/content/drive/MyDrive/probes"

# Model and dataset info
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DATASET = "Deception-Roleplaying"

# Output directory (can be in Drive to save permanently)
OUTPUT_DIR = "/content/drive/MyDrive/results/comparisons"

# ============================================================================
# OPTION A: Auto-discovery (if probes are organized by model/dataset/pooling)
# ============================================================================
print("Running comparison with auto-discovery...")
!python scripts/compare_pooling_layerwise.py \
    --probes_base_dir {PROBES_BASE_DIR} \
    --model {MODEL} \
    --dataset {DATASET} \
    --output_dir {OUTPUT_DIR}

# ============================================================================
# OPTION B: Manual paths (if probes are just in pooling directories)
# ============================================================================
print("\nAlternatively, with manual paths...")
!python scripts/compare_pooling_layerwise.py \
    --mean_dir {PROBES_BASE_DIR}/mean \
    --max_dir {PROBES_BASE_DIR}/max \
    --last_dir {PROBES_BASE_DIR}/last \
    --attn_dir {PROBES_BASE_DIR}/attn \
    --output_dir {OUTPUT_DIR}

# ============================================================================
# View the results
# ============================================================================
print("\n" + "="*80)
print("Results saved to:", OUTPUT_DIR)
print("="*80)

# Display the summary
summary_path = f"{OUTPUT_DIR}/pooling_comparison_summary.txt"
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        print(f.read())

# Display the plot
from IPython.display import Image, display
plot_path = f"{OUTPUT_DIR}/layerwise_pooling_comparison.png"
if os.path.exists(plot_path):
    display(Image(plot_path))

# ============================================================================
# QUICK ONE-LINER USAGE
# ============================================================================
"""
# Just run this in a Colab cell after mounting Drive:

!cd /content/Efficacy-of-ensemble-of-attention-probes && \
python scripts/compare_pooling_layerwise.py \
    --mean_dir /content/drive/MyDrive/probes/mean \
    --max_dir /content/drive/MyDrive/probes/max \
    --last_dir /content/drive/MyDrive/probes/last \
    --attn_dir /content/drive/MyDrive/probes/attn \
    --output_dir /content/drive/MyDrive/results/comparisons

# Then view the plot:
from IPython.display import Image, display
display(Image('/content/drive/MyDrive/results/comparisons/layerwise_pooling_comparison.png'))
"""
