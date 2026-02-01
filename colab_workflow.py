# %% [markdown]
# # SepsisRL Colab Workflow
#
# This notebook-style script is organized into **individual Colab cells** to:
# 1) compute descriptive statistics,
# 2) train each expert model,
# 3) evaluate + collect performance,
# 4) visualize results, and
# 5) monitor training progress (e.g., every 10k steps).

# %% [markdown]
# ## 0) Install runtime dependencies
# (Run once per Colab session.)

# %%
!pip install numpy pandas pyyaml jinja2 matplotlib

# %% [markdown]
# ## 1) Clone repo + set paths

# %%
# Update this to your repo URL (or skip clone if you've already uploaded the repo).
REPO_URL = "https://github.com/<ORG>/<REPO>.git"

!git clone {REPO_URL}
%cd /content/thework

import os

REPO_DIR = "/content/thework"
CONFIG_PATH = os.path.join(REPO_DIR, "final_config.yaml")

# Make repo importable
os.environ["PYTHONPATH"] = REPO_DIR

# %% [markdown]
# ## 2) (Optional) Mount Google Drive
# Use if you want datasets/outputs persisted.

# %%
# from google.colab import drive
# drive.mount("/content/drive")

# %% [markdown]
# ## 3) Descriptive statistics of the dataset
# - Summary stats for train/val/test
# - Mortality by SOFA bin (scripted)

# %%
import pandas as pd
from pathlib import Path

DATA_DIR = Path(REPO_DIR) / "data"
train_df = pd.read_csv(DATA_DIR / "rl_train_data_final_cont.csv")
val_df = pd.read_csv(DATA_DIR / "rl_val_data_final_cont.csv")
test_df = pd.read_csv(DATA_DIR / "rl_test_data_final_cont.csv")

# Basic schema overview
print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

# Descriptive statistics for key columns
key_cols = ["reward", "SOFA", "iv_input", "vaso_input", "mortality_90d"]
print(train_df[key_cols].describe(include="all"))

# %%
# Mortality summaries by SOFA bucket (train/val/test)
!PYTHONPATH=$REPO_DIR python explore_dataset_mortality.py --data-dir data

# %% [markdown]
# ## 4) Preprocessing (one-time)
# - Action bins
# - Cohort report

# %%
!PYTHONPATH=$REPO_DIR python preprocessing/action_bins.py --config final_config.yaml

# %%
!PYTHONPATH=$REPO_DIR python preprocessing/sepsis_cohort.py --config final_config.yaml --write_cohort_report

# %% [markdown]
# ## 5) Training with progress monitoring
#
# A helper to show progress **every 10k steps** by parsing stdout.

# %%
import re
import subprocess

def run_with_progress(cmd: str, every: int = 10000) -> None:
    step_pattern = re.compile(r"(?:Step|step)(?:\s+is)?\s+(\d+)")
    print(f"Running: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        match = step_pattern.search(line)
        if match:
            step = int(match.group(1))
            if step % every == 0:
                print(line.strip())
        elif "Saved Model" in line or "Average loss" in line:
            # keep key checkpoints visible
            print(line.strip())
    proc.wait()

# %% [markdown]
# ### 5.1) Physician (DDDQN variant)
# Logs appear every 500 steps by default; the helper surfaces each 10k step.
# (This trains the original physician DDDQN variant, not just the placeholder outputs.)

# %%
run_with_progress(
    "PYTHONPATH=/content/thework python expert/physician_v1.py --num_steps 100000",
    every=10000,
)

# %% [markdown]
# ### 5.2) CQL
# Use the training-specific script to set debug frequency.

# %%
run_with_progress(
    "PYTHONPATH=/content/thework python expert/cql_train.py --steps 80000 --debug-every 10000",
    every=10000,
)

# %% [markdown]
# ### 5.3) DDDQN
# Logs appear every 1000 steps; the helper surfaces each 10k step.

# %%
run_with_progress(
    "PYTHONPATH=/content/thework python expert/dddqn_train.py --steps 60000",
    every=10000,
)

# %% [markdown]
# ### 5.4) Model-based (MB, PPO)
# Logs appear every 2000 steps; the helper surfaces each 10k step.

# %%
run_with_progress(
    "PYTHONPATH=/content/thework python expert/mb_train.py --ppo-steps 40000",
    every=10000,
)

# %% [markdown]
# ### 5.5) MoE
# This script is lightweight and quick (no long training loop). Run as-is.

# %%
!PYTHONPATH=/content/thework python expert/moe.py --config final_config.yaml --train_or_load --eval_split val

# %% [markdown]
# ### 5.6) Hybrid
# Logs appear every 2000 steps; the helper surfaces each 10k step.

# %%
run_with_progress(
    "PYTHONPATH=/content/thework python expert/hybrid_train.py --ppo-steps 40000",
    every=10000,
)

# %% [markdown]
# ## 6) Policy outputs (val + test) for all policies
# This collects the per-policy action distributions and Q values for OPE.

# %%
!PYTHONPATH=/content/thework python expert/physician_v1.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/physician_v1.py --config final_config.yaml --train_or_load --eval_split test

!PYTHONPATH=/content/thework python expert/cql.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/cql.py --config final_config.yaml --train_or_load --eval_split test

!PYTHONPATH=/content/thework python expert/dddqn.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/dddqn.py --config final_config.yaml --train_or_load --eval_split test

!PYTHONPATH=/content/thework python expert/mb.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/mb.py --config final_config.yaml --train_or_load --eval_split test

!PYTHONPATH=/content/thework python expert/moe.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/moe.py --config final_config.yaml --train_or_load --eval_split test

!PYTHONPATH=/content/thework python expert/hybrid_train.py --config final_config.yaml --train_or_load --eval_split val
!PYTHONPATH=/content/thework python expert/hybrid_train.py --config final_config.yaml --train_or_load --eval_split test

# %% [markdown]
# ## 7) OPE metrics + safety checks

# %%
!PYTHONPATH=/content/thework python ope/run_ope.py --config final_config.yaml --split test
!PYTHONPATH=/content/thework python ope/safety_checks.py --config final_config.yaml --split test

# %% [markdown]
# ## 8) Action heatmaps + counts

# %%
!PYTHONPATH=/content/thework python results/action_heatmaps.py --config final_config.yaml --split test

# %% [markdown]
# ## 9) Visualize OPE + safety results

# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

results_dir = Path(REPO_DIR) / "results"

ope = pd.read_csv(results_dir / "ope_metrics_overall.csv")
safety = pd.read_csv(results_dir / "safety_plausibility.csv")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(ope["policy"], ope["phwdr"], label="PHWDR")
ax.bar(ope["policy"], ope["phwis"], alpha=0.6, label="PHWIS")
ax.set_title("OPE metrics (overall)")
ax.set_ylabel("Estimated return")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(safety["policy"], safety["jsd_vs_physician"], label="JSD vs physician")
ax.set_title("Safety/Plausibility")
ax.set_ylabel("JSD")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10) Visualize action heatmaps

# %%
from PIL import Image
import matplotlib.pyplot as plt

policy = "cql"  # change policy name as needed
heatmap_path = results_dir / f"action_heatmap_{policy}_overall.png"

img = Image.open(heatmap_path)
plt.figure(figsize=(4, 3))
plt.imshow(img)
plt.axis("off")
plt.title(f"{policy} action heatmap (overall)")
plt.show()
