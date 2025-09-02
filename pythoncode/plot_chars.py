import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# pvals CSVs dir
# pval_dir = "/home/jfriasna/thesis_output/new_reg_beta"

pval_dir = "/home/jori/Documents/QFIN/thesis_output/reg_beta"

# Collect CSVs for all Ks
files = sorted(glob.glob(os.path.join(pval_dir, "*_factors_pvals_*.csv")))

dfs = []
for f in files:
    # Parse K from filename (e.g., "2_factors_pvals_20_41.csv" -> 2)
    K = int(os.path.basename(f).split("_")[0])
    df = pd.read_csv(f)
    df["K"] = K
    dfs.append(df)

# Combine into one DataFrame
all_pvals = pd.concat(dfs, ignore_index=True)

# Pivot to wide format: rows=characteristics, cols=K
pivot = all_pvals.pivot_table(index="characteristic", columns="K", values="pval_beta", aggfunc="mean")

ordered_chars = [
    "mcap", "prc", "dh90",
    "beta", "ivol", "rvol_7d", "rvol_30d", "retvol", "var", "es_5", "delay",
    "volume", "volume_7d", "volume_30d", "turn", "turn_7d", "std_turn", "std_vol", "cv_vol",
    "bidask", "illiq", "sat", "dto", "volsh_15d", "volsh_30d",
    "r2_1", "r7_0", "r14_0", "r21_0", "r30_0", "r30_14", "r180_60", "alpha",
    "skew_7d", "skew_30d", "kurt_7d", "kurt_30d", "maxret_7d", "maxret_30d", "minret_7d", "minret_30d"
]

# Reindex pivot to match custom order (drop missing ones if necessary)
pivot = pivot.reindex(ordered_chars)

# Map p-values to significance scores
# 3 = p<0.01, 2 = p<0.05, 1 = p<0.10, 0 = not significant
sig_matrix = np.select(
    [pivot < 0.01, pivot < 0.05, pivot < 0.10],
    [3, 2, 1],
    default=0
)

# Custom colormap: white -> light green -> medium green -> dark green
cmap = ListedColormap(["white", "#b2df8a", "#33a02c", "#00441b"])

# Plot heatmap
plt.figure(figsize=(9, 12))
plt.imshow(sig_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3)

# Set ticks
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns)

# Labels
plt.xlabel("K")
plt.ylabel("Characteristics")
#plt.title("Significant Characteristics by Factor")

# Grid lines
nrows, ncols = sig_matrix.shape
for y in range(1, nrows):
    plt.hlines(y - 0.5, -0.5, ncols - 0.5, colors="lightgray", linestyles="dashed", linewidth=0.5)
for x in range(1, ncols):
    plt.vlines(x - 0.5, -0.5, nrows - 0.5, colors="lightgray", linestyles="dashed", linewidth=0.5)

# Colorbar
cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels([">=0.10", "<0.10", "<0.05", "<0.01"])

plt.tight_layout()
plt.show()
