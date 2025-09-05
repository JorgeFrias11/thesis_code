import os
import glob
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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
    "marketcap", "prc", "dh90",
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

pivot.rename(index=lambda x: x.replace("_0", "_1") if "_0" in x else x, inplace=True)

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
# plt.title("Significant Characteristics by Factor")

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

###############################################################################
# Plot characteristic contribution to factors for some K specification
###############################################################################

# 2 chars

K = 2

file_path = pval_dir + "/modelfit/2_factors_ipca_1_20.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Gamma DataFrame
gamma = data["model_fit"]["Gamma"]

# ---- Define groups (aligned with your tribble) ----
group_dict = {
    "Price & Size": ["marketcap", "prc", "dh90"],
    "Volatility & Risk": ["beta", "ivol", "rvol_7d", "rvol_30d", "retvol", "var", "es_5", "delay"],
    "Trading Activity": ["volume", "volume_7d", "volume_30d", "turn", "turn_7d", "std_turn", "std_vol", "cv_vol"],
    "Liquidity": ["bidask", "illiq", "sat", "dto", "volsh_15d", "volsh_30d"],
    "Past Returns": ["r2_1", "r7_1", "r14_1", "r21_1", "r30_1", "r30_14", "r180_60", "alpha"],
    "Distribution": ["skew_7d", "skew_30d", "kurt_7d", "kurt_30d",
                     "maxret_7d", "maxret_30d", "minret_7d", "minret_30d"]
}

# Order variables according to groups
ordered_chars = [c for grp in group_dict.values() for c in grp if c in gamma.index]
gamma_sorted = gamma.loc[ordered_chars]

# Assign colors by group
cmap = plt.get_cmap("tab10")
color_map = {grp: cmap(i) for i, grp in enumerate(group_dict)}
colors = []
for var in gamma_sorted.index:
    grp = next((g for g, vars_ in group_dict.items() if var in vars_), "Other")
    colors.append(color_map[grp])

n_factors = gamma.shape[1]
n_cols = 2   # with 2 factors, 2 cols works better
n_rows = 1

# Wider & shorter rectangle
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), sharey=True)
axes = axes.flatten()

# Plot each factor
for i, factor in enumerate(gamma.columns):
    ax = axes[i]
    ax.bar(range(len(gamma_sorted)), gamma_sorted.iloc[:, i].values, color=colors)
    ax.set_xticks(range(len(gamma_sorted)))
    ax.set_xticklabels(gamma_sorted.index, rotation=90, fontsize=10)
    ax.set_xlim(-0.5, len(gamma_sorted) - 0.5)
    # Y-axis ticks
    ax.tick_params(axis='y', labelsize=10)
    # Title above each subplot
    ax.set_title(f"Factor {i + 1}", fontsize=17, pad=10)
    ax.set_xlim(-0.5, len(gamma_sorted) - 0.5)


# Hide unused subplots if K < n_rows * n_cols
for j in range(n_factors, len(axes)):
    fig.delaxes(axes[j])

# Shared y-axis label
fig.text(0.04, 0.5, r"$\Gamma_\beta$ coefficient",
         va="center", rotation="vertical", fontsize=12)

# Legend centered across figure
handles = [Patch(color=color_map[grp], label=grp) for grp in group_dict.keys()]
fig.legend(
    handles=handles,
    title="Category",             # legend title
    title_fontsize=13,            # legend title font size
    loc="upper center",
    bbox_to_anchor=(0.5, -0.00),  # adjust closer/further below x-axis
    ncol=3,
    fontsize=12,
    frameon=False
)

# Adjust space so labels + legend fit nicely
plt.subplots_adjust(
    left=0.08, right=0.98, top=0.95, bottom=0.28,
    hspace=0.2, wspace=0.15
)

plt.show()


# K = 4

K = 4

file_path = pval_dir + "/modelfit/4_factors_ipca_1_20.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Gamma DataFrame
gamma = data["model_fit"]["Gamma"]

# ---- Define groups (aligned with your tribble) ----
group_dict = {
    "Price & Size": ["marketcap", "prc", "dh90"],
    "Volatility & Risk": ["beta", "ivol", "rvol_7d", "rvol_30d", "retvol", "var", "es_5", "delay"],
    "Trading Activity": ["volume", "volume_7d", "volume_30d", "turn", "turn_7d", "std_turn", "std_vol", "cv_vol"],
    "Liquidity": ["bidask", "illiq", "sat", "dto", "volsh_15d", "volsh_30d"],
    "Past Returns": ["r2_1", "r7_1", "r14_1", "r21_1", "r30_1", "r30_14", "r180_60", "alpha"],
    "Distribution": ["skew_7d", "skew_30d", "kurt_7d", "kurt_30d",
                     "maxret_7d", "maxret_30d", "minret_7d", "minret_30d"]
}

# Order variables according to groups
ordered_chars = [c for grp in group_dict.values() for c in grp if c in gamma.index]
gamma_sorted = gamma.loc[ordered_chars]

# Assign colors by group
cmap = plt.get_cmap("tab10")
color_map = {grp: cmap(i) for i, grp in enumerate(group_dict)}
colors = []
for var in gamma_sorted.index:
    grp = next((g for g, vars_ in group_dict.items() if var in vars_), "Other")
    colors.append(color_map[grp])


n_factors = gamma.shape[1]
n_cols = 2   # with 2 factors, 2 cols works better
n_rows = 2

# Wider & shorter rectangle
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), sharey=True)
axes = axes.flatten()

for i, factor in enumerate(gamma.columns):
    ax = axes[i]
    ax.bar(range(len(gamma_sorted)), gamma_sorted.iloc[:, i].values, color=colors)
    ax.set_xticks(range(len(gamma_sorted)))
    ax.set_xticklabels(gamma_sorted.index, rotation=90, fontsize=10)
    ax.set_xlim(-0.5, len(gamma_sorted) - 0.5)
    # Y-axis ticks
    ax.tick_params(axis='y', labelsize=10)
    # Title above each subplot
    ax.set_title(f"Factor {i + 1}", fontsize=17, pad=10)
    ax.set_xlim(-0.5, len(gamma_sorted) - 0.5)

# Hide unused subplots if K < n_rows*n_cols
for j in range(n_factors, len(axes)):
    fig.delaxes(axes[j])

# Shared y-axis label
fig.text(0.04, 0.5, r"$\Gamma_\beta$ coefficient", va="center", rotation="vertical", fontsize=12)

# Legend centered across figure (pushed further down)
# Legend centered across figure (with title "Category")
handles = [Patch(color=color_map[grp], label=grp) for grp in group_dict.keys()]
fig.legend(
    handles=handles,
    title="Category",             # legend title
    title_fontsize=13,            # legend title font size
    loc="upper center",
    bbox_to_anchor=(0.5, 0.16),  # adjust closer/further below x-axis
    ncol=3,
    fontsize=12,
    frameon=False
)

# Adjust space so labels + legend fit nicely
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.28,
                    hspace=0.75,
                    wspace=0.15)


plt.show()
