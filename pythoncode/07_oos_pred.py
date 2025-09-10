################################################################################
#  This code runs latent factor IPCA models (K=1,...,6) with out-of-sample
#  recursive estimation. Each SLURM array index corresponds to one K.
################################################################################

import sys
import pandas as pd
import ipca_pruitt
import pickle
import os
from timeit import default_timer as timer

starttime = timer()

K = int(sys.argv[1])
print(f"Running OOS IPCA model with {K} latent factors")

datapath = os.path.expanduser("~/thesis_data/data/")
coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")
coindata = coindata.sort_index(level=[0, 1])  # ensure MultiIndex is sorted

#  Fit OOS model
mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

# Cutoff date: June 1, 2023 (~70% of dataset for training, ~1826 days)
cutoff = 20230601
train_idx = model.Dates < cutoff
OOS_window_specs = train_idx.sum()

# Recursive OOS from June 1, 2023. 
oos_fit = model.fit(
    K=K,
    OOS=True,
    OOS_window='recursive',
    OOS_window_specs=OOS_window_specs,
    dispIters=False,
    minTol=mintol,
    maxIters=1000
)

results = {
    "spec": "IPCA",
    "K": K,
    "rfits_R2_Total": oos_fit["rfits"]["R2_Total"],
    "rfits_R2_Pred": oos_fit["rfits"]["R2_Pred"],
    "xfits_R2_Total": oos_fit["xfits"]["R2_Total"],
    "xfits_R2_Pred": oos_fit["xfits"]["R2_Pred"]
}

results_df = pd.DataFrame([results])
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print("\nOOS R2 Results")
print(results_df.to_string(index=False))

time = round(timer() - starttime, 2)
print(f"Model completed in: {time} seconds")

# Save results
output_dir = os.path.expanduser("~/thesis_output/oos_pred/alpha_latent")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"oos_ipca_K{K}.csv")
# Save CSV 
results_df.to_csv(output_file, index=False)

print(f"Saved OOS fit for K={K} to {output_file}")
