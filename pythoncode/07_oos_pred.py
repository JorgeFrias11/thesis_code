################################################################################
#  This code runs latent factor IPCA models (K=1,...,6) with out-of-sample
#  recursive estimation. Each SLURM array index corresponds to one K.
################################################################################

import sys
import pandas as pd
import ipca_pruitt
import pickle
from timeit import default_timer as timer

starttime = timer()

K = int(sys.argv[1])
print(f"Running OOS IPCA model with {K} latent factors")

datapath = "/home/jfriasna/thesis_data/data/"

coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")
coindata = coindata.sort_index(level=[0, 1])  # ensure MultiIndex is sorted

#  Fit OOS model
mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

# Recursive OOS from September 1, 2022. Training sample = 1553 days (~60% of dataset)
oos_fit = model.fit(
    K=K,
    OOS=True,
    OOS_window='recursive',
    OOS_window_specs=1553,
    dispIters=False,
    minTol=mintol,
    maxIters=1000
)

results = {
    "K": K,
    "rfits_R2_Total": oos_fit["rfits"]["R2_Total"],
    "rfits_R2_Pred": oos_fit["rfits"]["R2_Pred"],
    "xfits_R2_Total": oos_fit["xfits"]["R2_Total"],
    "xfits_R2_Pred": oos_fit["xfits"]["R2_Pred"]
}

results_df = pd.DataFrame([results])
print("\nOOS R2 Results")
print(results_df)

# -------------------------------
#  Save results
# -------------------------------
output_file = f"/home/jfriasna/thesis_output/oos_ipca_K{K}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results_df, f)

print(f"Saved OOS fit for K={K} to {output_file}")

time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")