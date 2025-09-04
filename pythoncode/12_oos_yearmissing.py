################################################################################
#  This script runs OOS IPCA models with latent factors (K=1..6).
#  Each SLURM array index corresponds to one K.
#  Stores results in both CSV and pickle.
################################################################################

import sys
import os
import pandas as pd
import ipca_pruitt
from timeit import default_timer as timer

starttime = timer()

# Job index: expects 1..6
job_id = int(sys.argv[1])
K = job_id
print(f"Running OOS IPCA model with {K} latent factors")

# Load data
datapath = os.path.expanduser("~/thesis_data/data/")
coindata = pd.read_pickle(datapath + "processed_daily_preds_1yearmissing.pkl")
coindata = coindata.sort_index(level=[0, 1])  # ensure MultiIndex sorted

# Fit OOS IPCA model
mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column="ret_excess", add_constant=False)

# Cutoff date: June 1, 2023 (~70% of dataset, ~1826 days)
cutoff = 20230601
train_idx = model.Dates < cutoff
OOS_window_specs = train_idx.sum()

oos_fit = model.fit(
    K=K,
    OOS=True,
    OOS_window="recursive",
    OOS_window_specs=OOS_window_specs,
    dispIters=False,
    minTol=mintol,
    maxIters=1000,
)

# Collect results
results = {
    "spec": "IPCA",
    "factors": f"K{K}",
    "K": K,
    "rfits_R2_Total": oos_fit["rfits"]["R2_Total"],
    "rfits_R2_Pred": oos_fit["rfits"]["R2_Pred"],
    "xfits_R2_Total": oos_fit["xfits"]["R2_Total"],
    "xfits_R2_Pred": oos_fit["xfits"]["R2_Pred"],
}
results_df = pd.DataFrame([results])
# Print full DataFrame
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print("\nOOS R2 Results")
print(results_df.to_string(index=False))

time = round(timer() - starttime, 2)
print(f"Model completed in: {time} seconds")

# Save outputs
output_dir = os.path.expanduser("~/thesis_output/yearmissing/oos_pred/latent")
os.makedirs(output_dir, exist_ok=True)

output_summary_file = os.path.join(output_dir, f"oos_ipca_K{K}_summary.csv")
# Save CSV 
results_df.to_csv(output_summary_file, index=False)

print(f"Saved summary table to {output_summary_file}")

