################################################################################
#  This script runs OOS IPCA models with observable factors only (SL)
#  and observable+latent factors (DL) for M=1..5 observable factors.
#  Stores all results in one DataFrame and saves to pickle.
#  Use with a job array: each array index runs a different spec.
################################################################################

import sys
import os
import pandas as pd
import pyreadr
import ipca_pruitt
import pickle
from timeit import default_timer as timer

starttime = timer()

# Factor model specifications
factor_jobs = [
    ("MKT", ["CMKT"], 0),
    ("MKT", ["CMKT"], 1),
    ("2_factors", ["CMKT", "MOM"], 0),
    ("2_factors", ["CMKT", "MOM"], 2),
    ("3_factors", ["CMKT", "MOM", "SMB"], 0),
    ("3_factors", ["CMKT", "MOM", "SMB"], 3),
    ("4_factors", ["CMKT", "MOM", "SMB", "LIQ"], 0),
    ("4_factors", ["CMKT", "MOM", "SMB", "LIQ"], 4),
    ("5_factors", ["CMKT", "MOM", "SMB", "LIQ", "VOL"], 0),
    ("5_factors", ["CMKT", "MOM", "SMB", "LIQ", "VOL"], 5),
]

# job from array index from 1 to N
job_id = int(sys.argv[1])
spec_name, fac_subset, K = factor_jobs[job_id - 1]

print(f"Running OOS IPCA: {spec_name}, factors={fac_subset}, K={K}")

# Load data
datapath = "~/thesis_data/data/"

coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")
coindata = coindata.sort_index(level=[0, 1])

factors = pyreadr.read_r(datapath + "factors.rds")[None]
factors['date'] = pd.to_datetime(factors['date'])
factors['date'] = factors['date'].dt.strftime('%Y%m%d').astype(int)
factors.set_index('date', inplace=True)

# Subset and transpose for gFac
gFac = factors[fac_subset].T

# IPCA model - Out-of-sample
mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

#cutoff = 20220901
cutoff = 20230601   # nearly 70% of the dataset, 1826 days
train_idx = model.Dates < cutoff
OOS_window_specs = train_idx.sum()

# Train data before September 1, 2022 (nearly 60% of the dataset)
oos_fit = model.fit(
    K=K,
    OOS=True,
    OOS_window='recursive',
    OOS_window_specs=OOS_window_specs,
    gFac=gFac,
    dispIters=False,
    minTol=mintol,
    maxIters=1000
)

results = {
    "spec": spec_name,
    "factors": ",".join(fac_subset),
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
output_dir = os.path.expanduser("~/thesis_output/oos_pred/obsfactors")
# create dir
os.makedirs(output_dir, exist_ok=True)

# File paths
output_summary_file = os.path.join(output_dir, f"OOS_{spec_name}_K{K}_summary.csv")
output_file = os.path.join(output_dir, f"OOS_{spec_name}_K{K}.pkl")

# Save df and model fit
results_df.to_csv(output_summary_file, index=False)
print(f"Saved summary table to {output_summary_file}")
