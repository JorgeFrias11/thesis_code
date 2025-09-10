################################################################################
#  Delta-bootstrap tests for each observable factor individually with K=1..6
################################################################################

import pyreadr
import pandas as pd
import ipca_pruitt_2
import pickle
import sys
import os
from timeit import default_timer as timer
from datetime import datetime

starttime = timer()

datapath = "/home/jfriasna/thesis_data/data/"
outpath = "/home/jfriasna/thesis_output/"

# Load coin data (returns and characteristics)
coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")

# Load observable factors
factors = pyreadr.read_r(datapath + 'factors.rds')[None]
factors['date'] = pd.to_datetime(factors['date'])
factors['date'] = factors['date'].dt.strftime('%Y%m%d').astype(int)
factors.set_index('date', inplace=True)

# Pre-specified observable factors
factor_list = ['CMKT', 'MOM', 'SMB', 'LIQ', 'VOL']

task_id = int(sys.argv[1]) - 1  
obs_factor = factor_list[task_id]

print(f"Running bootstrap delta test for factor: {obs_factor}")

# Pre-specified observable factor matrix (T × 1)
gFac = factors[[obs_factor]].T

# Init model
mintol = 1e-6
model = ipca_pruitt_2.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

results_summary = []

for k in range(1, 3):  # latent factors 1..6
    print(f"\nFactor={obs_factor}, K={k} latent factors")
    fit = model.fit(
        K=k,
        OOS=False,
        gFac=gFac,
        dispIters=True,
        dispItersInt=25,
        minTol=mintol,
        maxIters=100
    )

    # Run delta bootstrap
    pval = model.BS_Wdelta(ndraws=1000, n_jobs=-1, minTol=mintol, maxIters=1000)
    print(f"p-value for obs={obs_factor}, K={k}: {pval:.4f}")

    # Store results
    results_summary.append({
        "obs_factor": obs_factor,
        "K": k,
        "R2_Total": fit['rfits']['R2_Total'],
        "R2_Pred": fit['rfits']['R2_Pred'],
        "pval_delta": pval
    })

# Save results
results_df = pd.DataFrame(results_summary)
print("\nFinal results:")
print(results_df)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outfile = os.path.join(outpath, f"delta_bootstrap_{obs_factor}_{timestamp}.pkl")
with open(outfile, "wb") as f:
    pickle.dump(results_df, f)

print(f"\nResults saved to {outfile}")
time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")
