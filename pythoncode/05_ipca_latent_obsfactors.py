################################################################################
#  This code runs the models with latent + observable factors (DL)
#  for different observable factor sets and latent factors (K=1..6).
################################################################################

import pyreadr
import pandas as pd
import ipca_pruitt
import pickle
from timeit import default_timer as timer

starttime = timer()

datapath = "/home/jfriasna/thesis_data/data/"

coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")

# Load observable factors
factors = pyreadr.read_r(datapath + 'factors.rds')[None]
factors['date'] = pd.to_datetime(factors['date'])
factors['date'] = factors['date'].dt.strftime('%Y%m%d').astype(int)
factors.set_index('date', inplace=True)

# Factor specifications to test
factor_specs = {
    "MKT": ['CMKT'],
    "MKT_3": ['CMKT', 'MOM', 'SMB'],
    "MKT_5": ['CMKT', 'MOM', 'SMB', 'LIQ', 'VOL']
}

mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

# Store results
results_summary = []
results_models = {}

for spec_name, fac_subset in factor_specs.items():
    gFac = factors[fac_subset].T

    for k in range(1, 7):  # latent factor K=1 to 6
        print(f"Running {spec_name} with K={k} latent factors")

        fit_dl = model.fit(
            K=k,
            OOS=False,
            gFac=gFac,
            dispIters=False,
            minTol=mintol,
            maxIters=500
        )

        # Save only rfits in summary
        results_summary.append({
            "spec": spec_name,
            "factors": fac_subset,
            "K": k,
            "R2_Total": fit_dl['rfits']['R2_Total'],
            "R2_Pred": fit_dl['rfits']['R2_Pred']
        })

        # Save full model fit
        results_models[f"{spec_name}_K{k}"] = fit_dl

# Convert to DataFrame
results_df = pd.DataFrame(results_summary)
print(results_df)

# Save both summary and models
output_file = f'/home/jfriasna/thesis_output/reg_latent_obsfactors.pkl'
with open(output_file, "wb") as f:
    pickle.dump({"summary": results_df, "models": results_models}, f)

print(f"\nResults saved in {output_file}")

time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")
