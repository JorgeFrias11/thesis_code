################################################################################
#  This code runs the models with pre-specified observable factors, both with
#  and without characteristics instrumenting the dynamic loadings - Static
#  loadings (SL) or dynamic loadings (DL)
################################################################################

import pyreadr
import pandas as pd
import ipca_pruitt
import pickle
from timeit import default_timer as timer

starttime = timer()

datapath = "/home/jfriasna/thesis_data/data/"
# datapath = "/home/jori/Documents/QFIN/thesis_data/data/"

# Load coin data (returns and characteristics)
coindata = pd.read_pickle(datapath + "processed_daily_preds.pkl")

# Load observable factors
factors = pyreadr.read_r(datapath + 'factors.rds')[None]
factors['date'] = pd.to_datetime(factors['date'])
factors['date'] = factors['date'].dt.strftime('%Y%m%d').astype(int)
factors.set_index('date', inplace=True)

factor_list = ['CMKT', 'MOM', 'SMB', 'LIQ', 'VOL']

mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

# Store results
results_summary = []
results_models = {}

for k in range(1, len(factor_list) + 1):
    # select first k factors
    fac_subset = factor_list[:k]
    gFac = factors[fac_subset].T

    print(f"Running model with {k} factors: {fac_subset}")

    # Observable factors with STATIC loadings (K=0)
    fit_sl = model.fit(
        K=0,
        OOS=False,
        gFac=gFac,
        dispIters=True,
        dispItersInt=25,
        minTol=mintol,
        maxIters=1000
    )
    results_summary.append({
        "model": f"SL_{k}",
        "factors": fac_subset,
        "rt_R2_Total": fit_sl['rfits']['R2_Total'],
        "rt_R2_Pred": fit_sl['rfits']['R2_Pred'],
        "xt_R2_Total": fit_sl['xfits']['R2_Total'],
        "xt_R2_Pred": fit_sl['xfits']['R2_Pred']
    })
    results_models[f"SL_{k}"] = fit_sl

    # Observable factors with DYNAMIC loadings (K=k)
    fit_dl = model.fit(
        K=k,
        OOS=False,
        gFac=gFac,
        dispIters=False,
        minTol=mintol,
        maxIters=100
    )
    results_summary.append({
        "model": f"DL_{k}",
        "factors": fac_subset,
        "rt_R2_Total": fit_dl['rfits']['R2_Total'],
        "rt_R2_Pred": fit_dl['rfits']['R2_Pred'],
        "xt_R2_Total": fit_dl['xfits']['R2_Total'],
        "xt_R2_Pred": fit_dl['xfits']['R2_Pred']
    })
    results_models[f"DL_{k}"] = fit_dl

results_df = pd.DataFrame(results_summary)
print(results_df)

output_file = f'/home/jfriasna/thesis_output/reg_obsfactors.pkl'
with open(output_file, "wb") as f:
    pickle.dump({"summary": results_df, "models": results_models}, f)

print(f"\nResults saved in {output_file}")

time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")
