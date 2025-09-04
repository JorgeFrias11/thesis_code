################################################################################
#  This code fits all the model specifications: general IPCA and models with 
#  pre-specified observable factors, both with and without characteristics 
#  instrumenting the dynamic loadings - Static loadings (SL) or dynamic loadings (DL)
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
coindata = pd.read_pickle(datapath + "processed_daily_preds_1yearmissing.pkl")

# Load observable factors
factors = pyreadr.read_r(datapath + 'factors.rds')[None]
factors['date'] = pd.to_datetime(factors['date'])
factors['date'] = factors['date'].dt.strftime('%Y%m%d').astype(int)
factors.set_index('date', inplace=True)

factor_list = ['CMKT', 'MOM', 'SMB', 'LIQ', 'VOL']

# MATCH dates
coindata_dates = coindata.index.get_level_values("date").unique()
factors_aligned = factors.loc[factors.index.intersection(coindata_dates)]

mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)

model_alpha = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=True)

# Store results
results_summary = []
results_models = {}

#################################################################################
#  Fit general IPCA models
#################################################################################

for K in range(1, 7):
    print(f"Running gamma=0 model with {K} factors")
    fit_beta = model.fit(K=K,
                        OOS=False,
                        gFac=None,
                        dispIters=True,
                        dispItersInt=25,
                        minTol=mintol,
                        maxIters=1000)
    
    results_summary.append({
        "model": f"beta_{K}",
        "factors": K,
        "rt_R2_Total": fit_beta['rfits']['R2_Total'],
        "rt_R2_Pred": fit_beta['rfits']['R2_Pred'],
        "xt_R2_Total": fit_beta['xfits']['R2_Total'],
        "xt_R2_Pred": fit_beta['xfits']['R2_Pred']
    })
    results_models[f"beta_{K}"] = fit_beta

    print(f"Running gamma!=0 model with {K} factors")
    fit_alpha = model_alpha.fit(K=K,
                        OOS=False,
                        gFac=None,
                        dispIters=True,
                        dispItersInt=25,
                        minTol=mintol,
                        maxIters=1000)
    
    results_summary.append({
        "model": f"alpha_{K}",
        "factors": K,
        "rt_R2_Total": fit_alpha['rfits']['R2_Total'],
        "rt_R2_Pred": fit_alpha['rfits']['R2_Pred'],
        "xt_R2_Total": fit_alpha['xfits']['R2_Total'],
        "xt_R2_Pred": fit_alpha['xfits']['R2_Pred']
    })
    results_models[f"alpha_{K}"] = fit_alpha

#################################################################################
#  Fit models with observable factors
#################################################################################

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
        maxIters=1000
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
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(results_df.to_string(index=False))
print(results_df)

csv_file = '/home/jfriasna/thesis_output/yearmissing/reg_obsfactors_summary.csv'
results_df.to_csv(csv_file, index=False)
print(f"Results summary saved in {csv_file}")

output_file = f'/home/jfriasna/thesis_output/yearmissing/reg_obsfactors.pkl'
with open(output_file, "wb") as f:
    pickle.dump(results_models, f)

print(f"\nResults saved in {output_file}")

time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")
