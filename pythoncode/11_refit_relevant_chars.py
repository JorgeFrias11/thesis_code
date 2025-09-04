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

# Keep only significant characteristics (at 1% level) + return column
sig_chars = [
    "beta", "retvol", "r7_0", "r14_0", "r30_0", "r30_14",
    "alpha", "skew_7d", "maxret_7d", "minret_7d"
]

keep_cols = ["ret_excess"] + sig_chars
coindata = coindata[keep_cols]

mintol = 1e-6
model = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=False)
model_alpha = ipca_pruitt.ipca(RZ=coindata, return_column='ret_excess', add_constant=True)

# Store results
results_summary = []
results_models = {}

#################################################################################
#  Fit general IPCA models with selected characteristics only
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

# Results dataframe
results_df = pd.DataFrame(results_summary)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(results_df.to_string(index=False))

# Save CSV
csv_file = '/home/jfriasna/thesis_output/11_refit_relevant_chars.csv'
results_df.to_csv(csv_file, index=False)
print(f"Results summary saved in {csv_file}")

time = round(timer() - starttime, 2)
print(f"Total runtime: {time} seconds")
