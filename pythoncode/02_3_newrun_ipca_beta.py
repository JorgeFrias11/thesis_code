######################################################################################
# ONLY CHANGE range for loop, index, and output files. lines: 39, 49, 56, 65
# This code runs the IPCA models without a constant using Pruitts python implementation.
######################################################################################

import ipca_pruitt
import sys
import os
import pandas as pd
import pickle
from timeit import default_timer as timer

# SLURM Args: K start_idx end_idx
K = int(sys.argv[1])
start_idx = int(sys.argv[2])
end_idx = int(sys.argv[3])


# data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
# # data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")

# starttime = timer()

# print(f"Base model results:")

# mintol = 1e-6
# model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

# model_fit = model.fit(K=K,
#                       OOS=False,
#                       gFac=None,
#                       dispIters=True,
#                       dispItersInt=25,
#                       minTol=mintol,
#                       maxIters=2500)

# print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
# print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")


data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
mintol = 1e-6
starttime = timer()

# Output paths
output_file = f'/home/jfriasna/thesis_output/new_reg_beta/{K}_factors_ipca.pkl'

# ----------------------------------------------------
# Fit or load model
# ----------------------------------------------------
if os.path.exists(output_file):
    print(f"Model for K={K} already exists at {output_file}. Loading...")
    with open(output_file, "rb") as f:
        saved_obj = pickle.load(f)
        model = saved_obj["model"]
        model_fit = saved_obj["model_fit"]

else:
    print(f"Fitting new model for K={K}...")
    model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

    model_fit = model.fit(K=K,
                          OOS=False,
                          gFac=None,
                          dispIters=True,
                          dispItersInt=25,
                          minTol=mintol,
                          maxIters=2500)

    save_obj = {
        "model": model,
        "model_fit": model_fit
    }
    with open(output_file, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"Model and model_fit saved in {output_file}")

# Print R²s
print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")

# ----------------------------------------------------
# Run bootstrap on selected characteristics
# ----------------------------------------------------
print("\nStarting bootstrap for each characteristic: \n")
n_chars = range(start_idx, end_idx)
pvalues_beta = []
# 1000 draws following Kelly et al. (2019)
for i in n_chars:
    print(f"Testing significance of characteristic {i + 1}: {model.X.index[i]}...")
    #pval = model.BS_Wbeta([i], ndraws=1000, n_jobs=-1, minTol=mintol)
    pval = model.BS_Wbeta([i], ndraws=1000, n_jobs=-1, minTol=mintol)
    pvalues_beta.append(pval)

pval_df = pd.DataFrame({
    'characteristic': model.X.index[start_idx: end_idx],
    'pval_beta': pvalues_beta
})

print("p-values for each asset characteristic: \n\n", pval_df)

# Save results
output_file_pvalues = f'/home/jfriasna/thesis_output/new_reg_beta/{K}_factors_pvals_{start_idx}_{end_idx}.csv'
pval_df.to_csv(output_file_pvalues, index=False)
print(f"\n\nBootstrap p-values saved in {output_file_pvalues}")

# save_obj = {
#     "model_fit": model_fit,
#     "chars_pval": pval_df
# }

# output_file = f'/home/jfriasna/thesis_output/new_reg_beta/{K}_factors_ipca_{start_idx}_{end_idx}.pkl'
# with open(output_file, "wb") as f:
#     pickle.dump(save_obj, f)
# print(f"\nResults results saved in {output_file}")


time = round(timer() - starttime, 2)

print(f"\nCode finished after {time} seconds")
