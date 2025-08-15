######################################################################################
# This code runs the IPCA models with a constant using Pruitts python implementation
######################################################################################

import ipca_pruitt
import sys
import pandas as pd
import pickle

data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds_100mill.pkl")
# data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")

print(f"Base model results:")

K = int(sys.argv[1])
mintol = 1e-6

model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=True)

model_fit = model.fit(K=K,
                      OOS=False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=10000)

print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")

print("\nStarting bootstrap for alpha test: \n")
pval_alpha = model.BS_Walpha(ndraws=5000, n_jobs=-1, backend='loky', minTol=mintol)
print("\nAlpha p-value:", pval_alpha)

save_obj = {
    "model_fit": model_fit,
    "alpha_pval": pval_alpha
}

# Save file
output_file = f'/home/jfriasna/thesis_output/reg_alpha/{K}_factors_ipca.pkl'
with open(output_file, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\nResults saved in {output_file}")
