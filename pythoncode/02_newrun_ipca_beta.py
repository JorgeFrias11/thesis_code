######################################################################################
# This code runs the IPCA models without a constant using Pruitts python implementation.
######################################################################################

import ipca_pruitt
import sys
import pandas as pd
import pickle
from timeit import default_timer as timer

data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
# data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")

starttime = timer()

print(f"Base model results:")

K = int(sys.argv[1])
mintol = 1e-6

model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

model_fit = model.fit(K=K,
                      OOS=False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=2500)

print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")

# Run the bootstrap
#print("\nStarting bootstrap for each characteristic: \n")
print("\nStarting bootstrap for prc and r14_0: \n")
n_chars = [3, 5]
pvalues_beta = []
# 1000 draws following Kelly et al. (2019)
for i in n_chars:
    print(f"Testing significance of characteristic {i + 1}: {model.X.index[i]}...")
    #pval = model.BS_Wbeta([i], ndraws=1000, n_jobs=-1, minTol=mintol)
    pval = model.BS_Wbeta([i], ndraws=1000, n_jobs=-1, minTol=mintol)
    pvalues_beta.append(pval)

pval_df = pd.DataFrame({
    'characteristic': model.X.index[[3, 5]],
    'pval_beta': pvalues_beta
})

print("p-values for each asset characteristic: \n\n", pval_df)

# Save results
output_file_pvalues = f'/home/jfriasna/thesis_output/reg_beta/extra/extra_compute_{K}_factors_pvals.csv'
pval_df.to_csv(output_file_pvalues, index=False)
print(f"\n\nBootstrap p-values saved in {output_file_pvalues}")

save_obj = {
    "model_fit": model_fit,
    "chars_pval": pval_df
}

output_file = f'/home/jfriasna/thesis_output/reg_beta/extra/extra_compute_{K}_factors_ipca.pkl'
with open(output_file, "wb") as f:
    pickle.dump(save_obj, f)
print(f"\nResults results saved in {output_file}")


time = round(timer() - starttime, 2)

print(f"\nCode finished after {time} seconds")
