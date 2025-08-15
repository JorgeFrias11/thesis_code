######################################################################################
# This code runs the IPCA models without a constant using Pruitts python implementation.
######################################################################################

import ipca_pruitt
import sys
import pandas as pd

data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds_100mill.pkl")
# data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")

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
                      maxIters=10000)

print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")

model_fit['Gamma']
model_fit['Factor']

# Run the bootstrap

n_chars = model.X.shape[0]
pvalues_beta = []
n_chars = 2
for i in range(n_chars):
    print(f"Testing significance of characteristic {i + 1} of {n_chars}...")
    pval = model.BS_Wbeta([i], ndraws=5, n_jobs=3, minTol=1e-2)
    pvalues_beta.append(pval)

pval_df = pd.DataFrame({
    'characteristic': model.X.index,
    'pval_beta': pvalues_beta
})

print()

output_dir = '/home/jfriasna/thesis_output/reg_beta/'

pval_df.to_csv(output_dir, index=False)
print(f"Bootstrap results saved to {output_dir}")
