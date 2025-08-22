import pandas as pd
import ipca_utils
import ipca_pruitt
import sys
from ipca import InstrumentedPCA
import numpy as np


import importlib
importlib.reload(ipca_pruitt)  # RELOAD PACKAGE

# data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")
# data = data.reset_index(drop=True)
data2 = data.copy().reorder_levels(['coinName', 'date'])

data_y = data2['ret_excess']
data_x = data2.drop("ret_excess", axis=1)


# K = int(sys.argv[1])

# print("Running IPCA with intercept: \n")
# ipca_utils.run_ipca_alpha(data_x, data_y, K=K, max_iter = 1000, iter_tol=1e-06,
#                           save=False, bootstrap_draws=10, n_jobs=-1)


# print("\nRunning IPCA without intercept: \n")
# ipca_utils.run_ipca_beta(data_x, data_y, K=K, max_iter = 1000, iter_tol=1e-06,
#                          save=False, bootstrap_draws=10, n_jobs=-1)

#########################################################################
# Kelly output
#########################################################################

print("Kelly's output: ")
K = 2
regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=5,
                       iter_tol=1e-4, n_jobs=-1)
regr = regr.fit(X=data_x, y=data_y, data_type="panel")
gamma, factors = regr.get_factors(label_ind=True)


regr.Q
print(regr)
print(gamma)

#########################################################################
# Pruitt output
#########################################################################

print("\nPruitt's output:")
K = 2
model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

model_int = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=True)


model_fit = model.fit(K=K,
                      OOS=False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=5,
                      minTol=1e-2,
                      maxIters=5)


model_fit_int = model_int.fit(K=K,
                              OOS=False,
                              gFac=None,
                              dispIters=True,
                              dispItersInt=5,
                              minTol=1e-2,
                              maxIters=5)


pval_alpha = model_int.BS_Walpha(ndraws=10, n_jobs=3, minTol=1e-6)
print("Alpha p-value:", pval_alpha)


model.X

n_chars = model.X.shape[0]
pvalues_beta = []
n_chars = 2
for i in range(n_chars):
    print(f"Testing significance of characteristic {i + 1} of {n_chars}...")
    pval = model.BS_Wbeta([i], ndraws=5, n_jobs=3, minTol=1e-2)
    pvalues_beta.append(pval)

pvalues_beta

pval_df = pd.DataFrame({
    'characteristic': model.X.index[0:2],
    'pval_beta': pvalues_beta
})
pval_df

L = model.L  # or self.L
T = model.X.shape[1]
N = len(np.unique(data.index.to_frame().values[:, 1]))
