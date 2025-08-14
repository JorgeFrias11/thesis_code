import pandas as pd
import ipca_utils
import ipca_pruitt
import sys 
from ipca import InstrumentedPCA

data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
#data = data.reset_index(drop=True)
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

print(regr)
print(gamma)

#########################################################################
# Pruitt output
#########################################################################

print("\nPruitt's output:")
model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)
model_fit = model.fit(K=K,
                      OOS = False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=5,
                      minTol=1e-4,
                      maxIters=25)

print(model_fit)
#print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")

model_fit['Gamma'][1].values

# step 1 in W_Alpga test
Walpha = model_fit['Gamma'][1].values.T.dot(model_fit['Gamma'][1])

L = model.L  # or self.L
T = model.X.shape[1]
N = len(np.unique(data.index.to_frame().values[:, 1]))
