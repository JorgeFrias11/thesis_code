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


# step 1 in W_Alpga test
Walpha = model_fit['Gamma'][1].values.T.dot(model_fit['Gamma'][1])

pval_alpha = model_int.BS_Walpha(ndraws=10, n_jobs=3, minTol=1e-6)

pval_beta = model_int.BS_Walpha(ndraws=10, n_jobs=3, minTol=1e-3)

L = model.L  # or self.L
T = model.X.shape[1]
N = len(np.unique(data.index.to_frame().values[:, 1]))


def _BS_Walpha_sub(model, n, d):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp

            # convert to dataframe to match input in _fit()
            X_b_df = pd.DataFrame(X_b, index=model.Chars,
                                  columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b_df, W=model.W,
                                        minTol=1e-4, maxIters=5000,
                                        gFac=model.gFac)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                            Observation discarded.")
            pass

    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l, minTol=1e-4):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)
    # Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma)
    Gamma_beta_l[l, :] = 0

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors[:, t]) + d_temp

            # convert to dataframe to match input in _fit()
            X_b_df = pd.DataFrame(X_b, index=model.Chars,
                                  columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b, W=model.W,
                                        minTol=1e-4, maxIters=5000,
                                        gFac=model.gFac)

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b


# TESTING KELLY CODE
X = data_x
y = data_y

non_nan_ind = ~np.any(np.isnan(X), axis=1)
X = X[non_nan_ind]
y = y[non_nan_ind]

Q = np.full((L, T), np.nan)
W = np.full((L, L, T), np.nan)
val_obs = np.full((T), np.nan)

indices = regr.indices
sum(indices[:, 1] == 1)
model_fit['indices']

for t in range(T):
    ixt = (indices[:, 1] == t)
    val_obs[t] = np.sum(ixt)
    # Q[:, t] = X[ixt, :].T.dot(y[ixt])/val_obs[t]
    # W[:, :, t] = X[ixt, :].T.dot(X[ixt, :])/val_obs[t]

    # Ensure consistent ordering
W_df = model.W.sort_index()                     # rows sorted by (date, char)
chars = list(W_df.columns)                      # char order
dates = W_df.index.get_level_values(0).unique()  # unique dates in order
L, T = len(chars), len(dates)

# Must be grouped by date, then by char within each date
assert (W_df.index.get_level_values(0).value_counts().eq(L)).all()
assert list(W_df.columns) == chars

W_arr = W_df.values.reshape(T, L, L).transpose(1, 2, 0)  # shape (L, L, T)


def X_as_array(self):
    """Convert Pruitt's X DataFrame to a (L, T) NumPy array like Kelly's Q."""
    chars = list(self.X.index)
    dates = list(self.X.columns)
    # Ordering match self.Chars and self.metad["dates"]
    X_df_sorted = self.X.loc[chars, dates]
    return X_df_sorted.values


def W_as_array(self):
    """Convert Pruitt's W DataFrame to a (L, L, T) NumPy array."""
    chars = list(self.X.index)
    dates = list(self.X.columns)

    # Reorder W to match the chars/dates ordering
    W_df_sorted = self.W.loc[
        pd.MultiIndex.from_product([dates, chars], names=['date', 'char']),
        chars
    ]

    L = len(chars)
    T = len(dates)

    # Reshape to (T, L, L) then transpose to (L, L, T)
    W_array = (
        W_df_sorted.values
        .reshape(T, L, L)
        .transpose(1, 2, 0)
    )
    return W_array


X_array = X_as_array(model)

W_array = W_as_array(model)


model.X

pval_alpha = model_int.BS_Walpha(ndraws=10, n_jobs=3)


# TODO: ADJUST THESE FUNCTIONS

# ADJUSTED MODEL

def _BS_Walpha_sub(model, n, d):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp

            # convert to dataframe to match input in _fit()
            X_b_df = pd.DataFrame(X_b, index=model.Chars,
                                  columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b_df, W=model.W,
                                        gFac=model.gFac)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


# TEST

dates = model.Dates.values
d = np.full((L, T), np.nan)

for t_i, t in enumerate(dates):
    d[:, t_i] = X_array[:, t_i] - W_array[:, :, t_i].dot(model.Gamma)\
        .dot(model.Factors[:, t_i])


L, T = model.L, model.T
X_b = np.full((L, T), np.nan)

for t in range(model.T):
    d_temp = np.random.standard_t(5)
    d_temp *= d[:, np.random.randint(0, high=T)]
    X_b[:, t] = W_array[:, :, t].dot(model.Gamma[:, :-1])\
        .dot(model.Factors[:-1, t]) + d_temp

X_b_df = pd.DataFrame(X_b, index=model.Chars,
                      columns=model.Dates)

###

# ORIGINAL MODEL


def _BS_Walpha_sub(model, n, d):
    L, T = self.L, self.T
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b, W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True,
                                             data_type="portfolio")
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)
    # Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma)
    Gamma_beta_l[l, :] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors[:, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b, W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True,
                                             data_type="portfolio")

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b
