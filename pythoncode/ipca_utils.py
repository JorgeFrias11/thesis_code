###############################################################################
# This script includes the functions used in my python implementation of
# IPCA in cryptocurrency data
###############################################################################

from ipca import InstrumentedPCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import os
import math
import pickle
import pyreadr
import pandas as pd


def load_coindata(freq, data_path, cache_file=None,
                  daily_rds="daily_predictors.rds",
                  weekly_rds="weekly_predictors.rds",
                  ignore_cols=[],
                  pruitt = False,
                  save = False):
    # Loads the daily or weekly coindata
    data_path = Path(data_path)

    if freq == 'daily':
        date_idx = 'date'
    elif freq == 'weekly':
        date_idx = 'yyyyww'
    else:
        raise ValueError("freq must be 'daily' or 'weekly'")

    # If cache_file, try to load the data from cache
    if cache_file is not None:
        cache_path = data_path / cache_file
        if os.path.exists(cache_path):
            print("Data will be read...")
            # Load from cache
            with open(cache_path, "rb") as f:
                data_load = pickle.load(f)
            print("Loaded data from cache.")
            return data_load['data'], data_load['coin_id']

    # Load from .rds file
    print("Cache not used or not found. Loading data from predictors.rds...")
    if freq == 'daily':
        rds_path = data_path / daily_rds
        coins = pyreadr.read_r(rds_path)
        data = coins[None]
        data[date_idx] = pd.to_datetime(data[date_idx])
        # We need the index date column as int for ipca panel analysis to work
        data[date_idx] = data[date_idx].dt.strftime('%Y%m%d').astype(int)
    else:
        rds_path = data_path / weekly_rds
        coins = pyreadr.read_r(rds_path)
        data = coins[None]
        # We need the index date column as int for ipca panel analysis to work
        data[date_idx] = data[date_idx].astype(int)
        data = data.rename(columns={"ret_w": "ret"})

    # Assign unique coin IDs
    n = len(np.unique(data.coinName))
    coin_id = dict(zip(np.unique(data.coinName).tolist(), np.arange(1, n + 1)))
    data.coinName = data.coinName.apply(lambda x: coin_id[x])

    # Set index as the python package by Kelly et al.: level 0 Asset ID, level 1 date
    if pruitt == False:
        # Set index and transform columns
        data = data.set_index(['coinName', date_idx])
    elif pruitt:
        # Index as Pruitt's code, level 0 = date, level 1 = Asset ID
        data = data.set_index([date_idx, 'coinName'])

    # Ensure each pair is unique
    data = data[~data.index.duplicated(keep='first')]

    unneeded_cols = ['mcap', 'open', 'close', 'high', 'low', 'mktret', 'rf', 'cmkt']
    unneeded_cols = unneeded_cols + ignore_cols
    cols_to_remove = [col for col in unneeded_cols if col in data.columns]
    data.drop(columns=cols_to_remove, inplace=True)

    # Lag all characteristics so that ret_t corresponds to Z_t-1
    char_cols = data.columns.difference(['ret_excess'])
    data[char_cols] = data.groupby(level='coinName')[char_cols].shift(1)
    data.rename(columns={'ret': 'r2_1'}, inplace=True)

    # Drop rows with missing values
    data = data.dropna()

    # # Following the code of Kelly et al. (2019), keep dates with a min cross-section of 100 coins
    # min_coins = 100
    # obs_per_date = data.groupby(level=1).size()
    # # Only keep dates with enough coins
    # valid_dates = obs_per_date[obs_per_date >= min_coins].index
    # data = data[data.index.get_level_values(1).isin(valid_dates)]

    # Save preprocessed data to cache
    if save and cache_file is not None:
        with open(cache_path, "wb") as f:
            pickle.dump({'data': data, 'coin_id': coin_id}, f)
        print("Data processed and cached.")

    return data, coin_id


# def rank_scale(x):
#     # re-scale data as in Kelly et al (2019)
#     ranked = np.squeeze(x.rank(method="average"))
#     n = len(ranked)
#     return ((ranked - 1) / (n - 1)) - 0.5

# def rank_scale(x):
#     temp = x.dropna()
#     ranked = np.squeeze(temp.rank())
#     n = len(ranked)
#     x.loc[temp.index] = ranked - (1 + n) / 2
#     x.loc[x.isna()] = 0  # Missing values are replaced with 0 (mean), as in Kelly
#     x = x / (x[x > 0].sum() / len(x))
#     return x

# Scale as in Kelly et al. (2019)
def rank_scale(x):
    ranked = x.rank(method='average')
    normalized = (ranked - 1) / (ranked.max() - 1)
    shifted = normalized - 0.5
    return shifted

# Scanling on the (-1, 1) range
def rank_scale_one(x):
    ranked = x.rank(method='average')
    normalized = (ranked - 1) / (ranked.max() - 1)
    scaled = normalized * 2 - 1
    return scaled


def plot_gamma(gamma, save=False, file="gamma_plot.png"):
    # Code to produce a plot of the Gamma_alpha: contribution of
    # each characteristic to the factors

    # Code to group and plot, adding colors to each group
    group_dict = {
        "core": ["mcap", "prc", "r21_1", "r30_1", "r30_14"],
        "volrisk": ["beta", "ivol", "rvol", "retvol", "var", "delay"],
        "activity": ["lvol", "volscaled", "turn", "std_vol", "cv_vol"],
        "liquidity": ["bidask", "illiq", "sat", "dto", "volsh_30d"],
        "past_returns": ["r2_1", "r7_1", "r180_60", "dh90", "alpha"],
        "distribution": ["skew", "kurt", "maxret", "minret"],
        "news": ["nsi"]
    }
    group_order = list(group_dict)

    # Build maps
    group_map = {var: grp for grp, vars_ in group_dict.items() for var in vars_}
    group_rank = {grp: i for i, grp in enumerate(group_order)}
    default_rank = len(group_order)

    # Color map
    cmap = plt.get_cmap("tab10")

    vars_sorted = sorted(gamma.index, key=lambda v: (group_rank.get(group_map.get(v, ""), default_rank), v))
    gamma_sorted = gamma.loc[vars_sorted]

    unique_groups = list(dict.fromkeys(group_map.get(v, "other") for v in vars_sorted))
    color_map = {grp: cmap(i) for i, grp in enumerate(unique_groups)}

    n_factors = gamma.shape[1]
    n_rows = math.ceil(n_factors / 3)
    fig_height = 5 * n_rows

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, fig_height), sharey=True)
    axes = axes.flatten()  # flatten to 1D for indexing

    for i, factor in enumerate(gamma.columns):
        ax = axes[i]
        values = gamma_sorted[factor].values
        colors = [color_map.get(group_map.get(v, "other"), "lightgray")
                  for v in gamma_sorted.index]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(gamma_sorted.index, rotation=90, fontsize=7)
        ax.set_title(f'Factor {factor + 1}')
        ax.set_ylabel(r'$\Gamma_\beta$ coefficient')

    # Hide unused subplots
    for j in range(n_factors, len(axes)):
        fig.delaxes(axes[j])

    handles = [Patch(color=color_map[grp], label=grp) for grp in unique_groups]
    fig.legend(handles=handles, title="Group", loc='lower center',
               ncol=len(unique_groups), bbox_to_anchor=(0.5, -0.05))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25,
                        hspace=0.4, wspace=0.3)
    if save:
        plt.savefig(file, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# Main IPCA function with intercept (alpha)
def run_ipca_alpha(data_x, data_y, K, max_iter=250, iter_tol=1e-06, 
                   bootstrap_draws=10, n_jobs=-1, 
                   save=False, out_path="output"):

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------- Alpha (with intercept) -------------------
    alpha_model_path = out_path / f"ipca_{K}factor_alpha_result.pkl"
    print("Model with intercept...")

    if alpha_model_path.exists():
        print("Model already exists. Loading from file...")
        with open(alpha_model_path, 'rb') as f:
            results_alpha = pickle.load(f)
    else:
        print("Running model:")
        regr = InstrumentedPCA(n_factors=K, intercept=True, max_iter=max_iter,
                               iter_tol=iter_tol, n_jobs=n_jobs)
        regr = regr.fit(X=data_x, y=data_y, data_type="panel")
        gamma, factors = regr.get_factors(label_ind=True)
        print("IPCA done! Getting scores...\n")
        r2_total = regr.score(X=data_x, y=data_y)
        r2_pred = regr.score(X=data_x, y=data_y, mean_factor=True)
        # bootstrap
        pval_alpha = regr.BS_Walpha(ndraws=bootstrap_draws, n_jobs=n_jobs)

        results_alpha = {
            'model': regr,
            'pval': pval_alpha,
            'gamma': gamma,
            'factors': factors,
            'r2_total': r2_total,
            'r2_pred': r2_pred,
        }

        if save:
            with open(alpha_model_path, 'wb') as f:
                pickle.dump(results_alpha, f)
            print(f"Alpha model saved to {alpha_model_path}")

        print("Done!")

    print('Walpha p-value:', results_alpha['pval'])
    print("Total R2:", results_alpha['r2_total'])
    print("Predictive R2:", results_alpha['r2_pred'])


# Main IPCA without intercept. Estimations of beta
def run_ipca_beta(data_x, data_y, K, max_iter=250, iter_tol=1e-06, 
                  bootstrap_draws=10, n_jobs=-1, 
                  save=False, out_path="output"):

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    # ------------------- Beta (without intercept) -------------------
    print("Model without intercept...")
    beta_model_path = out_path / f"ipca_{K}factor_beta_result.pkl"

    if beta_model_path.exists():
        print("Model already exists. Loading from file...")
        with open(beta_model_path, 'rb') as f:
            results_beta = pickle.load(f)
        gamma = results_beta['Gamma']
        regr = results_beta['model']
    else:
        print("Running model:")
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=max_iter,
                               iter_tol=iter_tol, n_jobs=n_jobs)
        regr = regr.fit(X=data_x, y=data_y, data_type="panel")
        gamma, factors = regr.get_factors(label_ind=True)
        r2_total = regr.score(X=data_x, y=data_y)
        r2_pred = regr.score(X=data_x, y=data_y, mean_factor=True)

        results_beta = {
            'model': regr,
            'Gamma': gamma,
            'factors': factors,
            'r2_total': r2_total,
            'r2_pred': r2_pred,
        }

        if save:
            with open(beta_model_path, 'wb') as f:
                pickle.dump(results_beta, f)
            print(f"Beta model saved to {beta_model_path}")

        print("Done!")

    print("Total R2:", results_beta['r2_total'])
    print("Predictive R2:", results_beta['r2_pred'])

    # ------------------- Gamma Plot -------------------
    if save:
        figures_dir = out_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / f"gamma_{K}factor_plot.png"
        plot_gamma(gamma, save=True, file=plot_path)
        print(f"Gamma plot saved to {plot_path}")

    # -------------- Bootstrap for Individual Characteristics ----------------------
    print("Starting bootstrap...")
    pval_path = out_path / f"ipca_{K}factor_beta_pvals.csv"

    if pval_path.exists():
        print("Bootstrap p-values already exist. Loading from file...")
        pval_df = pd.read_csv(pval_path)
    else:
        n_chars = data_x.shape[1]
        pvalues_beta = []

        for i in range(n_chars):
            print(f"Testing significance of characteristic {i + 1} of {n_chars}...")
            pval = regr.BS_Wbeta([i], ndraws=bootstrap_draws, n_jobs=n_jobs)
            pvalues_beta.append(pval)

        pval_df = pd.DataFrame({
            'characteristic': data_x.columns,
            'pval_beta': pvalues_beta
        })

        if save:
            pval_df.to_csv(pval_path, index=False)
            print(f"Bootstrap results saved to {pval_path}")

    print("P-values for each characteristic:\n")
    print(pval_df)


# Function to run OOS predictions
def run_predictions(regr_path, pred_path, K, data_x, data_y, h):
    k_str = str(K)

    if os.path.exists(pred_path):
        with open(pred_path, "rb") as f:
            pred_results = pickle.load(f)
        print(f"Loaded prediction results for K={K} from cache.")
    else:
        # Load fitted model
        with open(regr_path, "rb") as f:
            regr_results = pickle.load(f)
        print(f"Loaded regr from {regr_path}.")

        regr = regr_results['model']

        # In-sample prediction
        Yhat_IS = regr.predict(X=data_x, mean_factor=True)
        r2_IS = r2_score(data_y, Yhat_IS)
        print(f"In-sample R^2 (K={K}):", r2_IS)

        # Out-of-sample prediction
        x_IS = data_x.iloc[:-h]
        x_OOS = data_x.iloc[-h:]
        y_IS = data_y.iloc[:-h]
        y_OOS = data_y.iloc[-h:]

        # Re-fit the model using only IS data
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=250)
        regr = regr.fit(X=x_IS, y=y_IS, data_type='panel')

        Ypred_OOS = regr.predictOOS(X=x_OOS, y=y_OOS, mean_factor=True)
        r2_OOS = r2_score(y_OOS, Ypred_OOS)
        print(f"Out-of-sample R^2 (K={K}):", r2_OOS)

        # Save predictions
        pred_results = {
            f'Yhat_{k_str}_IS': Yhat_IS,
            f'Ypred_{k_str}_OOS': Ypred_OOS,
            f'r2_{k_str}_IS': r2_IS,
            f'r2_{k_str}_OOS': r2_OOS
        }

        with open(pred_path, "wb") as f:
            pickle.dump(pred_results, f)
        print(f"Saved prediction results for K={K} to disk.")

    # Final print
    print(f"In-sample R2 (K={K}):", pred_results[f'r2_{k_str}_IS'])
    print(f"Out-of-sample R2 (K={K}):", pred_results[f'r2_{k_str}_OOS'])

    return pred_results
