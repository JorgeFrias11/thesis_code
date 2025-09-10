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

