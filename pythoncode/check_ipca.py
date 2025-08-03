import time
import sys
from pathlib import Path
import ipca
import numpy as np
import pandas as pd
from itertools import product
# Add the parent of the parent directory to the Python path

#current_dir = Path().resolve()
#sys.path.append(str(current_dir.parents[1]))

import ipca_utils

#import inspect
#print(inspect.getsource(ipca_utils.load_coindata))
#

datapath = "/home/jfriasna/thesis/data/new_data360"
data = ipca_utils.load_coindata('daily', datapath, cache_file = '02_cache_new360_daily.pkl',
                                daily_rds='new_daily_360_predictors.rds',
                                ignore_cols=['r180_60', 'logvol', 'volscaled'],
                                save = True)


########################### TEST CODE

### Filter marketcap
# Get the last observation for each coin
last_obs = data.groupby(level='coinName').tail(1)
# Filter those with marketcap > 10 million (more restrictive)
low_mcap_coins = last_obs[last_obs['marketcap'] > 10_000_000]
# Get the coin names (index level 0)
coin_ids = low_mcap_coins.index.get_level_values('coinName').unique()
# Filter the original data
filtered_data = data.loc[coin_ids]
# Number of coins: 242
len(filtered_data.index.get_level_values('coinName').unique())

# Remove coins with less than 365 observations
counts = filtered_data.groupby(level='coinName').size()
coins_with_365 = counts[counts > 365].index
# Number of coins: 230, 12 are removed
len(coins_with_365)
# Step 3: Filter the original DataFrame
filtered_data = filtered_data.loc[coins_with_365]

data = filtered_data
############################################################



data_y = data['ret_excess']
data_x = data.drop("ret_excess", axis=1)

# Rescale columns as per Kelly. Omit news factors!
news_cols = ['GPRD', 'GPRD_MA7', 'GPRD_MA30', 'nsi']
to_transform = data_x.drop(columns=news_cols)
to_keep = data_x[news_cols]

# Apply the transform only to the other columns
transformed = to_transform.groupby(level='date').transform(ipca_utils.rank_scale)

# Recombine columns
data_x = pd.concat([transformed, to_keep], axis=1)


# level for daily data is "date", for weekly change to "yyyyww"
#data_x = data_x.groupby(level='date').transform(ipca_utils.rank_scale)  # only scale X


#corr_matrix = data_x.corr()

# Stack into long format (row-wise pairs)
#corr_pairs = corr_matrix.stack()
# Filter: keep only values > 0.85 and < 1 (to exclude self-correlation)
#high_corr = corr_pairs[(abs(corr_pairs) > 0.85) & (abs(corr_pairs) < 1)]

# remove variable with high correlation
data_x = data_x.drop(["illiq", 'volume_30d', "std_vol", "maxdprc", "prcvol", "stdprcvol", "beta2",
                      "std_turn", "GPRD", "GPRD_MA7", "GPRD_MA30", "nsi"], axis=1)

print(data_x.columns)
print(len(data_x.columns))   # 32 characteristics

# Regressions

start_time = time.time()

K = int(sys.argv[1])


# Define hyperparameter grid
alpha_grid = [1, 5, 10, 25]
l1_ratio_grid = [0.0, 0.5, 1.0]  # ridge, elastic net, lasso

# Collect results here
results = []

for alpha, l1_ratio in product(alpha_grid, l1_ratio_grid):
    print(f"\nFitting IPCA with K={K}, alpha={alpha}, l1_ratio={l1_ratio}")

    regr = ipca.InstrumentedPCA(
        n_factors=K,
        intercept=False,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=2500,
        iter_tol=1e-6,
        n_jobs=-1,
        backend='loky'
    )

    # Fit model
    regr = regr.fit(X=data_x, y=data_y, data_type="panel")

    # Compute scores
    R2_total = regr.score(X=data_x, y=data_y)
    R2_pred = regr.score(X=data_x, y=data_y, mean_factor=True)

    print(f"alpha={alpha}, l1_ratio={l1_ratio} : R2_total={R2_total:.4f}, R2_pred={R2_pred:.4f}")

    # Append result
    results.append({
        "K": K,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "R2_total": R2_total,
        "R2_pred": R2_pred 
    })

# Convert to DataFrame and display sorted table
results_df = pd.DataFrame(results).sort_values(by="R2_pred", ascending=False)

print("\n=== IPCA Grid Search Results ===")
print(results_df.to_string(index=False))



