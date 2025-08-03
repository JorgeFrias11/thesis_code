import sys
from pathlib import Path
import pandas as pd

# Add the parent of the parent directory to the Python path

current_dir = Path().resolve()
sys.path.append(str(current_dir.parents[1]))

import new_ipca_utils
import ipca_pruitt

datapath = "/home/jfriasna/thesis/data/newer_data360"
data = new_ipca_utils.load_coindata('daily', datapath,
                                cache_file = 'cache_pruitt_360_daily_all.pkl',
                                daily_rds='daily_360_predictors.rds',
                                #ignore_cols=['r180_60', 'logvol'],
                                pruitt=True,
                                save = True)

print(data.columns)

### TEST CODE 

## Filter marketcap
# Get the last observation for each coin
last_obs = data.groupby(level='coinName').tail(1)
# Filter those with low marketcap 
lower = 1_000_000
low_mcap_coins = last_obs[last_obs['marketcap'] >= lower]
# Get the coin names (index level 0)
coin_ids = low_mcap_coins.index.get_level_values('coinName').unique()
# Filter the original data
idx = pd.IndexSlice
filtered_data = data.loc[idx[:, coin_ids], :]
# Number of coins: 684
len(filtered_data.index.get_level_values('coinName').unique())


### Keep these comments
# Remove coins with less than 365 observations
counts = filtered_data.groupby(level='coinName').size()
coins_with_365 = counts[counts > 365].index
print('Number of coins:', len(coins_with_365))

# Filter the original DataFrame
idx = pd.IndexSlice
filtered_data = filtered_data.loc[idx[:, coins_with_365], :]

data = filtered_data

# Following the code of Kelly et al. (2019), keep dates with a min cross-section of 100 coins

print("n rows before min cross-section:", data.shape)

min_coins = 100
obs_per_date = data.groupby(level=0).size() # for Pruitt code, date is level 0
# Only keep dates with enough coins
valid_dates = obs_per_date[obs_per_date >= min_coins].index
data = data[data.index.get_level_values(0).isin(valid_dates)]

print("nrows after removing low cross-section dates:", data.shape)

##################################################################################
## TEST 2: Shorten time period, and keep coins with at least 75% of total obs
## Period: 2020-01-01 to 2025-04-30
##################################################################################

# Step 1: Convert the integer date level to datetime
#date_ints = data.index.get_level_values('date')
#date_dt = pd.to_datetime(date_ints.astype(str), format="%Y%m%d")
#
## Step 2: Mask for date range (2020-01-01 to 2025-04-30)
#start_date = pd.to_datetime("2020-01-01")
#end_date = pd.to_datetime("2025-04-30")
#mask = (date_dt >= start_date) & (date_dt <= end_date)
#
## Step 3: Apply the mask to filter data
#data_in_range = data[mask]
#
## Step 4: Compute how many unique dates exist in the range
#total_dates = date_dt[mask].unique()
#num_days = len(total_dates)
#
## Step 5: Count number of observations per coin
#coin_counts = data_in_range.groupby(level='coinName').size()
#
## Step 6: Keep coins with >= 75% of possible observations
#min_obs_required = int(0.75 * num_days)
#eligible_coins = coin_counts[coin_counts >= min_obs_required].index
#
## 443 coins
#print("Total number of coins:", len(eligible_coins))
#
## Step 7: Filter the data
#idx = pd.IndexSlice
#filtered_data = data_in_range.loc[idx[:, eligible_coins], :]
#
#data = filtered_data


#data_y = data['ret_excess']
#data_x = data.drop("ret_excess", axis=1)

# remove variable with high correlation
#data = data.drop(["illiq", "std_vol", "maxdprc", "stdprcvol", "beta2",
#                      "std_turn", "GPRD", "GPRD_MA7", "GPRD_MA30", "nsi"], axis=1)

data = data.drop(["std_vol", "maxdprc", "stdprcvol", "beta2", "volscaled", "logvol",
                  "std_turn",  "GPRD_MA7", "GPRD_MA30", 'nsi'], axis=1)


# Rescale columns as per Kelly. Omit news factors!
#not_transform = ['ret_excess', 'GPRD', 'GPRD_MA7', 'GPRD_MA30', 'nsi']
not_transform = ['ret_excess', 'GPRD']
to_transform = data.drop(columns=not_transform)
to_keep = data[not_transform]

# Apply the transform only to the other columns
transformed = to_transform.groupby(level='date').transform(new_ipca_utils.rank_scale)

# Recombine columns
data = pd.concat([transformed, to_keep], axis=1)

print("Final list of characteristics (exclude ret_excess): ")
print(data.columns)

######################################################################################
## RUN THE MODEL
######################################################################################

K = int(sys.argv[1])
mintol = 1e-6

model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

print(f"Running model with {K} factors, minTol = {mintol}")

model_fit = model.fit(K=K,
                      OOS = False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=10000)

print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")
