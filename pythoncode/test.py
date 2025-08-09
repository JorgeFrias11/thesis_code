# Add the parent of the parent directory to the Python path
#current_dir = Path().resolve()
#sys.path.append(str(current_dir.parents[1]))

import pandas as pd
import ipca_utils

#datapath = "/home/jfriasna/thesis_data/data/"
datapath = "/home/jori/Documents/QFIN/thesis_data/data/"
data, coin_id = ipca_utils.load_coindata('daily', datapath,
                                cache_file = 'cache_daily_preds.pkl',
                                daily_rds='daily_predictors.rds',
                                ignore_cols=['logvol', 'nsi', 'GPRD', 'GPRD_MA7', 'GPRD_MA30'],
                                pruitt=True,
                                save = True)

print(data.columns)

##################################################################################
## TEST 2: Shorten time period, and keep coins with at least 75% of total obs
## Period: 2020-01-01 to 2025-07-31
##################################################################################

print("Number of coins in the whole sample:", len(data.index.get_level_values('coinName').unique()))

# Step 1: Convert the integer date level to datetime
date_ints = data.index.get_level_values('date')
date_dt = pd.to_datetime(date_ints.astype(str), format="%Y%m%d")

# Step 2: Mask for date range (2020-01-01 to 2025-07-31)
start_date = pd.to_datetime("2018-06-01")
end_date = pd.to_datetime("2025-07-31")
mask = (date_dt >= start_date) & (date_dt <= end_date)

# Step 3: Apply the mask to filter data
data_in_range = data[mask]

print("Number of coins in reduced sample period:", len(data_in_range.index.get_level_values('coinName').unique()))

# Step 4: Compute how many unique dates exist in the range
total_dates = date_dt[mask].unique()
num_days = len(total_dates)

# Step 5: Count number of observations per coin
coin_counts = data_in_range.groupby(level='coinName').size()

# Step 6: Keep coins with >= 75% of possible observations
#min_obs_required = int(0.5 * num_days)
min_obs_required = 730
eligible_coins = coin_counts[coin_counts >= min_obs_required].index

print("Total number of coins:", len(eligible_coins))

# Step 7: Filter the data
idx = pd.IndexSlice
filtered_data = data_in_range.loc[idx[:, eligible_coins], :]

filtered_data.to_pickle("/home/jori/Documents/QFIN/thesis_data/data/filtered_daily_preds.pkl")

data = filtered_data

##################################################################################
# Following the code of Kelly et al. (2019), keep dates with a min cross-section of 100 coins
##################################################################################

print("n rows before min cross-section:", data.shape)


##################################################################################
# Check correlation
##################################################################################

chars = data.drop(columns='ret_excess')
# Calculate correlation
corr_matrix = chars.corr()
# Stack into long format (row-wise pairs)
corr_pairs = corr_matrix.stack()
# Filter: keep only values > 0.85 and < 1 (to exclude self-correlation)
high_corr = corr_pairs[(abs(corr_pairs) > 0.85) & (abs(corr_pairs) < 1)]
print(high_corr)

# remove variable with high correlation
#data = data.drop(["illiq", "std_vol", "maxdprc", "stdprcvol", "beta2",
#                      "std_turn", "GPRD", "GPRD_MA7", "GPRD_MA30", "nsi"], axis=1)

data = data.drop(["maxdprc", "prcvol", "stdprcvol", "volscaled", "beta2"], axis=1)

##################################################################################
# Rescale columns to (-0.5, 0.5) as in Kelly et al. (2019)
##################################################################################

not_transform = ['ret_excess']
to_transform = data.drop(columns=not_transform)
to_keep = data[not_transform]

# Apply the transform only to the other columns
transformed = to_transform.groupby(level='date').transform(ipca_utils.rank_scale)

# Recombine columns
data = pd.concat([transformed, to_keep], axis=1)

print("Final list of characteristics (excluding ret_excess): ")
print(data.columns)

# Save the processed data for the empirical analysis
#data.to_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds.pkl")
data.to_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")
