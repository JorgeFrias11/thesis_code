
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import ipca_utils
import seaborn as sns
import numpy as np

datapath = "/home/jori/Documents/QFIN/thesis_data/"

mktcap = pd.read_csv(datapath + "CoinGecko-GlobalCryptoMktCap.csv")

mktcap.rename(columns={'snapped_at' : 'date'}, inplace=True)
mktcap['date'] = pd.to_datetime(mktcap['date'], unit='ms').dt.date
mktcap.drop('total_volume', axis=1, inplace=True)
mktcap = mktcap[mktcap['date'] >= datetime.date(2018, 6, 1)]


coindata = pd.read_pickle(datapath + "data/filtered_daily_preds.pkl")
#coindata = pd.read_pickle(datapath + "data/filtered_365_preds.pkl")
coindata = coindata.reset_index()
# Convert date column to date
coindata["date"] = pd.to_datetime(
    coindata["date"].astype(str),
    format="%Y%m%d"
)

data_mcap = coindata.groupby('date')['marketcap'].sum()

# Plot of market capitalization
plt.figure(figsize=(12, 6))  # To match my R figures' size
plt.plot(data_mcap.index, data_mcap.values / 1e9, label='Sample market capitalization')
plt.fill_between(data_mcap.index, data_mcap.values / 1e9, color='steelblue', alpha=0.2)
# Second series: mktcap dataFrame fron CoinGecko
plt.plot(mktcap['date'], mktcap['market_cap'] / 1e9, color='chocolate', alpha=0.75,
         linestyle='--', label='Total market capitalization')
#plt.fill_between(mktcap['date'], mktcap['market_cap'] / 1e9, color='chocolate', alpha=0.15)
plt.ylabel('Market capitalization\n(Billion USD)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12)  # tick labels
plt.tight_layout()
plt.show()


sample_num_coins = coindata.groupby('date').size().reset_index(name='n_obs')
plt.figure(figsize=(12, 4))
plt.plot(sample_num_coins['date'], sample_num_coins['n_obs'])
plt.ylabel('Number of unique coins', fontsize=9)
plt.xlabel('Date', fontsize=9)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()


# Boxplots
# Observations per coin
observations_per_coin = coindata.groupby('coinName').size().reset_index(name='n_obs')

# Size of the cross-section
cross_section_size = coindata.groupby('date').size().reset_index(name='n_obs')

# Plot in a single figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # 1 row, 2 columns
# First boxplot: observations per coin
axes[0].boxplot(observations_per_coin['n_obs'],
                notch=True,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                widths=0.25)
axes[0].set_ylabel("Number of observations for a given coin")
axes[0].set_yticks(range(0, observations_per_coin['n_obs'].max() + 500, 500))
axes[0].set_xticks([])
axes[0].grid(axis='y', alpha=0.5)

# Second boxplot: cross-section size
axes[1].boxplot(cross_section_size['n_obs'],
                notch=True,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                widths=0.25)
axes[1].set_ylabel("Number of unique coins at a given time")
axes[1].set_yticks(range(0, cross_section_size['n_obs'].max() + 200, 200))
axes[1].set_xticks([])
axes[1].grid(axis='y', alpha=0.5)

plt.tight_layout(w_pad=5.0)  # w_pad is in inches of space
#plt.subplots_adjust(wspace=0.5)  # Increase number for more space
plt.show()


# Summary table

#datapath = "/home/jori/Documents/QFIN/thesis_data/data/"
data, coin_id = ipca_utils.load_coindata('daily', datapath,
                                cache_file = 'data/cache_daily_preds.pkl',
                                daily_rds='data/daily_predictors.rds',
                                ignore_cols=['logvol', 'nsi', 'GPRD', 'GPRD_MA7', 'GPRD_MA30'],
                                pruitt=True,
                                save = False )

id_to_name = {v: k for k, v in coin_id.items()}

coindata['coinName'] = coindata['coinName'].map(id_to_name)

# Pick the last available date
last_date = coindata['date'].max()
# Filter for that date
last_day = coindata[coindata['date'] == last_date]
# Sort by marketcap and keep top 100
top100 = last_day.sort_values('marketcap', ascending=False).head(100)

# Use r2_1, which is the coin returns, but lagged one day.
def make_table(df, index_name):
    return pd.DataFrame({
        "No. Obs":        [len(df)],
        "Unique coins":   [df['coinName'].nunique()],
        #"Min No. Obs":   [df.groupby('coinName').size().min()],
        "Mean":           [df['r2_1'].mean()],
        "Std":            [df['r2_1'].std(ddof=1)],
        "P10":            [df['r2_1'].quantile(0.10)],
        "P25":            [df['r2_1'].quantile(0.25)],
        "P50":            [df['r2_1'].quantile(0.50)],
        "P75":            [df['r2_1'].quantile(0.75)],
        "P90":            [df['r2_1'].quantile(0.90)],
    }, index=[index_name])

# --- Full sample (you already have this as tbl) ---
tbl_all = make_table(coindata, "Sample")

# Top 100 coins
top100_coins = top100['coinName']
coindata_top100 = coindata[coindata['coinName'].isin(top100_coins)]
tbl_top100 = make_table(coindata_top100, "Top 100")

# --- Top 10 coins by market cap at last date ---
top10_coins = top100['coinName'].head(10).unique()
coindata_top10 = coindata[coindata['coinName'].isin(top10_coins)]
tbl_top10 = make_table(coindata_top10, "Top 10")

# --- Only BTC, ETH, XRP ---
# assuming your coin_id dict has 'bitcoin', 'ethereum', 'ripple' entries

btc = coindata[coindata['coinName'] == 'bitcoin']
eth = coindata[coindata['coinName'] == 'ethereum']
xrp = coindata[coindata['coinName'] == 'ripple']

tbl_btc = make_table(btc, "Bitcoin")
tbl_eth = make_table(eth, "Ethereum")
tbl_xrp = make_table(xrp, "Ripple")

# --- Combine all three tables ---
tbl_final = pd.concat([tbl_all, tbl_top100, tbl_top10, tbl_btc, tbl_eth, tbl_xrp])

# Pretty formatting: counts with commas, returns as percentages
percent_cols = ["Mean","Std","P10","P25","P50","P75","P90"]
tbl_fmt = tbl_final.copy()
for c in percent_cols:
    tbl_fmt[c] = (tbl_fmt[c]*100).map(lambda x: f"{x:,.2f}\\%")
for c in ["No. Obs","Unique coins"]:
    tbl_fmt[c] = tbl_fmt[c].map("{:,.0f}".format)

latex_code = tbl_fmt.to_latex(escape=False, index=True, column_format="l" + "r"*len(tbl_fmt.columns))
print(latex_code)

# Table of cross-section size

coindata['year'] = coindata['date'].dt.year
unique_coins = coindata.groupby('year')['coinName'].nunique()
# cross-section size per day
cross_section = coindata.groupby(['year', 'date'])['coinName'].nunique()
# minimum cross-section size per year
min_cross_section = cross_section.groupby('year').min()
# combine into one DataFrame with years as columns
tbl = pd.DataFrame([unique_coins, min_cross_section],
                   index=['Unique coins', 'Min cross section size'])
latex_tbl = tbl.to_latex(index=True, header=True, bold_rows=True,
                         caption="Number of unique coins and minimum cross-section size by year",
                         label="tab:cross_section")
print(latex_tbl)