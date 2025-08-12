
import pandas as pd
import matplotlib.pyplot as plt
import datetime
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
plt.figure(figsize=(12, 4))  # To match my R figures' size
plt.plot(data_mcap.index, data_mcap.values / 1e9, label='Sample market capitalization')
plt.fill_between(data_mcap.index, data_mcap.values / 1e9, color='steelblue', alpha=0.2)
# Second series: mktcap dataFrame fron CoinGecko
plt.plot(mktcap['date'], mktcap['market_cap'] / 1e9, color='chocolate', alpha=0.75,
         linestyle='--', label='Total market capitalization')
#plt.fill_between(mktcap['date'], mktcap['market_cap'] / 1e9, color='chocolate', alpha=0.15)
plt.ylabel('Market capitalization\n(Billion USD)', fontsize=9)
plt.xlabel('Date', fontsize=9)
plt.grid(True, alpha=0.5)
plt.legend(fontsize=9)
plt.tick_params(axis='both', labelsize=9)  # tick labels
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


# Table of descriptive statistics
min_no_obs = coindata.groupby('date').size().min()

# Summary row
tbl = pd.DataFrame({
    "No. Obs":        [len(coindata)],
    "Unique coins":   [coindata['coinName'].nunique()],
    "Min No. Obs":    [min_no_obs],
    "Mean":           [coindata['ret_excess'].mean()],
    "Std":            [coindata['ret_excess'].std(ddof=1)],
    "P10":            [coindata['ret_excess'].quantile(0.10)],
    "P25":            [coindata['ret_excess'].quantile(0.25)],
    "P50":            [coindata['ret_excess'].quantile(0.50)],
    "P75":            [coindata['ret_excess'].quantile(0.75)],
    "P90":            [coindata['ret_excess'].quantile(0.90)],
}, index=["Sample"])

# Pretty formatting: counts with commas, returns as percentages
percent_cols = ["Mean","Std","P10","P25","P50","P75","P90"]
tbl_fmt = tbl.copy()
for c in percent_cols:
    tbl_fmt[c] = (tbl_fmt[c]*100).map(lambda x: f"{x:,.2f}%")
for c in ["No. Obs","Unique coins","Min No. Obs"]:
    tbl_fmt[c] = tbl_fmt[c].map("{:,.0f}".format)


latex_code = tbl_fmt.to_latex(escape=False, index=False, column_format="l" + "r"*len(tbl_fmt.columns))
print(latex_code)



tbl.to_csv("output/sample_overview_table.csv", index=True)


df = pd.DataFrame(dict(name=['Raphael', 'Donatello'],
                       age=[26, 45],
                       height=[181.23, 177.65]))

print(df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.1f}".format,
))
