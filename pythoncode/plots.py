import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

datapath = "/home/jori/Documents/QFIN/thesis_data/"

coindata = pd.read_pickle(datapath + "data/filtered_daily_preds.pkl")
mktcap = pd.read_csv(datapath + "CoinGecko-GlobalCryptoMktCap.csv")

mktcap.rename(columns={'snapped_at' : 'date'}, inplace=True)
mktcap['date'] = pd.to_datetime(mktcap['date'], unit='ms').dt.date
mktcap.drop('total_volume', axis=1, inplace=True)
mktcap = mktcap[mktcap['date'] >= datetime.date(2018, 6, 1)]

coindata = coindata.reset_index()
# Convert date column to date
coindata["date"] = pd.to_datetime(
    coindata["date"].astype(str),
    format="%Y%m%d"
)

data_mcap = coindata.groupby('date')['marketcap'].sum()

# Plot of market capitalization
plt.figure(figsize=(10, 6))
plt.plot(data_mcap.index, data_mcap.values / 1e9, label='Sample market capitalization')
# Second series: mktcap dataFrame fron CoinGecko
plt.plot(mktcap['date'], mktcap['market_cap'] / 1e9, color='chocolate', alpha=0.75, linestyle='--', label='Total market capitalization')
plt.ylabel('Market capitalization (Billion USD)')
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()

# Boxplots
# Observations per coin
observations_per_coin = coindata.groupby('coinName').size().reset_index(name='n_obs')

# Boxplot
plt.figure(figsize=(6, 6))
plt.boxplot(observations_per_coin['n_obs'],
            notch=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            flierprops=dict(marker='o', markerfacecolor='black', markersize=5, linestyle='none'),
            showfliers=True
            )

plt.ylabel("Number of observations")
plt.yticks(range(0, observations_per_coin['n_obs'].max() + 500, 500))
plt.grid(axis='y', alpha=0.5)
plt.show()


observations_per_coin.max()
observations_per_coin.sort_values(by='n_obs', ascending=False)

observations_per_coin[observations_per_coin['coinName'] == 151]

coin_name_400 = next(name for name, cid in coin_id.items() if cid == 151)
print(coin_name_400)
