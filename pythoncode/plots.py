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

#################### REMOVE EXTREMES MCAP
btc_mcap = (
    coindata[coindata["coinName"] == "bitcoin"]
    .loc[:, ["date", "marketcap"]]
    .rename(columns={"marketcap": "btc_marketcap"})
)

# Join BTC marketcap to all coins
coins_vs_btc = coindata.merge(btc_mcap, on="date", how="left")

# Filter for coins with higher marketcap than BTC (excluding BTC itself)
coins_vs_btc = coins_vs_btc[
    (coins_vs_btc["marketcap"] > coins_vs_btc["btc_marketcap"]) &
    (coins_vs_btc["coinName"] != "bitcoin")
]

# Remove these observations from original data
coindata_filtered = coindata.merge(
    coins_vs_btc[["date", "coinName"]],
    on=["date", "coinName"],
    how="left",
    indicator=True
).query('_merge == "left_only"').drop(columns="_merge")

# Identify the row to remove
mask = (coindata_filtered["coinName"] == 480) & (coindata_filtered["date"] == pd.to_datetime("2021-10-22"))

# Drop it
coindata_filtered = coindata_filtered[~mask]

coindata_filtered[['date', 'coinName', 'marketcap']].sort_values(by='marketcap', ascending=False)

coindata = coindata_filtered.copy()
###########################################

data_mcap = coindata.groupby('date')['marketcap'].sum()

# Plot
plt.figure(figsize=(12,6))
plt.plot(data_mcap.index, data_mcap.values, color='steelblue')
plt.title('Total Market Cap Over Time')
plt.xlabel('Date')
plt.ylabel('Total Market Cap')
plt.grid(True)
plt.show()

# Plot with two series
# First series: sum of coin data
plt.figure(figsize=(12, 6))
plt.plot(data_mcap.index, data_mcap.values, color='steelblue', label='Total Market Cap (Coins)')
# Second series: mktcap DataFrame
plt.plot(mktcap['date'], mktcap['market_cap'], color='orange', label='External Market Cap')
# Labels and legend
plt.title('Total Market Cap Over Time')
plt.xlabel('Date')
plt.ylabel('Market Cap')
plt.grid(True)
plt.legend()
plt.show()
