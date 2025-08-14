######################################################################################
# This code runs the IPCA models without a constant using Pruitts python implementation.
######################################################################################

import ipca_pruitt
import sys
import pandas as pd

data = pd.read_pickle("/home/jfriasna/thesis_data/data/processed_daily_preds_100mill.pkl")
#data = pd.read_pickle("/home/jori/Documents/QFIN/thesis_data/data/processed_daily_preds.pkl")

print(f"Base model results:")

K = int(sys.argv[1])
mintol = 1e-6

model = ipca_pruitt.ipca(RZ=data, return_column='ret_excess', add_constant=False)

model_fit = model.fit(K=K,
                      OOS = False,
                      gFac=None,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=10000)

print(f"Total R2: {model_fit['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit['rfits']['R2_Pred']:.4f}")

######################################################################################
# Run second model with Geopolitical Risk Factor
######################################################################################

print(f"\n Model with Geopolitical risk index as prespecified factor:")

# Read Excel file (first sheet named "Sheet1")
gpr = pd.read_excel("/home/jfriasna/thesis_data/data_gpr_daily_recent.xlsx", sheet_name="Sheet1")
#gpr = pd.read_excel("/home/jori/Documents/QFIN/thesis_data/data_gpr_daily_recent.xlsx", sheet_name="Sheet1")

# Convert 'date' column to datetime
#gpr['date'] = pd.to_datetime(gpr['date'])
gpr['date'] = gpr['date'].dt.strftime('%Y%m%d').astype(int)
# Keep only relevant columns
gpr = gpr[['date', 'GPRD']]
min_date = data.index.get_level_values('date').min()
# Filter rows between 2014-01-01 and last_date
gpr = gpr[(gpr['date'] >= min_date) & (gpr['date'] <= 20250731)]
gpr.set_index('date', inplace=True)
gpr_factor = gpr.T

# RUUN IPCA with gFac: with GPR as a pre-specified factor
model_fit_gpr = model.fit(K=K,
                      OOS = False,
                      gFac=gpr_factor,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=10000)

print(f"Total R2: {model_fit_gpr['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit_gpr['rfits']['R2_Pred']:.4f}")


######################################################################################
# Run model with News sentiment index as pre-specified factor
######################################################################################

print(f"\n Model with News sentiment index as prespecified factor:")

# Read Excel file (first sheet named "Sheet1")
nsi = pd.read_excel("/home/jfriasna/thesis_data/news_sentiment_data.xlsx", sheet_name="Data")
#nsi = pd.read_excel("/home/jori/Documents/QFIN/thesis_data/news_sentiment_data.xlsx", sheet_name="Data")
# Convert 'date' column to datetime
nsi['date'] = nsi['date'].dt.strftime('%Y%m%d').astype(int)

nsi.rename(columns={'News Sentiment': 'nsi'}, inplace=True)
min_date = data.index.get_level_values('date').min()
# Filter rows between 2014-01-01 and last_date
nsi = nsi[(nsi['date'] >= min_date) & (nsi['date'] <= 20250731)]
nsi.set_index('date', inplace=True)
nsi_factor = nsi.T

# RUUN IPCA with gFac: with nsi as a pre-specified factor
model_fit_nsi = model.fit(K=K,
                      OOS = False,
                      gFac=nsi_factor,
                      dispIters=True,
                      dispItersInt=25,
                      minTol=mintol,
                      maxIters=10000)

print(f"Total R2: {model_fit_nsi['rfits']['R2_Total']:.4f}")
print(f"Predictive R2: {model_fit_nsi['rfits']['R2_Pred']:.4f}")

