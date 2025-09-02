import os
import pickle
import pandas as pd

latent_dir = os.path.expanduser("~/thesis_output/oos_pred/latent")
obs_dir = os.path.expanduser("~/thesis_output/oos_pred/obsfactors")

rows = []

# -------- Latent (IPCA, PKL files) --------
for K in range(1, 7):
    fname = os.path.join(latent_dir, f"oos_ipca_K{K}.pkl")
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)   # always a DataFrame
        row = data.iloc[0].to_dict()
        rows.append({
            "spec": "IPCA",
            "factors": f"K{K}",
            "K": K,
            "rfits_R2_Total": row["rfits_R2_Total"] * 100,
            "rfits_R2_Pred": row["rfits_R2_Pred"] * 100,
            "xfits_R2_Total": row["xfits_R2_Total"] * 100,
            "xfits_R2_Pred": row["xfits_R2_Pred"] * 100
        })

# -------- Observable factors (CSV files) --------
for fname in os.listdir(obs_dir):
    if fname.endswith("_summary.csv"):
        path = os.path.join(obs_dir, fname)
        data = pd.read_csv(path)
        row = data.iloc[0].to_dict()

        parts = fname.replace("_summary.csv", "").split("_")
        spec = parts[1]  # e.g. "MKT", "2", "3", "4", "5"
        K = int(parts[-1].replace("K", ""))

        rows.append({
            "spec": spec,
            "factors": "CMKT" if spec == "MKT" else f"{spec} factors",
            "K": K,
            "rfits_R2_Total": row["rfits_R2_Total"] * 100,
            "rfits_R2_Pred": row["rfits_R2_Pred"] * 100,
            "xfits_R2_Total": row["xfits_R2_Total"] * 100,
            "xfits_R2_Pred": row["xfits_R2_Pred"] * 100
        })

# -------- Build DataFrame --------
df = pd.DataFrame(rows).round(2)
df = df.sort_values(by=["spec", "K"]).reset_index(drop=True)

print(df.to_string(index=False))
