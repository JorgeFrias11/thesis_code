import os
import re
import pickle
import pandas as pd
import ipca_pruitt

directory = "/home/jfriasna/thesis_output/reg_alpha/"   

factor_files = {}

for fname in os.listdir(directory):
    if not fname.endswith(".pkl"):
        continue

    # Match files 
    match = re.match(r"(\d+)_factors.*\.pkl", fname)
    if match:
        factor_num = int(match.group(1))
        # keep only the first file for each factor
        if factor_num not in factor_files:
            factor_files[factor_num] = fname

results = []

for factor, fname in sorted(factor_files.items()):
    fpath = os.path.join(directory, fname)
    print(f"accessing factor {factor} file:")
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    try:
        r2_rfits_total = data["model_fit"]["rfits"]["R2_Total"] * 100
        r2_rfits_pred  = data["model_fit"]["rfits"]["R2_Pred"]  * 100
        r2_xfits_total = data["model_fit"]["xfits"]["R2_Total"] * 100
        r2_xfits_pred  = data["model_fit"]["xfits"]["R2_Pred"]  * 100

        results.append({
            "factor": factor,
            "file": fname,
            "R2_rfits_total": round(r2_rfits_total, 2),
            "R2_rfits_pred":  round(r2_rfits_pred, 2),
            "R2_xfits_total": round(r2_xfits_total, 2),
            "R2_xfits_pred":  round(r2_xfits_pred, 2)
        })
    except KeyError as e:
        print(f"Missing key in {fname}: {e}")


# Store in DF
df = pd.DataFrame(results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df.to_string(index=False))

