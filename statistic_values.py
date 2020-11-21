import utils
import pandas as pd

df = pd.read_pickle("timeseries_data/daily_data.pkl")

for column in df.columns:
    print(f"Mean for {column}:", df[column].mean())
    print(f"Std for {column}:", df[column].std())

d, _, _ = utils.standardize_data(df)

for column in [c for c in d.columns if "standardized" in c]:
    for col2 in [c for c in d.columns if "standardized" in c]:
        print(f"Correlation between {column} and {col2}: {d[column].corr(d[col2])}")

d["Change"] = (d["Close"] / d["Close"].shift(1) - 1).fillna(0)

print("Mean absolute daily change in closing price:", d["Change"].abs().mean())
print("Std for absolute daily change in closing price:", d["Change"].abs().std())
