import pandas as pd

import config
import utils

df = pd.read_pickle("timeseries_data/daily_data.pkl")
lstm_train = df[:int(len(df) * config.train_partition)]
lstm_val = df[int(len(df) * config.train_partition):-int(len(df) * config.test_partition)]
lstm_test = df[-int(len(df) * config.test_partition):]

for column in df.columns:
    print(f"Mean for {column} in entire dataset:", df[column].mean())
    print(f"Std for {column} in entire dataset:", df[column].std())
    print(f"Mean for {column} in train dataset:", lstm_train[column].mean())
    print(f"Std for {column} in train dataset:", lstm_train[column].std())
    print(f"Mean for {column} in validation dataset:", lstm_val[column].mean())
    print(f"Std for {column} in validation dataset:", lstm_val[column].std())
    print(f"Mean for {column} in test dataset:", lstm_test[column].mean())
    print(f"Std for {column} in test dataset:", lstm_test[column].std())
    print()

print()

d, _, _ = utils.standardize_data(df)

for column in [c for c in d.columns if "standardized" in c]:
    for col2 in [c for c in d.columns if "standardized" in c]:
        print(f"Correlation between {column} and {col2}: {d[column].corr(d[col2])}")

print()

d["Change"] = (d["Close"] / d["Close"].shift(1) - 1).fillna(0)

print()
print("Mean absolute daily change in closing price:", d["Change"].abs().mean())
print("Std for absolute daily change in closing price:", d["Change"].abs().std())
