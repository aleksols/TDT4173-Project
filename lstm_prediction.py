import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import config
import matplotlib.pyplot as plt

pd.options.display.width = 0

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

# Normalize data
data["Change"] = ((data["Close"] - data["Close"].shift(1)) / data["Close"]).fillna(0)
data["macd"] /= data["Close"]
data["macd_signal"] /= data["Close"]
data["macd_histogram"] /= data["Close"]
data["rsi"] /= 100

normalized_data = data[data.columns[1:]]

n = len(data)
print(n)
train_data = data[: int(n * config.train_partition)]
val_data = data[int(n * config.train_partition): int(n * (config.train_partition + config.val_partition))]
test_data = data[int(n * (1 - config.test_partition)):]

normalized_data.plot(subplots=True)
plt.show()