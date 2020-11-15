import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import config
import matplotlib.pyplot as plt
from window_generator import WindowGenerator

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
train_data = normalized_data[: int(n * config.train_partition)]
val_data = normalized_data[int(n * config.train_partition): int(n * (config.train_partition + config.val_partition))]
test_data = normalized_data[int(n * (1 - config.test_partition)):]

wg = WindowGenerator(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    observation_window=10,
    prediction_window=1,
    prediction_offset=1
)

train_dataset = wg.make_dataset(train_data)
val_dataset = wg.make_dataset(val_data)
test_dataset = wg.make_dataset(test_data)

normalized_data.plot(subplots=True)
plt.show()
