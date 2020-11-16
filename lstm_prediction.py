import tensorflow as tf
import pandas as pd
import config
import matplotlib.pyplot as plt
import utils
import numpy as np


pd.options.display.width = 0

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

# Normalize data
normalized_data = utils.normalize_data(data)

x, y = utils.label_data(normalized_data, config.observation_window, config.input_columns)

n = len(x)
train_x = x[: int(n * config.train_partition)]
train_y = y[: int(n * config.train_partition)]
val_x = x[int(n * config.train_partition): int(n * (config.train_partition + config.val_partition))]
val_y = y[int(n * config.train_partition): int(n * (config.train_partition + config.val_partition))]
test_x = x[-int(n * config.test_partition):]
test_y = y[-int(n * config.test_partition):]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=False))
model.add(tf.keras.layers.Dense(1))

history = utils.compile_and_fit(
    model=model,
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    patience=config.patience
)
import time
from os import mkdir

time_stamp = str(time.time())
mkdir(f"C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project/experiment_{time_stamp}")

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.savefig(f"C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project/experiment_{time_stamp}/history_{time_stamp}.png")
plt.show()
pred = model(test_x)

result = pd.DataFrame(data["Close"][-len(pred) - 1:])
result["Benchmark"] = result["Close"].shift()

pred_numpy = pred.numpy()
r = result["Close"][:-1].to_numpy() * (1 + pred.numpy().squeeze())
result = result.drop(result.index[0])
result["Predictions"] = r
result.rename(columns={"Close": "Actual"}, inplace=True)

result["Actual"].plot(label="Actual", marker=".")
plt.scatter(result.index, result["Predictions"], marker="X", c="orange", s=25, label="Predicted")
plt.scatter(result.index, result["Benchmark"], marker="o", c="green", s=25, label="Benchmark")
plt.legend()
plt.savefig(f"C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project/experiment_{time_stamp}/scatterplot_{time_stamp}.png")
plt.show()

plt.plot(normalized_data.index[-len(pred):], pred.numpy(), label="Predicted")
plt.plot(normalized_data["Change"][-len(pred):], label="Actual")
plt.legend()
plt.savefig(f"C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project/experiment_{time_stamp}/pred_vs_actual_{time_stamp}.png")
plt.show()

absolute_error_predicted = np.absolute(pred.numpy().squeeze() - test_y)
absolute_error_benchmark = np.absolute(test_y)
absolute_error_random = np.absolute(pred.numpy().squeeze() - (np.random.rand(len(pred)) / 20 - 0.025))
mae_predicted = absolute_error_predicted.mean()
mae_benchmark = absolute_error_benchmark.mean()
mae_random = absolute_error_random.mean()

print("Mean absolute error for predicted values:", mae_predicted)
print("Absolute error std for predicted values:", absolute_error_predicted.std())
print("Mean absolute error for benchmark:", mae_benchmark)
print("Absolute error std for benchmark:", absolute_error_benchmark.std())
print("Mean absolute error for random benchmark:", mae_random)
print("Absolute error std for random benchmark:", absolute_error_random.std())

with open(f"{config.statistics_root_directory}/experiment_{time_stamp}/stats_{time_stamp}.txt", "w") as file:
    file.write("Config:\n")
    s = "timeseries_interval = " + str(config.timeseries_interval) + "\n"
    s += "train_partition = " + str(config.train_partition) + "\n"
    s += "val_partition = " + str(config.val_partition) + "\n"
    s += "test_partition = " + str(config.test_partition) + "\n"
    s += "observation_window = " + str(config.observation_window) + "\n"
    s += "patience = " + str(config.patience) + "\n"
    s += "lstm_units = " + str(config.lstm_units) + "\n"
    s += "epochs = " + str(config.epochs) + "\n"
    s += "batch_size = " + str(config.batch_size) + "\n"
    s += "input_columns = " + str(config.input_columns) + "\n"
    file.write(s)
    file.write("\nStats:\n")
    file.write(f"Mean absolute error for predicted values: {mae_predicted}\n")
    file.write(f"Absolute error std for predicted values: {absolute_error_predicted.std()}\n")
    file.write(f"Mean absolute error for naive benchmark values: {mae_benchmark}\n")
    file.write(f"Absolute error std for naive benchmark values: {absolute_error_benchmark.std()}\n")
    file.write(f"Mean absolute error for random benchmark values: {mae_random}\n")
    file.write(f"Absolute error std for random benchmark values: {absolute_error_random.std()}\n")
