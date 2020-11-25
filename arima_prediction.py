import time
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

import config

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

data = data[["Close"]]  # We are only interested in the closing prices

# Split data into training and test datasets
n = len(data)
X = data.values.squeeze()
train_size = int(config.train_part * n)
test_size = int(config.test_part * n)
train, test = X[-(train_size + test_size):-test_size], X[-test_size:]

# Train the model
history = [x for x in train]
predictions = list()
for t in tqdm(range(len(test))):
    model = ARIMA(history, order=(config.p, config.d, config.q))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot predicted vs actual
plt.plot(test, label="Actual")
plt.plot(predictions, color='red', label="Predicted")
plt.legend()
plt.show()

# Calculate and print the metrics
absolute_error = np.absolute(np.array(predictions) - np.array(test))
abs_percentage_error = (absolute_error / np.array(test)) * 100
print("Mean absolute error:", absolute_error.mean())
print("Std of absolute error:", absolute_error.std())
print("Mean percentage absolute error:", abs_percentage_error.mean())
print("Std of absolute percentage error:", abs_percentage_error.std())

time_stamp = time.time()
mkdir(f"{config.statistics_root_directory}/ARIMA_experiment_{time_stamp}")

# Save config parameters and results to file
with open(f"{config.statistics_root_directory}/ARIMA_experiment_{time_stamp}/stats_{time_stamp}.txt", "w") as file:
    file.write("Config:\n")
    s = "timeseries_interval = " + str(config.timeseries_interval) + "\n"
    s += "train_part = " + str(config.train_part) + "\n"
    s += "test_part = " + str(config.test_part) + "\n"
    s += "p = " + str(config.p) + "\n"
    s += "i = " + str(config.d) + "\n"
    s += "q = " + str(config.q) + "\n"

    file.write(s)
    file.write("\nStats:\n")
    file.write(f"Mean absolute error for predicted values: {absolute_error.mean()}\n")
    file.write(f"Absolute error std for predicted values: {absolute_error.std()}\n")
    file.write(f"Mean absolute percentage error for predicted values: {abs_percentage_error.mean()}\n")
    file.write(f"Absolute percentage error std for predicted values: {abs_percentage_error.std()}\n")
