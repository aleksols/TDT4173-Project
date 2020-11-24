import tensorflow as tf
import pandas as pd
import config
import matplotlib.pyplot as plt
import utils
import time
from os import mkdir

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

# Split the data into train, validation and test datasets and standardize them individually to prevent data leakage
n = len(data)
train_df = data[: int(n * config.train_partition)]
val_df = data[int(n * config.train_partition): int(n * (config.train_partition + config.val_partition))]
test_df = data[-int(n * config.test_partition):]
standard_train, _, _ = utils.standardize_data(train_df)
standard_val, _, _ = utils.standardize_data(val_df)
standard_test, test_mean, test_std = utils.standardize_data(test_df)

# Label create labels for the train, validation and test datasets
train_x, train_y = utils.label_data(
    standard_train,
    config.observation_window,
    config.input_columns,
    label_column=config.label_column + "_standardized"
)

val_x, val_y = utils.label_data(
    standard_val,
    config.observation_window,
    config.input_columns,
    label_column=config.label_column + "_standardized"
)

test_x, test_y = utils.label_data(
    standard_test,
    config.observation_window,
    config.input_columns,
    label_column=config.label_column + "_standardized"
)

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=True))
model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=True))
model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=False))
model.add(tf.keras.layers.Dense(1))

# Compile and train model
history = utils.compile_and_fit(
    model=model,
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    patience=config.patience
)

# Create folder for saving results
time_stamp = str(time.time())
mkdir(f"{config.statistics_root_directory}/experiment_{time_stamp}")

# Plot the history from the training
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.savefig(f"{config.statistics_root_directory}/experiment_{time_stamp}/history_{time_stamp}.png")
plt.show()

# Use the model to predict the test set. The predicted values are compared to a benchmark that uses the naive
# approach of predicting yesterday's value as today's value.
pred = model(test_x)
result = pd.DataFrame(data["Close"][-len(pred) - 1:])
result["Benchmark"] = result["Close"].shift()
pred_unstandardized = pred.numpy().squeeze() * test_std + test_mean
result = result.drop(result.index[0])
result["Predictions"] = pred_unstandardized
result.rename(columns={"Close": "Actual"}, inplace=True)

# Plot actual values, predicted values and the benchmark values
result["Actual"].plot(label="Actual", marker=".")
plt.scatter(result.index, result["Predictions"], marker="X", c="orange", s=10, label="Predicted")
plt.scatter(result.index, result["Benchmark"], marker="o", c="green", s=10, label="Benchmark")
plt.legend()
plt.savefig(f"{config.statistics_root_directory}/experiment_{time_stamp}/scatterplot_{time_stamp}.png")
plt.show()

# Plot the raw, standardized output from the prediction and compare it to the standardized values of the actual values
plt.plot(standard_test.index[-len(pred):], pred.numpy(), label="Standardized predicted")
plt.plot(standard_test[config.label_column + "_standardized"][-len(pred):], label="Standardized actual")
plt.legend()
plt.savefig(f"{config.statistics_root_directory}/experiment_{time_stamp}/pred_vs_actual_{time_stamp}.png")
plt.show()

# Calculate mean absolute errors and percentage wise absolute errors to use for comparing performance
absolute_error_predicted = (result["Actual"] - result["Predictions"]).abs()
percentage_error_predicted = (1 - result["Predictions"]/result["Actual"]).abs() * 100
absolute_error_benchmark = (result["Actual"] - result["Benchmark"]).abs()
percentage_error_benchmark = (1 - result["Benchmark"]/result["Actual"]).abs() * 100
mae_predicted = absolute_error_predicted.mean()
avg_percentage_predicted = percentage_error_predicted.mean()
mae_benchmark = absolute_error_benchmark.mean()
avg_percentage_benchmark = percentage_error_benchmark.mean()

print("Mean absolute error for predicted values after un-standardizing:", mae_predicted)
print("Absolute error std for predicted values after un-standardizing:", absolute_error_predicted.std())
print("Mean average percentage wise error for predicted values after un-standarizing:", avg_percentage_predicted)
print("Mean average percentage wise error std for predicted values after un-standarizing:", percentage_error_predicted.std())
print("Mean absolute error for benchmark:", mae_benchmark)
print("Absolute error std for benchmark:", absolute_error_benchmark.std())
print("Mean average percentage wise error for benchmark values:", avg_percentage_benchmark)
print("Mean average percentage wise error std for benchmark values:", percentage_error_benchmark.std())

# Save config parameters and results to file
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
    s += "label_comumn = " + str(config.label_column) + "\n"

    file.write(s)
    file.write("\nStats:\n")
    file.write(f"Mean absolute error for predicted values: {mae_predicted}\n")
    file.write(f"Absolute error std for predicted values: {absolute_error_predicted.std()}\n")
    file.write(f"Mean absolute percentage error for predicted values: {percentage_error_predicted}\n")
    file.write(f"Absolute percentage error std for predicted values: {percentage_error_predicted.std()}\n")
    file.write(f"Mean absolute error for naive benchmark values: {mae_benchmark}\n")
    file.write(f"Absolute error std for naive benchmark values: {absolute_error_benchmark.std()}\n")
    file.write(f"Mean absolute percentage error for benchmark values: {percentage_error_predicted}\n")
    file.write(f"Absolute percentage error std for benchmark values: {percentage_error_predicted.std()}\n")
