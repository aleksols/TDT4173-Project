import tensorflow as tf
import config
import numpy as np
import pandas as pd


def compile_and_fit(model, train_x, train_y, val_x, val_y, patience):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.SGD(),
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping]
    )
    return history


def normalize_data(data):
    data["Change"] = ((data["Close"] - data["Close"].shift(1)) / data["Close"]).fillna(0)
    data["macd"] /= data["Close"]
    data["macd_signal"] /= data["Close"]
    data["macd_histogram"] /= data["Close"]
    data["rsi"] /= 100

    normalized_data = data[data.columns[1:]]
    return normalized_data


def label_data(data, observation_window, columns=None):
    if columns is not None:
        d = pd.DataFrame(data[columns])
    else:
        d = pd.DataFrame(data)
    d["y"] = data["Change"].shift(-1)
    values = d.to_numpy()
    x = []
    y = []
    for i in range(observation_window, len(values)):
        x.append(values[i - observation_window:i, :-1])
        y.append(values[i - 1, -1])

    x = np.stack(x)
    y = np.stack(y)
    return x, y

class Benchmark(tf.keras.Model):
    def call(self, x, **kwargs):
        return np.zeros(x.shape[0])

if __name__ == '__main__':
    import pandas as pd

    D = pd.read_pickle("timeseries_data/daily_data.pkl")
    norm = normalize_data(D)

    print(D)
    print(norm)
    print(norm["Change"].std())
    # X, Y = label_data(norm, config.observation_window, ["Change"])
    # print(X)
