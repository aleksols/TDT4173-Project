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


def standardize_data(data):
    d = pd.DataFrame(data[data.columns])
    for column in d.columns:
        d[column + "_standardized"] = (d[column] - d[column].mean()) / d[column].std()
    return d, d["Close"].mean(), d["Close"].std()


def label_data(data, observation_window, input_columns=None, label_column=None):
    if input_columns is not None:
        d = pd.DataFrame(data[input_columns])
    else:
        d = pd.DataFrame(data)
    d["y"] = d[label_column].shift(-1)
    values = d.to_numpy()
    x = []
    y = []
    for i in range(observation_window, len(values)):
        x.append(values[i - observation_window:i, :-1])
        y.append(values[i - 1, -1])

    x = np.stack(x)
    y = np.stack(y)
    return x, y


if __name__ == '__main__':
    # import pandas as pd
    import matplotlib.pyplot as plt

    D = pd.read_pickle("timeseries_data/daily_data.pkl")
    standardized = standardize_data(D)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(5, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam")
    a = np.reshape(np.arange(50000) / 10000, (50000, 1, 1))
    print(a.shape)
    model.fit(a, a.squeeze(), batch_size=32)
    pred = model(a)
    plt.plot(a.squeeze(), pred.numpy().squeeze())
    plt.show()
    print(model.layers[-1].weights)
