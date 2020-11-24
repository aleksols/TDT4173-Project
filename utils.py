import numpy as np
import pandas as pd
import tensorflow as tf

import config


def compile_and_fit(model, train_x, train_y, val_x, val_y, patience):
    """
    Compiles and trains the model on the training dataset. The trained is stopped early if the loss on the validation
    dataset does not improve for 5 epochs.

    :param model: The model to be trained
    :param train_x: The training dataset
    :param train_y: The label dataset for training
    :param val_x: The input validation dataset
    :param val_y: The label validation dataset
    :param patience: Patience before early stopping kicks in
    :return: The history returned from model.fit()
    """

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
    """
    Scales the values by subtracting the mean and dividing by the standard deviation. This is done to bring all the
    features to a common scale.
    :param data: A dataframe with features to be standardized
    :return: A copy of the input dataframe with standardized values, the mean of the
    """
    d = pd.DataFrame(data[data.columns])
    for column in d.columns:
        d[column + "_standardized"] = (d[column] - d[column].mean()) / d[column].std()
    return d, d[config.label_column].mean(), d[config.label_column].std()


def label_data(data, observation_window, input_columns=None, label_column=None):
    """
    Structure the data so it can be fed to the LSTM model. The input data (x) should have shape
    (None, observation, num_features). The first axis will be as long as the input data minus the length of the
    observation window.
    :param data: A dataframe with standardized input features
    :param observation_window: The number of time steps to include in one sample
    :param input_columns: A list of columns in the dataframe to be used as features
    :param label_column: The column to be used as label for the training data.
    :return:
    """
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
