import numpy as np
import tensorflow as tf

"""
This file is based on the following tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series
"""


class WindowGenerator:
    def __init__(self, train_data, val_data, test_data, observation_window, prediction_window, prediction_offset,
                 label_columns=None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.observation_window = observation_window
        self.prediction_window = prediction_window
        self.prediction_offset = prediction_offset

        self.total_window_size = observation_window + prediction_offset
        self.input_slice = slice(0, observation_window)
        self.observation_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_data.columns)}

        self.label_start = self.total_window_size - self.prediction_window
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.observation_window, None])
        labels.set_shape([None, self.prediction_window, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float)
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32
        )

        dataset = dataset.map(self.split)
        return dataset

if __name__ == '__main__':
    import config
    import pandas as pd

    data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")
    train_data = data[:int(0.7 * len(data))]
    print(train_data.head(10))
    val_data = data[int(0.7 * len(data)):int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]
    w = WindowGenerator(train_data, val_data, test_data, 8, 1, 1, label_columns=["Close"])
    example_window = tf.stack([np.array(train_data[0:w.total_window_size]),
                               np.array(train_data[9:9 + w.total_window_size]),
                               np.array(train_data[9:9 + w.total_window_size])])

    example_inputs, example_labels = w.split(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')

    test = w.make_dataset(train_data)
    print(test.element_spec)
