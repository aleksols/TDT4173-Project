"""
Configuration file for easy access to tweak parameters in the system
"""

# Parameters for data collection
# -------------------------------------------------------------------------
period_start = 0  # A unix timestamp. Use https://www.epochconverter.com/ to convert
period_end = 1604102400  # This is Oct 31 2020. Should not be changed for reproducibility reasons
data_folder = "timeseries_data"
# -------------------------------------------------------------------------


# General configurations
# -------------------------------------------------------------------------
timeseries_interval = "daily"  # Time series to use
# The variable below is the path to the root directory where the folder containing statistic information will be saved.
# This has to be changed when run on a different host
statistics_root_directory = "C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project"
# -------------------------------------------------------------------------


# Parameters for ARIMA
# -------------------------------------------------------------------------
# train_part + test_part can not be greater that 1
train_part = 0.2  # Partition of data to be used for initial training of the arima
test_part = 0.03  # Partition of data to be used for testing the arima
# train_part + test_part does not have to be 1. For example if train_part is 0.5 and test part is 0.1 we will use
# the last 10% of data for testing and a partition of size equal to 50% of the data as training data. That is, the last
# 60% of the data will be used in total.
p = 3  # The parameter for the AR part
i = 1  # The parameter for the integration part
q = 2  # The parameter for the MA part
# -------------------------------------------------------------------------


# Parameters for LSTM
# -------------------------------------------------------------------------
train_partition = 0.85
val_partition = 0.12
test_partition = 0.03
observation_window = 1  # The number of previous time steps to make up one input sample
patience = 1  # Number of epochs without improvement before early stopping kicks in
lstm_units = 20
epochs = 250
batch_size = 64

# A list of input features for the model to predict or None to use all features as input
input_columns = [
    "Close_standardized",
    "macd_standardized",
    "macd_histogram_standardized",
    "macd_signal_standardized",
    "rsi_standardized"
]

# NB: Not to be changed. This was meant to be a changeable variable, but the code in lstm_prediction.py does not
# support this as of now. Predicting the close price is also what this project is mainly about
label_column = "Close"
# -------------------------------------------------------------------------
