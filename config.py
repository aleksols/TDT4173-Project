"""
Configuration file for easy access to tweak parameters in the system
"""

# Parameters for data collection
# -------------------------------------------------------------------------
period_start = 0  # A unix timestamp. Use https://www.epochconverter.com/ to convert
period_end = 1604102400  # This is Oct 31 2020. Should not be changed for reproducibility reasons
data_folder = "timeseries_data"
# -------------------------------------------------------------------------


# Parameters for ARIMA
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# Parameters for LSTM
# -------------------------------------------------------------------------
timeseries_interval = "daily"
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

# The variable below is the path to the root directory where the folder containing statistic information will be saved.
# This has to be changed when run on a different host
statistics_root_directory = "C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project"
# -------------------------------------------------------------------------
