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
observation_window = 5
patience = 5
lstm_units = 10
epochs = 500
batch_size = 32
input_columns = None  # A list of input features or None to use all features as input
statistics_root_directory = "C:/Users/Aleksander/Google Drive/Skole/4. klasse/Høst/Maskinlæring/project"
# -------------------------------------------------------------------------
