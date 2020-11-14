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
train_partition = 0.7
val_partition = 0.2
test_partition = 0.1
# -------------------------------------------------------------------------
