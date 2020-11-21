# TDT4173-Project
This is the repository for the project in TDT4173 Machine Learning by Time Series Group 16.
Please read the instructions below carefully. 

### Downloading the datasets
Run data_collector.py with the following command to download the time series data.

<code>python data_collector.py</code>

This will download the nessecary time series data and save them as pickled pandas dataframes. 
The files will be stored in a new directory called "timeseries_data".

### Print statistics of the data
Run statistic_values.py to print relevant information about the the dataset such as the mean and standard deviation of features.

<code>python statistic_values.py</code>

### Configurations for ARIMA and LSTM
The file config.py can be used to tweak parameters such as the size of the training dataset and evaluation dataset for ARIMA.
For LSTM, in addition to tweak the sizes of the training, valuation and test datasets the configuration can be used to 
tweak hyperparameters for the LSTM.

##### IMPORTANT
Before running lstm_prediction.py the variable statistics_root_directory has to be changed to an existing directory on 
the host machine where the module can create subdirectories for storing log files. If this is not done lstm_predictions.py
will crash on line 61. 

### Run forecast analysis of ARIMA
Run arima_prediction.py to fit the model to a training dataset and evaluate it on a test dataset with the mean absolute error and mean absolute percentage error.

<code>python arima_prediction.py</code>

### Run forecast analysis of LSTM
Run lstm_prediction.py to fit the model to a training dataset and evaluate it on a test dataset with the mean absolute error and mean absolute percentage error.
The metrics for a naive approach is also output to use as a benchmark.  

<code>python lstm_prediction.py</code>