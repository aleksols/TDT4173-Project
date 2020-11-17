import pandas as pd
import config
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf




pd.options.display.width = 0

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

print(data.head())
#data.plot()
#plt.show()


data = data["Close"]

n = len(data)
print(n)

use_data = data[int(n*0.9):int(n*0.96)]

X = use_data.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
plot_acf(test)
plt.show()
plot_pacf(test)
plt.show()
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,0,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
