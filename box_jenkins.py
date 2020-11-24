import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import config

data = pd.read_pickle(f"{config.data_folder}/{config.timeseries_interval}_data.pkl")

data = data[["Close"]]
d = data.diff().dropna()  # Take the first difference to make the time series stationary

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
a = plot_acf(d, ax=axes[0])
plt.gca().set_xlabel("lag")
plot_pacf(d, ax=axes[1])
plt.gca().set_xlabel("lag")
plt.show()

# Find the best p and q with aic, aicc and bic
aic_values = {}
aicc_values = {}
bic_values = {}
for p in range(5):
    for q in range(5):
        model = ARIMA(data.values.squeeze(), order=(p, 1, q))
        model_fit = model.fit()
        aic = model_fit.aic
        aicc = model_fit.aicc
        bic = model_fit.bic
        aic_values[(p, q)] = aic
        aicc_values[(p, q)] = aicc
        bic_values[(p, q)] = bic
        print("p, q,", p, q, "AIC:", aic, "AICC:", aicc, "BIC:", bic)

print("Sorted after aic", sorted(list(aic_values.items()), key=lambda x: x[1]))
print("Sorted after aicc", sorted(list(aicc_values.items()), key=lambda x: x[1]))
print("Sorted after bic", sorted(list(bic_values.items()), key=lambda x: x[1]))
