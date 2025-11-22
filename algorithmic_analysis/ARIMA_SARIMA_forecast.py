# === ARIMA / SARIMA FORECAST TEMPLATE ===

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Choose one region + category or aggregate (recommended)
ts_df = master_df[(master_df["region"] == "West") & (master_df["category"] == "Dairy")]

# Build a price time series
ts_df = ts_df.sort_values("date")
ts = ts_df.set_index("date")["fmap_price"]

# -----------------------
# 1. Check stationarity
# -----------------------
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# -----------------------
# 2. Fit SARIMA
# Common starting point: SARIMA(1,1,1)(1,1,1,12)
# -----------------------
model = SARIMAX(
    ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print(results.summary())

# -----------------------
# 3. Forecast next 12 months
# -----------------------
forecast_steps = 12

forecast = results.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# -----------------------
# 4. Plot
# -----------------------
plt.figure(figsize=(10,6))
plt.plot(ts, label="Actual")
plt.plot(forecast_df["mean"], label="Forecast")
plt.fill_between(
    forecast_df.index,
    forecast_df["mean_ci_lower"],
    forecast_df["mean_ci_upper"],
    alpha=0.2
)
plt.title("SARIMA Forecast: Dairy Prices in the West")
plt.legend()
plt.show()
