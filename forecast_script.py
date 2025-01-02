import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def time_series_forecast(data, date_column, target_column, frequency, steps):
    """
    Perform time series analysis and forecasting using SARIMA and Holt-Winters models.

    Parameters:
        data (pd.DataFrame): Input dataset with date and target columns.
        date_column (str): Name of the date column.
        target_column (str): Name of the target column.
        frequency (int): Seasonal frequency (e.g., 12 for monthly data).
        steps (int): Number of steps to forecast.

    Returns:
        dict: Dictionary containing RMSE values and plots of forecasts.
    """
    # Preprocess the dataset
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data = data.dropna(subset=[date_column])
    data = data.sort_values(by=date_column)
    data.set_index(date_column, inplace=True)

    # Function to check stationarity using ADF test
    def test_stationarity(timeseries):
        result = adfuller(timeseries.dropna(), autolag='AIC')
        return result[1] <= 0.05

    # Perform seasonal decomposition
    def perform_decomposition(timeseries, frequency):
        decomposition = seasonal_decompose(timeseries, model='additive', period=frequency)
        decomposition.plot()
        plt.show()

    # Fit SARIMA model
    def fit_sarima(data, p, d, q, P, D, Q, frequency):
        sarima_model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, frequency))
        return sarima_model.fit(disp=False)

    # Fit Holt-Winters model
    def fit_holt_winters(data, frequency):
        return ExponentialSmoothing(
            data,
            seasonal_periods=frequency,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit()

    # Perform initial analysis
    timeseries = data[target_column]
    stationary = test_stationarity(timeseries)

    # Handle non-stationarity
    if not stationary:
        timeseries = timeseries.diff().dropna()

    # Perform decomposition
    perform_decomposition(timeseries, frequency)

    # Fit models
    sarima_model = fit_sarima(data[target_column], 1, 1, 1, 1, 1, 1, frequency)
    sarima_forecast = sarima_model.get_forecast(steps=steps).predicted_mean

    holt_model = fit_holt_winters(data[target_column], frequency)
    holt_forecast = holt_model.forecast(steps=steps)

    # Calculate RMSE for both models
    actual = data[target_column][-steps:]
    if len(actual) < steps:
        sarima_rmse = np.nan
        holt_rmse = np.nan
    else:
        sarima_rmse = sqrt(mean_squared_error(actual, sarima_forecast[:steps]))
        holt_rmse = sqrt(mean_squared_error(actual, holt_forecast[:steps]))

    return {
        "SARIMA_RMSE": sarima_rmse,
        "Holt_Winters_RMSE": holt_rmse,
        "SARIMA_Forecast": sarima_forecast,
        "Holt_Winters_Forecast": holt_forecast
    }
