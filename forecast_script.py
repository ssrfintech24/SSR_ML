import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

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
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)
    data.set_index(date_column, inplace=True)

    # Function to check stationarity using ADF test
    def test_stationarity(timeseries):
        result = adfuller(timeseries, autolag='AIC')
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        if result[1] <= 0.05:
            print("The series is stationary.")
            return True, result[0]
        else:
            print("The series is not stationary.")
            return False, result[0]

    # Function to perform seasonal decomposition
    def perform_decomposition(timeseries, frequency):
        decomposition = seasonal_decompose(timeseries, model='additive', period=frequency)
        decomposition.plot()
        plt.show()
        return decomposition

    # Function to plot ACF and PACF
    def plot_acf_pacf_dynamic(timeseries):
        lags = min(len(timeseries) // 2 - 1, 40)  # Dynamic lag adjustment
        lag_acf = acf(timeseries, nlags=lags)
        lag_pacf = pacf(timeseries, nlags=lags, method='ols')

        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(timeseries)), linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(timeseries)), linestyle='--', color='gray')
        plt.title('Autocorrelation Function')

        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(timeseries)), linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(timeseries)), linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')

        plt.tight_layout()
        plt.show()

    # Function to apply differencing
    def apply_differencing_refined(timeseries):
        differenced_series = timeseries.diff().dropna()
        print("\nTesting stationarity after differencing:")
        is_stationary, adf_stat = test_stationarity(differenced_series)
        plt.figure(figsize=(10, 6))
        plt.plot(differenced_series)
        plt.title('Differenced Series')
        plt.show()
        return differenced_series, is_stationary, adf_stat

    # Function to apply log transformation
    def apply_log_transformation(timeseries):
        log_transformed = np.log(timeseries.replace(0, np.nan)).dropna()
        print("\nTesting stationarity after log transformation:")
        is_stationary, adf_stat = test_stationarity(log_transformed)
        plt.figure(figsize=(10, 6))
        plt.plot(log_transformed)
        plt.title('Log Transformed Series')
        plt.show()
        return log_transformed, is_stationary, adf_stat

    # Function for time series analysis
    def time_series_analysis(data, value_column, frequency):
        timeseries = data[value_column]

        plt.figure(figsize=(10, 6))
        plt.plot(timeseries)
        plt.title('Time Series Data')
        plt.show()

        print("\nStationarity Test:")
        is_stationary, _ = test_stationarity(timeseries)

        if not is_stationary:
            print("\nAddressing Non-Stationarity:")
            differenced_series, diff_stationary, diff_adf_stat = apply_differencing_refined(timeseries)
            log_transformed_series, log_stationary, log_adf_stat = apply_log_transformation(timeseries)

            if diff_stationary and log_stationary:
                if diff_adf_stat < log_adf_stat:
                    print("Differencing showed a stronger impact. Selecting differencing.")
                    timeseries = differenced_series
                else:
                    print("Log transformation showed a stronger impact. Selecting log transformation.")
                    timeseries = log_transformed_series
            elif diff_stationary:
                print("Differencing achieved stationarity.")
                timeseries = differenced_series
            elif log_stationary:
                print("Log transformation achieved stationarity.")
                timeseries = log_transformed_series
            else:
                print("Neither method achieved perfect stationarity. Proceeding with differencing.")
                timeseries = differenced_series

        print("\nSeasonal Decomposition:")
        perform_decomposition(timeseries, frequency)

        print("\nACF and PACF:")
        plot_acf_pacf_dynamic(timeseries)

        return timeseries

    # Analyze the time series
    processed_series = time_series_analysis(data, value_column=target_column, frequency=frequency)
    # Determine AR and MA orders for SARIMA
    def find_p_q_acf_pacf(timeseries, lags=20):
        lag_acf = acf(timeseries, nlags=lags)
        lag_pacf = pacf(timeseries, nlags=lags, method='ols')
        p = next((i for i, x in enumerate(lag_pacf) if abs(x) < 1.96 / np.sqrt(len(timeseries))), 1)
        q = next((i for i, x in enumerate(lag_acf) if abs(x) < 1.96 / np.sqrt(len(timeseries))), 1)
        return p, q

    p, q = find_p_q_acf_pacf(processed_series)
    P, Q = find_p_q_acf_pacf(processed_series, lags=frequency)

    # Fit SARIMA model
    sarima_model = SARIMAX(data[target_column], order=(p, 1, q), seasonal_order=(P, 1, Q, frequency))
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fitted.get_forecast(steps=steps).predicted_mean

    # Fit Holt-Winters model
    holt_model = ExponentialSmoothing(
        data[target_column],
        seasonal_periods=frequency,
        trend='add',
        seasonal='add',
        damped_trend=True
    ).fit()
    holt_forecast = holt_model.forecast(steps=steps)

    # Calculate RMSE for both models
    sarima_rmse = sqrt(mean_squared_error(data[target_column][-steps:], sarima_forecast[:steps]))
    holt_rmse = sqrt(mean_squared_error(data[target_column][-steps:], holt_forecast[:steps]))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(data[target_column], label='Original Data', color='blue')
    plt.plot(sarima_forecast, label=f'SARIMA Forecast (RMSE={sarima_rmse:.2f})', color='purple')
    plt.plot(holt_forecast, label=f'Holt-Winters Forecast (RMSE={holt_rmse:.2f})', color='green')
    plt.title('Model Comparison')
    plt.legend()
    plt.show()

    # Return results
    return {
        "SARIMA_RMSE": sarima_rmse,
        "Holt_Winters_RMSE": holt_rmse,
        "SARIMA_Forecast": sarima_forecast,
        "Holt_Winters_Forecast": holt_forecast
    }