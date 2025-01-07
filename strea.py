# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from statsmodels.tsa.statespace.sarimax import SARIMAX
# # # from pmdarima import auto_arima
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import mean_squared_error
# # # from xgboost import XGBRegressor
# # # from llm import converting_date
# # # import streamlit as st
# # # from sklearn.model_selection import train_test_split, GridSearchCV

# # # # Helper Functions
# # # def preprocess_data(file_path):
# # #     data = pd.read_csv(file_path)
# # #     data = converting_date(data)
# # #     data.set_index('Date', inplace=True)
# # #     return data

# # # def sarima_model(data):
# # #     sarimax_model = SARIMAX(data['value'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
# # #     sarimax_fit = sarimax_model.fit(disp=False)
# # #     return sarimax_fit

# # # def autoarima_model(data):
# # #     auto_arima_model = auto_arima(data['value'], seasonal=True, m=12, trace=True, suppress_warnings=True)
# # #     return auto_arima_model

# # # def tune_xgboost(X_train, y_train):
# # #     """
# # #     Tune XGBoost hyperparameters using GridSearchCV.
# # #     """
# # #     param_grid = {
# # #         'n_estimators': [50, 100, 200],
# # #         'max_depth': [3, 5, 7],
# # #         'learning_rate': [0.01, 0.1, 0.2],
# # #         'subsample': [0.6, 0.8, 1.0],
# # #         'colsample_bytree': [0.6, 0.8, 1.0],
# # #     }

# # #     xgb_model = XGBRegressor(random_state=42)

# # #     grid_search = GridSearchCV(
# # #         estimator=xgb_model,
# # #         param_grid=param_grid,
# # #         scoring='neg_mean_squared_error',
# # #         cv=3,
# # #         verbose=1
# # #     )

# # #     grid_search.fit(X_train, y_train)
# # #     return grid_search.best_estimator_, grid_search.best_params_

# # # def hybrid_model(data, actual_2025_values, model_type):
# # #     if model_type == "SARIMA":
# # #         model = sarima_model(data)
# # #         forecast = model.get_forecast(steps=12).predicted_mean
# # #         fitted_values = model.fittedvalues
# # #     elif model_type == "AutoARIMA":
# # #         model = autoarima_model(data)
# # #         forecast = model.predict(n_periods=12)
# # #         fitted_values = model.predict_in_sample()

# # #     data['fitted_values'] = fitted_values
# # #     data['residual'] = data['value'] - data['fitted_values']

# # #     data['month'] = data.index.month
# # #     data['year'] = data.index.year
# # #     data['lag_1'] = data['value'].shift(1)
# # #     data['lag_2'] = data['value'].shift(2)
# # #     ml_data = data.dropna()

# # #     X = ml_data[['month', 'year', 'lag_1', 'lag_2']]
# # #     y = ml_data['residual']

# # #     # Train-Test Split
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # #     # Hyperparameter Tuning
# # #     st.write("Tuning XGBoost hyperparameters. This may take some time...")
# # #     best_xgb_model, best_params = tune_xgboost(X_train, y_train)
# # #     st.write(f"Best Parameters: {best_params}")

# # #     # Train XGBoost with the best parameters
# # #     best_xgb_model.fit(X_train, y_train)

# # #     # Predict Residuals for Future Dates
# # #     future_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]
# # #     future_months = future_dates.month
# # #     future_years = future_dates.year

# # #     xgb_input = pd.DataFrame({
# # #         'month': future_months,
# # #         'year': future_years,
# # #         'lag_1': [data['value'].iloc[-1]] * 12,
# # #         'lag_2': [data['value'].iloc[-2]] * 12
# # #     })
# # #     xgb_residual_forecast = best_xgb_model.predict(xgb_input)

# # #     # Hybrid Forecast
# # #     hybrid_forecast = forecast + xgb_residual_forecast

# # #     # Create Forecast DataFrame
# # #     forecast_df = pd.DataFrame({
# # #         'Date': future_dates,
# # #         'SARIMA/AutoARIMA Forecast': forecast,
# # #         'XGBoost Residual Forecast': xgb_residual_forecast,
# # #         'Hybrid Forecast': hybrid_forecast,
# # #         'Actual Values': actual_2025_values
# # #     })

# # #     # Calculate MAPE
# # #     forecast_df['Error'] = abs(forecast_df['Hybrid Forecast'] - forecast_df['Actual Values'])
# # #     forecast_df['APE'] = (forecast_df['Error'] / forecast_df['Actual Values']) * 100
# # #     mape = forecast_df['APE'].mean()

# # #     return forecast_df, mape, data

# # # # Streamlit App
# # # st.title("Forecasting Application with Hyperparameter Tuning")
# # # st.write("Upload a CSV file to forecast values for 2025.")

# # # uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# # # if uploaded_file is not None:
# # #     try:
# # #         data = preprocess_data(uploaded_file)

# # #         # Get unique IDs
# # #         unique_ids = data['ID'].unique()

# # #         # Select ID
# # #         selected_id = st.selectbox("Select ID to forecast:", unique_ids)

# # #         # Select Model Type
# # #         model_type = st.radio("Select Model Type:", ("SARIMA", "AutoARIMA"))

# # #         # Filter data by selected ID
# # #         filtered_data = data[data['ID'] == selected_id]

# # #         # Extract actual 2025 values
# # #         actual_2025_values = filtered_data.loc[filtered_data.index.year == 2025, 'value'].values

# # #         if len(actual_2025_values) == 12:
# # #             forecast_df, mape, historical_data = hybrid_model(filtered_data, actual_2025_values, model_type=model_type)

# # #             st.write("### Forecast Results")
# # #             st.write(forecast_df)

# # #             # Display MAPE
# # #             st.write(f"### Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# # #             # Plotting
# # #             plt.figure(figsize=(10, 6))
# # #             plt.plot(forecast_df['Date'], forecast_df['Actual Values'], label="Actual Values", marker="o")
# # #             plt.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", marker="o")
# # #             plt.title(f"Actual vs Forecasted Values for ID: {selected_id} using {model_type}")
# # #             plt.xlabel("Date")
# # #             plt.ylabel("Value")
# # #             plt.legend()
# # #             plt.grid()
# # #             st.pyplot(plt)

# # #         else:
# # #             st.error("The file must contain complete data for 2025 (12 months of actual values).")

# # #     except Exception as e:
# # #         st.error(f"An error occurred: {e}")


# # import os
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from statsmodels.tsa.statespace.sarimax import SARIMAX
# # from pmdarima import auto_arima
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.metrics import mean_squared_error
# # from xgboost import XGBRegressor
# # import streamlit as st

# # # Helper Functions
# # def preprocess_data(file_path):
# #     data = pd.read_csv(file_path)
# #     data['Date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
# #     data.set_index('Date', inplace=True)
# #     return data

# # def sarima_model(data):
# #     sarimax_model = SARIMAX(data['value'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
# #     sarimax_fit = sarimax_model.fit(disp=False)
# #     return sarimax_fit

# # def autoarima_model(data):
# #     auto_arima_model = auto_arima(data['value'], seasonal=True, m=12, trace=True, suppress_warnings=True)
# #     return auto_arima_model

# # def tune_xgboost(X_train, y_train):
# #     """
# #     Tune XGBoost hyperparameters using GridSearchCV.
# #     """
# #     param_grid = {
# #         'n_estimators': [50, 100, 200],
# #         'max_depth': [3, 5, 7],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'subsample': [0.6, 0.8, 1.0],
# #         'colsample_bytree': [0.6, 0.8, 1.0],
# #     }

# #     xgb_model = XGBRegressor(random_state=42)

# #     grid_search = GridSearchCV(
# #         estimator=xgb_model,
# #         param_grid=param_grid,
# #         scoring='neg_mean_squared_error',
# #         cv=3,
# #         verbose=1
# #     )

# #     grid_search.fit(X_train, y_train)
# #     return grid_search.best_estimator_, grid_search.best_params_

# # def identify_recurring_patterns(data):
# #     """
# #     Analyze recurring patterns for any ID and highlight consistent declines or patterns
# #     over periods of 1, 2, or 3 months every year based on historical data (till 2024).
# #     """
# #     # Filter historical data up to 2024
# #     historical_data = data[data.index.year <= 2024]
# #     recurring_patterns = {}

# #     for unique_id in historical_data['ID'].unique():
# #         id_data = historical_data[historical_data['ID'] == unique_id].copy()
# #         id_data['month'] = id_data.index.month
# #         id_data['year'] = id_data.index.year

# #         patterns = []

# #         # Check for declines or patterns across 1, 2, or 3 months
# #         for period in [1, 2, 3]:
# #             grouped = id_data.groupby('month')['value'].mean()
# #             std_dev = id_data.groupby('month')['value'].std()

# #             for month in range(1, 13 - period + 1):
# #                 next_months = [month + i for i in range(period)]
# #                 if all(m in grouped.index for m in next_months):
# #                     avg_values = grouped.loc[next_months].values
# #                     if np.all(np.diff(avg_values) < 0):  # Check for decline
# #                         patterns.append((month, period, "Decline"))
# #                     elif np.all(np.diff(avg_values) > 0):  # Check for growth
# #                         patterns.append((month, period, "Growth"))

# #         recurring_patterns[unique_id] = patterns

# #     return recurring_patterns


# # def hybrid_model(data, actual_2025_values, model_type):
# #     if model_type == "SARIMA":
# #         model = sarima_model(data)
# #         forecast = model.get_forecast(steps=12).predicted_mean
# #         fitted_values = model.fittedvalues
# #     elif model_type == "AutoARIMA":
# #         model = autoarima_model(data)
# #         forecast = model.predict(n_periods=12)
# #         fitted_values = model.predict_in_sample()

# #     data['fitted_values'] = fitted_values
# #     data['residual'] = data['value'] - data['fitted_values']

# #     data['month'] = data.index.month
# #     data['year'] = data.index.year
# #     data['lag_1'] = data['value'].shift(1)
# #     data['lag_2'] = data['value'].shift(2)
# #     ml_data = data.dropna()

# #     X = ml_data[['month', 'year', 'lag_1', 'lag_2']]
# #     y = ml_data['residual']

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Hyperparameter Tuning
# #     st.write("Tuning XGBoost hyperparameters. This may take some time...")
# #     best_xgb_model, best_params = tune_xgboost(X_train, y_train)
# #     st.write(f"Best Parameters: {best_params}")

# #     # Train XGBoost with the best parameters
# #     best_xgb_model.fit(X_train, y_train)

# #     # Predict Residuals for Future Dates
# #     future_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]
# #     future_months = future_dates.month
# #     future_years = future_dates.year

# #     xgb_input = pd.DataFrame({
# #         'month': future_months,
# #         'year': future_years,
# #         'lag_1': [data['value'].iloc[-1]] * 12,
# #         'lag_2': [data['value'].iloc[-2]] * 12
# #     })
# #     xgb_residual_forecast = best_xgb_model.predict(xgb_input)

# #     # Hybrid Forecast
# #     hybrid_forecast = forecast + xgb_residual_forecast

# #     # Create Forecast DataFrame
# #     forecast_df = pd.DataFrame({
# #         'Date': future_dates,
# #         'SARIMA/AutoARIMA Forecast': forecast,
# #         'XGBoost Residual Forecast': xgb_residual_forecast,
# #         'Hybrid Forecast': hybrid_forecast,
# #         'Actual Values': actual_2025_values
# #     })

# #     # Calculate MAPE
# #     forecast_df['Error'] = abs(forecast_df['Hybrid Forecast'] - forecast_df['Actual Values'])
# #     forecast_df['APE'] = (forecast_df['Error'] / forecast_df['Actual Values']) * 100
# #     mape = forecast_df['APE'].mean()

# #     return forecast_df, mape, data

# # # Streamlit App
# # st.title("Forecasting Application with Hyperparameter Tuning and Pattern Analysis")
# # st.write("Upload a CSV file to forecast values for 2025.")

# # uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# # if uploaded_file is not None:
# #     try:
# #         data = preprocess_data(uploaded_file)

# #         # Get unique IDs
# #         unique_ids = data['ID'].unique()

# #         # Select ID
# #         selected_id = st.selectbox("Select ID to forecast:", unique_ids)

# #         # Select Model Type
# #         model_type = st.radio("Select Model Type:", ("SARIMA", "AutoARIMA"))

# #         # Filter data by selected ID
# #         filtered_data = data[data['ID'] == selected_id]

# #         # Extract actual 2025 values
# #         actual_2025_values = filtered_data.loc[filtered_data.index.year == 2025, 'value'].values

# #         if len(actual_2025_values) == 12:
# #             forecast_df, mape, historical_data = hybrid_model(filtered_data, actual_2025_values, model_type=model_type)

# #             st.write("### Forecast Results")
# #             st.write(forecast_df)

# #             # Display MAPE
# #             st.write(f"### Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# #             # Plotting
# #             plt.figure(figsize=(10, 6))
# #             plt.plot(forecast_df['Date'], forecast_df['Actual Values'], label="Actual Values", marker="o")
# #             plt.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", marker="o")
# #             plt.title(f"Actual vs Forecasted Values for ID: {selected_id} using {model_type}")
# #             plt.xlabel("Date")
# #             plt.ylabel("Value")
# #             plt.legend()
# #             plt.grid()
# #             st.pyplot(plt)

# #             # Analyze recurring patterns
# #             st.write("### Recurring Patterns")
# #             recurring_patterns = identify_recurring_patterns(data)
# #             if recurring_patterns[selected_id]:
# #                 st.write(f"Patterns for ID {selected_id}: {recurring_patterns[selected_id]}")
# #             else:
# #                 st.write("No recurring patterns detected.")

# #         else:
# #             st.error("The file must contain complete data for 2025 (12 months of actual values).")

# #     except Exception as e:
# #         st.error(f"An error occurred: {e}")

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pmdarima import auto_arima
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error
# from llm import converting_date
# from xgboost import XGBRegressor
# import streamlit as st

# # Helper Functions
# def preprocess_data(file_path):
#     data = pd.read_csv(file_path)
#     data = converting_date(data)
#     data.set_index('Date', inplace=True)
#     return data

# def sarima_model(data):
#     sarimax_model = SARIMAX(data['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 6))
#     sarimax_fit = sarimax_model.fit(disp=False)
#     return sarimax_fit

# def autoarima_model(data):
#     auto_arima_model = auto_arima(data['value'], seasonal=True, m=12, trace=True, suppress_warnings=True)
#     return auto_arima_model

# def tune_xgboost(X_train, y_train):
#     """
#     Tune XGBoost hyperparameters using GridSearchCV.
#     """
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#     }

#     xgb_model = XGBRegressor(random_state=42)

#     grid_search = GridSearchCV(
#         estimator=xgb_model,
#         param_grid=param_grid,
#         scoring='neg_mean_squared_error',
#         cv=3,
#         verbose=1
#     )

#     grid_search.fit(X_train, y_train)
#     return grid_search.best_estimator_, grid_search.best_params_

# def hybrid_model(data, actual_2025_values, model_type):
#     if model_type == "SARIMA":
#         model = sarima_model(data)
#         forecast = model.get_forecast(steps=12).predicted_mean
#         fitted_values = model.fittedvalues
#     elif model_type == "AutoARIMA":
#         model = autoarima_model(data)
#         forecast = model.predict(n_periods=12)
#         fitted_values = model.predict_in_sample()

#     data['fitted_values'] = fitted_values
#     data['residual'] = data['value'] - data['fitted_values']

#     data['month'] = data.index.month
#     data['year'] = data.index.year
#     data['lag_1'] = data['value'].shift(1)
#     data['lag_2'] = data['value'].shift(2)
#     ml_data = data.dropna()

#     X = ml_data[['month', 'year', 'lag_1', 'lag_2']]
#     y = ml_data['residual']

#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Hyperparameter Tuning
#     st.write("Tuning XGBoost hyperparameters. This may take some time...")
#     best_xgb_model, best_params = tune_xgboost(X_train, y_train)
#     st.write(f"Best Parameters: {best_params}")

#     # Train XGBoost with the best parameters
#     best_xgb_model.fit(X_train, y_train)

#     # Predict Residuals for Future Dates
#     future_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]
#     future_months = future_dates.month
#     future_years = future_dates.year

#     xgb_input = pd.DataFrame({
#         'month': future_months,
#         'year': future_years,
#         'lag_1': [data['value'].iloc[-1]] * 12,
#         'lag_2': [data['value'].iloc[-2]] * 12
#     })
#     xgb_residual_forecast = best_xgb_model.predict(xgb_input)

#     # Hybrid Forecast
#     hybrid_forecast = forecast + xgb_residual_forecast

#     # Create Forecast DataFrame
#     forecast_df = pd.DataFrame({
#         'Date': future_dates,
#         'SARIMA/AutoARIMA Forecast': forecast,
#         'XGBoost Residual Forecast': xgb_residual_forecast,
#         'Hybrid Forecast': hybrid_forecast,
#         'Actual Values': actual_2025_values
#     })

#     # Calculate MAPE
#     forecast_df['Error'] = abs(forecast_df['Hybrid Forecast'] - forecast_df['Actual Values'])
#     forecast_df['APE'] = (forecast_df['Error'] / forecast_df['Actual Values']) * 100
#     mape = forecast_df['APE'].mean()

#     return forecast_df, mape, data

# # Streamlit App
# st.title("Forecasting Application with Hyperparameter Tuning and Pattern Analysis")
# st.write("Upload a CSV file to forecast values for 2025.")

# uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# if uploaded_file is not None:
#     try:
#         data = preprocess_data(uploaded_file)

#         # Get unique IDs
#         unique_ids = data['ID'].unique()

#         # Select ID
#         selected_id = st.selectbox("Select ID to forecast:", unique_ids)

#         # Select Model Type
#         model_type = st.radio("Select Model Type:", ("SARIMA", "AutoARIMA"))

#         # Filter data by selected ID
#         filtered_data = data[data['ID'] == selected_id]

#         # Extract actual 2025 values
#         actual_2025_values = filtered_data.loc[filtered_data.index.year == 2025, 'value'].values

#         if len(actual_2025_values) == 12:
#             forecast_df, mape, historical_data = hybrid_model(filtered_data, actual_2025_values, model_type=model_type)

#             st.write("### Forecast Results")
#             st.write(forecast_df)

#             # Display MAPE
#             st.write(f"### Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

#             # Plotting
#             plt.figure(figsize=(10, 6))
#             plt.plot(forecast_df['Date'], forecast_df['Actual Values'], label="Actual Values", marker="o")
#             plt.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", marker="o")
#             plt.title(f"Actual vs Forecasted Values for ID: {selected_id} using {model_type}")
#             plt.xlabel("Date")
#             plt.ylabel("Value")
#             plt.legend()
#             plt.grid()
#             st.pyplot(plt)

#             # Analyze recurring patterns
#             st.write("### Consistent Patterns")
#             consistent_patterns = identify_consistent_patterns(data)
#             if consistent_patterns[selected_id]:
#                 st.write(f"Consistent Patterns for ID {selected_id}: {consistent_patterns[selected_id]}")

#                 # Visualize patterns
#                 id_data = data[data['ID'] == selected_id]
#                 visualize_consistent_patterns(consistent_patterns[selected_id], id_data, selected_id)

#             else:
#                 st.write("No consistent patterns detected.")

#         else:
#             st.error("The file must contain complete data for 2025 (12 months of actual values).")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import streamlit as st

# Helper Functions
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data.set_index('Date', inplace=True)
    return data

def sarima_model(data):
    sarimax_model = SARIMAX(data['value'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit

def autoarima_model(data):
    auto_arima_model = auto_arima(data['value'], seasonal=True, m=12, trace=True, suppress_warnings=True)
    return auto_arima_model

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    xgb_model = XGBRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def identify_consistent_patterns(data):
    """
    Identify patterns that consistently occur every year during the same period.
    """
    # Filter historical data up to 2024
    historical_data = data[data.index.year <= 2024]
    consistent_patterns = {}

    for unique_id in historical_data['ID'].unique():
        id_data = historical_data[historical_data['ID'] == unique_id].copy()
        id_data['month'] = id_data.index.month
        id_data['year'] = id_data.index.year

        patterns = []

        # Check for consistent patterns over 1, 2, or 3 months
        for period in [1, 2, 3]:
            for start_month in range(1, 13 - period + 1):
                months = [(start_month + i - 1) % 12 + 1 for i in range(period)]

                yearly_trends = []
                for year in id_data['year'].unique():
                    values = id_data[(id_data['year'] == year) & (id_data['month'].isin(months))]['value']
                    if len(values) == period:
                        yearly_trends.append(values.values)

                # Check if the trend is consistent across all years
                if len(yearly_trends) == len(id_data['year'].unique()):
                    trend_diffs = [np.diff(year) for year in yearly_trends]
                    if all(np.all(diff < 0) for diff in trend_diffs):
                        patterns.append((start_month, period, "Decline"))
                    elif all(np.all(diff > 0) for diff in trend_diffs):
                        patterns.append((start_month, period, "Growth"))

        consistent_patterns[unique_id] = patterns

    return consistent_patterns

def visualize_consistent_patterns(patterns, id_data, selected_id):
    """
    Visualize consistent patterns on historical data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(id_data.index, id_data['value'], label='Value', marker='o')

    for pattern in patterns:
        start_month, period, trend_type = pattern
        label = f"{trend_type} ({start_month}-{start_month + period - 1})"

        for year in id_data['year'].unique():
            months = [(start_month + i - 1) % 12 + 1 for i in range(period)]
            indices = id_data[(id_data['year'] == year) & (id_data['month'].isin(months))].index
            plt.plot(indices, id_data.loc[indices, 'value'], marker='o', label=label)

    plt.title(f"Consistent Patterns for ID: {selected_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def hybrid_model(data, actual_2025_values, model_type):
    if model_type == "SARIMA":
        model = sarima_model(data[data.index.year < 2025])  # Train till 2024
        forecast = model.get_forecast(steps=12).predicted_mean
        fitted_values = model.fittedvalues
    elif model_type == "AutoARIMA":
        model = autoarima_model(data[data.index.year < 2025])  # Train till 2024
        forecast = model.predict(n_periods=12)
        fitted_values = model.predict_in_sample()

    data['fitted_values'] = fitted_values
    data['residual'] = data['value'] - data['fitted_values']

    data['month'] = data.index.month
    data['year'] = data.index.year
    data['lag_1'] = data['value'].shift(1)
    data['lag_2'] = data['value'].shift(2)
    ml_data = data[data.index.year < 2025].dropna()

    X = ml_data[['month', 'year', 'lag_1', 'lag_2']]
    y = ml_data['residual']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    st.write("Tuning XGBoost hyperparameters. This may take some time...")
    best_xgb_model, best_params = tune_xgboost(X_train, y_train)
    st.write(f"Best Parameters: {best_params}")

    # Train XGBoost with the best parameters
    best_xgb_model.fit(X_train, y_train)

    # Predict Residuals for Future Dates
    future_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]
    future_months = future_dates.month
    future_years = future_dates.year

    xgb_input = pd.DataFrame({
        'month': future_months,
        'year': future_years,
        'lag_1': [data['value'].iloc[-1]] * 12,
        'lag_2': [data['value'].iloc[-2]] * 12
    })
    xgb_residual_forecast = best_xgb_model.predict(xgb_input)

    # Hybrid Forecast
    hybrid_forecast = forecast + xgb_residual_forecast

    # Create Forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'SARIMA/AutoARIMA Forecast': forecast,
        'XGBoost Residual Forecast': xgb_residual_forecast,
        'Hybrid Forecast': hybrid_forecast,
        'Actual Values': actual_2025_values
    })

    # Calculate MAPE
    forecast_df['Error'] = abs(forecast_df['Hybrid Forecast'] - forecast_df['Actual Values'])
    forecast_df['APE'] = (forecast_df['Error'] / forecast_df['Actual Values']) * 100
    mape = forecast_df['APE'].mean()

    return forecast_df, mape, data

# Streamlit App
st.title("Forecasting Application with Hyperparameter Tuning and Pattern Analysis")
st.write("Upload a CSV file to forecast values for 2025.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = preprocess_data(uploaded_file)

        # Get unique IDs
        unique_ids = data['ID'].unique()

        # Select ID
        selected_id = st.selectbox("Select ID to forecast:", unique_ids)

        # Select Model Type
        model_type = st.radio("Select Model Type:", ("SARIMA", "AutoARIMA"))

        # Filter data by selected ID
        filtered_data = data[data['ID'] == selected_id]

        # Extract actual 2025 values
        actual_2025_values = filtered_data.loc[filtered_data.index.year == 2025, 'value'].values

        if len(actual_2025_values) == 12:
            forecast_df, mape, historical_data = hybrid_model(filtered_data, actual_2025_values, model_type=model_type)

            st.write("### Forecast Results")
            st.write(forecast_df)

            # Display MAPE
            st.write(f"### Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(forecast_df['Date'], forecast_df['Actual Values'], label="Actual Values", marker="o")
            plt.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", marker="o")
            plt.title(f"Actual vs Forecasted Values for ID: {selected_id} using {model_type}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            st.pyplot(plt)

            # Analyze recurring patterns
            st.write("### Consistent Patterns")
            consistent_patterns = identify_consistent_patterns(data)
            if consistent_patterns[selected_id]:
                st.write(f"Consistent Patterns for ID {selected_id}: {consistent_patterns[selected_id]}")

                # Visualize patterns
                id_data = data[data['ID'] == selected_id]
                visualize_consistent_patterns(consistent_patterns[selected_id], id_data, selected_id)

            else:
                st.write("No consistent patterns detected.")

        else:
            st.error("The file must contain complete data for 2025 (12 months of actual values).")

    except Exception as e:
        st.error(f"An error occurred: {e}")
