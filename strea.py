import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def hybrid_model(data, actual_2025_values):
    # AutoARIMA model
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
        'AutoARIMA Forecast': forecast,
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

        # Filter data by selected ID
        filtered_data = data[data['ID'] == selected_id]

        # Extract actual 2025 values
        actual_2025_values = filtered_data.loc[filtered_data.index.year == 2025, 'value'].values

        if len(actual_2025_values) == 12:
            forecast_df, mape, historical_data = hybrid_model(filtered_data, actual_2025_values)

            st.write("### Forecast Results")
            st.write(forecast_df)

            # Display MAPE
            st.write(f"### Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(forecast_df['Date'], forecast_df['Actual Values'], label="Actual Values", marker="o")
            plt.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", marker="o")
            plt.title(f"Actual vs Forecasted Values for ID: {selected_id}")
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
