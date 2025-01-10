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
        'subsample': [0.8, 1.0]
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

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Streamlit App
st.title("Forecasting Application with Hybrid Model and Correlated IDs")
st.write("Upload a CSV file to forecast values for 2025.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = preprocess_data(uploaded_file)

        # Prepare the data
        data['time'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

        # Dropdown to select the target ID
        unique_ids = data['ID'].unique()
        target_id = st.selectbox("Select Target ID for Forecasting:", unique_ids)

        # Select correlated IDs
        pivoted_data = data.pivot_table(index=['year', 'month'], columns='ID', values='value')
        correlation_matrix = pivoted_data.corr()
        highly_correlated_ids = correlation_matrix[target_id][correlation_matrix[target_id] >= 0.95].index.tolist()
        if target_id in highly_correlated_ids:
            highly_correlated_ids.remove(target_id)
        correlated_ids = highly_correlated_ids
        all_ids = [target_id] + correlated_ids
        pivot_data = data.pivot(index='time', columns='ID', values='value')[all_ids]

        # Feature Engineering
        for col in pivot_data.columns:
            pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1)
            pivot_data[f'{col}_lag2'] = pivot_data[col].shift(2)
            pivot_data[f'{col}_rolling_mean3'] = pivot_data[col].rolling(window=3).mean()
            pivot_data[f'{col}_exp_smooth'] = pivot_data[col].ewm(span=3, adjust=False).mean()

        pivot_data.dropna(inplace=True)

        # Separate training and testing data
        train_data = pivot_data[pivot_data.index.year < 2025]
        test_data = pivot_data[pivot_data.index.year == 2025]

        X_train = train_data.drop(columns=[target_id])
        y_train = train_data[target_id]
        X_test = test_data.drop(columns=[target_id])
        y_test = test_data[target_id]

        # AutoARIMA for baseline prediction
        auto_arima_model = auto_arima(y_train, seasonal=True, m=6, trace=True, suppress_warnings=True)
        y_pred_arima_train = auto_arima_model.predict_in_sample()
        y_pred_arima_test = auto_arima_model.predict(n_periods=len(y_test))

        # Calculate residuals for training XGBoost
        residuals_train = y_train - y_pred_arima_train

        # XGBoost Training
        xgb = XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, residuals_train)

        # Best model and prediction
        best_xgb = grid_search.best_estimator_
        y_pred_xgb_test = best_xgb.predict(X_test)

        # Combine predictions for final hybrid model
        hybrid_predictions = y_pred_arima_test + y_pred_xgb_test

        # Evaluate the models
        arima_rmse = mean_squared_error(y_test, y_pred_arima_test, squared=False)
        xgb_rmse = mean_squared_error(y_test, y_pred_xgb_test, squared=False)
        hybrid_rmse = mean_squared_error(y_test, hybrid_predictions, squared=False)

        arima_mape = calculate_mape(y_test, y_pred_arima_test)
        xgb_mape = calculate_mape(y_test, y_pred_xgb_test)
        hybrid_mape = calculate_mape(y_test, hybrid_predictions)

        # Display Results
        st.write(f"### RMSE (AutoARIMA): {arima_rmse:.2f}")
        st.write(f"### RMSE (XGBoost): {xgb_rmse:.2f}")
        st.write(f"### RMSE (Hybrid Model): {hybrid_rmse:.2f}")
        st.write(f"### MAPE (AutoARIMA): {arima_mape:.2f}%")
        st.write(f"### MAPE (XGBoost): {xgb_mape:.2f}%")
        st.write(f"### MAPE (Hybrid Model): {hybrid_mape:.2f}%")

        # Plotting Results
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index, y_test, label="Actual Values", marker="o")
        plt.plot(test_data.index, hybrid_predictions, label="Hybrid Forecast", marker="o")
        plt.title(f"Actual vs Forecasted Values for ID: {target_id}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
