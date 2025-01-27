import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\archive (7)\ITC.csv")
print("First few rows of the dataset:")
print(data.head())  # Debug: Check the first few rows of the dataset

# Strip any leading/trailing spaces from column names
data.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure the 'Date' column is present and formatted correctly
if 'Date' not in data.columns:
    raise KeyError("The 'Date' column is missing from the dataset.")

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Handle any invalid date formats
data.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
data = data.sort_values('Date')  # Sort data by date
print("Date column properly formatted.")

# Check for duplicate dates and remove them
duplicates = data.duplicated(subset=['Date']).sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate dates. Removing them.")
    data = data.drop_duplicates(subset=['Date'])

# Ensure the 'Close' column is present
if 'Close' not in data.columns:
    raise KeyError("The 'Close' column is missing from the dataset.")

# Filter out necessary columns for time series forecasting
time_series_data = data[['Date', 'Close']].set_index('Date')

# Train-Test Split
train_data = time_series_data[time_series_data.index.year <= 2020]
test_data = time_series_data[time_series_data.index.year == 2021]
print(f"Training Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}")

# SARIMAX training with grid search for best parameters
best_mape = float('inf')
best_params = None
best_sarimax_model = None

for p in range(2):  # Reduced parameter space for faster testing
    for d in range(3):
        for q in range(2):
            for P in range(2):
                for D in range(3):
                    for Q in range(2):
                        try:
                            model = SARIMAX(
                                train_data['Close'],
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 7)  # Weekly seasonality
                            )
                            results = model.fit(disp=False)
                            predictions = results.fittedvalues
                            mape = np.mean(np.abs((train_data['Close'] - predictions) / train_data['Close'])) * 100
                            if mape < best_mape:
                                best_mape = mape
                                best_params = (p, d, q, P, D, Q)
                                best_sarimax_model = results
                        except Exception as e:
                            print(f"Error with parameters {(p, d, q, P, D, Q)}: {e}")
                            continue

print(f"Best SARIMAX Parameters: {best_params}")
print(f"Training MAPE: {best_mape:.2f}%")

# Forecast for 2021
forecast_steps = len(test_data)
forecast = best_sarimax_model.forecast(steps=forecast_steps)

# Combine actual and forecasted values
forecast_results = pd.DataFrame({
    'Date': test_data.index,
    'Actual': test_data['Close'].values,
    'Forecast': forecast
})
forecast_results['Error'] = abs(forecast_results['Actual'] - forecast_results['Forecast'])
forecast_results['Percentage_Error'] = (forecast_results['Error'] / forecast_results['Actual']) * 100
test_mape = forecast_results['Percentage_Error'].mean()

print(f"Test MAPE for 2021: {test_mape:.2f}%")

# Plot the forecast results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Close'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Close'], label='Actual Data (2021)', color='green')
plt.plot(forecast_results['Date'], forecast_results['Forecast'], label='Forecast (2021)', color='orange')
plt.title('SARIMAX Forecast vs Actual (Daily)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

# Save forecast results to CSV
forecast_results.to_csv('SARIMAX_Forecast_Results_2021_Daily.csv', index=False)
print("Forecast results saved to 'SARIMAX_Forecast_Results_2021_Daily.csv'")


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the training dataset
# data = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\archive (7)\BAJFINANCE.csv")
# data['Date'] = pd.to_datetime(data['Date'])
# data = data.sort_values('Date')
# data.set_index('Date', inplace=True)

# # Load the 2022 dataset
# data_2022 = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\Quote-Equity-BAJAJFINSV-EQ-01-01-2022-to-31-12-2022.csv")
# data_2022.columns = data_2022.columns.str.strip()  # Strip extra spaces
# data_2022['Date'] = pd.to_datetime(data_2022['Date'], format='%d-%b-%Y')
# data_2022 = data_2022.sort_values('Date')  # Sort by date
# data_2022.set_index('Date', inplace=True)

# # Rename and clean the 'Close' column
# if 'close' in data_2022.columns:
#     data_2022 = data_2022.rename(columns={'close': 'Close'})
# data_2022['Close'] = data_2022['Close'].str.replace(',', '').astype(float)

# # Select the features for time series
# features = ['Open', 'High', 'Low', 'Last', 'Close']

# # Check for available features
# available_features = [col for col in features if col in data.columns and col in data_2022.columns]
# if 'Close' not in available_features:
#     raise ValueError("The 'Close' column is required for predictions but is missing.")

# print(f"Using features: {available_features}")

# # Split data into train (up to 2021) and test (2022 dataset)
# train = data.loc[data.index.year <= 2021]  # Rows up to 2021
# test = data_2022  # Use provided 2022 dataset for testing

# # Ensure the test dataset is not empty
# if test.empty:
#     raise ValueError("Test dataset for 2022 is empty. Please check the input data.")

# # Scale the data using only available features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_train = scaler.fit_transform(train[available_features])
# scaled_test = scaler.transform(test[available_features])

# # Prepare sequences for training and testing
# def create_sequences(data, target_idx=-1, time_steps=60):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps, target_idx])  # Target is the 'Close' price
#     return np.array(X), np.array(y)

# time_steps = 60
# X_train, y_train = create_sequences(scaled_train, target_idx=len(available_features) - 1, time_steps=time_steps)
# X_test, y_test = create_sequences(scaled_test, target_idx=len(available_features) - 1, time_steps=time_steps)

# # Build the LSTM model
# model = Sequential([
#     LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),
#     LSTM(100, return_sequences=True),
#     Dropout(0.2),
#     LSTM(50, return_sequences=False),
#     Dropout(0.2),
#     Dense(1)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

# # Predict the Close prices for 2022
# predictions_2022 = model.predict(X_test)
# predictions_2022 = scaler.inverse_transform(
#     np.hstack([np.zeros((len(predictions_2022), len(available_features) - 1)), predictions_2022])
# )[:, -1]  # Extract the Close price from scaled data

# # Align the predictions with the actual 2022 dataset
# comparison_df = pd.DataFrame({
#     'Date': data_2022.index[time_steps:],
#     'Actual_Close': data_2022['Close'].values[time_steps:],
#     'Predicted_Close': predictions_2022.flatten()
# }).set_index('Date')

# # Save the results to a CSV file
# output_file_path = r"predictions_vs_actual_2022.csv"
# comparison_df.to_csv(output_file_path)
# print(f"Predictions saved to: {output_file_path}")

# # Display RMSE and MAPE
# rmse = np.sqrt(mean_squared_error(comparison_df['Actual_Close'], comparison_df['Predicted_Close']))
# mape = np.mean(np.abs((comparison_df['Actual_Close'] - comparison_df['Predicted_Close']) / comparison_df['Actual_Close'])) * 100
# print(f'LSTM RMSE: {rmse}')
# print(f'LSTM MAPE: {mape:.2f}%')

# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(comparison_df.index, comparison_df['Actual_Close'], label='Actual Prices', color='blue')
# plt.plot(comparison_df.index, comparison_df['Predicted_Close'], label='Predicted Prices', color='red')
# plt.title('LSTM Model Predictions vs Actual Close Prices (2022)')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()
