

# import pandas as pd
# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pmdarima import auto_arima
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# # Load and preprocess data
# data = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\Book1.csv")  # Replace with your dataset file path
# data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
# data['time'] = data['date']  # Create a consistent time column for pivoting
# data = data.sort_values('date')

# # Step 1: Select Target ID and Correlated IDs
# target_id = 'S389088'  # Replace with your target ID
# pivoted_data = data.pivot_table(index=['year', 'month'], columns='ID', values='value')
# correlation_matrix = pivoted_data.corr()
# highly_correlated_ids = correlation_matrix[target_id][correlation_matrix[target_id] >= 0.95].index.tolist()
# highly_correlated_ids.remove(target_id)
# correlated_ids = highly_correlated_ids
# all_ids = [target_id] + correlated_ids

# # Pivot data for target and correlated IDs
# pivot_data = data.pivot(index='time', columns='ID', values='value')[all_ids]

# # Step 2: Feature Engineering
# # Adding lagged features, rolling averages, and exponential smoothing
# for col in pivot_data.columns:
#     pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1)
#     pivot_data[f'{col}_lag2'] = pivot_data[col].shift(2)
#     pivot_data[f'{col}_rolling_mean3'] = pivot_data[col].rolling(window=3).mean()
#     pivot_data[f'{col}_exp_smooth'] = pivot_data[col].ewm(span=3, adjust=False).mean()

# # Fill missing values after feature engineering
# pivot_data = pivot_data.fillna(method='bfill').fillna(method='ffill')

# # Prepare training and testing data
# train_data = pivot_data[pivot_data.index.year <= 2024]
# test_data = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')

# # Extract target series for SARIMAX
# train_target = train_data[target_id]

# # Step 3: SARIMAX Model
# print("Finding optimal SARIMAX parameters...")
# best_mape = float('inf')
# best_params = None
# best_sarimax_model = None

# for p in range(3):  # AR order
#     for d in range(2):  # Differencing order
#         for q in range(3):  # MA order
#             for P in range(3):  # Seasonal AR
#                 for D in range(2):  # Seasonal differencing
#                     for Q in range(3):  # Seasonal MA
#                         try:
#                             sarimax_model = SARIMAX(
#                                 train_target,
#                                 order=(p, d, q),
#                                 seasonal_order=(P, D, Q, 12),
#                                 exog=train_data.drop(columns=[target_id])
#                             )
#                             sarimax_fit = sarimax_model.fit(disp=False)
#                             predicted = sarimax_fit.fittedvalues
#                             mape = np.mean(np.abs((train_target - predicted) / train_target)) * 100
#                             if mape < best_mape:
#                                 best_mape = mape
#                                 best_params = (p, d, q, P, D, Q)
#                                 best_sarimax_model = sarimax_fit
#                         except:
#                             continue

# print(f"Best SARIMAX Params: {best_params} with MAPE: {best_mape}%")

# # Prepare exogenous variables for the forecast horizon
# exog_forecast = pivot_data.loc[test_data].drop(columns=[target_id], errors='ignore').fillna(method='bfill').fillna(method='ffill')

# # Forecast with SARIMAX
# sarimax_forecast = best_sarimax_model.forecast(
#     steps=len(test_data),
#     exog=exog_forecast
# )

# # Step 4: Load Actual 2025 Data
# actual_data_2025 = data[(data['year'] == 2025) & (data['ID'] == target_id)]
# actual_data_2025['time'] = pd.to_datetime(actual_data_2025[['year', 'month']].assign(day=1))

# # Merge actual 2025 values with SARIMAX forecast for alignment
# comparison_data = pd.DataFrame({
#     'date': test_data,
#     'sarimax_forecast': sarimax_forecast
# }).merge(actual_data_2025[['time', 'value']], left_on='date', right_on='time', how='left').rename(columns={'value': 'actual'})

# # Step 5: Visualization
# plt.figure(figsize=(12, 6))

# # Plot actual 2025 values
# plt.plot(comparison_data['date'], comparison_data['actual'], label="Actual 2025 Values", color="blue", linestyle="--")

# # Plot SARIMAX forecast
# plt.plot(comparison_data['date'], comparison_data['sarimax_forecast'], label="SARIMAX Forecast", color="orange")

# # Add labels, legend, and grid
# plt.title(f"Comparison of Actual vs SARIMAX Forecast for ID {target_id} (2025)")
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(alpha=0.3)

# # Show the plot
# plt.show()

# # Step 6: Save Results to CSV
# comparison_data[['date', 'sarimax_forecast', 'actual']].to_csv(f'sarimax_vs_actual_{target_id}.csv', index=False)

# print(f"SARIMAX forecast and actual values saved to 'sarimax_vs_actual_{target_id}.csv'")




import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv(r"C:\Users\Aparna Anand\Downloads\Book1.csv")  # Replace with your dataset file path
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
data['time'] = data['date']  # Create a consistent time column for pivoting
data = data.sort_values('date')

# Step 1: Select Target ID and Correlated IDs
target_id = 'S279218'  # Replace with your target ID
pivoted_data = data.pivot_table(index=['year', 'month'], columns='ID', values='value')
correlation_matrix = pivoted_data.corr()
highly_correlated_ids = correlation_matrix[target_id][correlation_matrix[target_id] >= 0.95].index.tolist()
highly_correlated_ids.remove(target_id)
correlated_ids = highly_correlated_ids
all_ids = [target_id] + correlated_ids

# Pivot data for target and correlated IDs
pivot_data = data.pivot(index='time', columns='ID', values='value')[all_ids]

# Step 2: Feature Engineering
# Adding lagged features, rolling averages, and exponential smoothing
for col in pivot_data.columns:
    pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1)
    pivot_data[f'{col}_lag2'] = pivot_data[col].shift(2)
    pivot_data[f'{col}_rolling_mean3'] = pivot_data[col].rolling(window=3).mean()
    pivot_data[f'{col}_exp_smooth'] = pivot_data[col].ewm(span=3, adjust=False).mean()

# Fill missing values after feature engineering
pivot_data = pivot_data.fillna(method='bfill').fillna(method='ffill')

# Prepare training and testing data
train_data = pivot_data[pivot_data.index.year <= 2024]
test_data = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')

# Extract target series for SARIMAX
train_target = train_data[target_id]

# Step 3: SARIMAX Model
print("Finding optimal SARIMAX parameters...")
best_mape = float('inf')
best_params = None
best_sarimax_model = None

for p in range(3):  # AR order
    for d in range(2):  # Differencing order
        for q in range(3):  # MA order
            for P in range(3):  # Seasonal AR
                for D in range(2):  # Seasonal differencing
                    for Q in range(3):  # Seasonal MA
                        try:
                            sarimax_model = SARIMAX(
                                train_target,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 12),
                                exog=train_data.drop(columns=[target_id])
                            )
                            sarimax_fit = sarimax_model.fit(disp=False)
                            predicted = sarimax_fit.fittedvalues
                            mape = np.mean(np.abs((train_target - predicted) / train_target)) * 100
                            if mape < best_mape:
                                best_mape = mape
                                best_params = (p, d, q, P, D, Q)
                                print(f"New Best Params: {best_params}")

                                best_sarimax_model = sarimax_fit
                        except:
                            continue

print(f"Best SARIMAX Params: {best_params} with Training MAPE: {best_mape:.2f}%")

# Prepare exogenous variables for the forecast horizon
exog_forecast = pivot_data.loc[test_data].drop(columns=[target_id], errors='ignore').fillna(method='bfill').fillna(method='ffill')

# Forecast with SARIMAX
sarimax_forecast = best_sarimax_model.forecast(
    steps=len(test_data),
    exog=exog_forecast
)

# Step 4: Load Actual 2025 Data
actual_data_2025 = data[(data['year'] == 2025) & (data['ID'] == target_id)]
actual_data_2025['time'] = pd.to_datetime(actual_data_2025[['year', 'month']].assign(day=1))

# Merge actual 2025 values with SARIMAX forecast for alignment
comparison_data = pd.DataFrame({
    'date': test_data,
    'sarimax_forecast': sarimax_forecast
}).merge(actual_data_2025[['time', 'value']], left_on='date', right_on='time', how='left').rename(columns={'value': 'actual'})

# Step 5: Calculate MAPE for Test Data
comparison_data['error'] = abs(comparison_data['actual'] - comparison_data['sarimax_forecast'])
comparison_data['percentage_error'] = comparison_data['error'] / comparison_data['actual'] * 100
test_mape = comparison_data['percentage_error'].mean()

print(f"MAPE for SARIMAX model on test data: {test_mape:.2f}%")

# Step 6: Visualization
plt.figure(figsize=(12, 6))

# Plot actual 2025 values
plt.plot(comparison_data['date'], comparison_data['actual'], label="Actual 2025 Values", color="blue", linestyle="--")

# Plot SARIMAX forecast
plt.plot(comparison_data['date'], comparison_data['sarimax_forecast'], label="SARIMAX Forecast", color="orange")

# Add labels, legend, and grid
plt.title(f"Comparison of Actual vs SARIMAX Forecast for ID {target_id} (2025)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.show()

# Step 7: Save Results to CSV
comparison_data[['date', 'sarimax_forecast', 'actual']].to_csv(f'sarimax_vs_actual_{target_id}.csv', index=False)

print(f"SARIMAX forecast and actual values saved to 'sarimax_vs_actual_{target_id}.csv'")
