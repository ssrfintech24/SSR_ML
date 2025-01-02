import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast_script import time_series_forecast  # Assuming your script is saved as 'forecast_script.py'


def main():
    st.title("Time Series Forecasting App")
    st.write("Upload your CSV file with 'Date' and 'Amount LC' columns for forecasting.")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)

            # Display data preview
            st.subheader("Uploaded Data Preview")
            st.dataframe(data.head())

            # Standardize column names
            data.columns = data.columns.str.strip().str.lower()

            # Process date column dynamically
            if 'date' in data.columns:
                st.write("Detected 'Date' column.")
                try:
                    data['Date'] = pd.to_datetime(data['date'], errors='coerce')
                    if data['Date'].isnull().any():
                        # If parsing fails, extract year and month manually
                        data['year'] = data['date'].str[:4].astype(int)
                        data['month'] = data['date'].str[5:7].astype(int)
                        data['Date'] = pd.to_datetime(data['year'].astype(str) + data['month'].astype(str), format='%Y%m')
                except Exception as e:
                    st.error(f"Error processing 'Date' column: {e}")
                    return
            elif 'year' in data.columns and 'month' in data.columns:
                st.write("Detected 'year' and 'month' columns.")
                data['Date'] = pd.to_datetime(data['year'].astype(str) + data['month'].astype(str), format='%Y%m')
            else:
                st.error("The uploaded file must contain either:\n"
                         "1. A 'Date' column (formatted as YYYY-MM-DD or YYYYMM), or\n"
                         "2. Separate 'year' and 'month' columns.")
                return

            # Ensure 'Amount LC' or equivalent column exists
            if 'amount lc' in data.columns:
                data.rename(columns={'amount lc': 'Amount LC'}, inplace=True)
            elif 'value' in data.columns:
                data.rename(columns={'value': 'Amount LC'}, inplace=True)
            else:
                st.error("The uploaded file must contain a column for the target values, such as 'Amount LC' or 'value'.")
                return

            # Keep only necessary columns
            data = data[['Date', 'Amount LC']]
            data = data.sort_values(by='Date')

            # Perform forecasting
            st.subheader("Time Series Forecasting Results")
            results = time_series_forecast(data, "Date", "Amount LC", frequency=12, steps=24)

            # Display RMSE
            st.write("Model Performance:")
            st.write(f"SARIMA RMSE: {results['SARIMA_RMSE']:.2f}")
            st.write(f"Holt-Winters RMSE: {results['Holt_Winters_RMSE']:.2f}")

            # Plot original data and forecasts
            st.subheader("Forecast Plots")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data["Date"], data["Amount LC"], label="Original Data", color="blue")
            forecast_dates = pd.date_range(start=data["Date"].iloc[-1], periods=24, freq="M")
            ax.plot(
                forecast_dates,
                results["SARIMA_Forecast"],
                label="SARIMA Forecast",
                color="purple"
            )
            ax.plot(
                forecast_dates,
                results["Holt_Winters_Forecast"],
                label="Holt-Winters Forecast",
                color="green"
            )
            ax.set_title("Forecast Comparison")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
