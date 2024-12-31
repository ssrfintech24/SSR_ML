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

            # Ensure 'Date' column is converted to datetime
            if 'Date' in data.columns and 'Amount LC' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
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

            else:
                st.error("The uploaded file must contain 'Date' and 'Amount LC' columns.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
