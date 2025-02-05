Here’s a README for your Streamlit-based Time Series Forecasting application:

---

# Time Series Forecasting: ARIMA & SARIMA

This Streamlit application enables users to perform time series forecasting on cryptocurrency price data using **ARIMA** and **SARIMA** models. Users can upload a CSV file containing historical cryptocurrency data, visualize it, perform stationarity tests, and generate forecasts.

## Features
- **Upload CSV Data**: Users can upload a CSV file containing cryptocurrency price data.
- **Select Cryptocurrency**: The app allows users to choose a specific cryptocurrency symbol from the dataset.
- **Data Processing**:
  - Converts the **Date** column to datetime format.
  - Removes duplicates and sorts the data by date.
  - Resamples missing dates to maintain daily frequency using **forward fill**.
- **Date Range Selection**: Users can specify a start and end date for analysis.
- **Candlestick Chart**: Interactive visualization of historical price data using **Plotly**.
- **Stationarity Test**: Performs the **Augmented Dickey-Fuller (ADF) Test** to determine whether the data is stationary.
- **Model Selection**:
  - **ARIMA (Auto-Regressive Integrated Moving Average)**
  - **SARIMA (Seasonal Auto-Regressive Integrated Moving Average)**
- **User-defined Model Parameters**: Users can specify model parameters for ARIMA/SARIMA:
  - ARIMA: `(p, d, q)`
  - SARIMA: `(p, d, q, Seasonal p, Seasonal d, Seasonal q, Seasonal period m)`
- **Forecasting**: The application allows forecasting future price movements for a user-specified number of days.
- **Model Summary**: Displays the trained model’s summary.
- **Forecast Visualization**: Plots actual vs. predicted closing prices using **Matplotlib**.

## Installation & Requirements
Ensure you have Python installed and install the required dependencies using:

```bash
pip install streamlit pandas plotly statsmodels matplotlib
```

## Running the Application
Run the Streamlit app using:

```bash
streamlit run app.py
```

## CSV File Format
The uploaded CSV file must contain the following columns:
- `Date`: Date of the record (YYYY-MM-DD format).
- `Symbol`: Cryptocurrency symbol (e.g., BTC, ETH).
- `Open`: Opening price.
- `High`: Highest price.
- `Low`: Lowest price.
- `Close`: Closing price.

Example CSV:

| Date       | Symbol | Open  | High  | Low   | Close |
|------------|--------|-------|-------|-------|-------|
| 2023-01-01 | BTC    | 45000 | 45500 | 44800 | 45200 |
| 2023-01-02 | BTC    | 45200 | 46000 | 45000 | 45800 |
| 2023-01-03 | ETH    | 3200  | 3300  | 3100  | 3250  |

## How to Use
1. **Upload a CSV file** with cryptocurrency price data.
2. **Select a cryptocurrency** from the dropdown.
3. **Choose a date range** using the slider.
4. **Visualize the candlestick chart** of the selected cryptocurrency.
5. **Perform a stationarity test** to check if differencing is needed.
6. **Choose ARIMA or SARIMA model** and set appropriate parameters.
7. **Specify the forecast period** (e.g., 30 days).
8. **Train the model & generate forecasts**.
9. **View forecast results and plots**.

## Notes
- Ensure the uploaded CSV file contains at least a few months of historical data.
- ARIMA is suitable for non-seasonal data, while SARIMA should be used for seasonal trends.
- If the stationarity test indicates non-stationary data (`p-value > 0.05`), increasing the `d` (differencing) parameter can help.

## Author
Developed using Streamlit, Pandas, Plotly, and Statsmodels.

---

Let me know if you need any modifications! 🚀
