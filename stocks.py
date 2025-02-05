import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings


warnings.filterwarnings("ignore")

st.title("Time Series Forecasting: ARIMA & SARIMA")

# File upload
uploaded_file = st.file_uploader("Upload your crypto CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Let user select cryptocurrency
    crypto_options = df['Symbol'].unique().tolist()
    selected_crypto = st.selectbox("Select Cryptocurrency", crypto_options)
    
    # Filter data for selected cryptocurrency
    df = df[df['Symbol'] == selected_crypto]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset=['Date', 'Symbol']).sort_values(by='Date')
    df.set_index('Date', inplace=True)
    
    # Handle missing dates by resampling to a daily frequency
    df = df.asfreq('D').fillna(method='ffill')
    
    # Allow user to select start and end date
    start_date, end_date = st.slider("Select Date Range", 
                                     min_value=df.index.min().date(), 
                                     max_value=df.index.max().date(), 
                                     value=(df.index.min().date(), df.index.max().date()))
    
    df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    
    st.write(f"### Data Preview for {selected_crypto}")
    st.write(df.head())
    
    # Plot Candlestick Chart
    st.write(f"### Candlestick Chart of {selected_crypto} Prices")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Market Data'
    ))
    fig.update_layout(
        title=f'{selected_crypto} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    st.plotly_chart(fig)
    
    # Stationarity test
    def check_stationarity(series):
        result = adfuller(series)
        return result[1]  # p-value
    
    st.write("### Stationarity Test")
    p_value = check_stationarity(df['Close'])
    if p_value <= 0.05:
        st.success(f"The data for {selected_crypto} is stationary (p-value: {p_value:.6f})")
    else:
        st.warning(f"The data for {selected_crypto} is NOT stationary (p-value: {p_value:.6f})")
    
    # Model selection
    model_type = st.radio("Select Model Type", ["ARIMA", "SARIMA"])
    
    # Get user input for parameters
    st.sidebar.header("Model Parameters")
    if model_type == "ARIMA":
        p = st.sidebar.number_input("p (Auto-regressive term)", min_value=0, max_value=5, value=2)
        d = st.sidebar.number_input("d (Differencing term)", min_value=0, max_value=2, value=1)
        q = st.sidebar.number_input("q (Moving average term)", min_value=0, max_value=5, value=2)
    
    elif model_type == "SARIMA":
        seasonal_p = st.sidebar.number_input("Seasonal p", min_value=0, max_value=5, value=1)
        seasonal_d = st.sidebar.number_input("Seasonal d", min_value=0, max_value=2, value=1)
        seasonal_q = st.sidebar.number_input("Seasonal q", min_value=0, max_value=5, value=1)
        seasonal_m = st.sidebar.number_input("Seasonal Period (m)", min_value=1, max_value=24, value=12)
    
    forecast_days = st.sidebar.number_input("Number of days to forecast", min_value=1, max_value=45, value=30)
    
    # Train model
    if st.button("Train Model & Forecast"):
        st.write(f"### Model Training & Forecasting for {selected_crypto}")
        
        if model_type == "ARIMA":
            model = ARIMA(df['Close'], order=(p, d, q))
        else:
            model = sm.tsa.statespace.SARIMAX(df['Close'], 
                                              order=(seasonal_p, seasonal_d, seasonal_q), 
                                              seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_m))
        
        model = model.fit()
        st.write(model.summary())
        
        # Ensure forecast starts after the last available date
        forecast_start = df.index[-1]
        forecast_index = pd.date_range(start=forecast_start, periods=forecast_days + 1, freq='D')[1:]
        predictions = pd.Series(model.forecast(steps=forecast_days), index=forecast_index)
        
        # Plot forecast
        plt.figure(figsize=(15, 10))
        plt.plot(df["Close"], label='Actual')
        plt.plot(predictions, color='red', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{model_type} Forecast for {selected_crypto}')
        plt.legend(loc='upper left')
        st.pyplot(plt)
        
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
