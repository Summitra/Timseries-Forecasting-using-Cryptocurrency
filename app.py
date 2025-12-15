import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crypto Time Series Forecasting",page_icon = ":bar_chart:", layout="wide")

# Sidebar
st.sidebar.title("Crypto Forecasting App")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)
page = st.sidebar.radio("Select Report", ["📊 EDA Dashboard", "📈 Forecasting Dashboard"])
crypto_symbol = "BTC-USD" 
period = "2y"   

# Fetch Data
@st.cache_data
def load_data(crypto_symbol, period):
    data = yf.download(crypto_symbol, period=period)
    data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    print(data.columns)
    
    data.reset_index(inplace=True)
    return data

data = load_data(crypto_symbol, period)

# ---------------- PAGE 1: EDA ----------------
if page == "📊 EDA Dashboard":
    st.title("📊 Cryptocurrency Time Series Analysis")
    st.subheader(f"Exploring {crypto_symbol} data ({period})")

    st.dataframe(data.head())
    csv_data = data.to_csv(index=False)
    st.download_button("Download Data", data= csv_data, file_name= "crypto_historical_data.csv", mime= "text/csv")

    st.write("### Closing Price Trend")
    fig = px.line(data, x="Date", y=f"Close {crypto_symbol}", title=f"{crypto_symbol} Closing Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

 
    st.write("### Candlestick Chart")
    candlestick = go.Figure(data=[go.Candlestick(x=data['Date'],
                                                 open=data['Open BTC-USD'], high=data['High BTC-USD'],
                                                 low=data['Low BTC-USD'], close=data['Close BTC-USD'])])
    candlestick.update_layout(title=f"{crypto_symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(candlestick, use_container_width=True)

    st.write("### Volume Over Time")
    fig_vol = px.area(data, x="Date", y="Volume BTC-USD", title="Trading Volume Over Time")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.write("### Statistical Summary")
    st.dataframe(data.describe())

# ---------------- PAGE 2: FORECASTING ----------------
else:
    st.title("📈 Cryptocurrency Price Forecasting")
    st.subheader(f"Forecasting {crypto_symbol} Closing Prices")

    model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])

    # Prepare data
    df = data[["Date", "Close BTC-USD"]].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    train_size = int(len(df) * 0.8)         
    # This calculates the 80% from the data. We have 731 rows so, 731*0.8 = 585
    train, test = df[:train_size], df[train_size:]
    # Here train data starts from 0 to 585 and test data starts from 585 to 730
    # train data = First 80% of data and test data = last 20% of data

    if model_choice == "ARIMA":
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        df_result = test.copy()
        df_result['Forecast'] = forecast.values

    elif model_choice == "SARIMA":
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
                                            # 1 → seasonal autoregressive term
                                            # 1 → seasonal differencing
                                            # 1  → seasonal moving average term
                                            # 12 → number of time steps(monthly)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))
        df_result = test.copy()
        df_result['Forecast'] = forecast.values

    elif model_choice == "Prophet":
        prophet_df = train.reset_index().rename(columns={"Date": "ds", "Close BTC-USD": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future) # This will forecast for all dates
        df_result = forecast.set_index('ds')[['yhat']].rename(columns={'yhat': 'Forecast'})
        # yhat is a predicted value for the datestamp
        df_result = df_result.join(df, how='left')

    elif model_choice == "LSTM":
        scaler = MinMaxScaler(feature_range=(0,1))  
        scaled_data = scaler.fit_transform(df)
        X, y = [], []
        look_back = 60 # 60 past timesteps and we have to predit 61th price trend
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # LSTM contains 3D model
        # X.shape[0] → number of samples (how many training examples)
        # X.shape[1] → number of time steps (look_back = 60)
        # 1 → number of features

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
            LSTM(50, return_sequences=False), # outputs final timestep
            Dense(25),
            Dense(1) # next day closing price bet 0 t0 1
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')  #Adaptive moment estimation
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)

        last_60 = scaled_data[-look_back:]
        X_test = np.array([last_60])  # Converting to 2D Sequence, even we are predicting for one value as well
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # X_test.shape[0] = number of samples → 1
            # X_test.shape[1] = number of time steps → 60
            # 1 = number of features 
               
        pred = model.predict(X_test) #Scaled bet 0 and 1 , not real price yet
        pred_price = scaler.inverse_transform(pred)  # Converting data into real world price
        df_result = pd.DataFrame({'Date': [df.index[-1] + timedelta(days=1)], 'Forecast': pred_price.flatten()})
        df_result.set_index('Date', inplace=True)

    # Visualization
    st.write("### Actual vs Predicted Prices")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close BTC-USD'], name='Actual', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=df_result.index, y=df_result['Forecast'], name='Forecast', line=dict(color='orange')))
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### Forecast Data")
    st.dataframe(df_result.tail())

    # Export forecast
    csv = df_result.to_csv().encode('utf-8')
    st.download_button("Download Forecast CSV", data=csv, file_name="crypto_forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("Developed by Sumitra Samal & Harsh Patel | Cryptocurrency Forecasting Project")
