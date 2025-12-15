import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO

# ---------------------- PAGE SETUP -----------------------
st.set_page_config(page_title="Time Series Analysis App", layout="wide")
st.title("📊 Cryptocurrency / Stock Market Time Series Analysis")

# ---------------------- FILE UPLOAD ----------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = df.sort_values("Date")

    st.subheader("📅 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------- BASIC VISUALIZATION ----------------------
    st.subheader("📈 Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Close Price", color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

    # ---------------------- DECOMPOSITION ----------------------
    st.subheader("🔍 Time Series Decomposition (Trend, Seasonality, Residual)")
    df.set_index("Date", inplace=True)
    result = seasonal_decompose(df["Close"], model="additive", period=3)
    result.plot()
    st.pyplot(plt.gcf())

    # ---------------------- FORECASTING ----------------------
    st.subheader("📅 ARIMA Forecasting")
    n_steps = st.slider("Select forecast days:", 5, 30, 10)

    model = ARIMA(df["Close"], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_steps)

    future_dates = pd.date_range(df.index[-1], periods=n_steps + 1, freq="D")[1:]
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.index, df["Close"], label="Historical", color="blue")
    ax2.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="red")
    ax2.legend()
    st.pyplot(fig2)

    # ---------------------- REPORT GENERATION ----------------------
    st.sidebar.header("Generate Reports")

    report_type = st.sidebar.selectbox("Choose Report Type:", ["Summary Report", "Forecast Report"])

    if report_type == "Summary Report":
        mean_price = df["Close"].mean()
        max_price = df["Close"].max()
        min_price = df["Close"].min()
        volatility = df["Close"].std()

        st.markdown("### 📘 Summary Report")
        st.write(f"**Average Closing Price:** {mean_price:.2f}")
        st.write(f"**Highest Closing Price:** {max_price:.2f}")
        st.write(f"**Lowest Closing Price:** {min_price:.2f}")
        st.write(f"**Volatility (Std Dev):** {volatility:.2f}")

    elif report_type == "Forecast Report":
        st.markdown("### 📗 Forecast Report")
        st.dataframe(forecast_df)

        # Export as CSV
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv, "forecast_report.csv", "text/csv")

else:
    st.info("Please upload a CSV file to start analysis.")
