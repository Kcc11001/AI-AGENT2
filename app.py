import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from dotenv import load_dotenv
import os
from groq import Groq
from io import BytesIO
from fpdf import FPDF

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="📈 Revenue Forecasting AI Agent", layout="wide")
st.title("📈 Revenue Forecasting AI Agent")

# API key check
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in a `.env` file or Streamlit Secrets.")
    st.stop()

# Upload Excel
uploaded_file = st.file_uploader("📥 Upload Excel file with 'Date' and 'Revenue' columns", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if "Date" not in df.columns or "Revenue" not in df.columns:
        st.error("❌ File must include 'Date' and 'Revenue' columns.")
        st.stop()

    # Preprocess
    df = df[["Date", "Revenue"]].dropna()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Historical Revenue")
    st.line_chart(df.set_index("ds"))

    # Forecast selector
    st.sidebar.header("🔧 Forecast Settings")
    forecast_days = st.sidebar.selectbox("Forecast period (days):", [30, 60, 90, 180], index=2)

    # Prophet forecast
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # KPI metrics
    hist_summary = df.tail(30).describe().round(2).to_dict()
    fut_summary = forecast[['ds', 'yhat']].tail(forecast_days).describe().round(2).to_dict()

    hist_avg = hist_summary['y']['mean']
    fut_avg = fut_summary['yhat']['mean']
    volatility = hist_summary['y']['std']
    forecast_change = ((fut_avg - hist_avg) / hist_avg * 100) if hist_avg != 0 else 0

    forecast['conf_width'] = forecast['yhat_upper'] - forecast['yhat_lower']
    conf_width_avg = forecast[['ds', 'conf_width']].tail(forecast_days)["conf_width"].mean()

    st.subheader("📌 Key Revenue Forecast KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Hist. Avg Revenue", f"${hist_avg:,.0f}")
    col2.metric("🔮 Forecast Avg Revenue", f"${fut_avg:,.0f}")
    col3.metric("📊 Forecast % Change", f"{forecast_change:.1f}%", delta=f"{forecast_change:.1f}%")
    col4.metric("🔁 Revenue Volatility", f"${volatility:,.0f}")
    col5.metric("📏 Confidence Range", f"${conf_width_avg:,.0f}")

    # --- Confidence Width Visualization ---
    st.subheader("📏 Forecast Confidence Width vs Historical Volatility")

    forecast_range = forecast.tail(forecast_days).copy()
    forecast_range["Confidence Width"] = forecast_range["yhat_upper"] - forecast_range["yhat_lower"]

    low_threshold = volatility * 0.75
    high_threshold = volatility * 1.25

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_range["ds"], forecast_range["Confidence Width"], label="Confidence Width", linewidth=2)
    ax.axhline(volatility, color="green", linestyle="--", label="Historical Volatility")
    ax.axhline(low_threshold, color="gray", linestyle=":", label="Low Threshold (0.75×)")
    ax.axhline(high_threshold, color="red", linestyle=":", label="High Threshold (1.25×)")

    ax.fill_between(forecast_range["ds"], 0, low_threshold, color="green", alpha=0.1, label="Acceptable Range")
    ax.fill_between(forecast_range["ds"], low_threshold, high_threshold, color="orange", alpha=0.1, label="Warning Zone")
    ax.fill_between(forecast_range["ds"], high_threshold, forecast_range["Confidence Width"],
                    where=forecast_range["Confidence Width"] > high_threshold,
                    color="red", alpha=0.2, label="High Risk")

    ax.set_ylabel("Confidence Width ($)")
    ax.set_title("Forecast Confidence Width vs Historical Revenue Volatility")
    ax.legend(loc="upper right")
    ax.grid(True)

    st.pyplot(fig)

    # Forecast chart
    st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    with st.expander("📉 View Forecast Components"):
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    # --- Excel Export with KPI sheet ---
    def generate_excel(df1,_


    except Exception as e:
        st.error(f"Groq API Error: {e}")

