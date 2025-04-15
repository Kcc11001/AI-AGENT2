import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit settings
st.set_page_config(page_title="📈 Revenue Forecasting with AI", page_icon="📊", layout="wide")
st.title("📈 Revenue Forecasting AI Agent")

# Validate API key
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in a `.env` file or Streamlit Secrets.")
    st.stop()

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File with 'Date' and 'Revenue' columns", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "Date" not in df.columns or "Revenue" not in df.columns:
        st.error("❌ The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Preprocessing
    df = df[["Date", "Revenue"]].dropna()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Historical Revenue")
    st.line_chart(df.set_index("ds"))

    # Forecasting
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    st.subheader("🔮 Forecast for Next 90 Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Components
    with st.expander("📉 View Forecast Components"):
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    # Prepare summarized data for AI
    st.subheader("🧠 AI-Generated Financial Commentary")

    # Limit to last 30 actuals and next 30 forecasted days for context
    historical_summary = df.tail(30).describe().round(2).to_dict()
    future_summary = forecast[['ds', 'yhat']].tail(30).describe().round(2).to_dict()

    prompt = f"""
    You are a senior FP&A analyst. Based on the summarized revenue data below, analyze the current state and outlook of the business.

    🔹 Historical Revenue Summary (last 30 days):
    Count: {historical_summary['y']['count']}
    Mean: {historical_summary['y']['mean']}
    Min: {historical_summary['y']['min']}
    Max: {historical_summary['y']['max']}

    🔹 Forecasted Revenue Summary (next 30 days):
    Count: {future_summary['yhat']['count']}
    Mean: {future_summary['yhat']['mean']}
    Min: {future_summary['yhat']['min']}
    Max: {future_summary['yhat']['max']}

    Please:
    - Identify revenue trends and inflection points
    - Highlight risks or seasonality
    - Provide a CFO-ready summary using the Pyramid Principle
    - Recommend 3 strategic actions
    """

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        ai_commentary = response.choices[0].message.content
        st.markdown(ai_commentary)
    except Exception as e:
        st.error(f"Groq API error: {e}")
