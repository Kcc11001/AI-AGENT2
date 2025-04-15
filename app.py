import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Revenue Forecasting AI Agent", page_icon="📈", layout="wide")
st.title("📈 Revenue Forecasting with Prophet & AI Insights")

# --- API Key Check ---
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# --- Upload File ---
uploaded_file = st.file_uploader("Upload Excel file with 'Date' and 'Revenue' columns", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Basic Validation ---
    if "Date" not in df.columns or "Revenue" not in df.columns:
        st.error("❌ The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # --- Preprocessing ---
    df = df[["Date", "Revenue"]].dropna()
    df.columns = ["ds", "y"]  # Rename for Prophet compatibility
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Historical Revenue Data")
    st.line_chart(df.set_index("ds")["y"])

    # --- Forecasting with Prophet ---
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    st.subheader("🔮 Forecasted Revenue (Next 90 Days)")
    fig = model.plot(forecast)
    st.pyplot(fig)

    # --- AI Commentary Generation ---
    st.subheader("🤖 AI-Generated Forecast Commentary")

    data_for_ai = df.to_json(orient="records", date_format="iso")

    prompt = f"""
    You are a top-tier FP&A professional. Analyze the historical and forecasted revenue data below.
    
    1. Identify key revenue trends and inflection points.
    2. Evaluate potential future risks or seasonality.
    3. Provide a summary using the Pyramid Principle.
    4. Suggest 3 strategic actions a CFO should consider.

    Historical and Forecasted Revenue JSON:
    {data_for_ai}
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an FP&A expert specializing in financial forecasting."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192"
    )
    ai_commentary = response.choices[0].message.content
    st.write(ai_commentary)
