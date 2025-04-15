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

# Set up Streamlit page
st.set_page_config(page_title="📈 Revenue Forecasting AI Agent", layout="wide")
st.title("📈 Revenue Forecasting AI Agent")

# API Key Check
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
        st.error("❌ File must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Prepare data
    df = df[["Date", "Revenue"]].dropna()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Historical Revenue")
    st.line_chart(df.set_index("ds"))

    # Forecast period selector
    st.sidebar.header("🔧 Forecast Settings")
    forecast_days = st.sidebar.selectbox("Forecast period (days):", [30, 60, 90, 180], index=2)

    # Prophet Forecast
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    with st.expander("📉 View Forecast Components"):
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    # --- Excel Export ---
    def generate_excel(df1, df2):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df1.to_excel(writer, index=False, sheet_name='Historical')
            df2.to_excel(writer, index=False, sheet_name='Forecast')
        output.seek(0)
        return output.read()

    excel_data = generate_excel(df, forecast)

    st.sidebar.subheader("📤 Export Options")
    st.sidebar.download_button(
        label="📥 Download Excel",
        data=excel_data,
        file_name="revenue_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --- AI Commentary ---
    st.subheader("🧠 AI-Generated Financial Commentary")

    hist_summary = df.tail(30).describe().round(2).to_dict()
    fut_summary = forecast[['ds', 'yhat']].tail(30).describe().round(2).to_dict()

    prompt = f"""
    You are a senior FP&A analyst. Based on the summarized revenue data below, analyze the current state and outlook of the business.

    🔹 Historical Revenue Summary (last 30 days):
    Count: {hist_summary['y']['count']}
    Mean: {hist_summary['y']['mean']}
    Min: {hist_summary['y']['min']}
    Max: {hist_summary['y']['max']}

    🔹 Forecasted Revenue Summary (next {forecast_days} days):
    Count: {fut_summary['yhat']['count']}
    Mean: {fut_summary['yhat']['mean']}
    Min: {fut_summary['yhat']['min']}
    Max: {fut_summary['yhat']['max']}

    Please:
    - Identify revenue trends and inflection points
    - Highlight risks or seasonality
    - Use the Pyramid Principle
    - Provide 3 CFO-level recommendations
    """

    estimated_tokens = len(prompt.split())
    st.sidebar.markdown(f"🧠 Estimated Tokens: `{estimated_tokens}` / 6000")

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

        # --- Generate PDF as bytes ---
        def create_summary_pdf(text: str) -> BytesIO:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in text.split('\n'):
                pdf.multi_cell(0, 10, line)
            pdf_bytes = pdf.output(dest='S').encode('latin1')  # FIXED LINE
            return BytesIO(pdf_bytes)

        pdf_bytes = create_summary_pdf(ai_commentary)

        st.sidebar.download_button(
            label="🧾 Download PDF Summary",
            data=pdf_bytes,
            file_name="forecast_summary.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Groq API Error: {e}")


