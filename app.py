import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from dotenv import load_dotenv
import os
from groq import Groq
from io import BytesIO
from fpdf import FPDF

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page setup
st.set_page_config(page_title="📈 Revenue Forecasting AI Agent", layout="wide")
st.title("📈 Revenue Forecasting AI Agent")

# Check API Key
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in a `.env` file or Streamlit Secrets.")
    st.stop()

# Upload Excel file
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

    df = df[["Date", "Revenue"]].dropna()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Historical Revenue")
    st.line_chart(df.set_index("ds"))

    # Forecast settings
    st.sidebar.header("🔧 Forecast Settings")
    forecast_days = st.sidebar.selectbox("Forecast period (days):", [30, 60, 90, 180], index=2)

    # Prophet forecasting
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

    # Confidence width plot
    st.subheader("📏 Forecast Confidence Width vs Historical Volatility")
    forecast_range = forecast.tail(forecast_days).copy()
    forecast_range["Confidence Width"] = forecast_range["yhat_upper"] - forecast_range["yhat_lower"]

    low_threshold = volatility * 0.75
    high_threshold = volatility * 1.25
    anomaly_threshold = volatility * 1.5

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

    # 🔍 Detect Anomalies
    st.subheader("🚨 Forecast Anomaly Detection")
    anomalies = forecast_range[forecast_range["Confidence Width"] > anomaly_threshold][["ds", "yhat", "Confidence Width"]]
    if anomalies.empty:
        st.success("✅ No anomalies detected in the forecast confidence intervals.")
    else:
        st.warning(f"⚠️ {len(anomalies)} anomalies detected (confidence width > 1.5× historical volatility)")
        st.dataframe(anomalies.rename(columns={"ds": "Date", "yhat": "Forecast", "Confidence Width": "Conf. Range Width"}))

    # Forecast chart
    st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    with st.expander("📉 View Forecast Components"):
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    # Excel export
    def generate_excel(df1, df2, kpi_dict):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df1.to_excel(writer, index=False, sheet_name='Historical')
            df2.to_excel(writer, index=False, sheet_name='Forecast')
            kpi_df = pd.DataFrame(kpi_dict.items(), columns=["KPI", "Value"])
            kpi_df.to_excel(writer, index=False, sheet_name='KPI Summary')
        output.seek(0)
        return output.read()

    # KPI export
    kpi_dict = {
        "Historical Average Revenue": f"${hist_avg:,.2f}",
        "Forecast Average Revenue": f"${fut_avg:,.2f}",
        "Forecast % Change": f"{forecast_change:.2f}%",
        "Revenue Volatility": f"${volatility:,.2f}",
        "Avg Confidence Range Width": f"${conf_width_avg:,.2f}"
    }

    excel_data = generate_excel(df, forecast, kpi_dict)

    st.sidebar.subheader("📤 Export Options")
    st.sidebar.download_button(
        label="📥 Download Excel",
        data=excel_data,
        file_name="revenue_forecast_with_kpis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # AI commentary with Groq
    st.subheader("🧠 AI-Generated Financial Commentary")

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

        def create_summary_pdf(text: str) -> BytesIO:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in text.split('\n'):
                pdf.multi_cell(0, 10, line)
            pdf_bytes = pdf.output(dest='S').encode('latin1')
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
