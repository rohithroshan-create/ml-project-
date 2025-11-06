import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Load models ---
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

catboost_delay_reg = load_model("catboost_delay_regression.pkl")
catboost_delivery_risk = load_model("catboost_delivery_risk.pkl")
module2_model = load_model("module2.pkl")
model5 = load_model("model5.pkl")
model6 = load_model("model6.pkl")

# --- Gemini assistant chatbot ---
def get_gemini_response(question):
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if genai is None or api_key is None:
        return "Gemini API not configured."
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = ("You are a helpful assistant AI, answer the user's questions clearly and completely.\n"
              f"User question: {question}")
    response = gemini_model.generate_content(prompt)
    return response.text if hasattr(response, "text") else str(response)

st.title("Supply Chain ML Modules + Gemini Chatbot")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Module 1: Delivery", "Module 2: Forecasting", "Module 3: Churn/Supplier", "Gemini Chatbot"]
)

with tab1:
    st.header("Module 1: Delivery Risk and Delay (CatBoost)")
    file = st.file_uploader("Upload Delivery Data (.csv/.xlsx)", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predict Delay (Regression)"):
                if catboost_delay_reg:
                    pred = catboost_delay_reg.predict(df)
                    st.success(f"Predicted Delays: {pred[:5]} ... (showing first 5)")
                else:
                    st.error("Delay regression model not loaded.")
        with col2:
            if st.button("Predict Risk (Classification)"):
                if catboost_delivery_risk:
                    pred = catboost_delivery_risk.predict(df)
                    st.success(f"Predicted Risks: {pred[:5]} ... (showing first 5)")
                else:
                    st.error("Delivery risk model not loaded.")

with tab2:
    st.header("Module 2: Demand Forecasting / Prediction")
    file = st.file_uploader("Upload Forecasting Data (.csv/.xlsx)", type=["csv", "xlsx"], key="mod2")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Run Forecast / Prediction"):
            if module2_model:
                pred = module2_model.predict(df)
                st.success(f"Forecasted/Predicted values: {pred[:5]} ... (first 5 rows)")
            else:
                st.error("Forecast model not loaded.")

with tab3:
    st.header("Module 3: Churn / Supplier Analysis")
    file1 = st.file_uploader("Upload Data for Churn/Supplier (.csv/.xlsx)", type=["csv","xlsx"], key="mod3")
    model_option = st.radio("Select Model", ["model5", "model6"])
    if file1:
        df = pd.read_csv(file1) if file1.name.endswith('.csv') else pd.read_excel(file1)
        if st.button("Predict Churn/Supplier"):
            model = model5 if model_option == "model5" else model6
            if model:
                pred = model.predict(df)
                st.success(f"Predicted labels: {pred[:5]} ...")
            else:
                st.error(f"{model_option} model not loaded.")

with tab4:
    st.header("Gemini-powered AI Chatbot")
    user_q = st.text_area("Ask anything. The AI assistant will answer helpfully.")
    if st.button("Get Answer"):
        if user_q.strip() == "":
            st.info("Type a question first.")
        else:
            answer = get_gemini_response(user_q)
            st.markdown(f"**AI Assistant:** {answer}")

st.info("If model predictions fail, check your input columns match the expected format in your training notebooks.")
