import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Supply Chain Risk AI", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

# ---- Gemini Config ----
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0")
        st.sidebar.success("✅ Gemini AI Connected")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini Error: {str(e)}")
else:
    st.sidebar.warning("⚠️ No Gemini API key found")

# ---- Model Loaders ----
def load_model_safe(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.warning(f"Error loading {path}: {e}")
    return None

catboost_delay_reg = load_model_safe("catboost_delay_regression.pkl")
catboost_delivery_risk = load_model_safe("catboost_delivery_risk.pkl")
module2_model = load_model_safe("module2.pkl")
model5 = load_model_safe("model5.pkl")
model6 = load_model_safe("model6.pkl")

# ---- Sidebar Navigation ----
st.sidebar.title("🏭 Supply Chain AI")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home",
        "📦 Delivery Risk",
        "📈 Demand Forecast",
        "👥 Churn & Supplier",
        "🤖 AI Chatbot"
    ],
    index=0
)

# ---- HOME PAGE ----
if page == "🏠 Home":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.title("🏭 Supply Chain Risk Predictor")
        st.markdown("""
        ### Advanced ML-Powered Analytics Platform
        - 📦 **Delivery Risk Assessment** (CatBoost)
        - 📈 **Demand Forecasting** (Custom Model)
        - 👥 **Customer Churn / Supplier Analysis** (Custom Models)
        - 🤖 **Chatbot** - General AI via Gemini
        """)
    with col2:
        st.info("""
        ### System Status
        - Models Loaded: 
        - Gemini AI: Active (if set)
        ### Features
        - Real-time ML predictions
        - Gemini Powered Assistant for all queries
        """)

# ---- MODULE 1: DELIVERY ----
elif page == "📦 Delivery Risk":
    st.header("Delivery Delay (Regression) & Risk (Classification) - CatBoost Models")
    file = st.file_uploader("Upload Delivery Data (.csv/.xlsx)", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Delay Prediction (Regression)")
            if st.button("Predict Delay"):
                if catboost_delay_reg:
                    try:
                        pred = catboost_delay_reg.predict(df)
                        st.success(f"First 5 predicted delays: {pred[:5]}")
                        st.markdown(f"Mean Delay: {np.mean(pred):.2f}, Max: {np.max(pred):.2f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Delay model not loaded.")
        with col2:
            st.markdown("### Delivery Risk (Classification)")
            if st.button("Predict Risk"):
                if catboost_delivery_risk:
                    try:
                        pred = catboost_delivery_risk.predict(df)
                        st.success(f"First 5 risk labels: {pred[:5]}")
                        st.markdown(f"Risk count: {np.sum(pred)} of {len(pred)}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Risk model not loaded.")

# ---- MODULE 2: DEMAND FORECAST ----
elif page == "📈 Demand Forecast":
    st.header("Time-Series Demand Forecast / Custom Model")
    file = st.file_uploader("Upload Data for Forecasting (.csv/.xlsx)", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Predict Forecast"):
            if module2_model:
                try:
                    pred = module2_model.predict(df)
                    st.success(f"First 5 forecast: {pred[:5]}")
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
            else:
                st.error("Forecast model not loaded.")

# ---- MODULE 3: CHURN & SUPPLIER ----
elif page == "👥 Churn & Supplier":
    st.header("Churn / Supplier Module")
    file = st.file_uploader("Upload Churn/Supplier Data (.csv/.xlsx)", type=["csv", "xlsx"])
    model_option = st.radio("Select Model:", ["model5 (churn/supplier)", "model6 (churn/supplier)"])
    model_selected = model5 if "model5" in model_option else model6
    if file and model_selected:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Predict (Churn/Supplier)"):
            try:
                pred = model_selected.predict(df)
                st.success(f"First 5 predictions: {pred[:5]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    elif file:
        st.warning("Model file not loaded for this selection.")

# ---- AI CHATBOT ----
elif page == "🤖 AI Chatbot":
    st.header("AI Chatbot Assistant (Gemini)")
    user_q = st.text_area("Ask anything! Answers from Gemini (unrestricted).")
    if st.button("Get Answer"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        elif not gemini_model:
            st.error("Gemini is not available. Configure your API key.")
        else:
            try:
                prompt = f"You are an excellent, helpful AI assistant. Answer with clarity and detail.\nUser: {user_q}\nAnswer:"
                answer = gemini_model.generate_content(prompt)
                st.markdown(answer.text if hasattr(answer, "text") else str(answer))
            except Exception as e:
                st.error(f"Gemini error: {e}")

st.info("Ensure columns in uploaded files match your training data. If models do not load, check pickle compatibility and Python/library version.")
