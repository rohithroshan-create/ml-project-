# ----------- DELIVERY MODULE -----------
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
                        # Show all predictions in a table, attached to input index
                        pred_df = df.copy()
                        pred_df["Predicted Delay"] = pred
                        st.dataframe(pred_df)
                        st.write(f"**Stats:** Mean = {np.mean(pred):.2f}, Std = {np.std(pred):.2f}, Min = {np.min(pred):.2f}, Max = {np.max(pred):.2f}")
                        st.bar_chart(pred_df["Predicted Delay"])
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
                        pred_df = df.copy()
                        pred_df["Predicted Risk"] = pred
                        st.dataframe(pred_df)
                        st.write(f"**Risk Counts:** {pd.Series(pred).value_counts().to_dict()}")   # e.g. {0: ..., 1: ...}
                        st.bar_chart(pd.Series(pred).value_counts())
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Risk model not loaded.")

# ----------- DEMAND FORECAST MODULE -----------
elif page == "📈 Demand Forecast":
    st.header("Time-Series Demand Forecast / Custom Model")
    file = st.file_uploader("Upload Data for Forecasting (.csv/.xlsx)", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Predict Forecast"):
            if module2_model:
                try:
                    pred = module2_model.predict(df)
                    pred_df = df.copy()
                    pred_df["Forecast"] = pred
                    st.dataframe(pred_df)
                    st.write(f"**Stats:** Mean = {np.mean(pred):.2f}, Std = {np.std(pred):.2f}, Min = {np.min(pred):.2f}, Max = {np.max(pred):.2f}")
                    st.line_chart(pred_df["Forecast"])
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
            else:
                st.error("Forecast model not loaded.")

# ----------- CHURN & SUPPLIER MODULE -----------
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
                pred_df = df.copy()
                pred_df["Prediction"] = pred
                st.dataframe(pred_df)
                st.write(f"**Prediction Counts:** {pd.Series(pred).value_counts().to_dict()}")
                st.bar_chart(pd.Series(pred).value_counts())
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    elif file:
        st.warning("Model file not loaded for this selection.")
