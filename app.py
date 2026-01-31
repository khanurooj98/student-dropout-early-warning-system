import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------- Load saved objects ----------
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Student Dropout Risk Dashboard", layout="wide")

st.title("🎓 Student Dropout Risk Prediction Dashboard")

# ---------- Upload CSV ----------
uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # ---------- Drop target if exists ----------
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # ---------- Preprocess ----------
    X_processed = preprocessor.transform(df)

    # ---------- Predictions ----------
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)

    risk_labels = label_encoder.inverse_transform(preds)

    result_df = df.copy()
    result_df["Risk_Class"] = preds
    result_df["Risk_Label"] = risk_labels
    result_df["Risk_Score"] = probs.max(axis=1)

    # ---------- Top 20 High-Risk ----------
    st.subheader("🚨 Top 20 High-Risk Students")
    high_risk = result_df[result_df["Risk_Label"] == "L"].sort_values(
        "Risk_Score", ascending=False
    ).head(20)

    st.dataframe(high_risk)

    # ---------- Select Student ----------
    st.subheader("👤 Select a Student")
    selected_index = st.selectbox("Choose Student Index", result_df.index)

    student = result_df.loc[selected_index]

    st.metric("Risk Level", student["Risk_Label"])
    st.metric("Risk Score", round(student["Risk_Score"], 3))

    # ---------- Feature Importance ----------
    st.subheader("📊 Top Reasons (Feature Importance)")

    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.head(10).set_index("Feature"))
    else:
        st.warning("Feature importance not available for this model.")
