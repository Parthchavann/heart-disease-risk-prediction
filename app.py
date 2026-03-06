"""
Streamlit frontend for Heart Disease Risk Prediction.
Run with: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config.settings import settings

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached model loading (avoids reloading on every interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading prediction model...")
def load_service():
    from src.prediction_service import HeartDiseasePredictionService
    return HeartDiseasePredictionService()


# --------------------------------------------------------------------------- #
# Helper: risk gauge
# --------------------------------------------------------------------------- #
def render_risk_gauge(probability: float, risk_level: str):
    colour = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}.get(risk_level, "#95a5a6")

    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=18, solid_capstyle="round")

    # Filled arc proportional to probability
    end_angle = np.pi - probability * np.pi
    theta_fill = np.linspace(np.pi, end_angle, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=colour, linewidth=18, solid_capstyle="round")

    ax.text(0, 0.15, f"{probability:.0%}", ha="center", va="center", fontsize=26, fontweight="bold", color=colour)
    ax.text(0, -0.25, risk_level, ha="center", va="center", fontsize=14, color=colour)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis("off")
    return fig


# --------------------------------------------------------------------------- #
# Helper: SHAP bar chart
# --------------------------------------------------------------------------- #
def render_shap_bar(risk_factors, protective_factors):
    items = []
    for f in risk_factors[:5]:
        items.append((f["feature"], abs(f["contribution"]), "#e74c3c"))
    for f in protective_factors[:3]:
        items.append((f["feature"], -abs(f["contribution"]), "#2ecc71"))

    if not items:
        return None

    items.sort(key=lambda x: x[1])
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colours = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(6, max(3, len(items) * 0.5)))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh(labels, values, color=colours, edgecolor="none", height=0.5)
    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP contribution", fontsize=10)
    ax.set_title("Feature Impact on Risk", fontsize=11, fontweight="bold")

    risk_patch = mpatches.Patch(color="#e74c3c", label="Increases risk")
    prot_patch = mpatches.Patch(color="#2ecc71", label="Decreases risk")
    ax.legend(handles=[risk_patch, prot_patch], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar - patient input form
# --------------------------------------------------------------------------- #
st.sidebar.title("Patient Data Input")
st.sidebar.markdown("Fill in the health indicators below:")

with st.sidebar.form("patient_form"):
    age = st.slider("Age", 18, 100, 55)
    sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    cp = st.selectbox(
        "Chest Pain Type",
        [(0, "Typical angina"), (1, "Atypical angina"), (2, "Non-anginal pain"), (3, "Asymptomatic")],
        format_func=lambda x: x[1],
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
    restecg = st.selectbox(
        "Resting ECG",
        [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")],
        format_func=lambda x: x[1],
    )
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
    slope = st.selectbox(
        "ST Slope",
        [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
        format_func=lambda x: x[1],
    )
    ca = st.selectbox("Major Vessels Coloured (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox(
        "Thalassemia",
        [(0, "Normal"), (1, "Fixed defect"), (2, "Reversable defect"), (3, "Unknown")],
        format_func=lambda x: x[1],
    )

    include_llm = st.checkbox("Include AI explanation (slower)", value=False)
    submitted = st.form_submit_button("Assess Risk", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.title("Heart Disease Risk Prediction")
st.markdown(
    "> **Disclaimer**: This tool is for **educational and preventive awareness purposes only**. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult with a qualified healthcare provider for medical decisions."
)

if not submitted:
    st.info("Fill in the patient data in the sidebar and click **Assess Risk** to get your result.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("Validation ROC-AUC", "0.9561")
    with col3:
        st.metric("Test ROC-AUC", "0.8706")

    st.markdown("### How it works")
    st.markdown(
        "1. Enter patient health indicators in the sidebar\n"
        "2. The ML model (XGBoost) predicts cardiovascular risk\n"
        "3. SHAP values explain which factors drive the risk\n"
        "4. Optional: AI generates a patient-friendly explanation\n"
    )

else:
    patient_data = {
        "age": age,
        "sex": sex[1],
        "cp": cp[0],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs[0],
        "restecg": restecg[0],
        "thalach": thalach,
        "exang": exang[0],
        "oldpeak": oldpeak,
        "slope": slope[0],
        "ca": ca,
        "thal": thal[0],
    }

    with st.spinner("Running prediction..."):
        try:
            service = load_service()
            result = service.predict(
                patient_data,
                include_explanation=True,
                include_llm_explanation=include_llm,
            )
        except Exception as e:
            st.error(f"Prediction service error: {e}")
            st.stop()

    if not result.get("success"):
        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        if result.get("validation_errors"):
            for err in result["validation_errors"]:
                st.warning(err)
        st.stop()

    risk_prob = result["risk_probability"]
    risk_level = result["risk_level"]
    ci = result.get("confidence_interval", (risk_prob - 0.1, risk_prob + 0.1))

    # ------------------------------------------------------------------ #
    # Row 1: gauge + summary cards
    # ------------------------------------------------------------------ #
    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        fig_gauge = render_risk_gauge(risk_prob, risk_level)
        st.pyplot(fig_gauge, use_container_width=True)
        st.caption(f"Confidence interval: [{ci[0]:.0%} – {ci[1]:.0%}]")

    with col_info:
        st.subheader("Prediction Summary")
        model_info = result.get("model_info", {})

        summary_data = {
            "Risk Probability": f"{risk_prob:.1%}",
            "Risk Level": risk_level,
            "Binary Prediction": "Disease Indicated" if result.get("prediction") == 1 else "No Disease Indicated",
            "Model": model_info.get("model_name", "N/A"),
            "Model Version": model_info.get("model_version", "N/A"),
        }
        for k, v in summary_data.items():
            cols = st.columns([1, 2])
            cols[0].markdown(f"**{k}**")
            cols[1].markdown(v)

    st.divider()

    # ------------------------------------------------------------------ #
    # Row 2: SHAP explanation
    # ------------------------------------------------------------------ #
    explanation = result.get("explanation")
    if explanation:
        detailed = explanation.get("detailed_explanation", {})
        risk_factors = detailed.get("top_risk_factors", [])
        protective_factors = detailed.get("top_protective_factors", [])

        col_shap, col_lists = st.columns([1.4, 1])

        with col_shap:
            st.subheader("Feature Impact (SHAP)")
            fig_shap = render_shap_bar(risk_factors, protective_factors)
            if fig_shap:
                st.pyplot(fig_shap, use_container_width=True)

        with col_lists:
            if risk_factors:
                st.subheader("Top Risk Factors")
                for f in risk_factors[:5]:
                    with st.expander(f["feature"].replace("_", " ").title()):
                        st.write(f.get("explanation", ""))
                        st.metric("Your Value", f"{f['feature_value']:.1f}", delta=f"{f['contribution']:+.3f} risk impact")

            if protective_factors:
                st.subheader("Protective Factors")
                for f in protective_factors[:3]:
                    with st.expander(f["feature"].replace("_", " ").title()):
                        st.write(f.get("explanation", ""))
                        st.metric("Your Value", f"{f['feature_value']:.1f}", delta=f"{f['contribution']:+.3f} risk impact")

        st.divider()

    # ------------------------------------------------------------------ #
    # Row 3: LLM explanation (if requested)
    # ------------------------------------------------------------------ #
    llm_exp = result.get("llm_explanation")
    if llm_exp:
        st.subheader("AI-Generated Explanation")
        st.info(llm_exp.get("risk_explanation", ""))

        col_rec, col_q = st.columns(2)

        with col_rec:
            st.subheader("Lifestyle Recommendations")
            for i, rec in enumerate(llm_exp.get("lifestyle_recommendations", []), 1):
                st.markdown(f"{i}. {rec}")

        with col_q:
            st.subheader("Questions to Ask Your Doctor")
            for i, q in enumerate(llm_exp.get("doctor_consultation_questions", []), 1):
                st.markdown(f"{i}. {q}")

        st.divider()

    # ------------------------------------------------------------------ #
    # Row 4: Input data table
    # ------------------------------------------------------------------ #
    with st.expander("View Raw Input Data"):
        st.dataframe(pd.DataFrame([patient_data]).T.rename(columns={0: "Value"}))

    # ------------------------------------------------------------------ #
    # Footer disclaimer
    # ------------------------------------------------------------------ #
    st.caption(result.get("medical_disclaimer", settings.MEDICAL_DISCLAIMER))
