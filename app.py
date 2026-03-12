"""
Heart Risk AI — Premium Streamlit Frontend
Run: PYTHONPATH=. python -m streamlit run app.py --server.port 8502
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.settings import settings

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Risk AI", page_icon="🫀", layout="wide",
                   initial_sidebar_state="auto")

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"page": "landing", "result": None, "inputs": {}, "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"] { display:none !important; }

/* ── Canvas ── */
.stApp { background: #f0f2f8; }
.main .block-container {
  padding: 0 !important; margin: 0 !important;
  max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f2f8; }
::-webkit-scrollbar-thumb { background: #c7d0e0; border-radius: 99px; }

/* ── Radio pill style ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
  display: flex; flex-wrap: wrap; gap: 8px;
}
div[data-testid="stRadio"] > div > label {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1.5px solid #e2e8f0; border-radius: 10px;
  padding: 9px 16px; cursor: pointer;
  background: white; color: #374151;
  font-size: .85rem; font-weight: 500;
  transition: all .15s ease;
}
div[data-testid="stRadio"] > div > label:hover {
  border-color: #818cf8; color: #4f46e5;
  box-shadow: 0 0 0 3px rgba(99,102,241,.1);
}
div[data-testid="stRadio"] > div > label[data-selected="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
  border-color: #6366f1; background: #6366f1; color: white;
  box-shadow: 0 2px 8px rgba(99,102,241,.35);
}
div[data-testid="stRadio"] input { position: absolute; opacity: 0; pointer-events: none; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  border: 1.5px solid #e2e8f0 !important; border-radius: 10px !important;
  background: white !important; font-size: .88rem !important;
  transition: border-color .15s;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.1) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: #6366f1 !important; border-color: #6366f1 !important;
  width: 18px !important; height: 18px !important;
  box-shadow: 0 2px 6px rgba(99,102,241,.4) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"] { display:none; }

/* ── Buttons ── */
.stButton > button {
  border-radius: 10px !important; font-weight: 600 !important;
  font-size: .88rem !important; letter-spacing: .01em !important;
  transition: all .15s ease !important; border: none !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(99,102,241,.4) !important;
  padding: .65rem 2rem !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(99,102,241,.5) !important;
}
.stButton > button[kind="secondary"] {
  background: white !important; color: #374151 !important;
  border: 1.5px solid #e2e8f0 !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: #6366f1 !important; color: #6366f1 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px; border-bottom: 1.5px solid #e5e7eb; padding-bottom: 0;
}
[data-testid="stTabs"] button[role="tab"] {
  border-radius: 8px 8px 0 0 !important; font-weight: 600 !important;
  font-size: .88rem !important; color: #6b7280 !important;
  padding: .6rem 1.25rem !important; border: none !important;
  background: transparent !important; transition: all .15s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #6366f1 !important;
  border-bottom: 2.5px solid #6366f1 !important;
  background: rgba(99,102,241,.05) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1.5px solid #e5e7eb !important; border-radius: 12px !important;
  background: white !important; overflow: hidden;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: #111827 !important; }
[data-testid="stMetricLabel"] { font-size: .78rem !important; color: #6b7280 !important; text-transform: uppercase; letter-spacing: .05em; }

/* ── Slider value label ── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] p { color: #374151 !important; }

/* ── Select slider ── */
[data-testid="stSelectSlider"] p,
[data-testid="stSelectSlider"] span { color: #374151 !important; }
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: #111827 !important; }

/* ── Toggle ── */
[data-testid="stToggle"] p,
[data-testid="stToggle"] span { color: #374151 !important; }

/* ── General text in white card contexts ── */
.stApp p { color: #374151; }
.stApp label { color: #374151 !important; }

/* ── Expander header text ── */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: #374151 !important; }

/* ── DataFrame text ── */
[data-testid="stDataFrame"] { color: #111827; }

/* ── Sidebar history panel ── */
[data-testid="stSidebar"] {
  background: #1e1b4b !important;
  border-right: 1px solid rgba(99,102,241,.2) !important;
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c7d2fe !important; }
[data-testid="stSidebar"] .stButton > button {
  background: rgba(99,102,241,.25) !important;
  color: #e0e7ff !important;
  border: 1px solid rgba(99,102,241,.4) !important;
  border-radius: 8px !important; font-size: .8rem !important;
  padding: .4rem .8rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(99,102,241,.45) !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
CP_LABELS   = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
ECG_LABELS  = ["Normal", "ST-T abnormality", "LV hypertrophy"]
SLOPE_LABELS = ["Upsloping", "Flat", "Downsloping"]
THAL_LABELS = ["Normal", "Fixed defect", "Reversible defect", "Unknown"]


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_service():
    from src.prediction_service import HeartDiseasePredictionService
    return HeartDiseasePredictionService()


def gauge(prob: float, level: str) -> go.Figure:
    colour = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}.get(level, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 52, "color": colour, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"color": "rgba(0,0,0,0)"}},
            "bar": {"color": colour, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "rgba(16,185,129,.12)"},
                {"range": [35, 65], "color": "rgba(245,158,11,.12)"},
                {"range": [65, 100],"color": "rgba(239,68,68,.12)"},
            ],
        },
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220,
    )
    return fig


DISPLAY_NAMES = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Blood Pressure", "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "ST Slope",
    "ca": "Major Vessels Blocked", "thal": "Thalassemia",
    "age_group": "Age Group", "bp_chol_ratio": "BP-Cholesterol Load",
    "hr_reserve": "Heart Rate Reserve", "multiple_risk_factors": "Multiple Risk Factors",
    "age_chol": "Age × Cholesterol", "bp_hr_ratio": "BP-Heart Rate Ratio",
    "oldpeak_exang": "ST Depression + Angina", "ca_thal_risk": "Vessel + Thalassemia Risk",
    "cp_exang_combo": "Chest Pain + Exercise Angina",
}
FEATURE_UNITS = {
    "age": "yrs", "trestbps": "mm Hg", "chol": "mg/dl",
    "thalach": "bpm", "oldpeak": "",
}
BASE_FEATURES = {"age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"}


def _fmt_value(feature: str, val) -> str:
    """Return a human-readable value string for a feature."""
    try:
        v = float(val)
        unit = FEATURE_UNITS.get(feature, "")
        # For base features show clean integer/decimal; engineered features show N/A
        if feature in BASE_FEATURES:
            display = str(int(v)) if v == int(v) else f"{v:.1f}"
            return f"{display} {unit}".strip()
        return "combined factor"
    except Exception:
        return str(val)


def shap_fig(risk_fs: list, prot_fs: list) -> go.Figure:
    rows = []
    for f in (risk_fs or [])[:6]:
        rows.append({"name": DISPLAY_NAMES.get(f["feature"], f["feature"].replace("_"," ").title()),
                     "val": abs(f["contribution"]), "col": "#ef4444",
                     "tip": f.get("explanation",""), "raw": _fmt_value(f["feature"], f["feature_value"])})
    for f in (prot_fs or [])[:4]:
        rows.append({"name": DISPLAY_NAMES.get(f["feature"], f["feature"].replace("_"," ").title()),
                     "val": -abs(f["contribution"]), "col": "#10b981",
                     "tip": f.get("explanation",""), "raw": _fmt_value(f["feature"], f["feature_value"])})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("val")
    fig = go.Figure(go.Bar(
        x=df["val"], y=df["name"], orientation="h",
        marker=dict(color=df["col"].tolist(), line_width=0),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1.5, line_dash="dot")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="← protective  |  risk →", gridcolor="#f1f5f9",
                   zeroline=False, title_font=dict(size=11, color="#94a3b8"),
                   tickfont=dict(size=11, color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color="#374151")),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(280, len(rows) * 44),
        font=dict(family="Inter"),
        showlegend=False,
    )
    return fig


def pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg};color:{color};padding:3px 10px;border-radius:99px;font-size:.75rem;font-weight:600;'>{text}</span>"


def card_open(extra_style=""):
    return f"<div style='background:white;border-radius:16px;padding:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);{extra_style}'>"

def card_close():
    return "</div>"


# Hide sidebar on landing and form pages
if st.session_state.page in ("landing", "form"):
    st.markdown("<style>[data-testid='stSidebar']{display:none!important;}</style>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    # ── Topbar ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:white;border-bottom:1px solid #f0f2f8;padding:.9rem 2.5rem;
                display:flex;align-items:center;gap:.6rem;position:sticky;top:0;z-index:100;'>
      <span style='font-size:1.3rem;'>🫀</span>
      <span style='font-weight:800;font-size:1.05rem;color:#111827;letter-spacing:-.01em;'>Heart Risk AI</span>
      <span style='margin-left:auto;font-size:.75rem;color:#9ca3af;'>v1.0 · Stacking Ensemble · AUC 0.9407</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#4c1d95 100%);
                padding:5rem 2.5rem 4rem;text-align:center;'>
      <div style='display:inline-flex;align-items:center;gap:.5rem;
                  background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
                  border-radius:99px;padding:.35rem 1rem;margin-bottom:1.5rem;'>
        <span style='width:6px;height:6px;background:#a5f3fc;border-radius:50%;display:inline-block;'></span>
        <span style='font-size:.78rem;color:#c7d2fe;font-weight:500;letter-spacing:.04em;'>
          POWERED BY STACKING ENSEMBLE · 5 AI MODELS
        </span>
      </div>
      <h1 style='font-size:3.2rem;font-weight:900;color:white;letter-spacing:-.03em;line-height:1.1;margin-bottom:1rem;'>
        Know your heart risk.<br>
        <span style='background:linear-gradient(90deg,#a5b4fc,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
          Before it's too late.
        </span>
      </h1>
      <p style='font-size:1.05rem;color:#a5b4fc;max-width:520px;margin:0 auto 2.5rem;line-height:1.7;'>
        Enter 13 health indicators. Get an AI-powered cardiovascular risk score
        with SHAP explanations in seconds.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats strip ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:white;border-bottom:1px solid #f0f2f8;padding:1.25rem 2.5rem;
                display:flex;gap:2.5rem;justify-content:center;flex-wrap:wrap;'>
      <div style='text-align:center;'>
        <div style='font-size:1.5rem;font-weight:800;color:#111827;letter-spacing:-.02em;'>94.1%</div>
        <div style='font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;'>ROC-AUC</div>
      </div>
      <div style='width:1px;background:#f0f2f8;'></div>
      <div style='text-align:center;'>
        <div style='font-size:1.5rem;font-weight:800;color:#111827;'>84.9%</div>
        <div style='font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;'>Accuracy</div>
      </div>
      <div style='width:1px;background:#f0f2f8;'></div>
      <div style='text-align:center;'>
        <div style='font-size:1.5rem;font-weight:800;color:#111827;'>90.5%</div>
        <div style='font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;'>Specificity</div>
      </div>
      <div style='width:1px;background:#f0f2f8;'></div>
      <div style='text-align:center;'>
        <div style='font-size:1.5rem;font-weight:800;color:#111827;'>2,690</div>
        <div style='font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;'>Training Samples</div>
      </div>
      <div style='width:1px;background:#f0f2f8;'></div>
      <div style='text-align:center;'>
        <div style='font-size:1.5rem;font-weight:800;color:#111827;'>22</div>
        <div style='font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;'>Features</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    st.markdown("<div style='text-align:center;padding:2.5rem 1rem 0;'>", unsafe_allow_html=True)
    col = st.columns([2,1,2])[1]
    with col:
        if st.button("Start Assessment →", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("<div style='max-width:900px;margin:3rem auto;padding:0 2rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='font-size:1.5rem;font-weight:800;color:#111827;text-align:center;
               letter-spacing:-.02em;margin-bottom:.5rem;'>How it works</h2>
    <p style='text-align:center;color:#6b7280;font-size:.9rem;margin-bottom:2rem;'>
      Four steps from input to insight
    </p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "#6366f1", "Enter Details", "13 clinical health indicators — no lab coat required."),
        ("02", "#8b5cf6", "AI Prediction", "5-model stacking ensemble calculates your risk score."),
        ("03", "#a855f7", "SHAP Analysis", "See exactly which factors drive your result."),
        ("04", "#ec4899", "Personalised Advice", "Optional Gemini AI generates tailored recommendations."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (num, color, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:16px;padding:1.5rem;
                        box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
                        height:100%;'>
              <div style='font-size:2rem;font-weight:900;color:{color};
                          font-variant-numeric:tabular-nums;margin-bottom:.75rem;
                          font-family:Inter;'>{num}</div>
              <div style='font-weight:700;color:#111827;font-size:.95rem;
                          margin-bottom:.4rem;'>{title}</div>
              <div style='color:#6b7280;font-size:.82rem;line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Model architecture ────────────────────────────────────────────────────
    st.markdown("<div style='max-width:900px;margin:0 auto 3rem;padding:0 2rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e1b4b);border-radius:20px;
                padding:2rem 2.5rem;margin-bottom:1.5rem;'>
      <div style='font-size:.72rem;color:#818cf8;text-transform:uppercase;letter-spacing:.1em;
                  margin-bottom:.5rem;'>Model Architecture</div>
      <div style='font-size:1.15rem;font-weight:700;color:white;margin-bottom:1.25rem;'>
        Stacking Ensemble — 5 base learners + LogReg meta
      </div>
      <div style='display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem;'>
        <span style='background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
                     color:#a5b4fc;padding:5px 12px;border-radius:8px;font-size:.8rem;font-weight:600;'>CatBoost · 300 trees</span>
        <span style='background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
                     color:#a5b4fc;padding:5px 12px;border-radius:8px;font-size:.8rem;font-weight:600;'>XGBoost · 267 trees</span>
        <span style='background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
                     color:#a5b4fc;padding:5px 12px;border-radius:8px;font-size:.8rem;font-weight:600;'>LightGBM · 200 trees</span>
        <span style='background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
                     color:#a5b4fc;padding:5px 12px;border-radius:8px;font-size:.8rem;font-weight:600;'>Random Forest · 200 trees</span>
        <span style='background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
                     color:#a5b4fc;padding:5px 12px;border-radius:8px;font-size:.8rem;font-weight:600;'>Extra Trees · 100 trees</span>
      </div>
      <div style='background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
                  border-radius:10px;padding:.75rem 1.25rem;font-size:.8rem;color:#94a3b8;'>
        ⚠️ For educational and preventive awareness purposes only. Not a substitute for professional medical advice.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT FORM
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "form":

    # Topbar
    st.markdown("""
    <div style='background:white;border-bottom:1px solid #f0f2f8;padding:.9rem 2.5rem;
                display:flex;align-items:center;gap:.6rem;'>
      <span style='font-size:1.3rem;'>🫀</span>
      <span style='font-weight:800;font-size:1.05rem;color:#111827;'>Heart Risk AI</span>
      <span style='margin-left:auto;font-size:.8rem;color:#9ca3af;'>Health Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div style='max-width:760px;margin:2.5rem auto 2rem;padding:0 1.5rem;'>
      <div style='font-size:.75rem;color:#6366f1;font-weight:600;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.5rem;'>Step 1 of 1</div>
      <h1 style='font-size:1.75rem;font-weight:800;color:#111827;letter-spacing:-.02em;'>
        Your Health Indicators
      </h1>
      <p style='color:#6b7280;font-size:.9rem;margin-top:.4rem;'>
        Fill in your clinical details below. All fields are required.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Form wrapper ──────────────────────────────────────────────────────────
    with st.container():
        st.markdown("<div style='max-width:760px;margin:0 auto;padding:0 1.5rem;'>", unsafe_allow_html=True)

        # ── Section: Personal ─────────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:1rem;padding-bottom:.5rem;
                    border-bottom:1px solid #f0f2f8;'>
          Personal Information
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.5rem;'>Age</div>", unsafe_allow_html=True)
            age = st.slider("", 18, 100, st.session_state.inputs.get("age", 52),
                            key="sl_age", label_visibility="collapsed")
            st.markdown(f"<div style='font-size:1.75rem;font-weight:800;color:#6366f1;margin-top:-.25rem;'>{age} <span style='font-size:.85rem;color:#9ca3af;font-weight:400;'>years</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Biological Sex</div>", unsafe_allow_html=True)
            sex_str = st.radio("sex_r", ["Male", "Female"], horizontal=True,
                               index=0 if st.session_state.inputs.get("sex", 1) == 1 else 1,
                               label_visibility="collapsed")
            sex = 1 if sex_str == "Male" else 0
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Section: Symptoms ─────────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.1em;margin:1.5rem 0 1rem;padding-bottom:.5rem;
                    border-bottom:1px solid #f0f2f8;'>
          Symptoms & History
        </div>
        """, unsafe_allow_html=True)

        # Chest pain
        st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Chest Pain Type</div>", unsafe_allow_html=True)
        cp_str = st.radio("cp_r", CP_LABELS, horizontal=True,
                          index=st.session_state.inputs.get("cp", 0),
                          label_visibility="collapsed")
        cp = CP_LABELS.index(cp_str)
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Exercise-Induced Angina</div>", unsafe_allow_html=True)
            exang_str = st.radio("exang_r", ["No", "Yes"], horizontal=True,
                                  index=st.session_state.inputs.get("exang", 0),
                                  label_visibility="collapsed")
            exang = 1 if exang_str == "Yes" else 0
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Fasting Blood Sugar &gt; 120 mg/dl</div>", unsafe_allow_html=True)
            fbs_str = st.radio("fbs_r", ["No", "Yes"], horizontal=True,
                                index=st.session_state.inputs.get("fbs", 0),
                                label_visibility="collapsed")
            fbs = 1 if fbs_str == "Yes" else 0
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Section: Measurements ─────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.1em;margin:1.5rem 0 1rem;padding-bottom:.5rem;
                    border-bottom:1px solid #f0f2f8;'>
          Measurements
        </div>
        """, unsafe_allow_html=True)

        def measure_card(label, unit, val, ref, color="#6366f1"):
            status_color = color
            return f"""
            <div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;
                        box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);
                        margin-bottom:1rem;'>
              <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.75rem;'>
                <div style='font-size:.8rem;font-weight:600;color:#374151;'>{label}</div>
                <div style='font-size:.72rem;color:#9ca3af;'>Normal: {ref}</div>
              </div>
              {{slider}}
              <div style='font-size:1.5rem;font-weight:800;color:{status_color};margin-top:.25rem;'>
                {val} <span style='font-size:.8rem;color:#9ca3af;font-weight:400;'>{unit}</span>
              </div>
            </div>"""

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;'><div style='font-size:.8rem;font-weight:600;color:#374151;'>Resting Blood Pressure</div><div style='font-size:.72rem;color:#9ca3af;'>Normal &lt;120</div></div>", unsafe_allow_html=True)
            trestbps = st.slider("", 80, 200, st.session_state.inputs.get("trestbps", 130),
                                  key="sl_bp", label_visibility="collapsed")
            bp_c = "#10b981" if trestbps < 120 else ("#f59e0b" if trestbps < 140 else "#ef4444")
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:{bp_c};'>{trestbps} <span style='font-size:.8rem;color:#9ca3af;font-weight:400;'>mm Hg</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;'><div style='font-size:.8rem;font-weight:600;color:#374151;'>Serum Cholesterol</div><div style='font-size:.72rem;color:#9ca3af;'>Desirable &lt;200</div></div>", unsafe_allow_html=True)
            chol = st.slider("", 100, 600, st.session_state.inputs.get("chol", 240),
                              key="sl_chol", label_visibility="collapsed")
            ch_c = "#10b981" if chol < 200 else ("#f59e0b" if chol < 240 else "#ef4444")
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:{ch_c};'>{chol} <span style='font-size:.8rem;color:#9ca3af;font-weight:400;'>mg/dl</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.5rem;'>Max Heart Rate Achieved</div>", unsafe_allow_html=True)
            thalach = st.slider("", 60, 220, st.session_state.inputs.get("thalach", 152),
                                 key="sl_hr", label_visibility="collapsed")
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:#6366f1;'>{thalach} <span style='font-size:.8rem;color:#9ca3af;font-weight:400;'>bpm</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;'><div style='font-size:.8rem;font-weight:600;color:#374151;'>ST Depression (oldpeak)</div><div style='font-size:.72rem;color:#9ca3af;'>Normal: 0</div></div>", unsafe_allow_html=True)
            oldpeak = st.slider("", 0.0, 6.2, st.session_state.inputs.get("oldpeak", 0.0),
                                 step=0.1, key="sl_op", label_visibility="collapsed")
            op_c = "#10b981" if oldpeak < 1 else ("#f59e0b" if oldpeak < 2.5 else "#ef4444")
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:{op_c};'>{oldpeak:.1f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Section: Diagnostics ──────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.1em;margin:1.5rem 0 1rem;padding-bottom:.5rem;
                    border-bottom:1px solid #f0f2f8;'>
          Diagnostic Results
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Resting ECG Result</div>", unsafe_allow_html=True)
            ecg_str = st.radio("ecg_r", ECG_LABELS, horizontal=False,
                                index=st.session_state.inputs.get("restecg", 0),
                                label_visibility="collapsed")
            restecg = ECG_LABELS.index(ecg_str)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>ST Slope</div>", unsafe_allow_html=True)
            slope_str = st.radio("slope_r", SLOPE_LABELS, horizontal=False,
                                  index=st.session_state.inputs.get("slope", 0),
                                  label_visibility="collapsed")
            slope = SLOPE_LABELS.index(slope_str)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Major Vessels Coloured (0–4)</div>", unsafe_allow_html=True)
            ca = st.select_slider("", options=[0,1,2,3,4],
                                   value=st.session_state.inputs.get("ca", 0),
                                   key="sl_ca", label_visibility="collapsed")
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:{'#ef4444' if ca >= 2 else '#6366f1'};'>{ca}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='background:white;border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);margin-bottom:1rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.75rem;'>Thalassemia</div>", unsafe_allow_html=True)
            thal_str = st.radio("thal_r", THAL_LABELS, horizontal=False,
                                 index=st.session_state.inputs.get("thal", 0),
                                 label_visibility="collapsed")
            thal = THAL_LABELS.index(thal_str)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Clinical range warnings ────────────────────────────────────────────
        warnings_list = []
        if trestbps >= 140:
            warnings_list.append(("Blood Pressure", f"{trestbps} mm Hg", "Stage 1+ hypertension (≥140 mm Hg). Consider lifestyle changes and consult your doctor."))
        if chol >= 240:
            warnings_list.append(("Cholesterol", f"{chol} mg/dl", "High cholesterol (≥240 mg/dl). Associated with increased arterial plaque risk."))
        if chol >= 200 and chol < 240:
            warnings_list.append(("Cholesterol", f"{chol} mg/dl", "Borderline high cholesterol (200–239 mg/dl). Diet and exercise can help."))
        if oldpeak >= 2.0:
            warnings_list.append(("ST Depression", f"{oldpeak}", "Significant ST depression (≥2.0). May indicate myocardial ischemia under stress."))
        if thalach < 100:
            warnings_list.append(("Max Heart Rate", f"{thalach} bpm", "Unusually low maximum heart rate (<100 bpm). Could indicate reduced cardiac fitness."))
        if age >= 65 and sex == 1:
            warnings_list.append(("Age & Sex", f"{age} yrs, Male", "Men aged 65+ have significantly elevated baseline cardiovascular risk."))
        if ca >= 2:
            warnings_list.append(("Major Vessels", f"{ca} blocked", "Multiple blocked vessels (≥2) is a strong indicator of coronary artery disease."))

        if warnings_list:
            st.markdown("""
            <div style='background:#fffbeb;border:1px solid #fcd34d;border-radius:14px;
                        padding:1.25rem 1.5rem;margin-top:.5rem;margin-bottom:1rem;'>
              <div style='font-size:.85rem;font-weight:700;color:#92400e;margin-bottom:.75rem;
                          display:flex;align-items:center;gap:.4rem;'>
                ⚠️ Clinical Range Alerts
              </div>
            """, unsafe_allow_html=True)
            for label, value, msg in warnings_list:
                st.markdown(f"""
              <div style='display:flex;gap:.75rem;margin-bottom:.6rem;align-items:flex-start;'>
                <div style='background:#fef3c7;border-radius:8px;padding:.25rem .6rem;
                            font-size:.75rem;font-weight:700;color:#b45309;white-space:nowrap;
                            min-width:fit-content;'>{label}: {value}</div>
                <div style='font-size:.8rem;color:#78350f;line-height:1.5;'>{msg}</div>
              </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── AI toggle + Submit ────────────────────────────────────────────────
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f5f3ff,#ede9fe);border-radius:14px;
                    padding:1.25rem 1.5rem;border:1px solid #ddd6fe;margin-top:.5rem;margin-bottom:1.5rem;'>
          <div style='font-size:.85rem;font-weight:600;color:#4c1d95;margin-bottom:.25rem;'>
            🤖 AI-Powered Explanation
          </div>
          <div style='font-size:.8rem;color:#6d28d9;line-height:1.5;'>
            Enable to get a personalised explanation and lifestyle recommendations powered by Gemini AI.
            Adds ~5 seconds to processing.
          </div>
        </div>
        """, unsafe_allow_html=True)
        include_llm = st.toggle("Include Gemini AI explanation",
                                 value=st.session_state.inputs.get("include_llm", False))

        c1, c2, _ = st.columns([1, 1.5, 1])
        with c1:
            if st.button("← Back", type="secondary", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()
        with c2:
            submit = st.button("Analyse Risk →", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit handler ────────────────────────────────────────────────────────
    if submit:
        patient_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        st.session_state.inputs = {**patient_data, "include_llm": include_llm}

        with st.spinner("Running AI prediction..."):
            try:
                service = load_service()
                result = service.predict(patient_data, include_explanation=True,
                                         include_llm_explanation=include_llm)
                st.session_state.result = result
                st.session_state.page = "results"
                # Save to in-session history (most recent first, cap at 10)
                from datetime import datetime as _dt
                history_entry = {
                    "timestamp": _dt.now().strftime("%H:%M:%S"),
                    "age": patient_data["age"],
                    "sex": "M" if patient_data["sex"] == 1 else "F",
                    "risk_level": result.get("risk_level", "?"),
                    "risk_pct": round(result.get("risk_probability", 0) * 100, 1),
                    "result": result,
                    "inputs": dict(patient_data),
                }
                st.session_state.history = ([history_entry] + st.session_state.history)[:10]
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Prediction History (shown on results page)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🫀 Session History")
    history = st.session_state.get("history", [])
    if not history:
        st.markdown("<p style='font-size:.82rem;color:#a5b4fc;'>No predictions yet.</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size:.78rem;color:#a5b4fc;margin-bottom:.75rem;'>"
                    f"{len(history)} prediction{'s' if len(history) > 1 else ''} this session</p>",
                    unsafe_allow_html=True)
        for i, h in enumerate(history):
            lvl = h["risk_level"]
            dot = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(lvl, "⚪")
            label = f"{dot} {h['timestamp']}  |  Age {h['age']}{h['sex']}  |  **{lvl} {h['risk_pct']}%**"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.result = h["result"]
                st.session_state.inputs = h["inputs"]
                st.rerun()
    st.markdown("---")
    if st.button("New Assessment", use_container_width=True):
        st.session_state.page = "form"
        st.rerun()
    if history and st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
result   = st.session_state.result or {}
inp      = st.session_state.inputs or {}

if not result or not result.get("success"):
    st.error("No result available. Please complete the assessment.")
    if st.button("← Go back"):
        st.session_state.page = "form"
        st.rerun()
    st.stop()

risk_prob   = result["risk_probability"]
risk_level  = result["risk_level"]
prediction  = result.get("prediction", 0)
ci          = result.get("confidence_interval", (max(0, risk_prob-.1), min(1, risk_prob+.1)))
explanation = result.get("explanation") or {}
risk_fs     = explanation.get("top_risk_factors", [])
prot_fs     = explanation.get("top_protective_factors", [])
llm_exp     = result.get("llm_explanation")
model_info  = result.get("model_info", {})

RISK_COLOR = {"Low":"#10b981","Moderate":"#f59e0b","High":"#ef4444"}.get(risk_level,"#6b7280")
RISK_BG    = {"Low":"#d1fae5","Moderate":"#fef3c7","High":"#fee2e2"}.get(risk_level,"#f3f4f6")
RISK_TEXT  = {"Low":"#065f46","Moderate":"#78350f","High":"#7f1d1d"}.get(risk_level,"#111827")

# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:white;border-bottom:1px solid #f0f2f8;padding:.9rem 2.5rem;
            display:flex;align-items:center;gap:.6rem;'>
  <span style='font-size:1.3rem;'>🫀</span>
  <span style='font-weight:800;font-size:1.05rem;color:#111827;'>Heart Risk AI</span>
  <span style='margin-left:auto;font-size:.8rem;color:#9ca3af;'>Assessment Complete</span>
</div>
""", unsafe_allow_html=True)

# ── Hero result banner ────────────────────────────────────────────────────────
pred_text  = "Heart Disease Indicated" if prediction == 1 else "No Disease Indicated"
pred_icon  = "⚠️" if prediction == 1 else "✅"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f172a,#1e1b4b);padding:2rem 2.5rem;
            display:flex;align-items:center;gap:2rem;flex-wrap:wrap;'>
  <div style='flex:1;min-width:220px;'>
    <div style='font-size:.75rem;color:#818cf8;text-transform:uppercase;letter-spacing:.08em;
                margin-bottom:.5rem;'>Risk Assessment Result</div>
    <div style='font-size:2rem;font-weight:800;color:white;letter-spacing:-.02em;line-height:1.2;'>
      {pred_icon} {pred_text}
    </div>
    <div style='margin-top:.75rem;display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;'>
      <span style='background:{RISK_BG};color:{RISK_TEXT};padding:5px 14px;border-radius:99px;
                   font-size:.82rem;font-weight:700;'>{risk_level} Risk</span>
      <span style='color:#94a3b8;font-size:.82rem;'>
        {risk_prob:.1%} probability · CI {ci[0]:.0%}–{ci[1]:.0%}
      </span>
    </div>
  </div>
  <div style='display:flex;gap:1rem;flex-wrap:wrap;'>
    <div style='background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                border-radius:12px;padding:.75rem 1.25rem;text-align:center;min-width:90px;'>
      <div style='font-size:1.3rem;font-weight:800;color:{RISK_COLOR};'>{risk_prob:.0%}</div>
      <div style='font-size:.7rem;color:#94a3b8;margin-top:.15rem;'>Risk Score</div>
    </div>
    <div style='background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                border-radius:12px;padding:.75rem 1.25rem;text-align:center;min-width:90px;'>
      <div style='font-size:1.3rem;font-weight:800;color:#60a5fa;'>0.9407</div>
      <div style='font-size:.7rem;color:#94a3b8;margin-top:.15rem;'>Model AUC</div>
    </div>
    <div style='background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                border-radius:12px;padding:.75rem 1.25rem;text-align:center;min-width:90px;'>
      <div style='font-size:1.3rem;font-weight:800;color:#a78bfa;'>22</div>
      <div style='font-size:.7rem;color:#94a3b8;margin-top:.15rem;'>Features Used</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:1.5rem 2.5rem 0;'>", unsafe_allow_html=True)
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button("← Edit Inputs", type="secondary"):
        st.session_state.page = "form"
        st.rerun()
with c2:
    if st.button("New Assessment", type="secondary"):
        st.session_state.result = None
        st.session_state.inputs = {}
        st.session_state.page   = "landing"
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:1rem 2.5rem 3rem;'>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  Overview  ", "  Feature Analysis  ", "  AI Guidance  "])

# ─────────────────────────────────────── TAB 1: OVERVIEW
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_s, col_v = st.columns([1, 1.2, 1.1], gap="large")

    # Gauge
    with col_g:
        st.markdown(f"""
        <div style='background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
                    border-top:3px solid {RISK_COLOR};text-align:center;'>
          <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:.5rem;'>Risk Probability</div>
        """, unsafe_allow_html=True)
        fig_g = gauge(risk_prob, risk_level)
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
          <div style='font-size:.8rem;color:#9ca3af;margin-top:-.75rem;'>
            Confidence interval: <strong>{ci[0]:.0%} – {ci[1]:.0%}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Summary card
    with col_s:
        ts = model_info.get("training_timestamp","20260309")[:8]
        st.markdown(f"""
        <div style='background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);height:100%;'>
          <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:1rem;'>Prediction Details</div>
          <div style='display:flex;flex-direction:column;gap:.85rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding-bottom:.85rem;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.85rem;color:#6b7280;'>Result</span>
              <span style='font-size:.85rem;font-weight:700;color:#111827;'>{pred_icon} {pred_text}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding-bottom:.85rem;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.85rem;color:#6b7280;'>Risk Level</span>
              <span style='background:{RISK_BG};color:{RISK_TEXT};padding:3px 10px;
                           border-radius:99px;font-size:.8rem;font-weight:700;'>{risk_level}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding-bottom:.85rem;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.85rem;color:#6b7280;'>Probability</span>
              <span style='font-size:.85rem;font-weight:700;color:{RISK_COLOR};'>{risk_prob:.1%}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding-bottom:.85rem;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.85rem;color:#6b7280;'>Model</span>
              <span style='font-size:.82rem;font-weight:600;color:#374151;'>Stacking Ensemble</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding-bottom:.85rem;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.85rem;color:#6b7280;'>Test AUC</span>
              <span style='font-size:.85rem;font-weight:700;color:#6366f1;'>0.9407</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
              <span style='font-size:.85rem;color:#6b7280;'>Trained</span>
              <span style='font-size:.82rem;color:#374151;'>{ts[:4]}-{ts[4:6]}-{ts[6:]}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Vitals card
    with col_v:
        bp_c  = "#10b981" if inp.get("trestbps",130)<120 else ("#f59e0b" if inp.get("trestbps",130)<140 else "#ef4444")
        bp_s  = "Normal" if inp.get("trestbps",130)<120 else ("Elevated" if inp.get("trestbps",130)<140 else "High")
        ch_c  = "#10b981" if inp.get("chol",240)<200 else ("#f59e0b" if inp.get("chol",240)<240 else "#ef4444")
        ch_s  = "Normal" if inp.get("chol",240)<200 else ("Borderline" if inp.get("chol",240)<240 else "High")
        op_c  = "#10b981" if inp.get("oldpeak",0)<1 else ("#f59e0b" if inp.get("oldpeak",0)<2.5 else "#ef4444")

        def vital_row(label, value, status, color):
            return f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:.7rem 0;border-bottom:1px solid #f9fafb;'>
              <span style='font-size:.83rem;color:#6b7280;'>{label}</span>
              <div style='text-align:right;'>
                <span style='font-size:.9rem;font-weight:700;color:{color};'>{value}</span>
                <span style='display:block;font-size:.7rem;color:{color};
                             background:{color}18;padding:1px 7px;border-radius:99px;
                             margin-top:2px;'>{status}</span>
              </div>
            </div>"""

        st.markdown(f"""
        <div style='background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);height:100%;'>
          <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:.75rem;'>Your Vitals</div>
          {vital_row("Blood Pressure", f"{inp.get('trestbps',130)} mm Hg", bp_s, bp_c)}
          {vital_row("Cholesterol", f"{inp.get('chol',240)} mg/dl", ch_s, ch_c)}
          {vital_row("Max Heart Rate", f"{inp.get('thalach',152)} bpm", "Recorded", "#6366f1")}
          {vital_row("ST Depression", f"{inp.get('oldpeak',0):.1f}", "Normal" if inp.get('oldpeak',0)<1 else "Elevated", op_c)}
          <div style='margin-top:.75rem;background:#f8fafc;border-radius:10px;padding:.7rem 1rem;
                      font-size:.8rem;color:#6b7280;'>
            <strong style='color:#374151;'>Age {inp.get("age",52)}</strong> ·
            {"Male" if inp.get("sex",1) else "Female"} ·
            {CP_LABELS[inp.get("cp",0)]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Risk context card
    st.markdown("<br>", unsafe_allow_html=True)
    context = {
        "Low": ("Your indicators suggest a low cardiovascular risk profile. Maintaining your current healthy habits will help keep your heart in great shape.", "#d1fae5", "#065f46", "#10b981"),
        "Moderate": ("Your indicators suggest a moderate cardiovascular risk. Some factors may need attention. Consider discussing preventive steps with your doctor.", "#fef3c7", "#78350f", "#f59e0b"),
        "High": ("Your indicators suggest an elevated cardiovascular risk. Several factors are contributing to this score. Please consult a healthcare professional soon.", "#fee2e2", "#7f1d1d", "#ef4444"),
    }.get(risk_level, ("", "#f3f4f6", "#111827", "#6b7280"))

    st.markdown(f"""
    <div style='background:{context[1]};border-radius:14px;padding:1.25rem 1.5rem;
                border-left:4px solid {context[3]};'>
      <div style='font-size:.78rem;font-weight:700;color:{context[2]};text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:.4rem;'>What This Means</div>
      <div style='font-size:.9rem;color:{context[2]};line-height:1.65;'>{context[0]}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────── TAB 2: FEATURE ANALYSIS
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    if not explanation:
        st.info("No SHAP explanation available.")
    else:
        col_chart, col_breakdown = st.columns([1.6, 1], gap="large")

        with col_chart:
            st.markdown("""
            <div style='background:white;border-radius:16px;padding:1.5rem;
                        box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);'>
              <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:.25rem;'>SHAP Feature Importance</div>
              <div style='font-size:.82rem;color:#6b7280;margin-bottom:.75rem;'>
                Each bar shows how much a feature shifts your risk probability.
              </div>
            """, unsafe_allow_html=True)
            fig_s = shap_fig(risk_fs, prot_fs)
            if fig_s:
                st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_breakdown:
            # Risk factors
            if risk_fs:
                st.markdown("""
                <div style='background:white;border-radius:16px;padding:1.5rem;
                            box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
                            margin-bottom:1rem;'>
                  <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:1rem;'>Top Risk Factors</div>
                """, unsafe_allow_html=True)
                for f in risk_fs[:5]:
                    name   = DISPLAY_NAMES.get(f["feature"], f["feature"].replace("_"," ").title())
                    contrib = f["contribution"]
                    pct    = min(100, int(abs(contrib)*900))
                    tip    = f.get("explanation","")
                    val_str = _fmt_value(f["feature"], f["feature_value"])
                    st.markdown(f"""
                    <div style='margin-bottom:.9rem;'>
                      <div style='display:flex;justify-content:space-between;align-items:baseline;
                                  margin-bottom:.3rem;'>
                        <span style='font-size:.85rem;font-weight:600;color:#111827;'>{name}</span>
                        <span style='font-size:.82rem;color:#6b7280;'>{val_str}</span>
                      </div>
                      <div style='background:#fee2e2;border-radius:99px;height:4px;'>
                        <div style='background:#ef4444;border-radius:99px;height:4px;width:{pct}%;'></div>
                      </div>
                      <div style='font-size:.74rem;color:#9ca3af;margin-top:.3rem;'>{tip}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Protective factors
            if prot_fs:
                st.markdown("""
                <div style='background:white;border-radius:16px;padding:1.5rem;
                            box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);'>
                  <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:1rem;'>Protective Factors</div>
                """, unsafe_allow_html=True)
                for f in prot_fs[:4]:
                    name    = DISPLAY_NAMES.get(f["feature"], f["feature"].replace("_"," ").title())
                    contrib = f["contribution"]
                    pct     = min(100, int(abs(contrib)*900))
                    tip     = f.get("explanation","")
                    val_str = _fmt_value(f["feature"], f["feature_value"])
                    st.markdown(f"""
                    <div style='margin-bottom:.9rem;'>
                      <div style='display:flex;justify-content:space-between;align-items:baseline;
                                  margin-bottom:.3rem;'>
                        <span style='font-size:.85rem;font-weight:600;color:#111827;'>{name}</span>
                        <span style='font-size:.82rem;color:#6b7280;'>{val_str}</span>
                      </div>
                      <div style='background:#d1fae5;border-radius:99px;height:4px;'>
                        <div style='background:#10b981;border-radius:99px;height:4px;width:{pct}%;'></div>
                      </div>
                      <div style='font-size:.74rem;color:#9ca3af;margin-top:.3rem;'>{tip}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Input summary table
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Full input summary"):
            readable = {
                "Age": inp.get("age",""), "Sex": "Male" if inp.get("sex")==1 else "Female",
                "Chest Pain": CP_LABELS[inp.get("cp",0)],
                "Blood Pressure": f"{inp.get('trestbps','')} mm Hg",
                "Cholesterol": f"{inp.get('chol','')} mg/dl",
                "Fasting Blood Sugar > 120": "Yes" if inp.get("fbs") else "No",
                "Resting ECG": ECG_LABELS[inp.get("restecg",0)],
                "Max Heart Rate": f"{inp.get('thalach','')} bpm",
                "Exercise Angina": "Yes" if inp.get("exang") else "No",
                "ST Depression": inp.get("oldpeak",""),
                "ST Slope": SLOPE_LABELS[inp.get("slope",0)],
                "Major Vessels": inp.get("ca",""),
                "Thalassemia": THAL_LABELS[inp.get("thal",0)],
            }
            df_in = pd.DataFrame(list(readable.items()), columns=["Feature","Value"])
            st.dataframe(df_in, use_container_width=True, hide_index=True)


# ─────────────────────────────────────── TAB 3: AI GUIDANCE
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    if llm_exp:
        # Risk explanation
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#f5f3ff,#ede9fe);border-radius:16px;
                    padding:1.5rem 1.75rem;border:1px solid #ddd6fe;margin-bottom:1.5rem;'>
          <div style='font-size:.75rem;font-weight:700;color:#7c3aed;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:.75rem;'>AI Analysis · Gemini 1.5 Flash</div>
          <div style='font-size:.92rem;color:#3b0764;line-height:1.75;'>
            {llm_exp.get("risk_explanation","No explanation available.")}
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_rec, col_q = st.columns(2, gap="large")

        with col_rec:
            st.markdown("""
            <div style='background:white;border-radius:16px;padding:1.5rem;
                        box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);'>
              <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:1rem;'>Lifestyle Recommendations</div>
            """, unsafe_allow_html=True)
            for i, rec in enumerate(llm_exp.get("lifestyle_recommendations", []), 1):
                st.markdown(f"""
                <div style='display:flex;gap:.75rem;padding:.8rem 0;
                            border-bottom:1px solid #f9fafb;align-items:flex-start;'>
                  <div style='background:#ede9fe;color:#6d28d9;border-radius:99px;
                              min-width:22px;height:22px;display:flex;align-items:center;
                              justify-content:center;font-size:.72rem;font-weight:700;
                              flex-shrink:0;margin-top:1px;'>{i}</div>
                  <div style='font-size:.86rem;color:#374151;line-height:1.55;'>{rec}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_q:
            st.markdown("""
            <div style='background:white;border-radius:16px;padding:1.5rem;
                        box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);'>
              <div style='font-size:.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:1rem;'>Questions for Your Doctor</div>
            """, unsafe_allow_html=True)
            for i, q in enumerate(llm_exp.get("doctor_consultation_questions", []), 1):
                st.markdown(f"""
                <div style='display:flex;gap:.75rem;padding:.8rem 0;
                            border-bottom:1px solid #f9fafb;align-items:flex-start;'>
                  <div style='background:#fef3c7;color:#92400e;border-radius:99px;
                              min-width:22px;height:22px;display:flex;align-items:center;
                              justify-content:center;font-size:.72rem;font-weight:700;
                              flex-shrink:0;margin-top:1px;'>?</div>
                  <div style='font-size:.86rem;color:#374151;line-height:1.55;'>{q}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 2rem;background:white;border-radius:16px;
                    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);'>
          <div style='font-size:2.5rem;margin-bottom:1rem;'>🤖</div>
          <div style='font-size:1rem;font-weight:700;color:#111827;margin-bottom:.5rem;'>
            AI Guidance Not Requested
          </div>
          <div style='font-size:.88rem;color:#6b7280;max-width:380px;margin:0 auto 1.5rem;line-height:1.6;'>
            Go back to the assessment form and enable the <strong>Gemini AI explanation</strong>
            toggle to receive personalised recommendations.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c = st.columns([2,1,2])[1]
        with c:
            if st.button("← Edit & Enable AI", type="primary", use_container_width=True):
                st.session_state.page = "form"
                st.rerun()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;
            padding:.9rem 1.25rem;font-size:.75rem;color:#9ca3af;line-height:1.6;
            margin-top:1.5rem;'>
  <strong style='color:#6b7280;'>Medical Disclaimer:</strong> {settings.MEDICAL_DISCLAIMER}
  &nbsp;·&nbsp; Model: Stacking Ensemble · Test AUC 0.9407 · 2,690 training samples
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
