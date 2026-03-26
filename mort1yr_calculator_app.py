import json
import os

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="TJA Mortality Risk",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Product UI styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #0a101a;
            --bg-2: #0f1725;
            --panel: rgba(15, 23, 37, 0.92);
            --panel-2: rgba(19, 31, 48, 0.94);
            --panel-3: rgba(12, 20, 32, 0.95);
            --border: rgba(148, 163, 184, 0.16);
            --border-strong: rgba(148, 163, 184, 0.28);
            --text: #e5edf7;
            --muted: #91a0b8;
            --subtle: #6e7d95;
            --blue: #7cb2ff;
            --blue-strong: #3b82f6;
            --success: #27c38a;
            --warning: #f3a43b;
            --danger: #ff6b73;
            --shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
            --radius-xl: 28px;
            --radius-lg: 22px;
            --radius-md: 16px;
            --radius-sm: 12px;
            --input-bg: rgba(255,255,255,0.03);
            --chip-bg: rgba(124,178,255,0.08);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.16), transparent 26%),
                radial-gradient(circle at top right, rgba(39,195,138,0.09), transparent 20%),
                linear-gradient(180deg, var(--bg) 0%, #08111d 100%);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div .app-shell) {
            width: 100%;
        }

        .app-shell {
            width: 100%;
        }

        .hero {
            padding: 30px 34px 26px 34px;
            border-radius: var(--radius-xl);
            background: linear-gradient(145deg, rgba(18, 28, 43, 0.96), rgba(12, 19, 31, 0.98));
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 22px;
        }

        .eyebrow {
            font-size: 0.76rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--blue);
            font-weight: 700;
            margin-bottom: 12px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.9fr);
            gap: 26px;
            align-items: end;
        }

        .hero-title {
            font-size: 2.3rem;
            line-height: 1.03;
            letter-spacing: -0.045em;
            font-weight: 760;
            margin: 0 0 12px 0;
            color: #f6f9ff;
        }

        .hero-copy {
            font-size: 1rem;
            line-height: 1.65;
            color: var(--muted);
            max-width: 860px;
            margin: 0;
        }

        .hero-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }

        .meta-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 14px 16px;
            backdrop-filter: blur(8px);
        }

        .meta-label {
            font-size: 0.74rem;
            color: var(--subtle);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
            font-weight: 700;
        }

        .meta-value {
            font-size: 1rem;
            color: var(--text);
            font-weight: 720;
            letter-spacing: -0.02em;
        }

        .panel {
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-3) 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 24px;
        }

        .panel + .panel { margin-top: 18px; }

        .section-label {
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--blue);
            font-weight: 700;
            margin-bottom: 10px;
        }

        .section-title {
            font-size: 1.16rem;
            font-weight: 730;
            color: #f4f7fc;
            letter-spacing: -0.03em;
            margin-bottom: 4px;
        }

        .section-copy {
            font-size: 0.92rem;
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 18px;
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0), var(--border), rgba(255,255,255,0));
            margin: 16px 0 18px 0;
        }

        .result-card {
            background: linear-gradient(180deg, rgba(18, 29, 45, 0.98), rgba(10, 18, 29, 0.98));
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-xl);
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .result-top {
            padding: 28px 26px 22px 26px;
            border-bottom: 1px solid var(--border);
            background: radial-gradient(circle at top, rgba(124,178,255,0.08), transparent 50%);
        }

        .result-kicker {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--subtle);
            font-weight: 700;
            margin-bottom: 12px;
        }

        .result-value {
            font-size: 4.8rem;
            line-height: 0.94;
            letter-spacing: -0.07em;
            font-weight: 820;
            color: #f8fbff;
            margin: 0 0 14px 0;
        }

        .risk-band {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            background: rgba(255,255,255,0.03);
        }

        .result-bottom {
            padding: 22px 24px 24px 24px;
        }

        .result-copy {
            font-size: 0.96rem;
            line-height: 1.75;
            color: #b7c4d7;
            margin-bottom: 18px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }

        .metric {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 14px 14px 12px 14px;
            background: rgba(255,255,255,0.025);
        }

        .metric-label {
            font-size: 0.72rem;
            color: var(--subtle);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.05rem;
            color: var(--text);
            font-weight: 730;
            letter-spacing: -0.02em;
        }

        .empty-state {
            min-height: 440px;
            border-radius: var(--radius-xl);
            border: 1px dashed rgba(148, 163, 184, 0.25);
            background: linear-gradient(180deg, rgba(18, 28, 43, 0.88), rgba(10, 18, 29, 0.92));
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 32px;
        }

        .empty-title {
            color: #f4f7fc;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }

        .empty-copy {
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.65;
            max-width: 360px;
        }

        .threshold-stack {
            display: grid;
            gap: 12px;
        }

        .threshold-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 13px 14px;
            border-radius: 14px;
            background: rgba(255,255,255,0.025);
            border: 1px solid var(--border);
        }

        .threshold-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .threshold-name {
            font-size: 0.88rem;
            color: #dfe8f5;
            font-weight: 650;
        }

        .threshold-sub {
            font-size: 0.78rem;
            color: var(--muted);
        }

        .threshold-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #f4f7fc;
        }

        .reference-shell {
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            overflow: hidden;
            box-shadow: var(--shadow);
            background: linear-gradient(180deg, rgba(16,25,39,0.96), rgba(10,18,29,0.98));
        }

        .ref-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
        }

        .ref-title {
            font-size: 1rem;
            font-weight: 730;
            color: #f4f7fc;
            letter-spacing: -0.02em;
            margin-bottom: 5px;
        }

        .ref-copy {
            font-size: 0.88rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .ref-table-wrap {
            padding: 0;
            overflow-x: auto;
        }

        table.ref-table {
            width: 100%;
            border-collapse: collapse;
        }

        .ref-table thead th {
            text-align: left;
            padding: 14px 18px;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--subtle);
            font-weight: 700;
            border-bottom: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
        }

        .ref-table tbody td {
            padding: 16px 18px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            font-size: 0.9rem;
            line-height: 1.6;
            vertical-align: top;
            color: #d7e1ef;
        }

        .ref-table tbody tr:hover td {
            background: rgba(124,178,255,0.035);
        }

        .rank-chip {
            display: inline-flex;
            width: 28px;
            height: 28px;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: rgba(124,178,255,0.10);
            border: 1px solid rgba(124,178,255,0.18);
            color: #dceaff;
            font-size: 0.8rem;
            font-weight: 800;
        }

        .footer {
            padding: 18px 2px 0 2px;
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.7;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            background: rgba(255,255,255,0.025) !important;
            overflow: hidden;
        }

        div[data-testid="stExpander"] details summary {
            padding: 0.7rem 1rem;
            color: #edf3fb !important;
            font-weight: 650;
        }

        div[data-testid="stExpander"] details > div {
            padding: 0.2rem 1rem 1rem 1rem;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.35rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            font-weight: 760;
            font-size: 0.98rem;
            letter-spacing: -0.01em;
            box-shadow: 0 16px 34px rgba(37,99,235,0.28);
        }

        .stButton > button:hover {
            filter: brightness(1.03);
        }

        div[data-baseweb="input"], div[data-baseweb="select"] > div {
            background: var(--input-bg) !important;
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        input, textarea {
            color: var(--text) !important;
        }

        label, .stNumberInput label, .stSelectbox label, .stCheckbox label {
            color: #dce7f6 !important;
            font-weight: 600 !important;
        }

        .stCheckbox {
            padding: 6px 0;
        }

        .stMarkdown p { color: inherit; }

        @media (max-width: 1150px) {
            .hero-grid { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------
base = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    import joblib
    model = joblib.load(os.path.join(base, "mort1yr_model_fitted_v7.joblib"))
    calibrator = joblib.load(os.path.join(base, "mort1yr_calibrator_v7.joblib"))
    with open(os.path.join(base, "mort1yr_imputer_fills_v7.json")) as f:
        fills = json.load(f)
    with open(os.path.join(base, "mort1yr_features_resolved_v7_nocosi.json")) as f:
        feats = json.load(f)
    return model, calibrator, fills, feats

try:
    model, calibrator, imputer_fills, feature_order = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_err = str(e)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
THR_YOUDEN = 0.00770
THR_DCA = 0.00500
THR_SENS90 = 0.00235
BASELINE = 0.005737
AUC = 0.7611
N_COHORT = 234252
N_EVENTS = 1344
WINSOR_CAPS = {"prior_hospitalizations_1yr": 3, "prior_ed_visits_1yr": 2}

COMORBIDITY_KEYS = [
    "has_diabetes","has_hypertension","has_heart_disease","has_copd",
    "has_anemia","has_sleep_apnea","has_liver_disease","has_thyroid_disease",
    "has_anticoagulant","has_bleeding_disorder","has_malignancy","has_ckd",
    "has_recent_ablation","has_thrombocytopenia","has_leukopenia",
    "has_kidney_transplant",
]

# -----------------------------------------------------------------------------
# Reference table data
# -----------------------------------------------------------------------------
REF_ROWS = [
    (1, "Age at surgery", "0.407", "Dominant predictor. Risk rises non-linearly with advancing age, reflecting progressive loss of physiologic reserve."),
    (2, "Sex (male)", "0.218", "Male sex independently contributes to higher 1-year mortality risk after TJA."),
    (3, "Creatinine", "0.162", "Renal dysfunction carries strong continuous signal, with risk accelerating as creatinine rises."),
    (4, "Hemoglobin", "0.153", "Preoperative anemia remains a strong long-horizon mortality marker rather than only a perioperative risk flag."),
    (5, "BMI", "0.145", "The relationship is non-linear: very low BMI and extreme obesity both increase predicted risk."),
    (6, "Systolic blood pressure", "0.142", "Both elevated and very low SBP values carry signal, capturing cardiovascular burden and frailty."),
    (7, "White blood cell count", "0.137", "Leukocytosis, leukopenia, and missing preoperative WBC all contribute predictive information."),
    (8, "Comorbidity count", "0.112", "Aggregate disease burden meaningfully outperforms any single isolated diagnosis."),
    (9, "Glucose", "0.100", "Hyperglycemia contributes risk even beyond the presence or absence of coded diabetes."),
    (10, "ASA physical status", "0.098", "ASA retains independent value as a global clinician-derived assessment of physiologic reserve."),
    (11, "Platelet count", "0.088", "Extremes in platelet count likely reflect marrow, inflammatory, hepatic, or systemic disease burden."),
    (12, "Albumin", "0.088", "Albumin provides graded information related to nutrition, inflammation, and hepatic synthetic function."),
    (13, "COPD", "0.086", "The strongest individual comorbidity flag, reflecting impaired pulmonary reserve."),
    (14, "Prior ED visits (12 mo)", "0.064", "Captures healthcare fragility and acute illness burden beyond static diagnosis coding."),
    (15, "Diastolic blood pressure", "0.063", "Adds incremental value beyond systolic pressure alone, especially at lower values."),
]

# -----------------------------------------------------------------------------
# Prediction logic (preserved)
# -----------------------------------------------------------------------------
def build_vector(inputs):
    v = {}
    v["age_at_surgery"] = float(inputs["age"])
    v["sex_encoded"] = 0.0 if inputs["sex"] == "Female" else 1.0
    v["proc_THA"] = 1.0 if inputs["procedure"] == "THA" else 0.0
    v["proc_TKA"] = 1.0 if inputs["procedure"] == "TKA" else 0.0
    v["bmi"] = float(inputs["bmi"])
    v["systolic_bp"] = float(inputs["sbp"])
    v["diastolic_bp"] = float(inputs["dbp"])
    v["asa_proxy"] = float(inputs["asa"])

    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        v[lab] = inputs.get(lab)

    v["creatinine_missing"] = 1.0 if v["creatinine"] is None else 0.0
    v["hemoglobin_missing"] = 1.0 if v["hemoglobin"] is None else 0.0
    v["wbc_missing"] = 1.0 if v["wbc"] is None else 0.0
    v["glucose_missing"] = 1.0 if v["glucose"] is None else 0.0
    v["platelets_missing"] = 1.0 if v["platelets"] is None else 0.0
    v["albumin_indicated_missing"] = 1.0 if (inputs.get("has_liver_disease") and v["albumin"] is None) else 0.0
    v["hba1c_indicated_missing"] = 1.0 if (inputs.get("has_diabetes") and v["hba1c"] is None) else 0.0
    v["inr_indicated_missing"] = 1.0 if (inputs.get("has_anticoagulant") and v["inr"] is None) else 0.0

    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        if v[lab] is None:
            v[lab] = float(imputer_fills.get(lab, 0.0))

    for k in COMORBIDITY_KEYS:
        v[k] = 1.0 if inputs.get(k, False) else 0.0
    v["comorbidity_count"] = sum(v[k] for k in COMORBIDITY_KEYS)

    hosp = min(float(inputs.get("prior_hosp", 0)), WINSOR_CAPS["prior_hospitalizations_1yr"])
    ed = min(float(inputs.get("prior_ed", 0)), WINSOR_CAPS["prior_ed_visits_1yr"])
    v["prior_hospitalizations_1yr"] = hosp
    v["prior_ed_visits_1yr"] = ed
    v["any_prior_hosp_1yr"] = 1.0 if hosp > 0 else 0.0
    v["any_prior_ed_1yr"] = 1.0 if ed > 0 else 0.0

    vec = np.array([float(v.get(f, float(imputer_fills.get(f, 0.0)))) for f in feature_order], dtype=np.float32).reshape(1, -1)
    for i, feat in enumerate(feature_order):
        if np.isnan(vec[0, i]):
            vec[0, i] = float(imputer_fills.get(feat, 0.0))
    return vec


def run_prediction(inputs):
    vec = build_vector(inputs)
    raw = float(model.predict_proba(vec)[0, 1])
    cal = float(np.clip(calibrator.predict([raw])[0], 0.0, 1.0))
    enr = cal / BASELINE
    comorbid = int(sum(1 for k in COMORBIDITY_KEYS if inputs.get(k, False)))

    if cal >= THR_YOUDEN:
        tier, color, note = "High risk", "var(--danger)", (
            f"Predicted 1-year mortality ({cal*100:.2f}%) exceeds the primary threshold ({THR_YOUDEN*100:.2f}%). "
            "This patient falls into the highest-risk stratum and merits elevated perioperative attention."
        )
    elif cal >= THR_DCA:
        tier, color, note = "Elevated risk", "var(--warning)", (
            f"Predicted risk ({cal*100:.2f}%) exceeds the decision-curve threshold ({THR_DCA*100:.2f}%) but remains below the primary threshold. "
            f"Risk is elevated relative to cohort baseline ({BASELINE*100:.3f}%)."
        )
    else:
        tier, color, note = "Average / lower risk", "var(--success)", (
            f"Predicted risk ({cal*100:.2f}%) is below the decision-curve threshold ({THR_DCA*100:.2f}%) "
            f"and at or below cohort baseline ({BASELINE*100:.3f}%)."
        )

    return {
        "cal": cal,
        "raw": raw,
        "pct": cal * 100,
        "enr": enr,
        "tier": tier,
        "color": color,
        "note": note,
        "comorbid": comorbid,
        "above_sens90": cal >= THR_SENS90,
        "above_dca": cal >= THR_DCA,
        "above_youden": cal >= THR_YOUDEN,
    }


def threshold_pill(value, active):
    bg = "rgba(255,107,115,0.16)" if active else "rgba(255,255,255,0.04)"
    bd = "rgba(255,107,115,0.26)" if active else "rgba(255,255,255,0.08)"
    fg = "#ffe7e9" if active else "#eef4fd"
    return f"<span class='threshold-pill' style='background:{bg}; border-color:{bd}; color:{fg};'>{value}</span>"


# -----------------------------------------------------------------------------
# App shell
# -----------------------------------------------------------------------------
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Ready-to-deploy risk product</div>
      <div class="hero-grid">
        <div>
          <h1 class="hero-title">TJA 1-Year Mortality Risk Estimator</h1>
          <p class="hero-copy">
            Calibrated patient-level mortality estimation after total joint arthroplasty using a restricted,
            internally validated XGBoost model. The interface is designed for production-style presentation:
            decisive output first, technical detail second.
          </p>
        </div>
        <div class="hero-meta">
          <div class="meta-card"><div class="meta-label">Model</div><div class="meta-value">XGBoost v7</div></div>
          <div class="meta-card"><div class="meta-label">Cross-validated AUC</div><div class="meta-value">{AUC:.3f}</div></div>
          <div class="meta-card"><div class="meta-label">Cohort</div><div class="meta-value">{N_COHORT:,} patients</div></div>
          <div class="meta-card"><div class="meta-label">Event rate</div><div class="meta-value">{BASELINE*100:.3f}%</div></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ok:
    st.error(f"Could not load model artifacts: {artifact_err}")
    st.info("Place the four model artifact files in the same directory as this app.")
    st.stop()

col_form, col_result = st.columns([1.18, 0.82], gap="large")

with col_form:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Patient inputs</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Clinical profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Enter the available patient information below. Missing laboratory values remain permitted and follow the model's original imputation behavior.</div>", unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    age = a1.number_input("Age at surgery", min_value=18, max_value=100, value=68)
    sex = a2.selectbox("Sex", ["Female", "Male"])
    proc = a3.selectbox("Procedure", ["TKA", "THA"])
    asa = a4.selectbox("ASA class", [1, 2, 3, 4, 5], index=1)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1rem;'>Vitals</div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    bmi = v1.number_input("BMI (kg/m²)", min_value=15.0, max_value=70.0, value=30.0, step=0.5)
    sbp = v2.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=130)
    dbp = v3.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1rem;'>Preoperative labs</div>", unsafe_allow_html=True)
    st.caption("Optional. Leave blank when not available.")
    l1, l2, l3, l4 = st.columns(4)
    creatinine = l1.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=15.0, value=None, placeholder="e.g. 0.9")
    hemoglobin = l2.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=None, placeholder="e.g. 13.5")
    wbc = l3.number_input("WBC (×10³/µL)", min_value=0.5, max_value=30.0, value=None, placeholder="e.g. 7.2")
    glucose = l4.number_input("Glucose (mg/dL)", min_value=50, max_value=600, value=None, placeholder="e.g. 95")
    l5, l6, l7, l8 = st.columns(4)
    platelets = l5.number_input("Platelets (×10³/µL)", min_value=10, max_value=1000, value=None, placeholder="e.g. 220")
    albumin = l6.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=None, placeholder="e.g. 4.0")
    hba1c = l7.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=None, placeholder="If diabetic")
    inr = l8.number_input("INR", min_value=0.5, max_value=10.0, value=None, placeholder="If anticoagulated")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1rem;'>Comorbidities</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    has_diabetes = c1.checkbox("Diabetes mellitus")
    has_hypertension = c1.checkbox("Hypertension")
    has_heart_disease = c1.checkbox("Heart disease")
    has_copd = c1.checkbox("COPD")
    has_anemia = c2.checkbox("Anemia")
    has_sleep_apnea = c2.checkbox("Sleep apnea")
    has_liver_disease = c2.checkbox("Liver disease")
    has_thyroid_disease = c2.checkbox("Thyroid disease")
    has_anticoagulant = c3.checkbox("Anticoagulation")
    has_bleeding_disorder = c3.checkbox("Bleeding disorder")
    has_malignancy = c3.checkbox("Active malignancy")
    has_ckd = c3.checkbox("Chronic kidney disease")
    has_recent_ablation = c4.checkbox("Recent cardiac ablation")
    has_thrombocytopenia = c4.checkbox("Thrombocytopenia")
    has_leukopenia = c4.checkbox("Leukopenia")
    has_kidney_transplant = c4.checkbox("Kidney transplant")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1rem;'>Healthcare utilization</div>", unsafe_allow_html=True)
    st.caption("Prior 12 months. Counts are capped at the model's original winsorization thresholds.")
    u1, u2 = st.columns(2)
    prior_hosp = u1.number_input(f"Inpatient hospitalizations (cap {WINSOR_CAPS['prior_hospitalizations_1yr']})", min_value=0, max_value=10, value=0)
    prior_ed = u2.number_input(f"Emergency department visits (cap {WINSOR_CAPS['prior_ed_visits_1yr']})", min_value=0, max_value=10, value=0)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    calc_btn = st.button("Calculate 1-Year Mortality Risk", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown("<div class='section-label'>Model output</div>", unsafe_allow_html=True)
    result_slot = st.empty()

    if not calc_btn:
        result_slot.markdown(
            """
            <div class="empty-state">
              <div>
                <div class="empty-title">Risk estimate will appear here</div>
                <div class="empty-copy">
                  Complete the patient profile and run the model to generate a calibrated 1-year mortality estimate,
                  threshold comparison, and risk context.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        inputs = dict(
            age=age, sex=sex, procedure=proc, asa=asa,
            bmi=bmi, sbp=sbp, dbp=dbp,
            creatinine=creatinine, hemoglobin=hemoglobin, wbc=wbc,
            glucose=glucose, platelets=platelets, albumin=albumin,
            hba1c=hba1c, inr=inr,
            has_diabetes=has_diabetes, has_hypertension=has_hypertension,
            has_heart_disease=has_heart_disease, has_copd=has_copd,
            has_anemia=has_anemia, has_sleep_apnea=has_sleep_apnea,
            has_liver_disease=has_liver_disease, has_thyroid_disease=has_thyroid_disease,
            has_anticoagulant=has_anticoagulant, has_bleeding_disorder=has_bleeding_disorder,
            has_malignancy=has_malignancy, has_ckd=has_ckd,
            has_recent_ablation=has_recent_ablation, has_thrombocytopenia=has_thrombocytopenia,
            has_leukopenia=has_leukopenia, has_kidney_transplant=has_kidney_transplant,
            prior_hosp=prior_hosp, prior_ed=prior_ed,
        )
        r = run_prediction(inputs)

        result_slot.markdown(
            f"""
            <div class="result-card">
              <div class="result-top">
                <div class="result-kicker">Predicted 1-year all-cause mortality</div>
                <div class="result-value">{r['pct']:.2f}%</div>
                <div class="risk-band" style="color:{r['color']}; border-color:color-mix(in srgb, {r['color']} 26%, transparent); background:color-mix(in srgb, {r['color']} 10%, transparent);">
                  {r['tier']}
                </div>
              </div>
              <div class="result-bottom">
                <div class="result-copy">{r['note']}</div>
                <div class="metric-grid">
                  <div class="metric">
                    <div class="metric-label">Cohort baseline</div>
                    <div class="metric-value">{BASELINE*100:.3f}%</div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Enrichment vs baseline</div>
                    <div class="metric-value">{r['enr']:.1f}×</div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Comorbidity burden</div>
                    <div class="metric-value">{r['comorbid']} / 16</div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Raw pre-calibration</div>
                    <div class="metric-value">{r['raw']*100:.2f}%</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='panel' style='margin-top:18px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Threshold positioning</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>The model output is shown against the three fixed anchors used in the original analysis pipeline.</div>", unsafe_allow_html=True)

    if calc_btn:
        st.markdown(
            f"""
            <div class="threshold-stack">
              <div class="threshold-row">
                <div class="threshold-left">
                  <div class="threshold-name">≥90% sensitivity anchor</div>
                  <div class="threshold-sub">High-sensitivity screening threshold</div>
                </div>
                {threshold_pill(f"{THR_SENS90*100:.3f}%", r['above_sens90'])}
              </div>
              <div class="threshold-row">
                <div class="threshold-left">
                  <div class="threshold-name">Decision-curve threshold</div>
                  <div class="threshold-sub">Net-benefit operating point</div>
                </div>
                {threshold_pill(f"{THR_DCA*100:.3f}%", r['above_dca'])}
              </div>
              <div class="threshold-row">
                <div class="threshold-left">
                  <div class="threshold-name">Primary threshold</div>
                  <div class="threshold-sub">Youden's J operating point</div>
                </div>
                {threshold_pill(f"{THR_YOUDEN*100:.3f}%", r['above_youden'])}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='section-copy' style='margin-bottom:0;'>Threshold interpretation will populate after a calculation is run.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Model diagnostics and deployment notes"):
        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric"><div class="metric-label">Architecture</div><div class="metric-value">XGBoost v7</div></div>
              <div class="metric"><div class="metric-label">Cross-validated AUC</div><div class="metric-value">{AUC:.3f} ± 0.016</div></div>
              <div class="metric"><div class="metric-label">Training cohort</div><div class="metric-value">{N_COHORT:,}</div></div>
              <div class="metric"><div class="metric-label">Events</div><div class="metric-value">{N_EVENTS:,}</div></div>
              <div class="metric"><div class="metric-label">Top-decile mortality</div><div class="metric-value">2.50% (4.4× baseline)</div></div>
              <div class="metric"><div class="metric-label">Features</div><div class="metric-value">43 (COSI excluded)</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Reference section
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
rows_html = "".join(
    f"<tr><td><span class='rank-chip'>{rank}</span></td><td><strong>{pred}</strong></td><td>{shap}</td><td>{interp}</td></tr>"
    for rank, pred, shap, interp in REF_ROWS
)

st.markdown(
    f"""
    <div class="reference-shell">
      <div class="ref-header">
        <div class="section-label" style="margin-bottom:8px;">Predictor reference</div>
        <div class="ref-title">Top 15 predictors by SHAP importance</div>
        <div class="ref-copy">
          Mean absolute SHAP values from the internally validated XGBoost mortality model. This section remains reference-oriented rather than explanatory, so the product surface stays focused on the patient-level estimate.
        </div>
      </div>
      <div class="ref-table-wrap">
        <table class="ref-table">
          <thead>
            <tr>
              <th style="width:8%;">Rank</th>
              <th style="width:24%;">Predictor</th>
              <th style="width:14%;">Mean |SHAP|</th>
              <th style="width:54%;">Clinical interpretation</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="footer">
      Research use only. Internally validated XGBoost model trained on de-identified multi-institutional administrative data.
      Cohort restricted to patients with at least 365 days of post-surgical follow-up or confirmed death within 365 days of surgery
      (n={N_COHORT:,}; {N_EVENTS:,} events; event rate {BASELINE*100:.3f}%). External validation has not been performed.
      Thresholds are preserved from the original workflow and are shown here without modification.
      This interface redesign does not change computational behavior, feature engineering, calibration, or threshold logic.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
