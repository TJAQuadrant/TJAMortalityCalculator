# ==============================================================================
# mort1yr_calculator_app.py
# Streamlit Risk Calculator — 1-Year All-Cause Mortality After TJA
# Model: XGBoost v7 (no COSI features, SPW disabled)
#
# DEPLOYMENT:
#   1. Place this file in the same directory as the model artifacts:
#        mort1yr_model_fitted_v7.joblib
#        mort1yr_calibrator_v7.joblib
#        mort1yr_imputer_fills_v7.json
#        mort1yr_features_resolved_v7_nocosi.json
#   2. streamlit run mort1yr_calculator_app.py
#
# ARTIFACTS REQUIRED (all in same directory as this file):
#   mort1yr_model_fitted_v7.joblib          — trained XGBoost model
#   mort1yr_calibrator_v7.joblib            — isotonic calibration object
#   mort1yr_imputer_fills_v7.json           — median imputation fills
#   mort1yr_features_resolved_v7_nocosi.json — ordered feature list (43 features)
#
# DISCLAIMER:
#   This tool is for RESEARCH PURPOSES ONLY.
#   Internally validated on a single multi-institutional cohort.
#   External validation has not been performed.
#   Not approved for clinical decision-making.
# ==============================================================================

import streamlit as st
import numpy as np
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TJA 1-Year Mortality Risk Calculator",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import joblib
    base = os.path.dirname(os.path.abspath(__file__))

    model      = joblib.load(os.path.join(base, "mort1yr_model_fitted_v7.joblib"))
    calibrator = joblib.load(os.path.join(base, "mort1yr_calibrator_v7.joblib"))

    with open(os.path.join(base, "mort1yr_imputer_fills_v7.json")) as f:
        imputer_fills = json.load(f)

    with open(os.path.join(base, "mort1yr_features_resolved_v7_nocosi.json")) as f:
        feature_order = json.load(f)

    return model, calibrator, imputer_fills, feature_order

try:
    model, calibrator, imputer_fills, feature_order = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    artifact_error   = str(e)

# ── Thresholds (from v7 cross-validated results) ──────────────────────────────
THR_YOUDEN   = 0.00770   # Primary: Youden's J mean (sens 54.8%, spec 83.9%)
THR_DCA      = 0.00500   # DCA net benefit (sens 68.7%, spec 70.2%)
THR_SENS90   = 0.00235   # ≥90% sensitivity anchor

COHORT_BASELINE = 0.005737   # Cohort 1-year mortality rate (restricted cohort)
AUC             = 0.7611
N_COHORT        = 234252
N_EVENTS        = 1344

# ── Winsorization caps ────────────────────────────────────────────────────────
WINSOR_CAPS = {
    "prior_hospitalizations_1yr": 3,
    "prior_ed_visits_1yr":        2,
}

# ── Helper: build feature vector ──────────────────────────────────────────────
def build_feature_vector(inputs: dict, feature_order: list,
                          imputer_fills: dict) -> np.ndarray:
    """
    Construct the 43-feature vector in model order.
    Applies:
      - Comorbidity count derivation
      - any_prior_* binary flags
      - Missing indicator flags
      - Median imputation for any remaining NaN
      - Winsorization caps
    """
    v = {}

    # ── Raw inputs ────────────────────────────────────────────────────────────
    v["age_at_surgery"]  = float(inputs["age"])
    v["sex_encoded"]     = 0.0 if inputs["sex"] == "Female" else 1.0
    v["proc_THA"]        = 1.0 if inputs["procedure"] == "THA" else 0.0
    v["proc_TKA"]        = 1.0 if inputs["procedure"] == "TKA" else 0.0
    v["bmi"]             = float(inputs["bmi"])
    v["systolic_bp"]     = float(inputs["sbp"])
    v["diastolic_bp"]    = float(inputs["dbp"])
    v["asa_proxy"]       = float(inputs["asa"])

    # Labs — None if not entered (will be imputed)
    v["creatinine"]  = inputs.get("creatinine")
    v["hemoglobin"]  = inputs.get("hemoglobin")
    v["wbc"]         = inputs.get("wbc")
    v["glucose"]     = inputs.get("glucose")
    v["platelets"]   = inputs.get("platelets")
    v["albumin"]     = inputs.get("albumin")
    v["hba1c"]       = inputs.get("hba1c")
    v["inr"]         = inputs.get("inr")

    # Missing indicator flags
    v["creatinine_missing"]        = 1.0 if v["creatinine"]  is None else 0.0
    v["hemoglobin_missing"]        = 1.0 if v["hemoglobin"]  is None else 0.0
    v["wbc_missing"]               = 1.0 if v["wbc"]         is None else 0.0
    v["glucose_missing"]           = 1.0 if v["glucose"]     is None else 0.0
    v["platelets_missing"]         = 1.0 if v["platelets"]   is None else 0.0
    # Indication-gated missing flags
    v["albumin_indicated_missing"] = (
        1.0 if (inputs.get("has_liver_disease") and v["albumin"] is None) else 0.0)
    v["hba1c_indicated_missing"]   = (
        1.0 if (inputs.get("has_diabetes") and v["hba1c"] is None) else 0.0)
    v["inr_indicated_missing"]     = (
        1.0 if (inputs.get("has_anticoagulant") and v["inr"] is None) else 0.0)

    # Replace None labs with median imputer fill
    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        if v[lab] is None:
            v[lab] = imputer_fills.get(lab, np.nan)

    # Comorbidity flags
    comorbidity_keys = [
        "has_diabetes","has_hypertension","has_heart_disease","has_copd",
        "has_anemia","has_sleep_apnea","has_liver_disease","has_thyroid_disease",
        "has_anticoagulant","has_bleeding_disorder","has_malignancy","has_ckd",
        "has_recent_ablation","has_thrombocytopenia","has_leukopenia",
        "has_kidney_transplant",
    ]
    for key in comorbidity_keys:
        v[key] = 1.0 if inputs.get(key, False) else 0.0

    # Derived features
    v["comorbidity_count"] = sum(v[k] for k in comorbidity_keys)

    # Utilization
    hosp_raw = float(inputs.get("prior_hospitalizations_1yr", 0))
    ed_raw   = float(inputs.get("prior_ed_visits_1yr", 0))
    v["prior_hospitalizations_1yr"] = min(hosp_raw, WINSOR_CAPS["prior_hospitalizations_1yr"])
    v["prior_ed_visits_1yr"]        = min(ed_raw,   WINSOR_CAPS["prior_ed_visits_1yr"])
    v["any_prior_hosp_1yr"]         = 1.0 if hosp_raw > 0 else 0.0
    v["any_prior_ed_1yr"]           = 1.0 if ed_raw   > 0 else 0.0

    # Build ordered vector
    vec = np.array(
        [float(v.get(f, imputer_fills.get(f, np.nan))) for f in feature_order],
        dtype=np.float32
    ).reshape(1, -1)

    # Final NaN safety — replace any remaining with median fills
    for i, feat in enumerate(feature_order):
        if np.isnan(vec[0, i]):
            vec[0, i] = float(imputer_fills.get(feat, 0.0))

    return vec

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(inputs: dict) -> dict:
    vec       = build_feature_vector(inputs, feature_order, imputer_fills)
    raw_prob  = float(model.predict_proba(vec)[0, 1])
    cal_prob  = float(calibrator.predict([raw_prob])[0])
    cal_prob  = float(np.clip(cal_prob, 0.0, 1.0))

    # Risk tier using Youden threshold as primary
    if cal_prob >= THR_YOUDEN:
        tier       = "HIGH RISK"
        tier_color = "#C0392B"
        tier_icon  = "🔴"
        tier_note  = (f"Predicted risk exceeds the Youden-optimal threshold "
                      f"({THR_YOUDEN*100:.2f}%). Enhanced perioperative surveillance "
                      f"may be warranted.")
    elif cal_prob >= THR_DCA:
        tier       = "ELEVATED RISK"
        tier_color = "#E76F51"
        tier_icon  = "🟠"
        tier_note  = (f"Predicted risk exceeds the DCA net-benefit threshold "
                      f"({THR_DCA*100:.2f}%) but is below the Youden threshold.")
    else:
        tier       = "AVERAGE / LOW RISK"
        tier_color = "#2A9D8F"
        tier_icon  = "🟢"
        tier_note  = (f"Predicted risk is below the DCA threshold "
                      f"({THR_DCA*100:.2f}%). Risk is at or below cohort average "
                      f"({COHORT_BASELINE*100:.2f}%).")

    enrichment = cal_prob / COHORT_BASELINE if COHORT_BASELINE > 0 else float("nan")

    return {
        "cal_prob":   cal_prob,
        "raw_prob":   raw_prob,
        "pct":        cal_prob * 100,
        "tier":       tier,
        "tier_color": tier_color,
        "tier_icon":  tier_icon,
        "tier_note":  tier_note,
        "enrichment": enrichment,
    }

# ==============================================================================
# UI
# ==============================================================================
st.title("🦴 TJA 1-Year Mortality Risk Calculator")
st.markdown(
    f"**XGBoost v7** · N = {N_COHORT:,} · Events = {N_EVENTS:,} "
    f"({COHORT_BASELINE*100:.3f}%) · AUC = {AUC:.3f} · Restricted cohort "
    f"(≥365d follow-up or confirmed death)"
)

if not artifacts_loaded:
    st.error(f"⚠️ Could not load model artifacts: {artifact_error}")
    st.info("Ensure all four artifact files are in the same directory as this app.")
    st.stop()

# Research disclaimer
st.warning(
    "**RESEARCH USE ONLY.** This calculator is based on internally validated results "
    "from a single multi-institutional administrative database. External validation has "
    "not been performed. Results should not be used to guide clinical decisions."
)

st.divider()

# ── Sidebar: threshold reference ─────────────────────────────────────────────
with st.sidebar:
    st.header("Threshold Reference")
    st.markdown(f"""
| Method | Threshold | Sens | Spec |
|--------|-----------|------|------|
| Youden's J *(primary)* | {THR_YOUDEN*100:.2f}% | 54.8% | 83.9% |
| DCA net benefit | {THR_DCA*100:.2f}% | 68.7% | 70.2% |
| ≥90% sensitivity | {THR_SENS90*100:.2f}% | 90.1% | 33.2% |
""")
    st.caption(
        f"Cohort baseline: {COHORT_BASELINE*100:.3f}%  \n"
        f"Top 10% observed: 2.50% (4.4× baseline)  \n"
        f"Top 5% observed: 3.57% (6.2× baseline)"
    )
    st.divider()
    st.header("Model Info")
    st.caption(
        f"**Features:** 43 (demographics, labs, vitals, comorbidities, utilization)  \n"
        f"**Validation:** Nested 5-fold CV, leakage-safe  \n"
        f"**Calibration:** Isotonic regression  \n"
        f"**SPW:** Disabled  \n"
        f"**COSI features:** Excluded (construct mismatch)"
    )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("calculator_form"):

    # Demographics
    st.subheader("Demographics & Procedure")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age  = col1.number_input("Age at surgery (years)", 18, 100, 68)
    with col2:
        sex  = col2.selectbox("Sex", ["Female", "Male"])
    with col3:
        proc = col3.selectbox("Procedure", ["TKA", "THA"])
    with col4:
        asa  = col4.selectbox("ASA class", [1, 2, 3, 4, 5], index=1)

    # Vitals
    st.subheader("Vitals")
    col1, col2, col3 = st.columns(3)
    bmi = col1.number_input("BMI (kg/m²)", 15.0, 70.0, 30.0, step=0.5)
    sbp = col2.number_input("Systolic BP (mmHg)", 70, 220, 130)
    dbp = col3.number_input("Diastolic BP (mmHg)", 40, 130, 80)

    # Labs
    st.subheader("Preoperative Labs")
    st.caption("Leave blank if not available — median imputation will be applied.")

    col1, col2, col3, col4 = st.columns(4)
    creatinine = col1.number_input("Creatinine (mg/dL)", 0.3, 15.0,
                                    value=None, placeholder="e.g. 0.9")
    hemoglobin = col2.number_input("Hemoglobin (g/dL)", 5.0, 20.0,
                                    value=None, placeholder="e.g. 13.5")
    wbc        = col3.number_input("WBC (×10³/µL)", 0.5, 30.0,
                                    value=None, placeholder="e.g. 7.2")
    glucose    = col4.number_input("Glucose (mg/dL)", 50, 600,
                                    value=None, placeholder="e.g. 95")

    col1, col2, col3, col4 = st.columns(4)
    platelets  = col1.number_input("Platelets (×10³/µL)", 10, 1000,
                                    value=None, placeholder="e.g. 220")
    albumin    = col2.number_input("Albumin (g/dL)", 1.0, 6.0,
                                    value=None, placeholder="e.g. 4.0")
    hba1c      = col3.number_input("HbA1c (%)", 4.0, 14.0,
                                    value=None, placeholder="If diabetic")
    inr        = col4.number_input("INR", 0.5, 10.0,
                                    value=None, placeholder="If anticoagulated")

    # Comorbidities
    st.subheader("Comorbidities")
    co1, co2, co3, co4 = st.columns(4)
    has_diabetes       = co1.checkbox("Diabetes mellitus")
    has_hypertension   = co1.checkbox("Hypertension")
    has_heart_disease  = co1.checkbox("Heart disease (CAD / HF / valvular)")
    has_copd           = co1.checkbox("COPD")
    has_anemia         = co2.checkbox("Anemia")
    has_sleep_apnea    = co2.checkbox("Obstructive sleep apnea")
    has_liver_disease  = co2.checkbox("Liver disease")
    has_thyroid_disease= co2.checkbox("Thyroid disease")
    has_anticoagulant  = co3.checkbox("On anticoagulation therapy")
    has_bleeding_disorder = co3.checkbox("Bleeding disorder")
    has_malignancy     = co3.checkbox("Active malignancy")
    has_ckd            = co3.checkbox("Chronic kidney disease")
    has_recent_ablation= co4.checkbox("Recent cardiac ablation")
    has_thrombocytopenia = co4.checkbox("Thrombocytopenia")
    has_leukopenia     = co4.checkbox("Leukopenia")
    has_kidney_transplant = co4.checkbox("Prior kidney transplant")

    # Utilization
    st.subheader("Healthcare Utilization (prior 12 months)")
    col1, col2 = st.columns(2)
    prior_hosp = col1.number_input(
        "Prior inpatient hospitalizations", 0, 10, 0,
        help=f"Capped at {WINSOR_CAPS['prior_hospitalizations_1yr']} for prediction")
    prior_ed   = col2.number_input(
        "Prior ED visits", 0, 10, 0,
        help=f"Capped at {WINSOR_CAPS['prior_ed_visits_1yr']} for prediction")

    submitted = st.form_submit_button("Calculate Risk", type="primary",
                                      use_container_width=True)

# ── Output ────────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "age": age, "sex": sex, "procedure": proc, "asa": asa,
        "bmi": bmi, "sbp": sbp, "dbp": dbp,
        "creatinine": creatinine, "hemoglobin": hemoglobin, "wbc": wbc,
        "glucose": glucose, "platelets": platelets, "albumin": albumin,
        "hba1c": hba1c, "inr": inr,
        "has_diabetes": has_diabetes, "has_hypertension": has_hypertension,
        "has_heart_disease": has_heart_disease, "has_copd": has_copd,
        "has_anemia": has_anemia, "has_sleep_apnea": has_sleep_apnea,
        "has_liver_disease": has_liver_disease,
        "has_thyroid_disease": has_thyroid_disease,
        "has_anticoagulant": has_anticoagulant,
        "has_bleeding_disorder": has_bleeding_disorder,
        "has_malignancy": has_malignancy, "has_ckd": has_ckd,
        "has_recent_ablation": has_recent_ablation,
        "has_thrombocytopenia": has_thrombocytopenia,
        "has_leukopenia": has_leukopenia,
        "has_kidney_transplant": has_kidney_transplant,
        "prior_hospitalizations_1yr": prior_hosp,
        "prior_ed_visits_1yr": prior_ed,
    }

    result = predict(inputs)

    st.divider()
    st.subheader("Predicted 1-Year Mortality Risk")

    # Main result metric
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f"<div style='padding:20px; border-radius:10px; "
            f"background-color:{result['tier_color']}22; "
            f"border-left: 5px solid {result['tier_color']};'>"
            f"<h2 style='color:{result['tier_color']}; margin:0;'>"
            f"{result['tier_icon']} {result['tier']}</h2>"
            f"<h1 style='font-size:3em; margin:8px 0; color:#1B3A5C;'>"
            f"{result['pct']:.2f}%</h1>"
            f"<p style='color:#6C757D; margin:0;'>{result['tier_note']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.metric("vs Cohort Baseline",
                  f"{COHORT_BASELINE*100:.2f}%",
                  f"{(result['pct'] - COHORT_BASELINE*100):+.2f}%")
        st.metric("Enrichment", f"{result['enrichment']:.1f}×",
                  help="Predicted risk ÷ cohort baseline mortality rate")
    with col3:
        st.metric("Comorbidity Count",
                  f"{sum(1 for k in ['has_diabetes','has_hypertension','has_heart_disease','has_copd','has_anemia','has_sleep_apnea','has_liver_disease','has_thyroid_disease','has_anticoagulant','has_bleeding_disorder','has_malignancy','has_ckd','has_recent_ablation','has_thrombocytopenia','has_leukopenia','has_kidney_transplant'] if inputs.get(k)):,}")
        st.metric("Raw Probability", f"{result['raw_prob']*100:.2f}%",
                  help="Pre-calibration model output")

    # Threshold context
    st.markdown("#### Risk Threshold Context")
    thr_data = {
        "Threshold": ["≥90% sensitivity (0.235%)", "DCA net benefit (0.500%)",
                      "Youden's J — primary (0.770%)", "This patient"],
        "Predicted probability": [
            f"{THR_SENS90*100:.3f}%", f"{THR_DCA*100:.3f}%",
            f"{THR_YOUDEN*100:.3f}%", f"**{result['pct']:.3f}%**"
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(thr_data), hide_index=True, use_container_width=True)

    # Disclaimer
    st.caption(
        "⚠️ Research use only. Internally validated on a single multi-institutional "
        "cohort (N=234,252; AUC=0.761). External validation required before "
        "clinical application. Thresholds derived via cross-validated Youden index, "
        "DCA net benefit maximization, and 90% sensitivity anchor."
    )
