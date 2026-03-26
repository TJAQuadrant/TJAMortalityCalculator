# ==============================================================================
# mort1yr_calculator_app.py
# 1-Year All-Cause Mortality Risk Calculator After TJA
# XGBoost v7 — no COSI features, SPW disabled, restricted cohort
#
# ARTIFACTS REQUIRED (same directory):
#   mort1yr_model_fitted_v7.joblib
#   mort1yr_calibrator_v7.joblib
#   mort1yr_imputer_fills_v7.json
#   mort1yr_features_resolved_v7_nocosi.json
#   style.css
# ==============================================================================

import streamlit as st
import numpy as np
import json, os

st.set_page_config(
    page_title="TJA Mortality Risk",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(base, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import joblib
    model      = joblib.load(os.path.join(base, "mort1yr_model_fitted_v7.joblib"))
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

# ── Constants ──────────────────────────────────────────────────────────────────
THR_YOUDEN  = 0.00770
THR_DCA     = 0.00500
THR_SENS90  = 0.00235
BASELINE    = 0.005737
AUC         = 0.7611
N_COHORT    = 234252
N_EVENTS    = 1344
WINSOR_CAPS = {"prior_hospitalizations_1yr": 3, "prior_ed_visits_1yr": 2}

COMORBIDITY_KEYS = [
    "has_diabetes","has_hypertension","has_heart_disease","has_copd",
    "has_anemia","has_sleep_apnea","has_liver_disease","has_thyroid_disease",
    "has_anticoagulant","has_bleeding_disorder","has_malignancy","has_ckd",
    "has_recent_ablation","has_thrombocytopenia","has_leukopenia",
    "has_kidney_transplant",
]

# ── Reference table HTML ───────────────────────────────────────────────────────
REF_TABLE_HTML = """
<table class="ref-table">
  <thead>
    <tr>
      <th style="width:4%">Rank</th>
      <th style="width:20%">Predictor</th>
      <th style="width:14%">SHAP Value</th>
      <th style="width:62%">Clinical Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span class="ref-rank">1</span></td>
      <td><span class="ref-crit">Age at surgery</span></td>
      <td><span class="ref-hi">0.407 — highest</span></td>
      <td>Dominant predictor. Risk increases non-linearly with age, reflecting cumulative physiologic reserve decline. Patients &ge;75 carry substantially elevated 1-year mortality risk.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">2</span></td>
      <td><span class="ref-crit">Sex (male)</span></td>
      <td>0.218</td>
      <td>Male sex independently predicts elevated 1-year mortality after TJA, consistent with published sex-based differences in cardiovascular and perioperative outcomes.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">3</span></td>
      <td><span class="ref-crit">Creatinine</span></td>
      <td>0.162</td>
      <td>Continuous renal function marker. Risk accelerates above 1.5 mg/dL, reflecting reduced physiologic reserve and increased vulnerability to perioperative hemodynamic stress.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">4</span></td>
      <td><span class="ref-crit">Hemoglobin</span></td>
      <td>0.153</td>
      <td>Graded risk across severity levels. Preoperative anemia independently predicts 1-year mortality beyond the 90-day window, likely reflecting underlying disease burden.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">5</span></td>
      <td><span class="ref-crit">BMI</span></td>
      <td>0.145</td>
      <td>Non-linear relationship. Both very low BMI (frailty/cachexia) and extreme obesity elevate risk. Mid-range obesity in an elective surgical cohort carries attenuated risk due to pre-operative selection.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">6</span></td>
      <td><span class="ref-crit">Systolic blood pressure</span></td>
      <td>0.142</td>
      <td>Bidirectional risk. Elevated SBP reflects cardiovascular burden; very low values may indicate frailty or reduced cardiac output. Both patterns independently predict 1-year mortality.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">7</span></td>
      <td><span class="ref-crit">White blood cell count</span></td>
      <td>0.137</td>
      <td>Both leukocytosis and leukopenia carry independent signal, reflecting infectious, inflammatory, and hematologic risk. Absent pre-operative WBC measurement also carries predictive weight.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">8</span></td>
      <td><span class="ref-crit">Comorbidity count</span></td>
      <td>0.112</td>
      <td>Aggregate burden outperforms any single diagnosis. Each additional comorbidity incrementally elevates 1-year mortality risk, independent of specific condition type or severity.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">9</span></td>
      <td><span class="ref-crit">Glucose</span></td>
      <td>0.100</td>
      <td>Continuous predictor capturing both diabetic and non-diabetic hyperglycemia. Elevated preoperative glucose reflects metabolic dysregulation independently predictive of 1-year mortality.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">10</span></td>
      <td><span class="ref-crit">ASA physical status</span></td>
      <td>0.098</td>
      <td>Carries independent weight beyond comorbidity count, reflecting the anesthesiologist's integrated assessment of physiologic reserve. The model corroborates ASA class as a valid mortality predictor.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">11</span></td>
      <td><span class="ref-crit">Platelet count</span></td>
      <td>0.088</td>
      <td>Both thrombocytopenia and thrombocytosis carry risk. Platelet count reflects hepatic synthetic function, bone marrow reserve, and inflammatory state &mdash; all independently relevant to survival.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">12</span></td>
      <td><span class="ref-crit">Albumin</span></td>
      <td>0.088</td>
      <td>Nutritional and hepatic synthetic marker. Values below 3.5 g/dL carry graded mortality risk, supporting albumin as a continuous rather than binary predictor of 1-year survival.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">13</span></td>
      <td><span class="ref-crit">COPD</span></td>
      <td>0.086</td>
      <td><span class="ref-excl">Strongest individual comorbidity flag.</span> COPD independently predicts 1-year mortality beyond its contribution to comorbidity count, reflecting severity of pulmonary reserve impairment.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">14</span></td>
      <td><span class="ref-crit">Prior ED visits (12 mo)</span></td>
      <td>0.064</td>
      <td>Leading utilization-based predictor. Prior ED visits independently capture healthcare fragility and acute illness burden not reflected in diagnosis codes alone. Risk increases with visit count.</td>
    </tr>
    <tr>
      <td><span class="ref-rank">15</span></td>
      <td><span class="ref-crit">Diastolic blood pressure</span></td>
      <td>0.063</td>
      <td>Provides incremental value beyond systolic BP alone, particularly at low values where wide pulse pressure signals advanced arterial stiffness and cardiovascular vulnerability.</td>
    </tr>
  </tbody>
</table>
<p style="font-size:11px;color:#64748b;margin-top:10px;line-height:1.5;">
  SHAP = SHapley Additive exPlanations; mean absolute SHAP value reflects average contribution to model
  prediction across a random sample of 5,000 patients. Predictors ranked by mean |SHAP| from XGBoost v7
  trained on 1-year all-cause mortality (n=234,252; restricted cohort, &ge;365-day follow-up or confirmed
  death). COSI ambulatory eligibility features excluded (construct mismatch for mortality outcome).
  Validation: nested 5-fold CV, leakage-safe isotonic calibration. Internally validated only.
</p>
"""

# ── Prediction logic ───────────────────────────────────────────────────────────
def build_vector(inputs):
    v = {}
    v["age_at_surgery"]  = float(inputs["age"])
    v["sex_encoded"]     = 0.0 if inputs["sex"] == "Female" else 1.0
    v["proc_THA"]        = 1.0 if inputs["procedure"] == "THA" else 0.0
    v["proc_TKA"]        = 1.0 if inputs["procedure"] == "TKA" else 0.0
    v["bmi"]             = float(inputs["bmi"])
    v["systolic_bp"]     = float(inputs["sbp"])
    v["diastolic_bp"]    = float(inputs["dbp"])
    v["asa_proxy"]       = float(inputs["asa"])

    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        v[lab] = inputs.get(lab)

    v["creatinine_missing"]        = 1.0 if v["creatinine"] is None else 0.0
    v["hemoglobin_missing"]        = 1.0 if v["hemoglobin"] is None else 0.0
    v["wbc_missing"]               = 1.0 if v["wbc"]        is None else 0.0
    v["glucose_missing"]           = 1.0 if v["glucose"]    is None else 0.0
    v["platelets_missing"]         = 1.0 if v["platelets"]  is None else 0.0
    v["albumin_indicated_missing"] = 1.0 if (inputs.get("has_liver_disease") and v["albumin"] is None) else 0.0
    v["hba1c_indicated_missing"]   = 1.0 if (inputs.get("has_diabetes")      and v["hba1c"]  is None) else 0.0
    v["inr_indicated_missing"]     = 1.0 if (inputs.get("has_anticoagulant") and v["inr"]    is None) else 0.0

    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        if v[lab] is None:
            v[lab] = float(imputer_fills.get(lab, 0.0))

    for k in COMORBIDITY_KEYS:
        v[k] = 1.0 if inputs.get(k, False) else 0.0
    v["comorbidity_count"] = sum(v[k] for k in COMORBIDITY_KEYS)

    hosp = min(float(inputs.get("prior_hosp", 0)), WINSOR_CAPS["prior_hospitalizations_1yr"])
    ed   = min(float(inputs.get("prior_ed",   0)), WINSOR_CAPS["prior_ed_visits_1yr"])
    v["prior_hospitalizations_1yr"] = hosp
    v["prior_ed_visits_1yr"]        = ed
    v["any_prior_hosp_1yr"]         = 1.0 if hosp > 0 else 0.0
    v["any_prior_ed_1yr"]           = 1.0 if ed   > 0 else 0.0

    vec = np.array(
        [float(v.get(f, float(imputer_fills.get(f, 0.0)))) for f in feature_order],
        dtype=np.float32
    ).reshape(1, -1)
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
        tier, color, icon = "HIGH RISK",     "#f43f5e", "🔴"
        note = (f"Predicted 1-year mortality ({cal*100:.2f}%) exceeds the Youden-optimal "
                f"threshold ({THR_YOUDEN*100:.2f}%). This patient is in the highest-risk "
                f"stratum; enhanced perioperative surveillance is supported by the model.")
    elif cal >= THR_DCA:
        tier, color, icon = "ELEVATED RISK", "#f59e0b", "🟠"
        note = (f"Predicted risk ({cal*100:.2f}%) exceeds the DCA net-benefit threshold "
                f"({THR_DCA*100:.2f}%) but is below the Youden threshold. Risk is elevated "
                f"relative to cohort baseline ({BASELINE*100:.3f}%).")
    else:
        tier, color, icon = "AVERAGE / LOW", "#14b8a6", "🟢"
        note = (f"Predicted risk ({cal*100:.2f}%) is below the DCA threshold "
                f"({THR_DCA*100:.2f}%) and at or below cohort baseline ({BASELINE*100:.3f}%). "
                f"Standard perioperative care pathway is supported.")

    # Threshold flags for colour-coding in result panel
    above_sens90  = cal >= THR_SENS90
    above_dca     = cal >= THR_DCA
    above_youden  = cal >= THR_YOUDEN

    return dict(cal=cal, raw=raw, pct=cal*100, enr=enr,
                tier=tier, color=color, icon=icon, note=note,
                comorbid=comorbid,
                above_sens90=above_sens90,
                above_dca=above_dca,
                above_youden=above_youden)

# ==============================================================================
# LAYOUT
# ==============================================================================

# Sticky header
st.markdown(f"""
<div class="app-hdr">
  <div class="app-hdr-title">🦴 TJA 1-Year Mortality Risk Calculator</div>
  <div class="app-hdr-sub">
    XGBoost v7 &nbsp;&middot;&nbsp; Restricted cohort (&ge;365d follow-up or confirmed death)
    &nbsp;&middot;&nbsp; N&nbsp;=&nbsp;{N_COHORT:,}
    &nbsp;&middot;&nbsp; Events&nbsp;=&nbsp;{N_EVENTS:,} ({BASELINE*100:.3f}%)
    &nbsp;&middot;&nbsp; AUC&nbsp;=&nbsp;{AUC:.3f}
    &nbsp;&middot;&nbsp; Research use only
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(f"Could not load model artifacts: {artifact_err}")
    st.info("Place all four artifact files in the same directory as this app.")
    st.stop()

# Two-column layout
col_form, col_result = st.columns([3, 2], gap="large")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
with col_form:

    # Demographics
    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-blue"></span>
      <span class="s-title">Demographics &amp; Procedure</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    age  = dc1.number_input("Age at surgery", 18, 100, 68, key="age")
    sex  = dc2.selectbox("Sex", ["Female", "Male"], key="sex")
    proc = dc3.selectbox("Procedure", ["TKA", "THA"], key="proc")
    asa  = dc4.selectbox("ASA class", [1, 2, 3, 4, 5], index=1, key="asa")
    st.markdown('</div>', unsafe_allow_html=True)

    # Vitals
    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-teal"></span>
      <span class="s-title">Vitals</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    vc1, vc2, vc3 = st.columns(3)
    bmi = vc1.number_input("BMI (kg/m²)", 15.0, 70.0, 30.0, step=0.5, key="bmi")
    sbp = vc2.number_input("Systolic BP (mmHg)", 70, 220, 130, key="sbp")
    dbp = vc3.number_input("Diastolic BP (mmHg)", 40, 130, 80, key="dbp")
    st.markdown('</div>', unsafe_allow_html=True)

    # Labs
    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-amber"></span>
      <span class="s-title">Preoperative Labs</span>
      <span class="s-note">&nbsp;&mdash; leave blank for median imputation</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    lc1, lc2, lc3, lc4 = st.columns(4)
    creatinine = lc1.number_input("Creatinine (mg/dL)", 0.3, 15.0,
                                   value=None, placeholder="e.g. 0.9", key="cr")
    hemoglobin = lc2.number_input("Hemoglobin (g/dL)",  5.0, 20.0,
                                   value=None, placeholder="e.g. 13.5", key="hgb")
    wbc        = lc3.number_input("WBC (×10³/µL)",       0.5, 30.0,
                                   value=None, placeholder="e.g. 7.2", key="wbc")
    glucose    = lc4.number_input("Glucose (mg/dL)",      50,  600,
                                   value=None, placeholder="e.g. 95", key="glc")
    lc5, lc6, lc7, lc8 = st.columns(4)
    platelets  = lc5.number_input("Platelets (×10³/µL)",  10, 1000,
                                   value=None, placeholder="e.g. 220", key="plt")
    albumin    = lc6.number_input("Albumin (g/dL)",       1.0,  6.0,
                                   value=None, placeholder="e.g. 4.0", key="alb")
    hba1c      = lc7.number_input("HbA1c (%)",            4.0, 14.0,
                                   value=None, placeholder="If diabetic", key="hba1c")
    inr        = lc8.number_input("INR",                  0.5, 10.0,
                                   value=None, placeholder="If anticoagulated", key="inr")
    st.markdown('</div>', unsafe_allow_html=True)

    # Comorbidities
    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-rose"></span>
      <span class="s-title">Comorbidities</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    has_diabetes          = cc1.checkbox("Diabetes mellitus",       key="has_diabetes")
    has_hypertension      = cc1.checkbox("Hypertension",            key="has_hypertension")
    has_heart_disease     = cc1.checkbox("Heart disease",           key="has_heart_disease")
    has_copd              = cc1.checkbox("COPD",                    key="has_copd")
    has_anemia            = cc2.checkbox("Anemia",                  key="has_anemia")
    has_sleep_apnea       = cc2.checkbox("Sleep apnea",             key="has_sleep_apnea")
    has_liver_disease     = cc2.checkbox("Liver disease",           key="has_liver_disease")
    has_thyroid_disease   = cc2.checkbox("Thyroid disease",         key="has_thyroid_disease")
    has_anticoagulant     = cc3.checkbox("Anticoagulation",         key="has_anticoagulant")
    has_bleeding_disorder = cc3.checkbox("Bleeding disorder",       key="has_bleeding_disorder")
    has_malignancy        = cc3.checkbox("Active malignancy",       key="has_malignancy")
    has_ckd               = cc3.checkbox("Chronic kidney disease",  key="has_ckd")
    has_recent_ablation   = cc4.checkbox("Recent cardiac ablation", key="has_recent_ablation")
    has_thrombocytopenia  = cc4.checkbox("Thrombocytopenia",        key="has_thrombocytopenia")
    has_leukopenia        = cc4.checkbox("Leukopenia",              key="has_leukopenia")
    has_kidney_transplant = cc4.checkbox("Kidney transplant",       key="has_kidney_transplant")
    st.markdown('</div>', unsafe_allow_html=True)

    # Utilization
    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-muted"></span>
      <span class="s-title">Healthcare Utilization</span>
      <span class="badge">Prior 12 months</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    uc1, uc2 = st.columns(2)
    prior_hosp = uc1.number_input(
        f"Inpatient hospitalizations  (capped at {WINSOR_CAPS['prior_hospitalizations_1yr']})",
        0, 10, 0, key="prior_hosp")
    prior_ed = uc2.number_input(
        f"Emergency department visits  (capped at {WINSOR_CAPS['prior_ed_visits_1yr']})",
        0, 10, 0, key="prior_ed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("▶  Calculate 1-Year Mortality Risk",
                         type="primary", use_container_width=True)

# ── RESULT PANEL ──────────────────────────────────────────────────────────────
with col_result:

    st.markdown("""
    <div class="s-hdr">
      <span class="dot dot-blue"></span>
      <span class="s-title">Risk Estimate</span>
    </div>""", unsafe_allow_html=True)

    result_slot = st.empty()

    if not calc_btn:
        result_slot.markdown("""
        <div class="res-card" style="text-align:center; padding:50px 20px;">
          <div style="font-size:40px; margin-bottom:16px; opacity:0.4;">🦴</div>
          <div style="color:#475569; font-size:15px; line-height:1.9;">
            Complete the patient data and click<br>
            <strong style="color:#64748b;">Calculate 1-Year Mortality Risk</strong>
          </div>
        </div>""", unsafe_allow_html=True)
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
            has_anticoagulant=has_anticoagulant,
            has_bleeding_disorder=has_bleeding_disorder,
            has_malignancy=has_malignancy, has_ckd=has_ckd,
            has_recent_ablation=has_recent_ablation,
            has_thrombocytopenia=has_thrombocytopenia,
            has_leukopenia=has_leukopenia,
            has_kidney_transplant=has_kidney_transplant,
            prior_hosp=prior_hosp, prior_ed=prior_ed,
        )
        r = run_prediction(inputs)

        def _thr_color(above):
            return "#f43f5e" if above else "#1e293b"
        def _thr_text(above):
            return "#f43f5e" if above else "#475569"

        result_slot.markdown(f"""
        <div class="verdict"
             style="background:{r['color']}15; border:1px solid {r['color']}40;">
          <div class="v-title" style="color:{r['color']};">
            {r['icon']}&nbsp;&nbsp;{r['tier']}
          </div>
          <div style="font-size:3.4em; font-weight:800; color:#ffffff;
                      letter-spacing:-1px; font-family:'Courier New',monospace;
                      margin:8px 0 4px;">{r['pct']:.2f}%</div>
          <div style="font-size:13px; color:#64748b; margin-bottom:10px;
                      font-family:'Courier New',monospace;">
            predicted 1-year all-cause mortality
          </div>
          <div class="v-body" style="color:#94a3b8;">{r['note']}</div>
        </div>

        <div class="info-card">
          <div class="info-ttl">Risk Context</div>
          <div class="t-row" style="margin-bottom:9px;">
            <span class="t-lbl">Cohort baseline (1-yr)</span>
            <span style="color:#e2e8f0;font-family:'Courier New',monospace;">
              {BASELINE*100:.3f}%</span>
          </div>
          <div class="t-row" style="margin-bottom:9px;">
            <span class="t-lbl">Enrichment vs baseline</span>
            <span style="color:#f59e0b;font-weight:700;
                         font-family:'Courier New',monospace;">{r['enr']:.1f}&times;</span>
          </div>
          <div class="t-row" style="margin-bottom:9px;">
            <span class="t-lbl">Comorbidity burden</span>
            <span style="color:#e2e8f0;font-family:'Courier New',monospace;">
              {r['comorbid']} / 16 conditions</span>
          </div>
          <div class="t-row">
            <span class="t-lbl">Raw (pre-calibration)</span>
            <span style="color:#475569;font-family:'Courier New',monospace;">
              {r['raw']*100:.2f}%</span>
          </div>
        </div>

        <div class="info-card">
          <div class="info-ttl">Threshold Comparison</div>
          <div class="t-row" style="margin-bottom:8px;">
            <span class="t-lbl">&ge;90% sensitivity anchor</span>
            <span style="color:{_thr_text(r['above_sens90'])};
                         font-family:'Courier New',monospace;
                         background:{_thr_color(r['above_sens90'])}22;
                         padding:1px 8px; border-radius:3px;">
              {THR_SENS90*100:.3f}%
              {"&nbsp;&#x2713;" if r['above_sens90'] else ""}</span>
          </div>
          <div class="t-row" style="margin-bottom:8px;">
            <span class="t-lbl">DCA net benefit</span>
            <span style="color:{_thr_text(r['above_dca'])};
                         font-family:'Courier New',monospace;
                         background:{_thr_color(r['above_dca'])}22;
                         padding:1px 8px; border-radius:3px;">
              {THR_DCA*100:.3f}%
              {"&nbsp;&#x2713;" if r['above_dca'] else ""}</span>
          </div>
          <div class="t-row" style="margin-bottom:8px;">
            <span class="t-lbl">Youden&apos;s J &mdash; primary</span>
            <span style="color:{_thr_text(r['above_youden'])};
                         font-family:'Courier New',monospace;
                         background:{_thr_color(r['above_youden'])}22;
                         padding:1px 8px; border-radius:3px;">
              {THR_YOUDEN*100:.3f}%
              {"&nbsp;&#x2713;" if r['above_youden'] else ""}</span>
          </div>
          <hr>
          <div class="t-row">
            <span class="t-lbl" style="color:#94a3b8;font-weight:700;">
              This patient</span>
            <span style="color:{r['color']};font-weight:800;font-size:1.1em;
                         font-family:'Courier New',monospace;">{r['pct']:.3f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model info — always visible
    st.markdown(f"""
    <div class="info-card" style="margin-top:14px;">
      <div class="info-ttl">Model Info</div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">Architecture</span>
        <span style="color:#e2e8f0;font-family:'Courier New',monospace;">XGBoost v7</span>
      </div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">AUC (cross-validated)</span>
        <span style="color:#4f90f6;font-weight:700;
                     font-family:'Courier New',monospace;">{AUC:.3f} &plusmn; 0.016</span>
      </div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">vs ASA class alone</span>
        <span style="color:#14b8a6;font-weight:700;
                     font-family:'Courier New',monospace;">+0.130 AUC</span>
      </div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">vs comorbidity count</span>
        <span style="color:#14b8a6;font-weight:700;
                     font-family:'Courier New',monospace;">+0.113 AUC</span>
      </div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">Training cohort</span>
        <span style="color:#e2e8f0;font-family:'Courier New',monospace;">
          {N_COHORT:,} patients</span>
      </div>
      <div class="t-row" style="margin-bottom:7px;">
        <span class="t-lbl">Top-decile mortality</span>
        <span style="color:#f59e0b;font-family:'Courier New',monospace;">
          2.50% (4.4&times; baseline)</span>
      </div>
      <div class="t-row">
        <span class="t-lbl">Features</span>
        <span style="color:#e2e8f0;font-family:'Courier New',monospace;">
          43 (no COSI criteria)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Reference table ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="s-hdr">
  <span class="dot dot-muted"></span>
  <span class="s-title">Predictor Reference &mdash; Top 15 by SHAP Importance</span>
  <span class="badge">1-Year All-Cause Mortality &middot; XGBoost v7</span>
</div>""", unsafe_allow_html=True)

st.markdown(REF_TABLE_HTML, unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div class="footer-txt">
  <span class="footer-em">Research use only.</span>
  Internally validated XGBoost model trained on de-identified multi-institutional
  administrative data (TriNetX). Cohort restricted to patients with &ge;365 days of
  post-surgical follow-up or confirmed death within 365 days of surgery
  (n={N_COHORT:,}; {N_EVENTS:,} events; event rate {BASELINE*100:.3f}%).
  External validation has not been performed. Thresholds derived via
  cross-validated Youden index, DCA net benefit maximization, and a 90%
  sensitivity anchor &mdash; all leakage-safe on out-of-fold predictions.
  <span class="footer-em">Not approved for clinical decision-making.
  Not intended to restrict access to surgery or guide treatment allocation.</span>
</div>
""", unsafe_allow_html=True)
