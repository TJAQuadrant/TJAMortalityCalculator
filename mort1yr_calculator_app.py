"""
TJA 1-Year Mortality Risk Calculator
=====================================
Usage:  streamlit run mort1yr_calculator_app.py
Requires in same directory:
    mort1yr_model_fitted_v7.joblib
    mort1yr_calibrator_v7.joblib
    mort1yr_imputer_fills_v7.json
    mort1yr_features_resolved_v7_nocosi.json
    style.css
    mort1yr_reftable.html
"""
import os, json
import numpy as np
import joblib
import streamlit as st

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Clinical thresholds (cross-validated, leakage-safe) ────────────────────────
THR_SENS90  = 0.00235   # ≥90% sensitivity anchor  -- sens 90.1%, spec 33.2%
THR_DCA     = 0.00500   # DCA net benefit           -- sens 68.7%, spec 70.2%
THR_YOUDEN  = 0.00770   # Youden's J primary        -- sens 54.8%, spec 83.9%
BASELINE    = 0.005737  # Cohort 1-year mortality rate (restricted cohort)
AUC         = 0.761
N_TRAIN     = 234_252

st.set_page_config(page_title="TJA Mortality Risk Calculator",
                   layout="wide", initial_sidebar_state="collapsed")

with open(os.path.join(MODEL_DIR, "style.css"), encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Artifacts ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model      = joblib.load(os.path.join(MODEL_DIR, "mort1yr_model_fitted_v7.joblib"))
    calibrator = joblib.load(os.path.join(MODEL_DIR, "mort1yr_calibrator_v7.joblib"))
    with open(os.path.join(MODEL_DIR, "mort1yr_imputer_fills_v7.json")) as f:
        fills = json.load(f)
    with open(os.path.join(MODEL_DIR, "mort1yr_features_resolved_v7_nocosi.json")) as f:
        feat_order = json.load(f)
    return model, calibrator, fills, feat_order

try:
    model, calibrator, imputer_fills, feature_order = load_artifacts()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    ARTIFACT_ERR = str(e)

# ── Keys / Labels ──────────────────────────────────────────────────────────────
COMORBIDITY_KEYS = [
    "has_diabetes","has_hypertension","has_heart_disease","has_copd",
    "has_anemia","has_sleep_apnea","has_liver_disease","has_thyroid_disease",
    "has_anticoagulant","has_bleeding_disorder","has_malignancy","has_ckd",
    "has_recent_ablation","has_thrombocytopenia","has_leukopenia",
    "has_kidney_transplant",
]
COMORB_LABELS = {
    "has_diabetes":           "Diabetes mellitus",
    "has_hypertension":       "Hypertension",
    "has_heart_disease":      "Heart disease (CAD / HF / valvular)",
    "has_copd":               "COPD",
    "has_anemia":             "Anemia",
    "has_sleep_apnea":        "Obstructive sleep apnea",
    "has_liver_disease":      "Liver disease",
    "has_thyroid_disease":    "Thyroid disease",
    "has_anticoagulant":      "Anticoagulation therapy",
    "has_bleeding_disorder":  "Bleeding disorder",
    "has_malignancy":         "Active malignancy",
    "has_ckd":                "Chronic kidney disease",
    "has_recent_ablation":    "Recent cardiac ablation",
    "has_thrombocytopenia":   "Thrombocytopenia",
    "has_leukopenia":         "Leukopenia",
    "has_kidney_transplant":  "Prior kidney transplant",
}
WINSOR_CAPS = {"prior_hospitalizations_1yr": 3, "prior_ed_visits_1yr": 2}

# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "needs_recalc" not in st.session_state:
    st.session_state.needs_recalc = False

def mark_dirty():
    st.session_state.needs_recalc = True

def do_reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.result       = None
    st.session_state.needs_recalc = False
    for k in COMORBIDITY_KEYS:
        st.session_state[f"cb_{k}"] = False
    st.rerun()

# ── Scoring ────────────────────────────────────────────────────────────────────
def score(inp):
    f = {}
    f["age_at_surgery"] = float(inp["age"])
    f["sex_encoded"]    = 1.0 if inp["sex"] == "Male" else 0.0
    f["proc_THA"]       = 1.0 if inp["procedure"] == "THA" else 0.0
    f["proc_TKA"]       = 1.0 if inp["procedure"] == "TKA" else 0.0
    f["bmi"]            = float(inp["bmi"])
    f["systolic_bp"]    = float(inp["systolic_bp"])
    f["diastolic_bp"]   = float(inp["diastolic_bp"])
    f["asa_proxy"]      = float(inp["asa"])

    # Continuous labs (None → impute)
    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        f[lab] = inp.get(lab)  # may be None

    # Missing indicator flags
    f["creatinine_missing"]        = 0.0 if f["creatinine"] is not None else 1.0
    f["hemoglobin_missing"]        = 0.0 if f["hemoglobin"] is not None else 1.0
    f["wbc_missing"]               = 0.0 if f["wbc"]        is not None else 1.0
    f["glucose_missing"]           = 0.0 if f["glucose"]    is not None else 1.0
    f["platelets_missing"]         = 0.0 if f["platelets"]  is not None else 1.0
    f["albumin_indicated_missing"] = float(inp.get("has_liver_disease", False) and f["albumin"]   is None)
    f["hba1c_indicated_missing"]   = float(inp.get("has_diabetes",      False) and f["hba1c"]    is None)
    f["inr_indicated_missing"]     = float(inp.get("has_anticoagulant", False) and f["inr"]      is None)

    # Impute None → median fill
    for lab in ["creatinine","hemoglobin","wbc","glucose","platelets","albumin","hba1c","inr"]:
        if f[lab] is None:
            f[lab] = float(imputer_fills.get(lab, 0.0))

    # Comorbidities
    for k in COMORBIDITY_KEYS:
        f[k] = float(inp.get(k, False))
    f["comorbidity_count"] = float(sum(f[k] for k in COMORBIDITY_KEYS))

    # Utilization
    hosp = min(int(inp.get("hospitalizations", 0)), WINSOR_CAPS["prior_hospitalizations_1yr"])
    ed   = min({"none":0,"1":1,"2plus":2}[inp["ed_cat"]], WINSOR_CAPS["prior_ed_visits_1yr"])
    f["prior_hospitalizations_1yr"] = float(hosp)
    f["prior_ed_visits_1yr"]        = float(ed)
    f["any_prior_hosp_1yr"]         = float(hosp > 0)
    f["any_prior_ed_1yr"]           = float(ed > 0)

    # Build feature vector in model order
    row = []
    for feat in feature_order:
        val = f.get(feat, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = float(imputer_fills.get(feat, 0.0))
        row.append(float(val))

    X        = np.array(row, dtype=np.float32).reshape(1, -1)
    raw_prob = float(model.predict_proba(X)[0][1])
    cal_prob = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))
    return raw_prob, cal_prob, f


def make_gauge(prob):
    """SVG semicircle gauge for 1-year mortality (capped display at 5%):
       Teal  0 – 0.50%   below DCA threshold — average / low risk
       Amber 0.50-0.77%  DCA to Youden — elevated risk
       Red   0.77-5%     above Youden — high risk
    """
    MAX_DISP = 0.05    # cap at 5% -- covers all clinically meaningful range
    W, H = 340, 195
    cx, cy, r_out, r_in = 170, 175, 150, 95

    def arc_pt(pct, radius):
        angle = np.pi * (1 - min(pct, MAX_DISP) / MAX_DISP)
        return cx + radius * np.cos(angle), cy - radius * np.sin(angle)

    zones = [
        (0,         THR_DCA,    "#0f2a27", "#14b8a6"),   # teal  — low/average
        (THR_DCA,   THR_YOUDEN, "#3b2700", "#f59e0b"),   # amber — elevated
        (THR_YOUDEN, MAX_DISP,  "#3b0a0e", "#f43f5e"),   # red   — high risk
    ]

    if prob <= THR_DCA:
        needle_color = "#14b8a6"
    elif prob <= THR_YOUDEN:
        needle_color = "#f59e0b"
    else:
        needle_color = "#f43f5e"

    needle_pct = min(prob, MAX_DISP)
    angle = np.pi * (1 - needle_pct / MAX_DISP)
    nx = cx + 128 * np.cos(angle)
    ny = cy - 128 * np.sin(angle)

    ticks = [(0, "0%"), (0.01, "1%"), (0.02, "2%"),
             (0.03, "3%"), (0.04, "4%"), (0.05, "5%")]

    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;display:block;margin:0 auto;">'
    svg += f'<rect width="{W}" height="{H}" fill="#0b0e14" rx="10"/>'

    # Background arc
    x0b, y0b = arc_pt(0, r_out);  x1b, y1b = arc_pt(MAX_DISP, r_out)
    x0i, y0i = arc_pt(0, r_in);   x1i, y1i = arc_pt(MAX_DISP, r_in)
    svg += f'<path d="M {x0b:.1f} {y0b:.1f} A {r_out} {r_out} 0 0 0 {x1b:.1f} {y1b:.1f} L {x1i:.1f} {y1i:.1f} A {r_in} {r_in} 0 0 1 {x0i:.1f} {y0i:.1f} Z" fill="#131822"/>'

    # Colored zones
    for z0, z1, fill_bg, fill_arc in zones:
        xo0,yo0 = arc_pt(z0, r_out); xo1,yo1 = arc_pt(z1, r_out)
        xi0,yi0 = arc_pt(z0, r_in);  xi1,yi1 = arc_pt(z1, r_in)
        large = 1 if (z1 - z0) / MAX_DISP > 0.5 else 0
        svg += f'<path d="M {xo0:.1f} {yo0:.1f} A {r_out} {r_out} 0 {large} 0 {xo1:.1f} {yo1:.1f} L {xi1:.1f} {yi1:.1f} A {r_in} {r_in} 0 {large} 1 {xi0:.1f} {yi0:.1f} Z" fill="{fill_bg}" opacity="0.8"/>'
        svg += f'<path d="M {xo0:.1f} {yo0:.1f} A {r_out} {r_out} 0 {large} 0 {xo1:.1f} {yo1:.1f}" fill="none" stroke="{fill_arc}" stroke-width="5" opacity="0.95"/>'

    # Threshold tick lines + percentage labels
    for thr_z, color_z, lbl in [
        (THR_DCA,    "#14b8a6", f"{THR_DCA*100:.2f}%"),
        (THR_YOUDEN, "#f43f5e", f"{THR_YOUDEN*100:.2f}%"),
    ]:
        if thr_z <= MAX_DISP:
            tx0z, ty0z = arc_pt(thr_z, r_out + 6)
            tx1z, ty1z = arc_pt(thr_z, r_in - 6)
            svg += f'<line x1="{tx0z:.1f}" y1="{ty0z:.1f}" x2="{tx1z:.1f}" y2="{ty1z:.1f}" stroke="{color_z}" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.9"/>'
            ttx, tty = arc_pt(thr_z, r_out + 24)
            svg += f'<text x="{ttx:.1f}" y="{tty:.1f}" fill="{color_z}" font-size="9" text-anchor="middle" font-family="Courier New">{lbl}</text>'

    # Baseline tick
    if BASELINE <= MAX_DISP:
        bx0, by0 = arc_pt(BASELINE, r_out + 4)
        bx1, by1 = arc_pt(BASELINE, r_in - 4)
        svg += f'<line x1="{bx0:.1f}" y1="{by0:.1f}" x2="{bx1:.1f}" y2="{by1:.1f}" stroke="#475569" stroke-width="1" stroke-dasharray="2,2" opacity="0.7"/>'

    # Tick marks
    for pct, lbl in ticks:
        tx, ty = arc_pt(pct, r_out + 16)
        ix, iy = arc_pt(pct, r_out + 2)
        ox, oy = arc_pt(pct, r_out - 4)
        svg += f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" stroke="#334155" stroke-width="1.5"/>'
        svg += f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#64748b" font-size="10" text-anchor="middle" dominant-baseline="middle" font-family="Courier New">{lbl}</text>'

    # Needle
    svg += f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{needle_color}" stroke-width="3.5" stroke-linecap="round"/>'
    svg += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{needle_color}"/>'
    svg += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#0b0e14"/>'

    # Center readout
    pct_str = f"{100*prob:.2f}%"
    if prob <= THR_DCA:
        zone_str = "Average / Low Risk"
    elif prob <= THR_YOUDEN:
        zone_str = "Elevated Risk"
    else:
        zone_str = "High Risk"
    svg += f'<text x="{cx}" y="{cy-28}" fill="{needle_color}" font-size="26" font-weight="700" text-anchor="middle" font-family="Courier New">{pct_str}</text>'
    svg += f'<text x="{cx}" y="{cy-8}" fill="#64748b" font-size="10" text-anchor="middle" font-family="Courier New">{zone_str}</text>'

    svg += "</svg>"
    return svg


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-hdr">
    <div class="app-hdr-title">TJA 1-Year Mortality Risk Calculator</div>
    <div class="app-hdr-sub">XGBoost v7 &middot; AUC {AUC} &middot; n={N_TRAIN:,} &middot; Isotonic Calibration &middot; Restricted cohort (&ge;365d follow-up or confirmed death) &middot; Research use only</div>
</div>
""", unsafe_allow_html=True)

if not ARTIFACTS_OK:
    st.error(f"Could not load model artifacts: {ARTIFACT_ERR}")
    st.stop()

col_form, col_result = st.columns([11, 6], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# FORM COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_form:
    st.markdown('<div style="padding:16px 6px 0 6px;">', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    # ── Demographics & Procedure ──────────────────────────────────────────────
    st.markdown("""<div class="s-hdr" style="margin-top:0;">
        <div class="dot dot-blue"></div>
        <span class="s-title">Demographics &amp; Procedure</span>
        <span class="s-note">&nbsp;— age rank 1, sex rank 2</span>
    </div>""", unsafe_allow_html=True)

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1: age          = st.number_input("Age (years)",         min_value=18,  max_value=100,  value=68,   step=1,   on_change=mark_dirty)
    with dc2: bmi          = st.number_input("BMI (kg/m²)",         min_value=15.0, max_value=80.0, value=30.0, step=0.5, on_change=mark_dirty)
    with dc3: systolic_bp  = st.number_input("Systolic BP (mmHg)",  min_value=60,  max_value=250,  value=130,  step=1,   on_change=mark_dirty)
    with dc4: diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40,  max_value=150,  value=80,   step=1,   on_change=mark_dirty)

    pc1, pc2, pc3 = st.columns(3)
    with pc1: sex       = st.radio("Sex",       ["Female","Male"],     horizontal=True, on_change=mark_dirty)
    with pc2: procedure = st.radio("Procedure", ["TKA","THA"],         horizontal=True, on_change=mark_dirty)
    with pc3: asa_lbl   = st.radio("ASA Class", ["I","II","III","IV"], horizontal=True, index=1, on_change=mark_dirty)
    asa = ["I","II","III","IV"].index(asa_lbl) + 1

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Laboratory Values ─────────────────────────────────────────────────────
    st.markdown("""<div class="s-hdr">
        <div class="dot dot-amber"></div>
        <span class="s-title">Laboratory Values</span>
        <span class="s-note">&nbsp;— creatinine rank 3, hemoglobin rank 4, WBC rank 7</span>
    </div>""", unsafe_allow_html=True)

    has_anemia_now  = st.session_state.get("cb_has_anemia",       False)
    has_liver_now   = st.session_state.get("cb_has_liver_disease", False)
    has_anticoag_now= st.session_state.get("cb_has_anticoagulant", False)
    has_diabetes_now= st.session_state.get("cb_has_diabetes",      False)

    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=15.0,
                                     value=None, placeholder="e.g. 0.9", on_change=mark_dirty)
    with lc2:
        st.markdown('<span class="lab-lbl">Hemoglobin</span>', unsafe_allow_html=True)
        if has_anemia_now:
            hemoglobin = st.number_input("hgb_val", min_value=5.0, max_value=20.0,
                                         value=None, placeholder="g/dL",
                                         label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Anemia" below to activate</span>', unsafe_allow_html=True)
            hemoglobin = None
    with lc3:
        wbc = st.number_input("WBC (×10³/µL)", min_value=0.5, max_value=30.0,
                               value=None, placeholder="e.g. 7.2", on_change=mark_dirty)
    with lc4:
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=600,
                                   value=None, placeholder="e.g. 95", on_change=mark_dirty)

    lc5, lc6, lc7, lc8 = st.columns(4)
    with lc5:
        platelets = st.number_input("Platelets (×10³/µL)", min_value=10, max_value=1000,
                                     value=None, placeholder="e.g. 220", on_change=mark_dirty)
    with lc6:
        st.markdown('<span class="lab-lbl">Albumin</span>', unsafe_allow_html=True)
        if has_liver_now:
            albumin = st.number_input("alb_val", min_value=1.0, max_value=6.0,
                                       value=None, placeholder="g/dL",
                                       label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Liver disease" below to activate</span>', unsafe_allow_html=True)
            albumin = None
    with lc7:
        st.markdown('<span class="lab-lbl">HbA1c</span>', unsafe_allow_html=True)
        if has_diabetes_now:
            hba1c = st.number_input("hba1c_val", min_value=4.0, max_value=14.0,
                                     value=None, placeholder="%",
                                     label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Diabetes" below to activate</span>', unsafe_allow_html=True)
            hba1c = None
    with lc8:
        st.markdown('<span class="lab-lbl">INR</span>', unsafe_allow_html=True)
        if has_anticoag_now:
            inr = st.number_input("inr_val", min_value=0.5, max_value=10.0,
                                   value=None, placeholder="e.g. 1.1",
                                   label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Anticoagulation" below to activate</span>', unsafe_allow_html=True)
            inr = None

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Comorbidities ─────────────────────────────────────────────────────────
    cc_count  = sum(1 for k in COMORBIDITY_KEYS if st.session_state.get(f"cb_{k}", False))
    badge_cls = "badge-hi" if cc_count >= 4 else "badge"
    st.markdown(f"""<div class="s-hdr">
        <div class="dot dot-teal"></div>
        <span class="s-title">Comorbidities</span>
        <span class="s-note">&nbsp;— comorbidity count rank 8, COPD rank 13</span>
        <span class="{badge_cls}">{cc_count} active</span>
    </div>""", unsafe_allow_html=True)

    cmorb_items = list(COMORB_LABELS.items())
    cm1, cm2, cm3, cm4 = st.columns(4)
    comorbidity_vals = {}
    cols = [cm1, cm2, cm3, cm4]
    for i, (key, label) in enumerate(cmorb_items):
        with cols[i % 4]:
            comorbidity_vals[key] = st.checkbox(label, value=False, key=f"cb_{key}", on_change=mark_dirty)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Healthcare Utilization ────────────────────────────────────────────────
    st.markdown("""<div class="s-hdr">
        <div class="dot dot-muted"></div>
        <span class="s-title">Healthcare Utilization (prior 12 months)</span>
        <span class="s-note">&nbsp;— prior ED visits rank 14</span>
    </div>""", unsafe_allow_html=True)

    uc1, uc2 = st.columns(2)
    with uc1:
        hospitalizations = st.number_input(
            f"Inpatient hospitalizations (capped at {WINSOR_CAPS['prior_hospitalizations_1yr']})",
            min_value=0, max_value=20, value=0, step=1, on_change=mark_dirty)
    with uc2:
        ed_sel = st.radio(
            f"ED visits (capped at {WINSOR_CAPS['prior_ed_visits_1yr']})",
            ["None","1","2+"], horizontal=True, on_change=mark_dirty)
    ed_cat = {"None":"none","1":"1","2+":"2plus"}[ed_sel]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Action Buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns([3, 2, 2])
    with btn1:
        calc_clicked = st.button("Calculate Mortality Risk", type="primary", use_container_width=True)
    with btn2:
        if st.button("Reset All", use_container_width=True):
            do_reset()
    with btn3:
        if st.session_state.needs_recalc and st.session_state.result is not None:
            st.markdown('<div class="stale-warn">Inputs changed — recalculate</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close form-card

    # ── Reference Table ───────────────────────────────────────────────────────
    ref_path = os.path.join(MODEL_DIR, "mort1yr_reftable.html")
    with st.expander("Reference: Predictor Importance & Clinical Interpretation  —  ranked by SHAP value"):
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as _rf:
                st.markdown(_rf.read(), unsafe_allow_html=True)
        else:
            st.caption("mort1yr_reftable.html not found in model directory.")

    st.markdown(f"""
    <div class="footer-txt" style="margin-top:14px;">
        <span class="footer-em">Research use only.</span>
        {N_TRAIN:,} patients, restricted to &ge;365-day post-surgical follow-up or confirmed death.
        Leakage-safe nested 5-fold cross-validation with isotonic calibration.
        Model outperforms ASA class alone by +0.130 AUC and comorbidity count alone by +0.113 AUC.
        Not approved for clinical decision-making or to restrict surgical access.
    </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_result:
    st.markdown('<div style="padding:16px 0 0 0;">', unsafe_allow_html=True)

    inp = dict(
        age=age, bmi=bmi, sex=sex, procedure=procedure, asa=asa,
        systolic_bp=systolic_bp, diastolic_bp=diastolic_bp,
        creatinine=creatinine, hemoglobin=hemoglobin, wbc=wbc,
        glucose=glucose, platelets=platelets, albumin=albumin,
        hba1c=hba1c, inr=inr,
        hospitalizations=hospitalizations, ed_cat=ed_cat,
    )
    inp.update(comorbidity_vals)

    if calc_clicked:
        try:
            raw_prob, cal_prob, feats = score(inp)
            st.session_state.result = dict(
                cal_prob=cal_prob, raw_prob=raw_prob, feats=feats
            )
            st.session_state.needs_recalc = False
        except Exception as e:
            st.error(f"Scoring error: {e}")

    res = st.session_state.result

    if res is None:
        st.markdown("""
        <div class="res-card" style="text-align:center;padding:40px 20px;">
            <div style="font-size:18px;color:#475569;margin-bottom:8px;">No result yet</div>
            <div style="font-size:14px;color:#334155;">Complete the form and press<br>
            <strong style="color:#4f90f6;">Calculate Mortality Risk</strong></div>
        </div>""", unsafe_allow_html=True)
    else:
        cal_prob = res["cal_prob"]
        feats    = res["feats"]

        # Gauge
        st.markdown(make_gauge(cal_prob), unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # Verdict
        if cal_prob <= THR_DCA:
            v_bg, v_bd, v_tc = "#0f2a27", "#0f766e", "#14b8a6"
            v_title = "Average / Low Risk"
            v_body  = (f"Predicted 1-year mortality ({100*cal_prob:.2f}%) is below the "
                       f"DCA net-benefit threshold ({100*THR_DCA:.2f}%) and at or near "
                       f"the cohort baseline ({100*BASELINE:.3f}%). Standard perioperative pathway supported.")
        elif cal_prob <= THR_YOUDEN:
            v_bg, v_bd, v_tc = "#2d1f00", "#92400e", "#f59e0b"
            v_title = "Elevated Risk — Clinician Discretion"
            v_body  = (f"Predicted 1-year mortality ({100*cal_prob:.2f}%) exceeds the DCA "
                       f"threshold ({100*THR_DCA:.2f}%) but is below the Youden-optimal "
                       f"threshold ({100*THR_YOUDEN:.2f}%). Enhanced perioperative monitoring may be warranted.")
        else:
            v_bg, v_bd, v_tc = "#1f0a0e", "#991b1b", "#f43f5e"
            v_title = "High Risk — Enhanced Surveillance Supported"
            v_body  = (f"Predicted 1-year mortality ({100*cal_prob:.2f}%) exceeds the "
                       f"Youden-optimal threshold ({100*THR_YOUDEN:.2f}%, sensitivity 54.8%, "
                       f"specificity 83.9%). This patient is in the highest-risk stratum; "
                       f"enhanced perioperative surveillance is supported by the model.")

        st.markdown(f"""
        <div class="verdict" style="background:{v_bg};border:1px solid {v_bd};">
            <div class="v-title" style="color:{v_tc};">{v_title}</div>
            <div class="v-body" style="color:#94a3b8;">{v_body}</div>
        </div>""", unsafe_allow_html=True)

        # Threshold context card
        def _trow(label, thr, above):
            col  = "#f43f5e" if above else "#475569"
            mark = "&nbsp;&#x2713;" if above else ""
            return (f'<div class="t-row" style="margin-bottom:8px;">'
                    f'<span class="t-lbl">{label}</span>'
                    f'<span style="color:{col};font-family:\'Courier New\',monospace;">'
                    f'{thr*100:.3f}%{mark}</span></div>')

        st.markdown(f"""
        <div class="info-card">
            <div class="info-ttl">Threshold Comparison</div>
            {_trow("≥90% sensitivity anchor", THR_SENS90,  cal_prob >= THR_SENS90)}
            {_trow("DCA net benefit",          THR_DCA,     cal_prob >= THR_DCA)}
            {_trow("Youden's J — primary",     THR_YOUDEN,  cal_prob >= THR_YOUDEN)}
            <hr>
            <div class="t-row">
                <span class="t-lbl" style="color:#94a3b8;font-weight:700;">This patient</span>
                <span style="color:{v_tc};font-weight:800;font-family:'Courier New',monospace;">
                    {100*cal_prob:.3f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Key model inputs summary
        comorb_n = int(feats.get("comorbidity_count", 0))
        enr      = cal_prob / BASELINE if BASELINE > 0 else float("nan")
        st.markdown(f"""
        <div class="info-card">
            <div class="info-ttl">Key Model Inputs</div>
            <div style="font-size:15px;color:#94a3b8;line-height:2.1;">
                <span style="color:#e2e8f0;">Comorbidity count:</span>&nbsp;&nbsp;{comorb_n} / 16<br>
                <span style="color:#e2e8f0;">Enrichment vs baseline:</span>&nbsp;&nbsp;
                    <span style="color:#f59e0b;">{enr:.1f}&times;</span><br>
                <span style="color:#e2e8f0;">Prior hospitalizations:</span>&nbsp;&nbsp;{int(feats.get("prior_hospitalizations_1yr",0))}<br>
                <span style="color:#e2e8f0;">Prior ED visits:</span>&nbsp;&nbsp;{int(feats.get("prior_ed_visits_1yr",0))}<br>
                <span style="color:#e2e8f0;">Procedure:</span>&nbsp;&nbsp;{"THA" if feats.get("proc_THA") else "TKA"}<br>
                <span style="color:#e2e8f0;">Raw (pre-calibration):</span>&nbsp;&nbsp;{100*res["raw_prob"]:.2f}%
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.needs_recalc:
            st.markdown('<div class="stale-warn" style="margin-top:4px;">Inputs changed since last calculation</div>',
                        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
