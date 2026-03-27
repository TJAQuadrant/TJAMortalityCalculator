
# Updated UI version with fixed CSS

import streamlit as st
import numpy as np
import json, os

st.set_page_config(layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(180deg, #020617 0%, #031224 100%);
    color: #ffffff !important;
}

* { color: #ffffff !important; }

div[data-testid="stVerticalBlock"] > div:empty,
div[data-testid="stHorizontalBlock"] > div:empty,
div[data-testid="stMarkdownContainer"]:empty,
div[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

div[style*="border-radius: 999px"] {
    display: none !important;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background: rgba(8,23,42,0.95) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}

div[data-baseweb="input"] > div {
    background: transparent !important;
    border: none !important;
}

input {
    color: #ffffff !important;
    background: transparent !important;
}

input::placeholder {
    color: #94a3b8 !important;
}

.stCheckbox > label {
    font-size: 18px !important;
    color: #ffffff !important;
}

.stCheckbox div[data-baseweb="checkbox"] {
    transform: scale(1.25);
}

</style>
""", unsafe_allow_html=True)

st.title("TJA 1-Year Mortality Risk Estimator")

col1, col2 = st.columns([3,2])

with col1:
    age = st.number_input("Age", 18, 100, 65)
    sex = st.selectbox("Sex", ["Female","Male"])
    bmi = st.number_input("BMI", value=30.0)

    st.subheader("Comorbidities")
    dm = st.checkbox("Diabetes")
    htn = st.checkbox("Hypertension")
    copd = st.checkbox("COPD")

with col2:
    if st.button("Calculate"):
        st.markdown("<div style='font-size:48px;'>2.4%</div>", unsafe_allow_html=True)
