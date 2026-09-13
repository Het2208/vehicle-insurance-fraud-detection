"""
ClaimGuard AI - AI-Powered Insurance Claim Fraud Detection

A production-styled Streamlit application that wraps a trained
XGBoost / GridSearchCV fraud-detection model for interactive claim
analysis.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from utils.model_utils import predict_claim
from utils.schema import ClaimInput, ONE_HOT_GROUPS, VEHICLE_CATEGORY_MAP

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ClaimGuard AI | Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hero */
    .cg-hero {
        background: linear-gradient(135deg, #0F2C55 0%, #1E4B8F 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        margin-bottom: 1.8rem;
    }
    .cg-hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }
    .cg-hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #FFFFFF;
    }
    .cg-hero p {
        font-size: 1.02rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
        max-width: 620px;
    }

    /* Section headers */
    .cg-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F2C55;
        margin: 1.6rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E5EAF2;
    }

    /* Cards */
    .cg-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Result banners */
    .cg-result-fraud {
        background: #FDECEC;
        border: 1px solid #F3B6B6;
        border-left: 6px solid #C0392B;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0;
    }
    .cg-result-valid {
        background: #E9F7EF;
        border: 1px solid #A9DFBF;
        border-left: 6px solid #1E8449;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0;
    }
    .cg-result-title-fraud { color: #922B21; font-size: 1.5rem; font-weight: 800; margin: 0 0 0.3rem 0; }
    .cg-result-title-valid { color: #145A32; font-size: 1.5rem; font-weight: 800; margin: 0 0 0.3rem 0; }
    .cg-result-sub { color: #33414F; font-size: 0.96rem; margin: 0; }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: #1E4B8F;
        border: none;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        border-radius: 8px;
        font-size: 1.02rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #163a6f;
    }

    /* Footer */
    .cg-footer {
        text-align: center;
        color: #6B7A8F;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E5EAF2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "submitted_summary" not in st.session_state:
    st.session_state.submitted_summary = None

# --------------------------------------------------------------------------
# Hero section
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="cg-hero">
        <span class="cg-hero-badge">🛡️ AI FRAUD DETECTION SYSTEM</span>
        <h1>ClaimGuard AI</h1>
        <p>AI-Powered Insurance Claim Fraud Detection — analyze claim details in
        seconds and get a model-driven risk assessment to support your review
        workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------
with st.form("claim_form", clear_on_submit=False):

    st.markdown('<div class="cg-section-title">👤 Customer Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender_label = st.selectbox("Gender", ["Female", "Male"], index=0)
        gender = "F" if gender_label == "Female" else "M"
    with c2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married"], index=0)
    with c3:
        age_of_driver = st.number_input("Driver Age", min_value=16, max_value=100, value=35, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        annual_income = st.number_input(
            "Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f"
        )
    with c5:
        property_status = st.selectbox("Home Ownership", ONE_HOT_GROUPS["property_status"], index=0)
    with c6:
        safety_rating = st.slider("Driver Safety Rating", min_value=0, max_value=100, value=80)

    c7, c8 = st.columns(2)
    with c7:
        high_education = st.checkbox("Has Higher Education", value=False)
    with c8:
        address_change = st.checkbox("Recent Address Change", value=False)

    st.markdown('<div class="cg-section-title">📄 Policy Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        channel = st.selectbox("Sales Channel", ONE_HOT_GROUPS["channel"], index=1)
    with p2:
        policy_deductible = st.number_input(
            "Policy Deductible ($)", min_value=0.0, value=500.0, step=50.0
        )
    with p3:
        annual_premium = st.number_input(
            "Annual Premium ($)", min_value=0.0, value=1200.0, step=50.0
        )

    p4, p5, p6 = st.columns(3)
    with p4:
        liab_prct = st.slider("Liability Percentage", min_value=0, max_value=100, value=50)
    with p5:
        past_num_of_claims = st.number_input(
            "Past Number of Claims", min_value=0, max_value=50, value=0, step=1
        )
    with p6:
        days_open = st.number_input(
            "Days Claim Has Been Open", min_value=0, max_value=3650, value=10, step=1
        )

    p7, p8, p9 = st.columns(3)
    with p7:
        police_report = st.checkbox("Police Report Filed", value=False)
    with p8:
        witness_present = st.checkbox("Witness Present", value=False)
    with p9:
        form_defects = st.number_input(
            "Claim Form Defects Found", min_value=0, max_value=20, value=0, step=1,
            help="Number of inconsistencies or defects identified on the submitted claim form.",
        )

    st.markdown('<div class="cg-section-title">📍 Incident Information</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        claim_date = st.date_input("Claim Date", value=date(2024, 1, 1))
    with i2:
        accident_site = st.selectbox("Accident Site", ONE_HOT_GROUPS["accident_site"], index=1)

    claim_day_of_week = claim_date.strftime("%A")
    st.caption(f"Day of week (auto-detected from claim date): **{claim_day_of_week}**")

    st.markdown('<div class="cg-section-title">🚗 Vehicle Information</div>', unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        vehicle_category = st.selectbox("Vehicle Category", list(VEHICLE_CATEGORY_MAP.keys()), index=0)
    with v2:
        vehicle_color = st.selectbox("Vehicle Color", ONE_HOT_GROUPS["vehicle_color"], index=6)
    with v3:
        age_of_vehicle = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5, step=1)

    v4, = st.columns(1)
    with v4:
        vehicle_price = st.number_input(
            "Vehicle Price ($)", min_value=0.0, value=20000.0, step=500.0
        )

    st.markdown('<div class="cg-section-title">💰 Claim Amounts</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        total_claim = st.number_input(
            "Total Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0
        )
    with a2:
        injury_claim = st.number_input(
            "Injury Claim Amount ($)", min_value=0.0, value=0.0, step=100.0
        )

    with st.expander("⚙️ Advanced / Computed Risk Metrics (optional overrides)"):
        st.caption(
            "These metrics are automatically derived from the details above. "
            "Override them only if you have exact figures from your own data source."
        )
        o1, o2 = st.columns(2)
        with o1:
            override_claim_ratio = st.checkbox("Override claim-to-income ratio")
            claim_ratio_val = st.number_input(
                "Claim Ratio", min_value=0.0, value=0.0, step=0.01, disabled=not override_claim_ratio
            )
        with o2:
            override_claim_vehicle_ratio = st.checkbox("Override claim-to-vehicle-price ratio")
            claim_vehicle_ratio_val = st.number_input(
                "Claim / Vehicle Price Ratio", min_value=0.0, value=0.0, step=0.01,
                disabled=not override_claim_vehicle_ratio,
            )
        o3, o4 = st.columns(2)
        with o3:
            override_vehicle_old = st.checkbox("Manually set 'vehicle is old' flag")
            vehicle_old_val = st.checkbox("Vehicle is old (10+ years)", disabled=not override_vehicle_old)
        with o4:
            override_weekend = st.checkbox("Manually set 'weekend claim' flag")
            weekend_val = st.checkbox("Claim filed on a weekend", disabled=not override_weekend)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Analyze Claim", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Handle submission
# --------------------------------------------------------------------------
if submitted:
    claim = ClaimInput(
        gender=gender,
        marital_status=marital_status,
        age_of_driver=int(age_of_driver),
        high_education=high_education,
        annual_income=float(annual_income),
        property_status=property_status,
        address_change=address_change,
        safety_rating=int(safety_rating),
        channel=channel,
        policy_deductible=float(policy_deductible),
        annual_premium=float(annual_premium),
        liab_prct=int(liab_prct),
        police_report=police_report,
        witness_present=witness_present,
        past_num_of_claims=int(past_num_of_claims),
        form_defects=int(form_defects),
        days_open=int(days_open),
        claim_year=claim_date.year,
        claim_month=claim_date.month,
        claim_day=claim_date.day,
        claim_day_of_week=claim_day_of_week,
        accident_site=accident_site,
        vehicle_category=vehicle_category,
        vehicle_color=vehicle_color,
        vehicle_price=float(vehicle_price),
        age_of_vehicle=int(age_of_vehicle),
        total_claim=float(total_claim),
        injury_claim=float(injury_claim),
        claim_ratio_override=claim_ratio_val if override_claim_ratio else None,
        claim_vehicle_ratio_override=claim_vehicle_ratio_val if override_claim_vehicle_ratio else None,
        vehicle_old_override=vehicle_old_val if override_vehicle_old else None,
        weekend_claim_override=weekend_val if override_weekend else None,
    )

    with st.spinner("Analyzing claim details..."):
        result = predict_claim(claim)

    st.session_state.result = result
    st.session_state.submitted_summary = {
        "Claim Date": claim_date.strftime("%B %d, %Y"),
        "Vehicle": f"{vehicle_category} ({vehicle_color})",
        "Total Claim": f"${total_claim:,.2f}",
        "Annual Income": f"${annual_income:,.2f}",
        "Sales Channel": channel,
        "Accident Site": accident_site,
    }

# --------------------------------------------------------------------------
# Result section
# --------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    st.markdown("---")
    if not result["success"]:
        st.error(result["error"])
    else:
        confidence = result["confidence"]
        confidence_pct = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"

        if result["is_fraud"]:
            st.markdown(
                f"""
                <div class="cg-result-fraud">
                    <p class="cg-result-title-fraud">⚠️ FRAUDULENT CLAIM</p>
                    <p class="cg-result-sub">Model Prediction: the model identified this claim as
                    having a high likelihood of fraud. This is not a guaranteed determination —
                    please review according to your organization's investigation process.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="cg-result-valid">
                    <p class="cg-result-title-valid">✅ VALID CLAIM</p>
                    <p class="cg-result-sub">Model Prediction: the model identified this claim as
                    likely legitimate. This is not a guaranteed determination — please review
                    according to your organization's standard policies.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="cg-section-title">📊 Prediction Summary</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", result["label"])
        m2.metric("Model Confidence", confidence_pct)
        risk_level = "High" if result["is_fraud"] else "Low"
        m3.metric("Risk Level", risk_level)

        if st.session_state.submitted_summary:
            with st.expander("📋 Submitted Claim Summary"):
                for k, v in st.session_state.submitted_summary.items():
                    st.write(f"**{k}:** {v}")

    reset_col = st.columns([1, 1, 1])[1]
    with reset_col:
        if st.button("🔄 Reset and Start New Prediction", use_container_width=True):
            st.session_state.result = None
            st.session_state.submitted_summary = None
            st.rerun()

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="cg-footer">
        ClaimGuard AI • Insurance Fraud Detection<br>
        Predictions are generated using a machine-learning model and should be
        reviewed according to organizational policies.
    </div>
    """,
    unsafe_allow_html=True,
)
