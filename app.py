import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# VEHICLE INSURANCE FRAUD DETECTION
# Streamlit frontend + inference backend
# Model: Logistic Regression (.pkl)
# ============================================================

MODEL_PATH = Path(__file__).parent / "veh_ins_fraud.pkl"

st.set_page_config(
    page_title="InsureGuard AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,.12), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(14,165,233,.10), transparent 25%),
        #f7f8fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #312e81 100%);
    color: white;
    box-shadow: 0 18px 50px rgba(15,23,42,.18);
    margin-bottom: 1.4rem;
}

.hero-kicker {
    color: #a5b4fc;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .45rem;
}

.hero h1 {
    font-size: 2.35rem;
    line-height: 1.1;
    margin: 0 0 .55rem 0;
    font-weight: 800;
}

.hero p {
    color: #dbeafe;
    margin: 0;
    font-size: 1rem;
}

.section-card {
    background: rgba(255,255,255,.92);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1.35rem 1.35rem .8rem 1.35rem;
    margin: .9rem 0 1rem 0;
    box-shadow: 0 8px 28px rgba(15,23,42,.055);
}

.section-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: .12rem;
}

.section-subtitle {
    color: #64748b;
    font-size: .84rem;
    margin-bottom: 1rem;
}

[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-weight: 600;
    color: #334155;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 11px;
    border-color: #dbe1ea;
    min-height: 42px;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,.10);
}

.stButton > button {
    border-radius: 12px;
    min-height: 48px;
    font-weight: 800;
    border: 0;
    background: linear-gradient(135deg, #4f46e5, #2563eb);
    color: white;
    box-shadow: 0 8px 22px rgba(37,99,235,.20);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(37,99,235,.28);
}

.result-card {
    border-radius: 22px;
    padding: 1.45rem;
    margin-top: 1.2rem;
    border: 1px solid #e2e8f0;
    background: white;
    box-shadow: 0 12px 35px rgba(15,23,42,.08);
}

.safe {
    border-left: 6px solid #16a34a;
}

.risk {
    border-left: 6px solid #dc2626;
}

.result-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: .35rem;
}

.result-copy {
    color: #64748b;
    margin-bottom: 1rem;
}

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: .9rem 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #111827;
}

.metric-label {
    font-size: .74rem;
    color: #64748b;
    margin-top: .15rem;
}

.small-note {
    color: #64748b;
    font-size: .76rem;
    line-height: 1.45;
}

.sidebar-title {
    font-weight: 800;
    font-size: 1rem;
    margin-bottom: .4rem;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 12px 14px;
    border-radius: 14px;
}

hr {
    border-color: #e5e7eb;
}

/* Make labels and controls breathe */
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: .08rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Keyboard navigation
# Press Enter to move to next field.
# Ctrl/⌘ + Enter submits the form.
# -----------------------------
st.markdown("""
<script>
(function () {
    const attach = () => {
        const root = window.parent.document;
        if (root.__insureguard_enter_handler) return;
        root.__insureguard_enter_handler = true;

        root.addEventListener("keydown", function(e) {
            if (e.key !== "Enter") return;

            const target = e.target;
            const tag = (target.tagName || "").toLowerCase();

            if (!["input", "select", "textarea"].includes(tag)) return;

            // Allow Ctrl/Command + Enter to submit.
            if (e.ctrlKey || e.metaKey) return;

            const fields = Array.from(root.querySelectorAll(
                'input:not([disabled]), select:not([disabled]), textarea:not([disabled])'
            )).filter(el => el.offsetParent !== null);

            const idx = fields.indexOf(target);
            if (idx === -1) return;

            e.preventDefault();

            if (idx + 1 < fields.length) {
                fields[idx + 1].focus();
                if (fields[idx + 1].select) fields[idx + 1].select();
            }
        }, true);
    };

    setTimeout(attach, 500);
})();
</script>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(
        "Could not load the model. Put `veh_ins_fraud.pkl` in the same folder as `app.py`."
    )
    st.exception(exc)
    st.stop()

FEATURES = list(getattr(model, "feature_names_in_", []))

if len(FEATURES) != 54:
    st.warning(
        f"The loaded model reports {len(FEATURES)} input features. "
        "This app was built from the feature names stored inside your uploaded PKL."
    )

# -----------------------------
# Helpers
# -----------------------------
def one_hot_row(prefix: str, selected: str, columns: list[str]) -> dict:
    return {col: int(col == f"{prefix}{selected}") for col in columns}

def build_input(
    gender,
    property_status,
    claim_day,
    accident_site,
    channel,
    vehicle_color,
    income_group,
    vehicle_category,
    age_of_driver,
    marital_status,
    safety_rating,
    annual_income,
    high_education,
    address_change,
    past_num_of_claims,
    witness_present,
    liab_prct,
    police_report,
    age_of_vehicle,
    vehicle_price,
    total_claim,
    injury_claim,
    policy_deductible,
    annual_premium,
    days_open,
    form_defects,
    claim_year,
    claim_month,
    claim_day_num,
    claim_ratio,
    vehicle_old,
    weekend_claim,
    claim_vehicle_ratio,
):
    row = {feature: 0.0 for feature in FEATURES}

    # One-hot categorical features stored in the PKL
    mappings = {
        "ohe__gender_": gender,
        "ohe__property_status_": property_status,
        "ohe__claim_day_of_week_": claim_day,
        "ohe__accident_site_": accident_site,
        "ohe__channel_": channel,
        "ohe__vehicle_color_": vehicle_color,
        "ohe__income_group_": income_group,
    }

    for prefix, selected in mappings.items():
        key = prefix + selected
        if key in row:
            row[key] = 1.0

    # Ordinal feature. The training notebook did not expose the label mapping,
    # so this control intentionally uses the encoded values used by the model.
    if "ord__vehicle_category" in row:
        row["ord__vehicle_category"] = float(vehicle_category)

    numeric = {
        "remainder__age_of_driver": age_of_driver,
        "remainder__marital_status": marital_status,
        "remainder__safety_rating": safety_rating,
        "remainder__annual_income": annual_income,
        "remainder__high_education": high_education,
        "remainder__address_change": address_change,
        "remainder__past_num_of_claims": past_num_of_claims,
        "remainder__witness_present": witness_present,
        "remainder__liab_prct": liab_prct,
        "remainder__police_report": police_report,
        "remainder__age_of_vehicle": age_of_vehicle,
        "remainder__vehicle_price": vehicle_price,
        "remainder__total_claim": total_claim,
        "remainder__injury_claim": injury_claim,
        "remainder__policy_deductible": policy_deductible,
        "remainder__annual_premium": annual_premium,
        "remainder__days_open": days_open,
        "remainder__form_defects": form_defects,
        "remainder__claim_year": claim_year,
        "remainder__claim_month": claim_month,
        "remainder__claim_day": claim_day_num,
        "remainder__claim_ratio": claim_ratio,
        "remainder__vehicle_old": vehicle_old,
        "remainder__weekend_claim": weekend_claim,
        "remainder__claim_vehicle_ratio": claim_vehicle_ratio,
    }

    for key, value in numeric.items():
        if key in row:
            row[key] = float(value)

    return pd.DataFrame([row], columns=FEATURES)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-kicker">AI-Powered Vehicle Insurance Analytics</div>
    <h1>🛡️ InsureGuard AI</h1>
    <p>Vehicle insurance fraud risk assessment powered by your trained Logistic Regression model.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Model Information</div>', unsafe_allow_html=True)
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Input features:** 54")
    st.write("**Classes:** 0 = Not Fraud · 1 = Fraud")
    st.divider()
    st.markdown("### ⌨️ Fast form navigation")
    st.caption(
        "Press **Enter** to move through fields. "
        "Use **Ctrl + Enter** (or **⌘ + Enter**) to submit."
    )
    st.divider()
    st.markdown("### Model reference")
    st.metric("Notebook accuracy", "74.62%")
    st.caption(
        "The accuracy shown here comes from the uploaded Jupyter notebook "
        "and is not a guarantee for a new claim."
    )

# -----------------------------
# Form
# -----------------------------
with st.form("fraud_prediction_form", clear_on_submit=False):

    st.markdown("""
    <div class="section-card">
        <div class="section-title">01 · Policy & claimant profile</div>
        <div class="section-subtitle">Start with the claimant, policy and vehicle classification.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        gender = st.selectbox("Gender", ["F", "M"], index=0)
        property_status = st.selectbox("Property status", ["Own", "Rent"], index=0)
        vehicle_category = st.selectbox(
            "Vehicle category · encoded",
            [0, 1, 2],
            index=1,
            help="The PKL stores vehicle_category as an ordinal encoded feature. "
                 "The uploaded notebook does not contain the original label-to-number mapping."
        )
        age_of_driver = st.number_input("Age of driver", 16, 100, 35, 1)

    with c2:
        marital_status = st.selectbox("Marital status · encoded", [0, 1], index=1)
        safety_rating = st.number_input("Safety rating", 0.0, 100.0, 50.0, 0.1)
        annual_income = st.number_input("Annual income", 0.0, 1000000.0, 60000.0, 1000.0)
        high_education = st.selectbox("Higher education · encoded", [0, 1], index=1)

    with c3:
        address_change = st.selectbox("Recent address change · encoded", [0, 1], index=0)
        past_num_of_claims = st.number_input("Past number of claims", 0, 50, 1, 1)
        witness_present = st.selectbox("Witness present · encoded", [0, 1], index=1)
        liab_prct = st.number_input("Liability percentage", 0.0, 100.0, 50.0, 1.0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">02 · Incident details</div>
        <div class="section-subtitle">Describe when, where and how the incident was reported.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        claim_day = st.selectbox(
            "Claim day",
            ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"],
            index=0
        )
        accident_site = st.selectbox(
            "Accident site",
            ["Highway", "Local", "Parking Lot"],
            index=1
        )
        channel = st.selectbox(
            "Claim channel",
            ["Broker", "Online", "Phone"],
            index=1
        )

    with c2:
        police_report = st.selectbox("Police report · encoded", [0, 1], index=1)
        age_of_vehicle = st.number_input("Age of vehicle", 0.0, 50.0, 5.0, 0.5)
        days_open = st.number_input("Days claim remained open", 0.0, 1000.0, 30.0, 1.0)
        form_defects = st.number_input("Form defects", 0.0, 50.0, 0.0, 1.0)

    with c3:
        claim_year = st.number_input("Claim year", 2000, 2100, 2024, 1)
        claim_month = st.number_input("Claim month", 1, 12, 6, 1)
        claim_day_num = st.number_input("Claim day of month", 1, 31, 15, 1)
        weekend_claim = st.selectbox("Weekend claim · encoded", [0, 1], index=0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">03 · Vehicle & financial information</div>
        <div class="section-subtitle">Enter claim amounts and vehicle/policy financial attributes.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        vehicle_color = st.selectbox(
            "Vehicle color",
            ["black", "blue", "gray", "other", "red", "silver", "white"],
            index=5
        )
        vehicle_price = st.number_input("Vehicle price", 0.0, 10000000.0, 500000.0, 5000.0)
        policy_deductible = st.number_input("Policy deductible", 0.0, 1000000.0, 50000.0, 1000.0)

    with c2:
        total_claim = st.number_input("Total claim", 0.0, 10000000.0, 100000.0, 1000.0)
        injury_claim = st.number_input("Injury claim", 0.0, 10000000.0, 25000.0, 1000.0)
        annual_premium = st.number_input("Annual premium", 0.0, 1000000.0, 15000.0, 500.0)

    with c3:
        claim_ratio = st.number_input("Claim ratio", 0.0, 100.0, 20.0, 0.1)
        vehicle_old = st.selectbox("Vehicle old · encoded", [0, 1], index=0)
        claim_vehicle_ratio = st.number_input("Claim / vehicle ratio", 0.0, 100.0, 20.0, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">04 · Income segmentation</div>
        <div class="section-subtitle">Select the annual-income band used by the trained preprocessing.</div>
    """, unsafe_allow_html=True)

    income_group = st.selectbox(
        "Income group",
        [
            "(-1.601, 56899.2]",
            "(56899.2, 60899.2]",
            "(60899.2, 64699.2]",
            "(64699.2, 257313.6]"
        ],
        index=1
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    submitted = st.form_submit_button(
        "🔍  Analyze Insurance Claim",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    try:
        X_input = build_input(
            gender=gender,
            property_status=property_status,
            claim_day=claim_day,
            accident_site=accident_site,
            channel=channel,
            vehicle_color=vehicle_color,
            income_group=income_group,
            vehicle_category=vehicle_category,
            age_of_driver=age_of_driver,
            marital_status=marital_status,
            safety_rating=safety_rating,
            annual_income=annual_income,
            high_education=high_education,
            address_change=address_change,
            past_num_of_claims=past_num_of_claims,
            witness_present=witness_present,
            liab_prct=liab_prct,
            police_report=police_report,
            age_of_vehicle=age_of_vehicle,
            vehicle_price=vehicle_price,
            total_claim=total_claim,
            injury_claim=injury_claim,
            policy_deductible=policy_deductible,
            annual_premium=annual_premium,
            days_open=days_open,
            form_defects=form_defects,
            claim_year=claim_year,
            claim_month=claim_month,
            claim_day_num=claim_day_num,
            claim_ratio=claim_ratio,
            vehicle_old=vehicle_old,
            weekend_claim=weekend_claim,
            claim_vehicle_ratio=claim_vehicle_ratio,
        )

        prediction = int(model.predict(X_input)[0])
        probabilities = model.predict_proba(X_input)[0]
        fraud_probability = float(probabilities[list(model.classes_).index(1)])
        normal_probability = float(probabilities[list(model.classes_).index(0)])

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card risk">
                <div class="result-title">🚨 Fraud Risk Detected</div>
                <div class="result-copy">
                    The trained model classified this insurance claim as <b>potentially fraudulent</b>.
                    This is a model prediction, not a final investigation decision.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card safe">
                <div class="result-title">✅ No Fraud Signal Detected</div>
                <div class="result-copy">
                    The trained model classified this insurance claim as <b>not fraudulent</b>.
                    Continue normal verification procedures.
                </div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Fraud probability", f"{fraud_probability * 100:.2f}%")

        with m2:
            st.metric("Non-fraud probability", f"{normal_probability * 100:.2f}%")

        with m3:
            st.metric("Predicted class", "Fraud" if prediction == 1 else "Not Fraud")

        st.progress(fraud_probability, text=f"Fraud probability · {fraud_probability * 100:.1f}%")

        with st.expander("View model input vector"):
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as exc:
        st.error("Prediction failed. Check the feature values and model compatibility.")
        st.exception(exc)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="small-note">InsureGuard AI · Built with Streamlit · '
    'Prediction powered by the supplied Logistic Regression PKL. '
    'The uploaded notebook reports 74.62% test accuracy.</div>',
    unsafe_allow_html=True
)
