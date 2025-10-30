# Shim classes for unpickling (MUST be BEFORE any imports that use pickle)
class ChurnXGBCWConfig: 
    pass

class ChurnXGBCWPipeline: 
    pass

import streamlit as st
from utils.brand import inject_brand_css, brand_header
from utils.io import get_expected_cols, load_pipeline
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ALPHA Churn Predictor",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject branding
inject_brand_css(dark=True)

# Main page
brand_header("Customer Churn Predictor")

# Load pipeline (cached)
pipeline, model_name, model_path = load_pipeline()

# Initialize session state
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

# Sidebar
with st.sidebar:
    if Path("assets/logo.png").exists():
        st.image("assets/logo.png", width=120)
    else:
        st.markdown("## 🔷 ALPHA")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, step=0.01,
        value=st.session_state.get("threshold", 0.50),
        help="Customers with churn probability above this threshold will be classified as 'Churn'."
    )
    st.session_state.threshold = threshold

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    # Model label (fallback ke nama default kalau loader tidak menyuplai)
    model_label = model_name or "XGBoost (class-weight)"
    st.markdown(f"**Model:** {model_label}")
    st.markdown("**Balancing:** Class-weight (no resampling)")
    st.markdown("**Primary Metric:** F₂-score (β = 2)")
    st.markdown(f"**Threshold:** {threshold:.2f}")

    if model_path:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        st.markdown(f"**Size:** {size_mb:.2f} MB")

st.markdown(f"""
<div class="alpha-card">
  <h3>Welcome to ALPHA Customer Intelligence Suite</h3>
  <p>
    Predict customer churn with gradient-boosted trees. This app uses
    <strong>XGBoost</strong> with <em>class-weight balancing</em> (no resampling)
    and evaluates performance with <strong>F₂-score</strong> as the primary metric.
  </p>
</div>
""", unsafe_allow_html=True)

# CTA Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="alpha-card">
        <h3>🧮 Single Scoring</h3>
        <p>Score individual customers with detailed risk analysis and SHAP explanations.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Single Scoring", key="btn1", use_container_width=True):
        st.switch_page("pages/2_Single_Scoring.py")

with col2:
    st.markdown("""
    <div class="alpha-card">
        <h3>📦 Batch Scoring</h3>
        <p>Upload CSV files to score multiple customers at once with downloadable results.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Batch Scoring", key="btn2", use_container_width=True):
        st.switch_page("pages/3_Batch_Scoring.py")

with col3:
    st.markdown("""
    <div class="alpha-card">
        <h3>🎚️ Threshold Tuning</h3>
        <p>Optimize decision threshold to maximize F₂ or meet business constraints.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Threshold Tuning", key="btn3", use_container_width=True):
        st.switch_page("pages/4_Threshold_Tuning.py")

# Model summary
st.markdown("---")
st.markdown("### 📈 Model Summary")

expected, num_cols, cat_cols = get_expected_cols(pipeline)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Features", len(expected))
col2.metric("Numeric Features", len(num_cols))
col3.metric("Categorical Features", len(cat_cols))
col4.metric("Current Threshold", f"{st.session_state.threshold:.2f}")

# Info metrik
st.info("Primary evaluation metric: **F₂-score (β=2)** – recall-weighted for churn prevention.")

# Info banner
st.info("💡 **Tip:** Use the template CSV download in Batch Scoring to ensure your data matches the required schema.")