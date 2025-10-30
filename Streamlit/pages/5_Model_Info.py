import streamlit as st
import pandas as pd
import sys
import graphviz as gv
from pathlib import Path
from utils.brand import inject_brand_css, brand_header
from utils.io import load_pipeline, get_expected_cols

st.set_page_config(page_title="Model Info", page_icon="🏗️", layout="wide")
inject_brand_css(dark=True)
brand_header("Pipeline Architecture")

# Load pipeline
pipeline, model_label, model_path = load_pipeline()

st.markdown(f"""
The model uses an end-to-end pipeline for preprocessing and prediction.
Balancing is handled via **class-weight**. Primary evaluation metric: **F₂-score (β=2)**.
""")


# graphviz diagram 
dot = gv.Digraph("pipeline", format="svg")
dot.attr(rankdir="LR", bgcolor="#111111")       # Layout: TB/LR
dot.attr("edge", color="#FFFFFF")

node_style = {
    "shape": "box",
    "style": "rounded,filled",
    "color": "#FFFFFF",
    "fontcolor": "#FFFFFF"
}

# Nodes 
dot.node("A", "Raw Data", fillcolor="#111111", **node_style)

# ColumnTransformer & its branches
dot.node("B", "ColumnTransformer", fillcolor="#111111", **node_style)
dot.node("B1", "Numerical Features", fillcolor="#111111", **node_style)
dot.node("B2", "Categorical Features", fillcolor="#111111", **node_style)

# Numeric sub-steps (vertical chain under Numeric Branch)
dot.node("C1", "IterativeImputer", fillcolor="#111111", **node_style)
dot.node("C2", "RobustScaler", fillcolor="#111111", **node_style)

# Category sub-step
dot.node("C3", "OneHotEncoder", fillcolor="#111111", **node_style)

# Remaining pipeline
dot.node("E", f"{model_label}", fillcolor="#3B82F6", **node_style)
dot.node("F", "Predictions", fillcolor="#3B82F6", **node_style)

# Edges 
dot.edge("A", "B")
dot.edge("B", "B1")
dot.edge("B", "B2")

# numeric chain
dot.edge("B1", "C1")
dot.edge("C1", "C2")
dot.edge("C2", "E")

# categorical chain
dot.edge("B2", "C3")
dot.edge("C3", "E")

# remainder
dot.edge("E", "F")

st.graphviz_chart(dot, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Preprocessing Steps
    1. **ColumnTransformer**: Separates numeric and categorical features
    2. **IterativeImputer**: Fills missing values using iterative approach
    3. **RobustScaler**: Scales numeric features (robust to outliers)
    4. **OneHotEncoder**: Encodes categorical variables
    """)

with col2:
    st.markdown(f"""
    ### Model Training
    5. **XGBoost (class-weight)**: Gradient-boosted trees with **scale_pos_weight** to handle imbalance.
    6. **Probability output**: Model returns churn probability for thresholding.
    """)

# Hyperparameters
st.markdown("---")
st.markdown("## ⚙️ Model Hyperparameters")

# Best XGBoost hyperparameters
best_params_xgb = {
    'model__subsample': 0.9,
    'model__n_estimators': 150,
    'model__min_child_weight': 3,
    'model__max_depth': 9,
    'model__learning_rate': 0.2,
    'model__colsample_bytree': 0.8
}

try:
    model = pipeline.named_steps.get("model", None)
    if model is not None:
        params = model.get_params()
        important_params = {
            "n_estimators":       "Number of trees",
            "max_depth":          "Maximum tree depth",
            "learning_rate":      "Learning rate",
            "min_child_weight":   "Min sum of instance weight in a child",
            "subsample":          "Row sampling ratio",
            "colsample_bytree":   "Column sampling ratio",
            "reg_alpha":          "L1 regularization",
            "reg_lambda":         "L2 regularization",
            "scale_pos_weight":   "Class-weight ratio (neg/pos)",
        }

        param_data = []
        for param, description in important_params.items():
            if param in params:
                # Check if we have the best value from the XGBoost model
                param_value = best_params_xgb.get(f"model__{param}", str(params[param]))
                param_data.append({
                    "Parameter": param,
                    "Value": param_value,
                    "Description": description
                })
        
        if param_data:
            df_params = pd.DataFrame(param_data)
            st.dataframe(df_params, use_container_width=True, hide_index=True)
        else:
            st.info("Hyperparameters not available in standard format.")
    else:
        st.warning("Could not extract model from pipeline.")
        
except Exception as e:
    st.error(f"Error extracting hyperparameters: {e}")

# Expected Features
st.markdown("---")
st.markdown("## 📋 Expected Features")

expected, num_cols, cat_cols = get_expected_cols(pipeline)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🔢 Numeric Features ({len(num_cols)})")
    if num_cols:
        num_df = pd.DataFrame({"Feature": num_cols})
        st.dataframe(num_df, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No numeric features found.")

with col2:
    st.markdown(f"### 🏷️ Categorical Features ({len(cat_cols)})")
    if cat_cols:
        cat_df = pd.DataFrame({"Feature": cat_cols})
        st.dataframe(cat_df, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No categorical features found.")

# Feature descriptions
st.markdown("---")
st.markdown("## 📖 Feature Descriptions")

feature_descriptions = {
    "CustomerID": "Unique identifier for each customer",
    "Tenure": "Number of months the customer has been with the company",
    "Complain": "Whether customer has complained (1=Yes, 0=No)",
    "OrderCount": "Total number of orders has been places in last month",
    "DaySinceLastOrder": "Number of days since the last order",
    "CashbackAmount": "Average cashback in last month",
    "HourSpendOnApp": "Average hours spent on the mobile app daily",
    "OrderAmountHikeFromlastYear": "Percentage increase in order amount from last year",
    "WarehouseToHome": "Distance from warehouse to customer's home (km)",
    "CouponUsed": "Total number of coupon has been used in last month",
    "NumberOfDeviceRegistered": "Number of devices registered to the account",
    "NumberOfAddress": "Number of addresses registered",
    "SatisfactionScore": "Customer satisfaction score (1-5)",
    "CityTier": "City tier classification (1=Metro, 3=Urban)",
    "Gender": "Customer gender",
    "MaritalStatus": "Marital status of the customer",
    "PreferredLoginDevice": "Preferred device for logging in",
    "PreferredPaymentMode": "Preferred payment method",
    "PreferedOrderCat": "Preferred order category",
}

desc_data = []
for feature in expected:
    desc_data.append({
        "Feature": feature,
        "Type": "Numeric" if feature in num_cols else "Categorical",
        "Description": feature_descriptions.get(feature, "No description available")
    })

df_desc = pd.DataFrame(desc_data)
st.dataframe(df_desc, use_container_width=True, hide_index=True, height=400)

# System Information
st.markdown("---")
st.markdown("## 💻 System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📦 Model File")
    if model_path:
        st.markdown(f"**Path:** `{model_path}`")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        st.markdown(f"**Size:** {size_mb:.2f} MB")

with col2:
    st.markdown("### 🐍 Python Environment")
    st.markdown(f"**Python:** {sys.version.split()[0]}")
    
    try:
        import sklearn
        st.markdown(f"**scikit-learn:** {sklearn.__version__}")
    except:
        pass
    
    try:
        import imblearn
        st.markdown(f"**imbalanced-learn:** {imblearn.__version__}")
    except:
        pass
    
    try:
        import xgboost
        st.markdown(f"**XGBoost:** {xgboost.__version__}")
    except:
        pass

with col3:
    st.markdown("### 📊 Pipeline Steps")
    steps = list(pipeline.named_steps.keys())
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. `{step}`")

# Performance Characteristics
st.markdown("---")
st.markdown("## ⚡ Performance Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features", len(expected))
    st.caption("After one-hot encoding, this expands to more features")

with col2:
    st.metric("Model Complexity", f"{model.get_params().get('n_estimators', 'N/A')} trees" if model else "N/A")
    st.caption("More trees = better accuracy but slower inference")

with col3:
    st.metric("Prediction Speed", "< 100ms")
    st.caption("Typical single prediction latency")

# Best Practices
st.markdown("---")
st.markdown("## 🎯 Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Do's
    - Use the template CSV to ensure schema compatibility
    - Validate data types before uploading
    - Handle missing values gracefully (pipeline handles them)
    - Tune threshold based on business costs
    - Monitor model performance over time
    - Retrain periodically with new data
    """)

with col2:
    st.markdown("""
    ### ❌ Don'ts
    - Don't change column names arbitrarily
    - Don't remove expected features
    - Don't use categorical values outside training set
    - Don't ignore extreme probability values (0.99+)
    - Don't deploy without threshold tuning
    - Don't forget to validate on holdout data
    """)

# Model Limitations
st.markdown("---")
st.markdown("## ⚠️ Model Limitations & Considerations")

st.markdown("""
- **Label Definition**: Churn is inferred from inactivity without a strict business rule, which can cause **label bias or misclassification** (e.g., active users misidentified as churners).
- **Metric Choice (F₂)**: Model is optimized for **F₂-Score** (recall-heavy). If future business priorities emphasize **precision** or cost-based targeting, threshold recalibration is required.
- **Seasonality**: The model does not account for **time-based or promotional seasonality**, so performance may vary during campaigns or holiday periods.
- **Monetary Features**: Missing financial indicators such as **CLV, AOV, or total spend** limit the model’s ability to prioritize high-value customers in retention strategies.
- **Data Drift**: Model performance may degrade if **customer behavior or churn patterns shift** significantly; regular monitoring and retraining are recommended.
- **Class Imbalance**: Addressed using **class-weight (scale_pos_weight)** during training; monitor for changes in churn ratio over time.
- **Feature Availability**: All required features must be present during inference; missing data is imputed but may affect prediction accuracy.
- **Threshold Sensitivity**: Deployed model uses a **CV-derived threshold**; recalibration should be done periodically as business costs or data distributions evolve.
- **Interpretability**: Tree-based models are generally interpretable but can still exhibit **non-linear and interacting effects** that are not easily observable.
- **Ethical Considerations**: Regularly audit predictions to ensure **no bias or unfair treatment** against protected demographic groups.
""")

# Footer
st.markdown("---")
st.caption(f"""
**ALPHA Customer Intelligence Suite** · Model: {model_label}.
""")