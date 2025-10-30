import streamlit as st
import pandas as pd
from utils.brand import inject_brand_css, brand_header
from utils.io import load_pipeline, get_expected_cols, align_df, score, prepare_for_shap, fix_xgb_base_score
from utils.viz import gauge
import matplotlib.pyplot as plt
import shap
import numpy as np

st.set_page_config(page_title="Single Scoring", page_icon="🧮", layout="wide")
inject_brand_css(dark=True)
brand_header("Single Customer Scoring")

# Load pipeline + guard
pipeline, model_label, _ = load_pipeline()  
pipeline.named_steps["model"] = fix_xgb_base_score(pipeline.named_steps["model"])
threshold = st.session_state.get("threshold", 0.5)

st.markdown(f"""
Enter customer information below to get an instant churn risk assessment.
""")

# Form with grouped fields
with st.form("single_scoring_form"):
    st.markdown("### 👤 Customer Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=999999, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col4:
        city_tier = st.selectbox("City Tier", [1, 2, 3])
    
    st.markdown("---")
    st.markdown("### 📱 App Usage")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hour_spend = st.number_input("Hours Spend on App", min_value=0.0, max_value=24.0, value=2.5, step=0.1)
    with col2:
        num_devices = st.number_input("Number of Devices", min_value=0, max_value=10, value=2)
    with col3:
        login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
    
    st.markdown("---")
    st.markdown("### 🛒 Order Behavior")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        order_count = st.number_input("Order Count", min_value=0, max_value=1000, value=5)
    with col2:
        days_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=999, value=15)
    with col3:
        order_cat = st.selectbox("Preferred Order Category", 
                                 ["Mobile Phone", "Laptop & Accessory", "Fashion", "Grocery", "Others"])
    with col4:
        coupon_used = st.number_input("Coupons Used", min_value=0, max_value=999, value=1)
    
    st.markdown("---")
    st.markdown("### 💳 Financial")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cashback = st.number_input("Cashback Amount", min_value=0.0, max_value=1000000.0, value=120.0, step=10.0)
    with col2:
        order_hike = st.number_input("Order Amount Hike (%)", min_value=0.0, max_value=100.0, value=12.0, step=1.0) / 100
    with col3:
        payment_mode = st.selectbox("Preferred Payment Mode", 
                                    ["Credit Card", "Debit Card", "UPI", "E Wallet", "Cash on Delivery"])
    with col4:
        warehouse_dist = st.number_input("Warehouse to Home (km)", min_value=0, max_value=999, value=12)
    
    st.markdown("---")
    st.markdown("### ⚠️ Churn Factors")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=6)
    with col2:
        complain = st.selectbox("Complain", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        satisfaction = st.number_input("Satisfaction Score", min_value=1, max_value=5, value=3)
    with col4:
        num_address = st.number_input("Number of Addresses", min_value=0, max_value=10, value=1)
    
    submit = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

if submit:
    # Create dataframe
    data = {
        "CustomerID": customer_id,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "CityTier": city_tier,
        "HourSpendOnApp": hour_spend,
        "NumberOfDeviceRegistered": num_devices,
        "PreferredLoginDevice": login_device,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_last_order,
        "PreferedOrderCat": order_cat,
        "CouponUsed": coupon_used,
        "CashbackAmount": cashback,
        "OrderAmountHikeFromlastYear": order_hike,
        "PreferredPaymentMode": payment_mode,
        "WarehouseToHome": warehouse_dist,
        "Tenure": tenure,
        "Complain": complain,
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": num_address
    }
    
    df = pd.DataFrame([data])
    df_aligned = align_df(df, pipeline)
    
    # Score
    probs, preds = score(df_aligned, pipeline, threshold)
    prob = probs[0]
    pred = preds[0]
    
    # Display results
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gauge(prob, threshold)
    
    with col2:
        st.markdown("### Decision")
        if pred == 1:
            st.markdown('<span class="metric-badge danger">⚠️ CHURN RISK</span>', unsafe_allow_html=True)
            st.error(f"This customer is likely to churn (probability: {prob:.1%})")
        else:
            st.markdown('<span class="metric-badge success">✅ RETAINED</span>', unsafe_allow_html=True)
            st.success(f"This customer is likely to stay (churn probability: {prob:.1%})")
        
        st.markdown("---")
        st.metric("Churn Probability", f"{prob * 100:.1f}%", f"{prob:.3f}")
        st.metric("Decision Threshold", f"{threshold * 100:.1f}%")
    
    # SHAP explanation (optional)
    try:        
        st.markdown("---")
        st.markdown("### 🔎 Feature Importance (SHAP)")
            
        # siapkan data ter-transform + feature names + model
        X_proc, feature_names, model = prepare_for_shap(pipeline, df_aligned)

        # ambil satu baris yang sedang di-score (pastikan X_proc shape (1, n_features))
        X1 = X_proc[:1]

        # Build explainer (new API -> fallback old) 
        explanation = None
        base_value = None
        shap_values = None  # numpy array for bar chart

        try:
            explainer = shap.Explainer(model)
            explanation = explainer(X1)              # Explanation object
            base_value = explanation.base_values[0] if hasattr(explanation, "base_values") else None
            shap_values = explanation.values         # (1, M)
        except Exception:
            # Fallback ke TreeExplainer (versi lama)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X1)  # (1, M)
            base_value = explainer.expected_value if hasattr(explainer, "expected_value") else None
            # Bungkus manual jadi Explanation untuk waterfall
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=base_value if np.ndim(base_value)==0 else np.array(base_value).mean(),
                data=X1[0],
                feature_names=feature_names,
            )

        # Top-k bar chart (global-ish view for this record) 
        from utils.viz import shap_bar
        st.altair_chart(shap_bar(shap_values, feature_names, top_k=12), use_container_width=True)

        # Waterfall (local explanation for this record) 
        st.markdown("#### 📻 SHAP Waterfall (this customer)")

        # 🎨 Dark style patch for SHAP plot
        plt.style.use("default")
        plt.rcParams.update({
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "axes.edgecolor": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "text.color": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
            "savefig.facecolor": "#0e1117",
        })

        # Ringkas nama fitur lebih dulu supaya tidak perlu plot 2x
        short_names = [n if len(n) <= 20 else n[:17] + "..." for n in feature_names]
        try:
            # kalau Explanation object, update feature_names-nya
            explanation.feature_names = short_names
        except Exception:
            # jaga-jaga tipe lain; silent pass
            pass

        # 🧩 Waterfall plot (single render)
        fig = plt.figure(figsize=(8.5, 6))
        shap.plots.waterfall(
            explanation[0] if isinstance(explanation, shap._explanation.Explanation) else explanation,
            max_display=12,
            show=False
        )
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.info(f"SHAP analysis unavailable: {e}")