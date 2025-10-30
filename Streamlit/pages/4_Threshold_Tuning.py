import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from utils.brand import inject_brand_css, brand_header
from utils.io import load_pipeline, align_df, persist_threshold
from utils.viz import pr_threshold_chart, confusion_matrix_chart

st.set_page_config(page_title="Threshold Tuning", page_icon="🎚️", layout="wide")
inject_brand_css(dark=True)
brand_header("Decision Threshold Optimization")

# Load pipeline
pipeline, model_label, _ = load_pipeline()
current_threshold = st.session_state.get("threshold", 0.5)

st.markdown(f"""
Optimize the decision threshold based on your business requirements.
Upload a labeled validation dataset to see how thresholds affect your results.
""")

# File upload
st.markdown("### 📤 Upload Labeled Data")
st.info("💡 Your CSV must include a 'Churn' column with actual labels (0 or 1).")

uploaded_file = st.file_uploader(
    "Choose a CSV file with labels",
    type=["csv"],
    help="Upload validation/test data with actual churn labels"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for Churn column
        if "Churn" not in df.columns:
            st.error("❌ The CSV must contain a 'Churn' column with actual labels!")
            st.stop()
        
        y_true = df["Churn"].values
        
        # Remove Churn from features and align
        df_features = df.drop(columns=["Churn"])
        df_aligned = align_df(df_features, pipeline)
        
        # Get probabilities
        y_probs = pipeline.predict_proba(df_aligned)[:, 1]
        
        st.success(f"✅ Loaded {len(df)} samples with labels.")
        
        # Display class distribution
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(y_true))
        col2.metric("Churn (Class 1)", (y_true == 1).sum())
        col3.metric("No Churn (Class 0)", (y_true == 0).sum())
        
        st.markdown("---")
        
        # Compute metrics for different thresholds
        thresholds = np.arange(0.10, 0.91, 0.01)
        precision_scores, recall_scores, f2_scores = [], [], []

        for thr in thresholds:
            y_pred = (y_probs >= thr).astype(int)

            precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
            f2_scores.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        # Plot
        st.markdown("### 📈 Metrics vs Threshold")
        st.altair_chart(
            pr_threshold_chart(thresholds, precision_scores, recall_scores, f2_scores),
            use_container_width=True
        )
                
        # Interactive threshold selector
        st.markdown("---")
        st.markdown("### 🎚️ Threshold Selector")

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_threshold = st.slider(
                "Select Threshold",
                min_value=0.10, max_value=0.90, step=0.01,
                value=st.session_state["threshold"]
            )
        with col2:
            if st.button("✅ Set as Default", use_container_width=True):
                st.session_state.threshold = selected_threshold
                persist_threshold(selected_threshold)  # simpan ke artifacts
                st.success(f"Threshold updated to {selected_threshold:.2f}")
                st.rerun()
                
        # Show metrics for selected threshold
        y_pred_selected = (y_probs >= selected_threshold).astype(int)

        precision_sel = precision_score(y_true, y_pred_selected, zero_division=0)
        recall_sel    = recall_score(y_true, y_pred_selected, zero_division=0)
        f2_sel        = fbeta_score(y_true, y_pred_selected, beta=2, zero_division=0)
        
        st.markdown(f"### 📊 Metrics at Threshold = {selected_threshold:.2f}")

        col1, col2, col3 = st.columns(3)  # ← 3 kolom (tanpa F1)
        col1.metric("F₂ Score (β=2)", f"{f2_sel:.3f}")
        col2.metric("Precision", f"{precision_sel:.3f}")
        col3.metric("Recall", f"{recall_sel:.3f}")
        
        # Confusion matrix
        st.markdown("---")
        st.markdown("### 🔢 Confusion Matrix")
        
        cm = confusion_matrix(y_true, y_pred_selected)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.altair_chart(
                confusion_matrix_chart(cm),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Interpretation")
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown(f"- **True Negatives (TN):** {tn} - Correctly predicted no churn")
            st.markdown(f"- **False Positives (FP):** {fp} - Incorrectly predicted churn")
            st.markdown(f"- **False Negatives (FN):** {fn} - Missed churn (⚠️ costly!)")
            st.markdown(f"- **True Positives (TP):** {tp} - Correctly predicted churn")
            
            st.markdown("---")
            
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            st.metric("Accuracy", f"{accuracy:.3f}")
            
            if (tp + fn) > 0:
                actual_churn_rate = (tp + fn) / (tn + fp + fn + tp)
                st.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if precision_sel < 0.5:
            st.warning("⚠️ **Low Precision**: Many false alarms. Consider increasing threshold to reduce false positives.")
        
        if recall_sel < 0.5:
            st.warning("⚠️ **Low Recall**: Missing many churners. Consider decreasing threshold to catch more at-risk customers.")
        
        if f2_sel > 0.7:
            st.success("✅ **Good F2 Score**: Balanced performance with emphasis on recall (catching churners).")
        
        # Find optimal thresholds
        st.markdown("---")
        st.markdown("### 🎯 Suggested Thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_f2_idx = int(np.nanargmax(f2_scores))
            best_f2_threshold = float(np.round(thresholds[best_f2_idx], 2))
            
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                st.session_state["last_uploaded_name"] = uploaded_file.name
                st.session_state["threshold"] = best_f2_threshold
    
            st.markdown("**Best F2 Score**")
            st.metric("Threshold", f"{best_f2_threshold:.2f}")
            st.metric("F2 Score", f"{f2_scores[best_f2_idx]:.3f}")
            
            st.caption(f"Applied default: Best F2 ({best_f2_threshold:.2f}) for this dataset")
        
        with col2:
            # Find threshold where precision ≈ recall (balanced)
            balance_idx = np.argmin(np.abs(np.array(precision_scores) - np.array(recall_scores)))
            balanced_threshold = thresholds[balance_idx]
            st.markdown("**Balanced (P≈R)**")
            st.metric("Threshold", f"{balanced_threshold:.2f}")
            st.metric("Precision", f"{precision_scores[balance_idx]:.3f}")
            st.metric("Recall", f"{recall_scores[balance_idx]:.3f}")
        
        with col3:
            # High recall threshold (catch most churners)
            high_recall_idx = np.where(np.array(recall_scores) >= 0.8)[0]
            if len(high_recall_idx) > 0:
                high_recall_threshold = thresholds[high_recall_idx[0]]
                st.markdown("**High Recall (≥80%)**")
                st.metric("Threshold", f"{high_recall_threshold:.2f}")
                st.metric("Recall", f"{recall_scores[high_recall_idx[0]]:.3f}")
            else:
                st.markdown("**High Recall Target**")
                st.info("No threshold achieves ≥80% recall")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload a labeled CSV file to begin threshold optimization.")
    
    st.markdown("---")
    st.markdown("### 📖 How It Works")
    
    st.markdown("""
    **Threshold tuning** helps you find the optimal decision boundary for your business context:
    
    - **Precision**: Of customers we predict will churn, what % actually churn?
      - *Higher precision = fewer false alarms, but may miss some churners*
    
    - **Recall**: Of customers who actually churn, what % do we catch?
      - *Higher recall = catch more churners, but more false alarms*
    
    - **F2 Score**: Weighted average favoring recall (2x more important than precision)
      - *Good for churn where missing a churner is costly*
    
    **Trade-offs:**
    - **Lower threshold (e.g., 0.3)** → High recall, low precision (cast wide net)
    - **Higher threshold (e.g., 0.7)** → High precision, low recall (be selective)
    - **Balanced (e.g., 0.5)** → Moderate precision and recall
    """)