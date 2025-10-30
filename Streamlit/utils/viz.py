import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

def gauge(prob: float, threshold: float = 0.5):
    """Create a gauge chart for churn probability"""
    # Determine color based on threshold
    if prob >= threshold:
        color = "#EF4444"  # Red
        label = "HIGH RISK"
    elif prob >= threshold - 0.15:
        color = "#F59E0B"  # Orange
        label = "MEDIUM RISK"
    else:
        color = "#10B981"  # Green
        label = "LOW RISK"
    
    # Create gauge visualization
    fig_html = f"""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 48px; font-weight: 700; color: {color};">
            {prob:.1%}
        </div>
        <div style="font-size: 18px; font-weight: 600; color: {color}; margin-top: 8px;">
            {label}
        </div>
        <div style="width: 100%; height: 8px; background: #333; border-radius: 4px; margin-top: 16px; overflow: hidden;">
            <div style="width: {prob*100}%; height: 100%; background: {color}; transition: width 0.5s;"></div>
        </div>
    </div>
    """
    st.markdown(fig_html, unsafe_allow_html=True)

def hist_probs(probs: np.ndarray, threshold: float = 0.5):
    """Histogram of churn probabilities"""
    df = pd.DataFrame({"Churn Probability": probs})
    
    hist = alt.Chart(df).mark_bar(color="#FFFFFF", opacity=0.7).encode(
        alt.X("Churn Probability:Q", bin=alt.Bin(maxbins=30), title="Churn Probability"),
        alt.Y("count()", title="Count")
    ).properties(
        width=600,
        height=300,
        title="Distribution of Churn Probabilities"
    )
    
    # Add threshold line
    threshold_line = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
        color="#EF4444", strokeWidth=2, strokeDash=[5, 5]
    ).encode(x="threshold:Q")
    
    return hist + threshold_line

def pred_count_bar(preds: np.ndarray):
    """Bar chart of prediction counts"""
    df = pd.DataFrame({
        "Prediction": ["No Churn", "Churn"],
        "Count": [(preds == 0).sum(), (preds == 1).sum()]
    })
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Prediction:N", title=""),
        y=alt.Y("Count:Q", title="Number of Customers"),
        color=alt.Color("Prediction:N", scale=alt.Scale(
            domain=["No Churn", "Churn"],
            range=["#10B981", "#EF4444"]
        ), legend=None)
    ).properties(
        width=300,
        height=250,
        title="Predictions Summary"
    )
    
    return chart

def pr_threshold_chart(thresholds, precision, recall, f2):
    """Precision-Recall-F2 vs Threshold chart"""
    df = pd.DataFrame({
        "Threshold": np.tile(thresholds, 3),
        "Score": np.concatenate([precision, recall, f2]),
        "Metric": ["Precision"]*len(thresholds) + ["Recall"]*len(thresholds) + ["F2"]*len(thresholds)
    })
    
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("Threshold:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
        color=alt.Color("Metric:N", scale=alt.Scale(
            domain=["Precision", "Recall", "F2"],
            range=["#3B82F6", "#10B981", "#F59E0B"]
        ))
    ).properties(
        width=700,
        height=400,
        title="Metrics vs Decision Threshold"
    )
    
    return chart

def confusion_matrix_chart(cm, labels=("No Churn", "Churn")):
    """Confusion matrix heatmap"""
    df = pd.DataFrame({
        "Actual": [labels[0], labels[0], labels[1], labels[1]],
        "Predicted": [labels[0], labels[1], labels[0], labels[1]],
        "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
    })
    
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("Predicted:N", title="Predicted"),
        y=alt.Y("Actual:N", title="Actual"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual", "Predicted", "Count"]
    ).properties(
        width=300,
        height=300,
        title="Confusion Matrix"
    )
    
    text = chart.mark_text(baseline="middle", fontSize=16, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > cm.max() / 2,
            alt.value("white"),
            alt.value("black")
        )
    )
    
    return chart + text

def shap_bar(shap_values, feature_names, top_k=12):
    """Return Altair bar chart of mean(|SHAP|) per feature."""
    vals = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": feature_names, "importance": vals})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_k)

    chart = (
        alt.Chart(df_imp)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Mean |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
        )
        .properties(height=max(240, 24 * len(df_imp)))
    )
    return chart