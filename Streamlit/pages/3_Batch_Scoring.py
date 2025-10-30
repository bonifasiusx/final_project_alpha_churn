import streamlit as st
import pandas as pd
import numpy as np
from utils.brand import inject_brand_css, brand_header
from utils.io import load_pipeline, score, template_csv
from utils.viz import hist_probs, pred_count_bar

st.set_page_config(page_title="Batch Scoring", page_icon="📦", layout="wide")
inject_brand_css(dark=True)
brand_header("Batch Customer Scoring")

# Load pipeline
pipeline, model_label, _ = load_pipeline()
threshold = st.session_state.get("threshold", 0.5)

st.markdown(f"""
Upload a CSV file with customer data to score multiple customers at once.
""")

# Template download
col1, col2 = st.columns([3, 1])
with col1:
    st.info("💡 **Tip:** Download the template CSV below to ensure your data matches the required format.")
with col2:
    template_df = template_csv(pipeline, n_rows=3)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download Template",
        data=csv_template,
        file_name="churn_template.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file with customer data. Max size: 200MB"
)

def _normalize_percent_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Jika ada nilai > 1 di kolom persen, anggap unitnya % lalu bagi 100."""
    if col in df.columns:
        try:
            vals = pd.to_numeric(df[col], errors="coerce")
            if np.nanmax(vals) > 1.0:
                df[col] = vals / 100.0
        except Exception:
            pass
    return df

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Auto-normalisasi kolom persen agar konsisten dengan training/Single Scoring
        df = _normalize_percent_column(df, "OrderAmountHikeFromlastYear")

        st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")

        # Preview
        with st.expander("📄 Preview Data (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        # Align & score (io.score sudah melakukan align internal)
        with st.spinner("🔄 Aligning schema and scoring..."):
            probs, preds = score(df, pipeline, threshold)

        # Results dataframe
        result_df = df.copy()
        result_df["churn_prob"] = probs
        result_df["churn_pred"] = preds

        # Risk banding proporsional pada threshold (low ≤ 0.8*thr < medium ≤ thr < high)
        low_cut = max(0.0, 0.8 * threshold)
        result_df["churn_risk"] = pd.cut(
            probs,
            bins=[0.0, low_cut, threshold, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
            right=True
        )

        st.markdown("---")
        st.markdown("## 📊 Scoring Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Predicted Churn", f"{int((preds == 1).sum())} ({(preds == 1).mean()*100:.1f}%)")
        col3.metric("Mean Churn Probability", f"{probs.mean():.1%}")
        col4.metric(f"High Risk (>{threshold:.0%})", int((probs > threshold).sum()))

        # Visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Distribution of Churn Probabilities")
            st.altair_chart(hist_probs(probs, threshold), use_container_width=True)

        with col2:
            st.markdown("### Prediction Summary")
            st.altair_chart(pred_count_bar(preds), use_container_width=True)

        # Results table + filters
        st.markdown("---")
        st.markdown("### 📋 Detailed Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            show_filter = st.selectbox("Filter by Prediction", ["All", "Churn Only", "Retained Only"])
        with col2:
            risk_filter = st.multiselect("Filter by Risk Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        with col3:
            sort_by = st.selectbox("Sort by", ["churn_prob (High to Low)", "churn_prob (Low to High)", "Original Order"])

        filtered_df = result_df.copy()
        if show_filter == "Churn Only":
            filtered_df = filtered_df[filtered_df["churn_pred"] == 1]
        elif show_filter == "Retained Only":
            filtered_df = filtered_df[filtered_df["churn_pred"] == 0]
        if risk_filter:
            filtered_df = filtered_df[filtered_df["churn_risk"].isin(risk_filter)]

        if sort_by == "churn_prob (High to Low)":
            filtered_df = filtered_df.sort_values("churn_prob", ascending=False)
        elif sort_by == "churn_prob (Low to High)":
            filtered_df = filtered_df.sort_values("churn_prob", ascending=True)

        st.dataframe(
            filtered_df.style.background_gradient(subset=["churn_prob"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=400
        )

        # Download results
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 💾 Export Results")
            st.markdown(f"Download the scored dataset with {len(filtered_df)} rows (filtered) or {len(result_df)} rows (all).")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download All Results",
                data=result_df.to_csv(index=False),
                file_name="churn_scored_all.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download Filtered Results",
                data=filtered_df.to_csv(index=False),
                file_name="churn_scored_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV matches the template format.")
else:
    st.info("👆 Upload a CSV file to begin batch scoring.")