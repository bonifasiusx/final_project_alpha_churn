# Shim classes for unpickling (ensure available before pickle.load)
class ChurnXGBCWConfig:
    pass

class ChurnXGBCWPipeline:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import pickle, sys, importlib
from pathlib import Path
from typing import Tuple, List

# Column alias mapping for typos
ALIAS_MAP = {
    "PreferredOrderCat": "PreferedOrderCat",
    "OrderAmountHikeFromLastYear": "OrderAmountHikeFromlastYear",
    "City Tier": "CityTier",
    "Gender ": "Gender",
    "Marital Status": "MaritalStatus",
}

def _ensure_pickle_compat():
    main = sys.modules.get("__main__")
    if main is None:
        return
    if not hasattr(main, "ChurnXGBCWConfig"):
        class ChurnXGBCWConfig: ...
        setattr(main, "ChurnXGBCWConfig", ChurnXGBCWConfig)
    if not hasattr(main, "ChurnXGBCWPipeline"):
        class ChurnXGBCWPipeline: ...
        setattr(main, "ChurnXGBCWPipeline", ChurnXGBCWPipeline)

# compatibility patch for sklearn pickles across versions
try:
    ct_mod = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compat shim so old pickles with _RemainderColsList can be loaded."""
            pass
        ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    pass

def fix_xgb_base_score(xgb_model):
    """
    Fix XGBoost base_score that might be stored as string '[5E-1]'.
    This is a common issue when loading older XGBoost models.
    """
    try:
        if hasattr(xgb_model, "get_booster"):
            booster = xgb_model.get_booster()
            
            # Try to get base_score from attributes
            base_score = booster.attributes().get("base_score", None)
            
            if base_score and isinstance(base_score, str):
                # Remove brackets and convert: '[5E-1]' -> 0.5
                base_score_clean = base_score.strip('[]').strip()
                base_score_float = float(base_score_clean)
                
                # Set the corrected base_score
                booster.set_param({"base_score": str(base_score_float)})
                print(f"✅ Fixed XGBoost base_score: {base_score} -> {base_score_float}")
                
    except Exception as e:
        # Silent fail - model might still work
        print(f"⚠️ Could not fix base_score (model may still work): {e}")
    
    return xgb_model

@st.cache_resource
def load_pipeline(path: str = "artifacts/churn_xgb_cw.sav"):
    """
    Load final pipeline (XGBoost + class-weight, no resampling).
    Returns: (pipeline, model_label, path_obj)
    """
    base_dir = Path(__file__).resolve().parent.parent  # -> .../Streamlit
    path_obj = (base_dir / path).resolve()

    if not path_obj.exists():
        st.error(f"❌ Model file not found: {path_obj}")
        st.stop()

    _ensure_pickle_compat()
    with open(path_obj, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "pipe" in obj:
        pipe = obj["pipe"]
        cfg  = obj.get("cfg", None)
    else:
        pipe = obj
        cfg  = None

    # 🔧 Fix XGBoost base_score issue before any predictions
    if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
        pipe.named_steps["model"] = fix_xgb_base_score(pipe.named_steps["model"])

    # label
    model_label = "XGBoost (Class-weight • Balanced)"
    
    # 🎯 Set threshold from config (always override to ensure consistency)
    if cfg and hasattr(cfg, "threshold"):
        threshold_value = float(cfg.threshold)
        st.session_state["threshold"] = threshold_value
        st.success(f"✅ Model loaded: {model_label} (threshold: {threshold_value:.2f})")
    else:
        # Fallback to default only if not already set
        if "threshold" not in st.session_state:
            st.session_state["threshold"] = 0.50
        st.success(f"✅ Model loaded: {model_label}")

    return pipe, model_label, path_obj

def get_expected_cols(pipeline) -> Tuple[List[str], List[str], List[str]]:
    """Extract expected raw input columns from ColumnTransformer."""
    try:
        transformer = pipeline.named_steps["transformer"]
        num_cols, cat_cols = [], []
        for name, trans, cols in transformer.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
        expected = num_cols + cat_cols
        return expected, num_cols, cat_cols
    except Exception as e:
        st.error(f"Cannot extract columns from pipeline: {e}")
        return [], [], []

def align_df(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Align incoming dataframe to match training schema (typo fix, add-missing, reorder)."""
    expected, num_cols, cat_cols = get_expected_cols(pipeline)
    df = df.copy()

    # 1) alias mapping (perbaiki ejaan supaya cocok dengan training)
    df.rename(columns={c: ALIAS_MAP.get(c, c) for c in df.columns}, inplace=True)

    # 2) tambahkan kolom yang hilang
    for col in expected:
        if col not in df.columns:
            df[col] = (np.nan if col in num_cols else "Unknown")

    # 3) urutkan kolom sesuai training
    return df[expected]

def prepare_for_shap(pipeline, df_aligned):
    """
    Return (X_proc, feature_names, model) for SHAP.
    """
    pre = pipeline.named_steps["transformer"]
    X_proc = pre.transform(df_aligned)

    try:
        raw_names = pre.get_feature_names_out()
    except Exception:
        raw_names = [f"f{i}" for i in range(X_proc.shape[1])]

    cleaned = []
    for n in raw_names:
        n = n.replace("num__", "").replace("cat__", "")
        if "|" in n: n = n.split("|")[-1]
        if "_" in n and len(n) > 25: n = n.split("_")[-1]
        cleaned.append(n)

    model = pipeline.named_steps["model"]
    return X_proc, cleaned, model

def score(df: pd.DataFrame, pipeline, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Align -> predict_proba -> thresholding."""
    try:
        df_aligned = align_df(df, pipeline)
        probs = pipeline.predict_proba(df_aligned)[:, 1]
        preds = (probs >= threshold).astype(int)
        return probs, preds
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.array([]), np.array([])

def template_csv(pipeline, n_rows: int = 3) -> pd.DataFrame:
    """Generate template CSV with realistic dummy rows."""
    expected, num_cols, cat_cols = get_expected_cols(pipeline)
    templates = [
        # JELAS CHURN
        {
            "CustomerID": 999101,
            "Tenure": 0,
            "Complain": 1,
            "OrderCount": 5,
            "DaySinceLastOrder": 18,
            "CashbackAmount": 230.0,
            "HourSpendOnApp": 4.7,
            "PreferredLoginDevice": "Computer",
            "PreferredPaymentMode": "UPI",
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Single",
            "CityTier": 3,
            "WarehouseToHome": 50,
            "CouponUsed": 4,
            "SatisfactionScore": 4,
            "NumberOfDeviceRegistered": 2,
            "NumberOfAddress": 2,
            "Gender": "Male",
            "OrderAmountHikeFromlastYear": 0.30
        },

        # JELAS LOYAL
        {
            "CustomerID": 999102,
            "Tenure": 36,
            "Complain": 0,
            "OrderCount": 60,
            "DaySinceLastOrder": 0,
            "CashbackAmount": 800.0,
            "HourSpendOnApp": 9.0,
            "PreferredLoginDevice": "Computer",
            "PreferredPaymentMode": "Credit Card",
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Married",
            "CityTier": 1,
            "WarehouseToHome": 1,
            "CouponUsed": 6,
            "SatisfactionScore": 5,
            "NumberOfDeviceRegistered": 4,
            "NumberOfAddress": 3,
            "Gender": "Female",
            "OrderAmountHikeFromlastYear": 0.40
        },

        # 50 : 50 (borderline)
        {
            "CustomerID": 999103,
            "Tenure": 3,
            "Complain": 1,
            "OrderCount": 4,
            "DaySinceLastOrder": 150,
            "CashbackAmount": 50.0,
            "HourSpendOnApp": 1.5,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "Cash on Delivery",
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Single",
            "CityTier": 3,
            "WarehouseToHome": 120,
            "CouponUsed": 2,
            "SatisfactionScore": 2,
            "NumberOfDeviceRegistered": 2,
            "NumberOfAddress": 1,
            "Gender": "Male",
            "OrderAmountHikeFromlastYear": 0.15
        }
    ]
    return pd.DataFrame(templates[:n_rows])