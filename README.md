

# 🏢 Alpha Company — Customer Churn Prediction

**👨‍💻 Authors:** [Alfriando C. Vean](https://github.com/alfcvean) · [Ardinata Jeremy Kingstone Tambun](https://github.com/ardinatatambun) · [Bonifasius Sinurat](https://github.com/bonifasiusx)

📅 *Purwadhika Final Project — JCDS-3004 Cohort*

---

## 🎯 1. Business Objective

Alpha Company is a mid-scale e-commerce facing a **critical churn problem** — customers stop transacting or move to competitors.

This project aims to **predict and prevent churn** by:

* 🔍 Identifying **high-risk customers** before they leave
* 🎯 Enabling **targeted & cost-efficient retention** campaigns
* 💰 Delivering **measurable financial impact** and ROI improvements

---

## 📊 2. Data Overview

* **Source:** `E Commerce Dataset.xlsx`
* **Target Variable:** `Churn` (binary: 1 = churned, 0 = active)
* **Sample Size:** 4,656 customers (after cleaning & imputations)
* **Key Features:**
  * `Tenure`, `Complain`, `DaySinceLastOrder`, `PreferredPaymentMode`,

    `PreferredLoginDevice`, `CityTier`, `SatisfactionScore`, etc.
* **Note:** Data is anonymized for analytics and modeling.

---

## ⚙️ 3. Methodology

| Step                     | Description                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Preprocessing**  | Missing-value imputation (IterativeImputer) + optional scaling (RobustScaler) + One-Hot Encoding |
| **Modeling**       | Gradient boosting with**XGBoost (class-weight balanced)**                                  |
| **Validation**     | 5-Fold CV with hyperparameter tuning using**F2-score**as main metric                       |
| **Explainability** | SHAP (global, local, waterfall) for interpretability                                             |
| **Business Layer** | ROI simulation using CAC–CRC economics                                                          |

### 🧠 Pipeline Overview

![Pipeline Overview](images/pipeline_overview.png)

---

## 📈 4. Model Performance

| **Metric**              | **Cross-Validation** | **Test Set** |
| ----------------------------- | -------------------------: | -----------------: |
| 🧮**F₂-Score**         |                     0.8894 |   **0.9759** |
| 🎯**Recall (Churn)**    |                      0.986 |    **0.989** |
| 🎯**Precision (Churn)** |                      0.963 |    **0.964** |
| 📊**AUC-PR**            |                      0.993 |    **0.993** |

**Confusion Matrix (Test Set)**

> TN=929 FP=7 FN=4 TP=186

![Confusion Matrix](images/confusion_matrix.png)

---

## 🧩 5. Explainability — SHAP & Feature Importance

### 🔝 Key Drivers of Churn

1. 🕒 **Tenure** — shorter tenure strongly increases churn likelihood
2. 😠 **Complain** — complaint history ≈ 2–3× higher churn odds
3. 📆 **DaySinceLastOrder** — longer inactivity = higher churn risk
4. 💳 **PreferredPaymentMode** — COD users churn more than e-wallet users
5. 📱 **PreferredLoginDevice** — mobile app users are more loyal

### 🔍 SHAP Global Summary

![SHAP Summary](images/shap_summary.png)

### 💡 Feature Importances (Model Perspective)

![Feature Importances](images/feature_importance_bar.png)

### 📊 Example — Local SHAP Waterfall (Churn Case)

![SHAP Waterfall](images/shap_waterfall.png)

---

## 💵 6. Business Impact & ROI

### Assumptions

| Parameter                     | Value ($) | Description                                |
| ----------------------------- | --------: | ------------------------------------------ |
| **CAC**                 |        80 | Cost to acquire new customer               |
| **CRC**                 |        20 | Cost to retain one customer                |
| **Net Retention Value** |        60 | Savings per successfully retained customer |

### Impact Summary

| Component                      |        Value ($) | Notes                     |
| ------------------------------ | ---------------: | ------------------------- |
| 💰**Savings (TP)**       |           11,160 | 186 × (80 − 20)         |
| 💸**Cost (FP)**          |              140 | 7 × 20                   |
| 😓**Loss (FN)**          |              320 | 4 × 80                   |
| 🧾**Net Impact**         | **10,700** | 11,160 − (140 + 320)     |
| 📈**ROI (baseline)**     | **78.7×** | (11,160 − 140) / 140     |
| 🔁**ROI (↓ churn 5pp)** | **55.6×** | If churn drops 17% → 12% |

✅ **Result:** precision retention strategy — focus on  *true churners* , reduce waste in retention budget.

---

## 🚀 7. Deployment & Operations

**Deployment Options:**

* 🧭 **Streamlit App** — Interactive scoring, SHAP-based explanations
* ⚙️ **REST API** — For integration with CRM / customer pipeline

![Streamlit App](images/streamlit_screenshot.png)

---

## 📊 8. Tableau Story — *The 90-Day Churn Reduction Playbook*

📍 [View on Tableau Public](https://public.tableau.com/views/alpha_churn_dashboards/The90-DayChurnReductionPlaybook?:language=en-US&publish=yes&:redirect=auth)

![Tableau Story](images/tableau_story.png)

---

## 🧱 9. Repository Structure

```
Final Project/
├─ Dataset/
│  ├─ Cleaned Dataset Analysis/	   # cleaned after EDA
│  ├─ Processed Data/              # final train/test CSVs
│  └─ Raw Dataset/ 		   # original input
├─ images/                         # PNGs for README
├─ Streamlit/                      # Streamlit app (UI & serving)
│  ├─ .streamlit/                  # config.toml, secrets.toml
│  ├─ artifacts/                   # churn_xgb_cw.sav
│  ├─ assets/                      # css, icons, small UI images
│  ├─ pages/                       # multipage Streamlit
│  ├─ utils/                       # I/O, metrics, plotting, loaders
│  ├─ app.py 
│  └─ requirements.txt
├─ Tableau/
│  └─ alpha_churn_dashboards.twbx  # Tableau workbook
├─ alpha_churn_notebook.ipynb
├─ experimental_notebook.ipynb
└─ README.md
```

---

## 🧪 10. Reproducibility & Environment

**Python:** ≥ 3.10

**Core Packages:**

`xgboost`, `lightgbm`, `scikit-learn`, `imbalanced-learn`, `shap`, `pandas`, `numpy`, `matplotlib`, `streamlit`

### Setup

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook alpha_churn_notebook.ipynb
```

### Run Streamlit App

```bash
streamlit run Streamlit/app.py
```

> ⚠️ Ensure model & encoder paths in `app.py` are correct.

---

## 🔍 11. Monitoring & Risks

* 📊 **Data Drift:** Track key distributions (`Tenure`, `DaySinceLastOrder`)
* ⚖️ **Class Imbalance:** Adjust threshold if churn ratio shifts
* 🧱 **Feature Availability:** Schema must match training features
* 🧑‍⚖️ **Ethical Use:** Ensure fair treatment across customer segments

---

## 🪪 12. License & Credits

**License:** MIT License © 2025 Group Alpha

**Contributors:**

👤 Alfriando C. Vean

👤 Ardinata Jeremy Kingstone Tambun

👤 Bonifasius Sinurat

---

💡 *“Data-driven retention — empowering businesses to act before customers leave.”*
