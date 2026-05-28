# 📉 Customer Churn Prediction

> Binary Classification | EDA | SMOTE | Threshold Tuning | Feature Importance  
> **Best Model: Random Forest | F1 (Churn): 0.61 | Recall (Churn): 0.79 | ROC-AUC: 0.8261**

---

## 📋 Overview

Customer churn is one of the most costly problems in subscription-based businesses. This project builds a machine learning pipeline to predict which telecom customers are likely to churn, using the **IBM Telco Customer Churn Dataset** (7,043 customers, 20 features).

The focus goes beyond accuracy — handling class imbalance with SMOTE, tuning the decision threshold for business-optimal recall, and extracting actionable feature importance for business stakeholders.

---

## 📊 Results

### Cross-Validation (SMOTE-balanced train set)

| Model | CV F1 (churn) | CV F1 (std) | CV ROC-AUC |
|-------|---------------|-------------|------------|
| Logistic Regression | 0.7830 | ±0.0063 | 0.8549 |
| Gradient Boosting | 0.8517 | ±0.0114 | 0.9324 |
| XGBoost | 0.8484 | ±0.0035 | 0.9297 |
| **Random Forest — Final Model** | **0.8567** | **±0.0061** | **0.9256** |

### Final Test Performance (real distribution, threshold = 0.33)

| Metric | No Churn | Churn |
|--------|----------|-------|
| Precision | 0.90 | 0.50 |
| Recall | 0.71 | **0.79** |
| F1-Score | 0.80 | 0.61 |
| Accuracy | 0.7324 | — |
| ROC-AUC | — | 0.8261 |

> **Note:** CV ROC-AUC (0.9256) is higher than test ROC-AUC (0.8261) because SMOTE creates synthetic samples on balanced data, inflating CV scores. The test set reflects the real 26% churn distribution — always evaluate final performance on real, unbalanced test data.

---

## 🗂️ Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,043 customers · 20 features · 26.5% churn rate · No missing values (after cleaning)

| Feature | Description |
|---------|-------------|
| `tenure` | Months the customer has stayed |
| `Contract` | Month-to-month, one year, two year |
| `MonthlyCharges` | Current monthly charge |
| `TotalCharges` | Total amount charged |
| `InternetService` | DSL, Fiber optic, No |
| `OnlineSecurity` | Whether online security is enabled |
| `TechSupport` | Whether tech support is enabled |
| `PaymentMethod` | Payment type |
| `Churn` | **TARGET** — Yes / No |

---

## ⚙️ Project Pipeline

```
Load Data → Clean (TotalCharges fix) → EDA → 
Encode → Split → Scale → SMOTE → 
Model Comparison (5-fold CV) → Threshold Tuning → 
GridSearchCV → Final Evaluation → Feature Importance
```

### Key Steps

- **TotalCharges fix:** Blank strings converted to NaN → filled with 0 (new customers with tenure=0)
- **SMOTE applied on training data only** — synthetic minority samples generated after split, never on test set
- **Stratified K-Fold CV** — preserves class distribution in every fold
- **Threshold tuned from 0.50 → 0.33** — prioritizes recall on churn class (catching churners matters more than avoiding false alarms)
- **4-model comparison** with F1 as primary metric — accuracy is misleading on imbalanced data
- **Built-in feature importance** from Random Forest — top drivers identified for business use

---

## 🔍 Key Findings

- **Contract type** is the strongest predictor — month-to-month customers churn at ~42% vs ~3% for two-year contracts
- **Tenure** is the most powerful numerical feature — churn risk peaks in the first 12 months
- **Fiber optic + high monthly charges** correlate with higher churn — price sensitivity is a major driver
- **Customers without OnlineSecurity or TechSupport** churn significantly more
- **Threshold tuning** boosted churn recall from ~0.65 → 0.79 at the cost of lower precision

---

## 💼 Business Recommendations

- Target **early-tenure customers (0–12 months)** with onboarding incentives and loyalty rewards
- Promote **annual or two-year contracts** — the single biggest lever to reduce churn
- Bundle **OnlineSecurity and TechSupport** into base plans — reduces perceived value gap
- Review **fiber optic pricing** — high charges are driving away this segment

---

## 🏗️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)
![Pandas](https://img.shields.io/badge/Pandas-latest-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuhammadAliEjaz1/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook customer_churn_prediction.ipynb
   ```

---

## 📁 Repository Structure

```
customer-churn-prediction/
├── customer_churn_prediction.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## 👤 Author

**Muhammad Ali Ejaz**  
[![Kaggle](https://img.shields.io/badge/Kaggle-chaliejaz-blue?logo=kaggle)](https://www.kaggle.com/chaliejaz)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-muhammad--ali--ejaz-0077B5?logo=linkedin)](https://linkedin.com/in/muhammad-ali-ejaz-855117341)

---

*If you found this project helpful, please ⭐ the repository!*

