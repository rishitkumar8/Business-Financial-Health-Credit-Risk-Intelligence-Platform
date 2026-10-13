# 💳 CreditLens — Corporate Credit Intelligence Dashboard
### Premium Machine Learning Platform for Financial Risk Analytics & Solvency Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.65+-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-6.0+-3F4F75?logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![Status](https://img.shields.io/badge/Deployment-Ready-success)

---

## 📌 Executive Overview

**CreditLens** is an end-to-end corporate credit intelligence platform that converts multi-source financial ratios into predictive credit risk metrics, solvency classifications, and unsupervised corporate archetypes. 

Built on an empirical corpus of **7,805 rating observations across 1,377 global corporations**, CreditLens pairs industry-grade predictive modeling with a modern fintech dashboard design.

The platform operationalizes three machine learning tracks:
1. 📈 **Continuous Credit Rating Score Prediction (Regression):** Predicts continuous rating grade scores (0=D to 22=AAA) with fine-grained ordinal resolution ($R^2 = 0.706$).
2. 🏷️ **Solvency & Default Risk Classification (Classification):** Categorizes corporations into Investment Grade ($1$) vs. Speculative Grade ($0$) with high precision ($92.4\%$ Accuracy, $0.963$ ROC-AUC).
3. 🔵 **Unsupervised Financial Archetype Profiling (Clustering):** Discovers latent corporate profiles (High-Margin, Capital-Intensive, Distressed) without relying on credit labels during training.

---

## 🎯 Key Capabilities & Dashboard Modules

CreditLens features **seven interconnected functional modules**:

| Module | Core Purpose & Analytical Features |
|---|---|
| **1. Overview** | Executive summary dashboard featuring real-time KPI metrics (7,805 records, 1,377 corporations, 12 sectors, 99.04% data completeness), grade distributions, sector comparisons, and empirical observations. |
| **2. Financial Explorer** | Interactive cohort screening by corporation, ticker, credit rating, rating agency, and sector. Explores ratio distributions across 6 categories (Liquidity, Leverage, Profitability, Returns, Asset Utilization, Cash Flow), scatter plots, and CSV export. |
| **3. Prediction Center** | Real-time dual-track prediction engine supporting both **Mode A (Manual Entry)** and **Mode B (Existing Corporation Search)**. Transforms inputs through exact pipelines and presents continuous rating scores, letter grades, solvency classes, and class probabilities in a dark navy card. |
| **4. Regression Lab** | Comparative diagnostic suite for **12 regression models**. Displays verified test-set metrics ($R^2$, RMSE, MAE), actual-vs-predicted plots, residual plots, error distributions, and Gini feature importances. |
| **5. Classification Lab** | Benchmark suite for **12 classification models**. Displays test-set metrics (Accuracy, Precision, Recall, Weighted F1, ROC-AUC), interactive confusion matrices, per-class evaluations, and ROC curves. |
| **6. Clustering Studio** | Unsupervised segmentation comparing **K-Means ($k=3$)** and **Agglomerative Hierarchical ($k=2$)**. Features 2D PCA projections, inertia elbow curve, dendrogram tree, financial ratio radar profiles, and an interactive cluster assignment tool. |
| **7. Dataset & EDA** | Audit and exploratory analysis containing shape metrics, missing value audit, duplicate analysis, descriptive statistics, categorical breakdowns, target distributions, and comprehensive correlation heatmaps. |

---

## 🗂️ Fixed Project Structure

```text
Business-Financial-Health-Credit-Risk-Intelligence-Platform/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── corporateCreditRatingWithFinancialRatios.csv
│   ├── train_regression_preprocessed.csv
│   ├── test_regression_preprocessed.csv
│   ├── train_classification_preprocessed.csv
│   ├── test_classification_preprocessed.csv
│   └── clustering_preprocessed.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
│
├── models/
│   ├── regression_models.pkl
│   ├── classification_models.pkl
│   └── clustering_models.pkl
│
└── app/
    └── app.py
```

---

## 💻 Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) (1.65+)
- **Interactive Visualizations:** [Plotly](https://plotly.com/) Express & Graph Objects (6.0+)
- **Data Engineering:** [Pandas](https://pandas.pydata.org/) (2.0+), [NumPy](https://numpy.org/) (1.24+)
- **Machine Learning & Pipeline:** [Scikit-learn](https://scikit-learn.org/) (1.7+), [SciPy](https://scipy.org/) (1.11+)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/) (1.3+)

---

## 🚀 Installation & Local Execution

### 1. Clone or Navigate to the Repository

```bash
cd Business-Financial-Health-Credit-Risk-Intelligence-Platform
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📂 Dataset Specification

The project uses the **Corporate Credit Rating with Financial Ratios** dataset:

- **Total Records:** 7,805
- **Unique Corporations:** 1,377
- **Features:** 25 raw columns (17 financial ratios + identifiers and targets)
- **Data Completeness:** 99.04%
- **Rating Agencies:** Egan-Jones Ratings, Standard & Poor's, Moody's, Fitch Ratings, DBRS, Japan Credit Rating Agency, HR Ratings de Mexico.
- **Sectors (12):** BusEq (Business Equipment), Chems (Chemicals), Durbl (Durables), Enrgy (Energy), Hlth (Healthcare), Manuf (Manufacturing), Money (Financials), NoDur (Non-Durables), Other, Shops (Retail/Wholesale), Telcm (Telecommunications), Utils (Utilities).

### Financial Feature Categorization

1. **Liquidity:** Current Ratio
2. **Leverage:** Long-term Debt / Capital, Debt/Equity Ratio
3. **Profitability:** Gross Margin, Operating Margin, EBIT Margin, EBITDA Margin, Pre-Tax Profit Margin, Net Profit Margin (and engineered `Average Profit Margin`)
4. **Returns:** Return on Equity (ROE), Return on Tangible Equity, Return on Assets (ROA), Return on Investment (ROI)
5. **Asset Utilization:** Asset Turnover
6. **Cash Flow:** Operating Cash Flow Per Share, Free Cash Flow Per Share

---

## 🤖 Model Artifacts & Benchmark Performance

All models are saved inside `models/` and loaded dynamically via `joblib`:

### 1. Regression Models (`models/regression_models.pkl`)
12 trained estimators predicting continuous `Rating_Score` (0=D to 22=AAA):

| Model | $R^2$ Score | RMSE | MAE | Tuned Status |
|---|---:|---:|---:|:---:|
| **Random Forest - Tuned** | **0.7062** | **2.0355** | **1.4939** | **Tuned** |
| Random Forest Regressor | 0.6990 | 2.0602 | 1.5122 | Baseline |
| KNN Regressor - Tuned | 0.6385 | 2.2578 | 1.5543 | Tuned |
| KNN Regressor | 0.6298 | 2.2848 | 1.5869 | Baseline |
| Support Vector Regressor (SVR) | 0.5352 | 2.5601 | 1.9232 | Baseline |
| Gradient Boosting Regressor | 0.5226 | 2.5946 | 2.0032 | Baseline |
| Polynomial Regression (Degree 2) | 0.4544 | 2.7737 | 2.1351 | Baseline |
| Decision Tree Regressor | 0.4122 | 2.8791 | 2.2264 | Baseline |
| Ridge Regression | 0.3474 | 3.0336 | 2.3208 | Baseline |
| Linear Regression | 0.3474 | 3.0336 | 2.3208 | Baseline |
| ElasticNet Regression | 0.2227 | 3.3107 | 2.6072 | Baseline |
| Lasso Regression | 0.2014 | 3.3558 | 2.6739 | Baseline |

### 2. Classification Models (`models/classification_models.pkl`)
12 trained estimators predicting `Binary Rating` (1 = Investment Grade, 0 = Speculative Grade):

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest - Tuned** | **0.9238** | **0.9234** | **0.9238** | **0.9231** | **0.9634** |
| Random Forest Classifier | 0.9142 | 0.9137 | 0.9142 | 0.9135 | 0.9576 |
| Bagging Classifier | 0.9135 | 0.9131 | 0.9135 | 0.9129 | 0.9587 |
| Bagging - Tuned | 0.9116 | 0.9114 | 0.9116 | 0.9104 | 0.9587 |
| MLP Classifier | 0.8860 | 0.8851 | 0.8860 | 0.8851 | 0.9406 |
| KNN Classifier | 0.8661 | 0.8650 | 0.8661 | 0.8640 | 0.9172 |
| Support Vector Classifier (SVC) | 0.8450 | 0.8437 | 0.8450 | 0.8432 | 0.9070 |
| Gradient Boosting Classifier | 0.8443 | 0.8426 | 0.8443 | 0.8414 | 0.9056 |
| Decision Tree Classifier | 0.8341 | 0.8336 | 0.8341 | 0.8338 | 0.8226 |
| Logistic Regression | 0.7719 | 0.7679 | 0.7719 | 0.7656 | 0.8242 |
| Gaussian Naive Bayes | 0.7636 | 0.7584 | 0.7636 | 0.7548 | 0.8122 |
| AdaBoost Classifier | 0.7559 | 0.7533 | 0.7559 | 0.7542 | 0.8037 |

### 3. Clustering Models (`models/clustering_models.pkl`)
Unsupervised models evaluated on the preprocessed financial matrix (28 features):

| Algorithm | Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Out-of-Sample Prediction |
|---|:---:|---:|---:|---:|:---:|
| **K-Means** | 3 | **0.1776** | **1.7440** | **1998.38** | **Yes (`predict()`)** |
| Agglomerative Hierarchical | 2 | 0.1684 | 1.8458 | 1866.80 | No (Transductive) |

---

## ☁️ Streamlit Community Cloud Deployment

CreditLens is configured for deployment to **Streamlit Community Cloud**:

1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository:
   - **Repository:** `Business-Financial-Health-Credit-Risk-Intelligence-Platform`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Click **Deploy**. Dependencies from `requirements.txt` will install automatically.

---

## ⚖️ Analytical Disclaimer

> **Notice:** Predictions, scores, and cluster assignments produced by CreditLens are statistical estimates generated by machine learning models trained on historical corporate ratios. They do not constitute official credit ratings issued by NRSRO rating agencies (S&P, Moody's, Fitch) nor do they constitute investment, financial, or legal advice.
