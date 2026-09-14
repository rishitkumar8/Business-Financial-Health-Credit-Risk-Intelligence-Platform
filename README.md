# 💳 Business-Financial-Health-Credit-Risk-Intelligence-Platform

### 📊 An End-to-End Machine Learning Approach for Corporate Financial Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Status](https://img.shields.io/badge/Project-Academic-success)

---

## 📌 About the Project

Credit rating is an important indicator of a corporation's financial strength and creditworthiness. It helps investors, financial institutions, and other stakeholders understand the financial risk associated with a company.

This project uses **Machine Learning and corporate financial ratios** to study patterns in corporate credit ratings and develop predictive models.

The project follows a complete Machine Learning workflow:

**Data Collection → EDA → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Model Comparison → Interpretation**

Three Machine Learning approaches are explored:

- 📈 **Regression**
- 🏷️ **Classification**
- 🔵 **Clustering**

The objective is not only to build predictive models, but also to understand how financial characteristics relate to corporate creditworthiness.

---

# 🎯 Problem Statement

Traditional corporate credit assessment requires analysing several financial indicators simultaneously, including liquidity, debt, profitability, asset efficiency, and cash flow.

Analysing these indicators manually can make it difficult to identify relationships and patterns across a large number of corporations.

### The problem addressed in this project is:

> **How effectively can Machine Learning techniques use corporate financial ratios to analyse and predict credit-related outcomes and identify meaningful groups of corporations based on their financial characteristics?**

To address this problem, the project:

- Analyses corporate financial ratios and credit ratings.
- Identifies relationships between financial indicators.
- Builds multiple predictive models.
- Compares different Machine Learning algorithms.
- Determines the best-performing models using appropriate evaluation metrics.
- Uses clustering to discover groups of financially similar corporations.

---

# 🎯 Project Objectives

### 🔍 1. Understand the Dataset
Explore the structure, distributions, missing values, relationships, and characteristics of the corporate financial data.

### 🧹 2. Prepare the Data
Clean the dataset, handle missing values and duplicates, analyse outliers, encode categorical variables, and scale features where required.

### ⚙️ 3. Engineer Useful Features
Create meaningful derived features from existing financial indicators where they can provide additional information to the models.

### 🤖 4. Build Machine Learning Models
Implement multiple regression, classification, and clustering algorithms.

### 📊 5. Compare Model Performance
Evaluate all models using appropriate metrics and identify the strongest-performing approaches.

### 🔬 6. Discover Financial Patterns
Use clustering techniques to identify groups of corporations with similar financial characteristics.

### 🏆 7. Select the Best Models
Perform hyperparameter tuning and validation to improve and select suitable models.

---

# 📂 Dataset

The project uses a **Corporate Credit Rating with Financial Ratios** dataset.

### Dataset Statistics

| Property | Value |
|---|---:|
| 📌 Number of Records | 7,805 |
| 📌 Number of Features | 25 |
| 🏢 Sectors | 12 |
| ⭐ Credit Rating Categories | 23 |
| 🎯 Binary Rating Classes | 2 |

The dataset contains corporate information, credit ratings, industry information, and several financial ratios.

---

## 📋 Dataset Features

### 🏢 Corporate Information

- `Rating Agency`
- `Corporation`
- `Rating`
- `Rating Date`
- `CIK`
- `SIC Code`
- `Sector`
- `Ticker`

### ⭐ Credit Rating Information

- `Rating`
- `Binary Rating`

The `Rating` field represents the credit rating assigned to a corporation, while `Binary Rating` provides a binary representation of the rating.

### 💰 Financial Ratios

- `Current Ratio`
- `Long-term Debt / Capital`
- `Debt/Equity Ratio`
- `Gross Margin`
- `Operating Margin`
- `EBIT Margin`
- `EBITDA Margin`
- `Pre-Tax Profit Margin`
- `Net Profit Margin`
- `Asset Turnover`
- `ROE - Return On Equity`
- `Return On Tangible Equity`
- `ROA - Return On Assets`
- `ROI - Return On Investment`
- `Operating Cash Flow Per Share`
- `Free Cash Flow Per Share`

---

# 🔎 Exploratory Data Analysis

The EDA stage focuses on understanding the financial dataset before applying Machine Learning algorithms.

### The analysis includes:

- 📐 Dataset shape and structure
- 🔢 Data types
- ❌ Missing-value analysis
- 🔁 Duplicate-value analysis
- 📊 Feature distributions
- ⭐ Credit-rating distribution
- 🔥 Correlation analysis
- 📈 Feature relationships
- 📉 Outlier analysis
- 💰 Financial ratio analysis

### Visualisations

The project includes:

- Distribution plots
- Correlation heatmap
- Target distribution
- Feature-target scatter plots
- Box plots for outlier analysis
- Other relevant financial feature visualisations

Each major visualisation is accompanied by an observation explaining the pattern identified from the data.

---

# 🧹 Data Preprocessing

Before model training, the dataset undergoes appropriate preprocessing.

### Missing Values

Missing values are identified and handled using suitable imputation strategies.

### Duplicate Records

Duplicate observations are checked and removed where appropriate.

### Outliers

Numerical features are analysed for extreme observations and treated where necessary.

### Categorical Variables

Categorical features are converted into numerical representations using suitable encoding techniques.

### Feature Scaling

Scaling is applied to algorithms that are sensitive to feature magnitude, such as:

- K-Nearest Neighbors
- Support Vector Machine
- Support Vector Regression
- MLP Classifier

### Train-Test Split

A consistent train-test split is maintained within each track to ensure a fair comparison between algorithms.

---

# ⚙️ Feature Engineering

Feature engineering is performed to create additional information from the existing financial variables.

The engineered features are selected based on their financial meaning and potential usefulness to the Machine Learning models.

Each engineered feature is documented in the corresponding notebook along with the reasoning behind its creation.

---

# 📈 Regression

The regression track compares **10 different regression algorithms**.

### Algorithms Implemented

| # | Algorithm |
|---:|---|
| 1️⃣ | Linear Regression |
| 2️⃣ | Ridge Regression |
| 3️⃣ | Lasso Regression |
| 4️⃣ | ElasticNet Regression |
| 5️⃣ | Polynomial Regression |
| 6️⃣ | Decision Tree Regressor |
| 7️⃣ | Random Forest Regressor |
| 8️⃣ | Gradient Boosting Regressor |
| 9️⃣ | Support Vector Regressor |
| 🔟 | K-Nearest Neighbors Regressor |

### 📏 Evaluation Metrics

The models are evaluated using:

- **R² Score**
- **RMSE**
- **MAE**
- **5-Fold Cross-Validated R²**

The two best-performing models are further evaluated using cross-validation.

---

# 🏷️ Classification

Classification is used to predict the **credit-rating class** of a corporation from its available financial characteristics.

The classification track is divided into Part A and Part B as specified by the capstone requirements.

---

## Part A

### Algorithms

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine

---

## Part B

### Algorithms

6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. MLP Classifier

---

## 📏 Classification Evaluation

Each classification model is evaluated using:

- 🎯 Accuracy
- 🎯 Precision
- 🎯 Recall
- 🎯 Weighted F1-Score
- 📊 Confusion Matrix
- 📈 ROC-AUC

For multi-class classification, **One-vs-Rest (OvR)** is used for ROC-AUC evaluation.

The final comparison contains all **10 classification algorithms**.

---

# 🔵 Clustering

Clustering is used to discover groups of corporations that exhibit similar financial characteristics without using credit-rating labels during model fitting.

### Algorithms

#### 🔹 K-Means Clustering

The optimal number of clusters is investigated using the **Elbow Method**.

#### 🔹 Agglomerative Hierarchical Clustering

Hierarchical relationships between observations are explored using a **Dendrogram** and different linkage strategies.

---

## 📏 Clustering Evaluation

The clustering models are evaluated using:

| Metric | Purpose |
|---|---|
| Silhouette Score | Measures cluster separation and cohesion |
| Davies-Bouldin Index | Measures similarity between clusters |
| Calinski-Harabasz Index | Measures cluster separation |

### 📊 Cluster Visualisation

The project includes:

- 📉 K-Means Elbow Curve
- 🌳 Hierarchical Dendrogram
- 📊 PCA 2D Cluster Visualisation
- 🔬 t-SNE Visualisation

Ground-truth credit-rating labels are **not used during clustering**. They are used only after clustering for interpretation or validation.

---

# 🎛️ Hyperparameter Tuning

Hyperparameter tuning is performed to improve the performance of selected Machine Learning models.

Techniques include:

- `GridSearchCV`
- `RandomizedSearchCV`

The tuned models are compared with their baseline versions to determine whether the selected parameters improve model performance.

---

# 🏆 Model Evaluation & Comparison

All models within a track are evaluated using a consistent dataset split and appropriate metrics.

## 📈 Regression Results

| Rank | Model | R² | RMSE | MAE |
|---:|---|---:|---:|---:|
| 🥇 1 | — | — | — | — |
| 🥈 2 | — | — | — | — |
| 🥉 3 | — | — | — | — |
| 4 | — | — | — | — |
| 5 | — | — | — | — |
| 6 | — | — | — | — |
| 7 | — | — | — | — |
| 8 | — | — | — | — |
| 9 | — | — | — | — |
| 10 | — | — | — | — |

> 📌 Final values will be updated after the complete regression evaluation.

---

## 🏷️ Classification Results

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 1 | — | — | — | — | — | — |
| 🥈 2 | — | — | — | — | — | — |
| 🥉 3 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — |
| 6 | — | — | — | — | — | — |
| 7 | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — |
| 9 | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — |

> 📌 Final consolidated classification results will be added after completing all 10 algorithms.

---

## 🔵 Clustering Results

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---:|---:|---:|
| K-Means | — | — | — |
| Agglomerative Clustering | — | — | — |

---

# 🏆 Final Model Selection

The final model is selected based on:

- Overall predictive performance
- Evaluation metrics
- Cross-validation performance
- Generalisation ability
- Hyperparameter tuning results
- Model complexity
- Interpretability where relevant

Rather than selecting a model based on a single metric, the project considers the overall performance and suitability of the model for the given problem.

---

# 🗂️ Project Structure

```text
Corporate-Credit-Rating-ML/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📁 data/
│   └── corporateCreditRatingWithFinancialRatios.csv
│
├── 📁 notebooks/
│   ├── 📓 regression.ipynb
│   ├── 📓 classification.ipynb
│   └── 📓 clustering.ipynb
│
├── 📁 models/
│   └── saved_models/
│
└── 📁 app/
    └── app.py
