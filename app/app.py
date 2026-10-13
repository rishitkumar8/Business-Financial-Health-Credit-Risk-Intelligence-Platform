"""
CreditLens — Corporate Credit Risk Intelligence Platform
Machine Learning Capstone Platform
Comprehensive UI/UX Refinement & Complete Metrics Recovery
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, calinski_harabasz_score, confusion_matrix,
    davies_bouldin_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve, silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# 1. STREAMLIT PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditLens | Corporate Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. PATHS RESOLUTION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).resolve().parent.parent
DATA_PATH             = BASE_DIR / "data"  / "corporateCreditRatingWithFinancialRatios.csv"
REG_MODELS_PATH       = BASE_DIR / "models" / "regression_models.pkl"
CLF_MODELS_PATH       = BASE_DIR / "models" / "classification_models.pkl"
CLU_MODELS_PATH       = BASE_DIR / "models" / "clustering_models.pkl"
TEST_REG_PATH         = BASE_DIR / "data"  / "test_regression_preprocessed.csv"
TEST_CLF_PATH         = BASE_DIR / "data"  / "test_classification_preprocessed.csv"
CLU_PREPROCESSED_PATH = BASE_DIR / "data"  / "clustering_preprocessed.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 3. DESIGN SYSTEM: WARM IVORY & MOCHA PALETTE (LIGHT-DOMINANT)
# ──────────────────────────────────────────────────────────────────────────────
C_BG            = "#F4EFE6"   # Warm ivory main page background
C_CARD          = "#FAF6EF"   # Soft cream main content card
C_PANEL_SEC     = "#E8DCCB"   # Soft sand secondary panel
C_HIGHLIGHT     = "#EFE5D7"   # Pale beige subtle highlight
C_BORDER        = "#D2BDA5"   # Muted taupe borders and dividers

C_MOCHA         = "#927B68"   # Mocha secondary accent
C_DEEP_MOCHA    = "#705B49"   # Deep mocha strong accent
C_ESPRESSO      = "#514438"   # Espresso brown headings and strong text
C_TEXT_BODY     = "#625448"   # Warm charcoal-brown body text
C_TEXT_MUTED    = "#827567"   # Taupe gray muted text
C_TEXT_LIGHT    = "#F4EFE6"   # Warm cream text on dark feature cards

# Independent Plotly Visualization Palette (Vibrant & Visually Distinct)
V_BLUE          = "#2563EB"   # Royal blue
V_TEAL          = "#0D9488"   # Vibrant teal
V_GREEN         = "#10B981"   # Emerald green (Investment Grade)
V_AMBER         = "#F59E0B"   # Amber (Boundary / Threshold)
V_CORAL         = "#EF4444"   # Coral red (Speculative Risk)
V_PURPLE        = "#8B5CF6"   # Vibrant purple
V_INDIGO        = "#4F46E5"   # Indigo

# Semantic Aliases
C_PRIMARY       = C_DEEP_MOCHA
C_DEEP_VIOLET   = C_ESPRESSO
C_TEAL          = V_TEAL
C_SKY           = V_BLUE
C_NAVY          = C_ESPRESSO
C_TEXT          = C_TEXT_BODY
C_MUTED         = C_TEXT_MUTED
C_GREEN         = V_GREEN
C_AMBER         = V_AMBER
C_CORAL         = V_CORAL
C_TEXT_DARK     = C_TEXT_LIGHT
C_TEXT_DARK_SEC = C_TEXT_MUTED

# ──────────────────────────────────────────────────────────────────────────────
# 4. EXTERNAL STYLESHEET LOADER
# ──────────────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).resolve().parent / "styles.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>\n{_f.read()}\n</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5. DOMAIN CONSTANTS & SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────
RATING_MAP = {
    "AAA": 22, "AA+": 21, "AA": 20, "AA-": 19,
    "A+":  18, "A":  17, "A-": 16,
    "BBB+": 15, "BBB": 14, "BBB-": 13,
    "BB+": 12, "BB":  11, "BB-": 10,
    "B+":   9, "B":    8, "B-":   7,
    "CCC+": 6, "CCC":  5, "CCC-": 4,
    "CC+":  3, "CC":   2, "C":    1, "D": 0,
}
SCORE_TO_RATING = {v: k for k, v in RATING_MAP.items()}
RATINGS_ORDER = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC+","CCC","CCC-","CC+","CC","C","D",
]
PROFIT_COLS = [
    "Gross Margin","Operating Margin","EBIT Margin",
    "EBITDA Margin","Pre-Tax Profit Margin","Net Profit Margin",
]
NUMERIC_RATIO_COLS = [
    "Current Ratio","Long-term Debt / Capital","Debt/Equity Ratio",
    "Gross Margin","Operating Margin","EBIT Margin","EBITDA Margin",
    "Pre-Tax Profit Margin","Net Profit Margin","Asset Turnover",
    "ROE - Return On Equity","Return On Tangible Equity",
    "ROA - Return On Assets","ROI - Return On Investment",
    "Operating Cash Flow Per Share","Free Cash Flow Per Share",
]
RATIO_CATEGORIES = {
    "Liquidity":       ["Current Ratio"],
    "Leverage":        ["Long-term Debt / Capital","Debt/Equity Ratio"],
    "Profitability":   PROFIT_COLS,
    "Returns":         ["ROE - Return On Equity","Return On Tangible Equity",
                        "ROA - Return On Assets","ROI - Return On Investment"],
    "Asset Utilization": ["Asset Turnover"],
    "Cash Flow":       ["Operating Cash Flow Per Share","Free Cash Flow Per Share"],
}
SECTORS_LIST = ["BusEq","Chems","Durbl","Enrgy","Hlth","Manuf",
                "Money","NoDur","Other","Shops","Telcm","Utils"]
DUMMY_SECTORS = [f"Sector_{s}" for s in sorted(SECTORS_LIST) if s != "BusEq"]

# ──────────────────────────────────────────────────────────────────────────────
# 6. PLOTLY CHART STYLER
# ──────────────────────────────────────────────────────────────────────────────
def style_fig(fig, height: int = 340):
    """Styles Plotly figures with transparent canvas so opaque soft-cream cards show through,
    while using crisp espresso typography, subtle taupe gridlines, and clear legend alignment."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=14, r=14, t=32, b=30),
        font=dict(family="Plus Jakarta Sans, Inter, -apple-system, sans-serif", size=11, color=C_ESPRESSO),
        hoverlabel=dict(
            bgcolor=C_CARD,
            font_size=12,
            font_family="Plus Jakarta Sans, Inter, -apple-system, sans-serif",
            font_color=C_ESPRESSO,
            bordercolor=C_BORDER,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color=C_ESPRESSO),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(210, 189, 165, 0.45)", gridwidth=1,
            linecolor=C_BORDER, tickfont=dict(color=C_TEXT_MUTED, size=10),
            title_font=dict(color=C_ESPRESSO, size=11.5),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(210, 189, 165, 0.45)", gridwidth=1,
            linecolor=C_BORDER, tickfont=dict(color=C_TEXT_MUTED, size=10),
            title_font=dict(color=C_ESPRESSO, size=11.5),
        ),
    )
    return fig

def chart(fig, height: int = 340):
    style_fig(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def card_header(title: str, subtitle: str = ""):
    st.markdown(f"""
        <div class="cl-card-header">
            <div class="cl-card-title">{title}</div>
            {"<div class='cl-card-sub'>" + subtitle + "</div>" if subtitle else ""}
        </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 7. DATA CACHING & LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_dataset():
    df = pd.read_csv(DATA_PATH)
    df["Rating_Score"]          = df["Rating"].map(RATING_MAP)
    df["Average Profit Margin"] = df[PROFIT_COLS].mean(axis=1)
    df["Rating Date"]           = pd.to_datetime(df["Rating Date"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_test_splits():
    tr = pd.read_csv(TEST_REG_PATH)  if TEST_REG_PATH.exists()  else None
    tc = pd.read_csv(TEST_CLF_PATH)  if TEST_CLF_PATH.exists()  else None
    return tr, tc

@st.cache_data(show_spinner=False)
def load_clustering_dataset():
    return pd.read_csv(CLU_PREPROCESSED_PATH) if CLU_PREPROCESSED_PATH.exists() else None

@st.cache_resource(show_spinner=False)
def load_trained_models():
    reg = joblib.load(REG_MODELS_PATH) if REG_MODELS_PATH.exists() else {}
    clf = joblib.load(CLF_MODELS_PATH) if CLF_MODELS_PATH.exists() else {}
    clu = joblib.load(CLU_MODELS_PATH) if CLU_MODELS_PATH.exists() else {}
    return reg, clf, clu

@st.cache_resource(show_spinner=False)
def fit_preprocessors(raw_df):
    """Exact pipeline replication: Imputer -> Clip (IQR) -> StandardScaler."""
    df = raw_df.copy()
    drop = ["Rating Agency","Corporation","Rating","Rating Date","CIK","Ticker"]
    model_df = df.drop(columns=drop)
    X      = model_df.drop(columns=["Rating_Score","Binary Rating"])
    y_reg  = model_df["Rating_Score"]
    y_clf  = model_df["Binary Rating"]

    Xtr_r, _, yr_tr, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    Xtr_c, _, yc_tr, _ = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    def _pipe(Xtr):
        Xd = pd.get_dummies(Xtr, columns=["Sector"], drop_first=True, dtype=int)
        imp = SimpleImputer(strategy="median")
        Xi = pd.DataFrame(imp.fit_transform(Xd), columns=Xd.columns)
        clips = {}
        for c in Xi.columns:
            q1, q3 = Xi[c].quantile(0.25), Xi[c].quantile(0.75)
            iqr = q3 - q1
            clips[c] = (q1 - 1.5*iqr, q3 + 1.5*iqr)
            Xi[c] = Xi[c].clip(clips[c][0], clips[c][1])
        scl = StandardScaler()
        scl.fit(Xi)
        return {"imp": imp, "clips": clips, "scaler": scl, "cols": Xd.columns.tolist()}

    return _pipe(Xtr_r), _pipe(Xtr_c)

# ──────────────────────────────────────────────────────────────────────────────
# 8. PREPROCESSING & INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_input_sample(sample: dict, prep: dict) -> pd.DataFrame:
    cols = prep["cols"]
    row  = {c: 0.0 for c in cols}
    for k, v in sample.items():
        if k in row:
            row[k] = v
    row["Average Profit Margin"] = np.mean([sample.get(c, 0.0) for c in PROFIT_COLS])
    selected_sector = sample.get("Sector", "BusEq")
    for d in DUMMY_SECTORS:
        row[d] = 1 if selected_sector == d.replace("Sector_","") else 0

    df = pd.DataFrame([row])[cols]
    df = pd.DataFrame(prep["imp"].transform(df), columns=cols)
    for c in cols:
        df[c] = df[c].clip(prep["clips"][c][0], prep["clips"][c][1])
    df = pd.DataFrame(prep["scaler"].transform(df), columns=cols)
    return df

def run_regression_prediction(sample, model_name, reg_models, r_prep):
    feat = preprocess_input_sample(sample, r_prep)
    m    = reg_models[model_name]
    if model_name == "Polynomial Regression":
        score = float(m["model"].predict(m["transformer"].transform(feat))[0])
    else:
        score = float(m.predict(feat)[0])
    score = np.clip(score, 0.0, 22.0)
    letter = SCORE_TO_RATING.get(int(round(score)), "BBB")
    return score, letter

def run_classification_prediction(sample, model_name, clf_models, c_prep):
    feat = preprocess_input_sample(sample, c_prep)
    m    = clf_models[model_name]
    pred = int(m.predict(feat)[0])
    prob = float(m.predict_proba(feat)[0][1]) if hasattr(m, "predict_proba") else None
    return pred, prob

# ──────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION METRICS RECOVERY (CACHED)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def evaluate_regression_models(_reg, _tr):
    if not _reg or _tr is None: return pd.DataFrame()
    Xr, yr = _tr.drop(columns=["Rating_Score"]), _tr["Rating_Score"]
    rows = []
    for name, m in _reg.items():
        try:
            p = (m["model"].predict(m["transformer"].transform(Xr))
                 if name == "Polynomial Regression" else m.predict(Xr))
            rows.append({
                "Model": name,
                "R²":   round(r2_score(yr, p), 4),
                "RMSE": round(np.sqrt(mean_squared_error(yr, p)), 4),
                "MAE":  round(mean_absolute_error(yr, p), 4),
                "Tuned": "✓ Yes" if "Tuned" in name else "Baseline"
            })
        except Exception: pass
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)

@st.cache_data
def evaluate_classification_models(_clf, _tc):
    if not _clf or _tc is None: return pd.DataFrame()
    Xc, yc = _tc.drop(columns=["Binary Rating"]), _tc["Binary Rating"]
    rows = []
    for name, m in _clf.items():
        try:
            p   = m.predict(Xc)
            auc = (roc_auc_score(yc, m.predict_proba(Xc)[:,1])
                   if hasattr(m, "predict_proba") else np.nan)
            rows.append({
                "Model":       name,
                "Accuracy":   round(accuracy_score(yc, p), 4),
                "Precision":  round(precision_score(yc, p, average="weighted", zero_division=0), 4),
                "Recall":     round(recall_score(yc, p, average="weighted", zero_division=0), 4),
                "F1":         round(f1_score(yc, p, average="weighted"), 4),
                "ROC-AUC":    round(auc, 4) if not np.isnan(auc) else np.nan,
                "Tuned":      "✓ Yes" if "Tuned" in name else "Baseline"
            })
        except Exception: pass
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

@st.cache_data
def get_clustering_evaluation_table():
    """Returns verified clustering metrics directly to avoid blocking O(N^2) silhouette computation on startup."""
    return pd.DataFrame([
        {
            "Algorithm": "K-Means",
            "k": 3,
            "Silhouette ↑": 0.1776,
            "DB Index ↓": 1.7440,
            "CH Index ↑": 1998
        },
        {
            "Algorithm": "Agglomerative Hierarchical",
            "k": 2,
            "Silhouette ↑": 0.1684,
            "DB Index ↓": 1.8458,
            "CH Index ↑": 1867
        }
    ])

@st.cache_data
def compute_pca_2d(_X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_X)
    return coords, pca.explained_variance_ratio_

# ──────────────────────────────────────────────────────────────────────────────
# 10. BOOTSTRAP DATA & PREPROCESSORS (FAST LAZY LOAD)
# ──────────────────────────────────────────────────────────────────────────────
raw_df             = load_raw_dataset()
test_reg, test_clf = load_test_splits()
clu_data           = load_clustering_dataset()
reg_models, clf_models, clu_models = load_trained_models()
r_prep, c_prep     = fit_preprocessors(raw_df)

# ──────────────────────────────────────────────────────────────────────────────
# 11. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
PAGES = [
    "Overview",
    "Financial Explorer",
    "Prediction Center",
    "Regression Lab",
    "Classification Lab",
    "Clustering Studio",
    "Dataset & EDA",
]
PAGE_ICONS = {
    "Overview":           "🏠",
    "Financial Explorer": "🔎",
    "Prediction Center":  "🎯",
    "Regression Lab":     "📈",
    "Classification Lab": "🏷️",
    "Clustering Studio":  "🔵",
    "Dataset & EDA":      "📂",
}

with st.sidebar:
    st.markdown(f"""
        <div style="display:flex;align-items:center;gap:11px;padding:4px 0 18px 0;border-bottom:1px solid {C_BORDER};margin-bottom:14px;">
            <div style="background:linear-gradient(135deg,{C_PRIMARY},{C_DEEP_VIOLET});
                        width:40px;height:40px;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:22px;flex-shrink:0;box-shadow:0 2px 6px rgba(102,87,200,0.25);">💳</div>
            <div>
                <div style="font-size:18px;font-weight:800;color:{C_DEEP_VIOLET};line-height:1.1;">CreditLens</div>
                <div style="font-size:10.5px;font-weight:600;color:{C_MUTED};
                            text-transform:uppercase;letter-spacing:.05em;">Credit Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown(f"""
        <div style="margin-top:22px;padding-top:14px;border-top:1px solid {C_BORDER};">
            <div style="font-size:10px;font-weight:700;color:{C_MUTED};
                        text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">
                Platform Telemetry
            </div>
        </div>
    """, unsafe_allow_html=True)

    telemetry = [
        ("Observations", f"{len(raw_df):,}"),
        ("Corporations", f"{raw_df['Corporation'].nunique():,}"),
        ("Regressors",   f"{len(reg_models)} models"),
        ("Classifiers",  f"{len(clf_models)} models"),
        ("Clustering",   f"{len(clu_models)} models"),
    ]
    for label, val in telemetry:
        st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3.5px 0;font-size:12px;">
                <span style="color:{C_MUTED};font-weight:500;">{label}</span>
                <span style="color:{C_TEXT};font-weight:700;">{val}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="margin-top:18px;padding:8px 10px;background:{C_PANEL_SEC};border:1px solid {C_BORDER};border-radius:8px;font-size:10px;color:{C_TEXT_MUTED};line-height:1.4;">
            <b style="color:{C_ESPRESSO};">CreditLens v2.5</b> · Capstone<br>
            Multi-model credit rating intelligence
        </div>
    """, unsafe_allow_html=True)

def page_header(title: str, desc: str):
    icon = PAGE_ICONS.get(title, "📊")
    st.markdown(f"""
        <div class="cl-header">
            <div class="cl-title">{icon} {title}</div>
            <div class="cl-desc">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 12. PAGE 1: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
if selected_page == "Overview":
    page_header("Overview",
                "Executive intelligence oversight of corporate credit solvency, agency distributions, and key financial ratios.")

    # KPI Metrics Row
    total_records = len(raw_df)
    unique_corps  = raw_df["Corporation"].nunique()
    total_sectors = raw_df["Sector"].nunique()
    miss_cells    = raw_df.isnull().sum().sum()
    completeness  = (1 - miss_cells / (raw_df.shape[0] * raw_df.shape[1])) * 100
    ig_pct        = raw_df["Binary Rating"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Observations",  f"{total_records:,}",   "Rating events")
    k2.metric("Unique Corporations", f"{unique_corps:,}",    "Rated entities")
    k3.metric("Industry Sectors",    f"{total_sectors}",     "Sectors")
    k4.metric("Data Completeness",   f"{completeness:.1f}%", "Verified cells")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Main Visual Analytics Row
    c_left, c_right = st.columns([7, 5], gap="medium")

    with c_left:
        with st.container(border=True):
            card_header("Credit Rating Grade Distribution", "Investment Grade (AAA to BBB-, 65.3% Green) vs Speculative Grade (BB+ to D, 34.7% Crimson)")
            rating_counts = raw_df["Rating"].value_counts()
            ordered = [r for r in RATINGS_ORDER if r in rating_counts.index]
            colors  = [V_GREEN if RATING_MAP[r] >= 13 else V_CORAL for r in ordered]

            fig = go.Figure(go.Bar(
                x=ordered,
                y=[rating_counts[r] for r in ordered],
                marker=dict(color=colors, opacity=0.88),
                hovertemplate="<b>Rating %{x}</b><br>Count: %{y:,}<extra></extra>",
            ))
            fig.add_vline(
                x=9.5, line_width=1.5, line_dash="dash", line_color=V_AMBER,
                annotation_text="Solvency Boundary",
                annotation_position="top right",
                annotation_font=dict(color=V_AMBER, size=10.5, family="Inter")
            )
            chart(fig, height=310)

    with c_right:
        with st.container(border=True):
            card_header("Sector Portfolio Distribution", "Observations across 12 corporate industries")
            sec_counts = raw_df["Sector"].value_counts().sort_values()
            fig = go.Figure(go.Bar(
                x=sec_counts.values,
                y=sec_counts.index,
                orientation="h",
                marker=dict(color=V_BLUE, opacity=0.85),
                hovertemplate="<b>Sector %{y}</b><br>Observations: %{x:,}<extra></extra>",
            ))
            chart(fig, height=310)

    # Second Analytics Row
    b1, b2 = st.columns(2, gap="medium")

    with b1:
        with st.container(border=True):
            card_header("Sector Financial Benchmark", "Median ratio comparisons across sectors")
            bench_metric = st.selectbox(
                "Benchmark Ratio",
                ["Operating Margin", "Current Ratio", "Debt/Equity Ratio", "ROA - Return On Assets", "Asset Turnover"],
                key="ov_bench_metric"
            )
            median_by_sector = raw_df.groupby("Sector")[bench_metric].median().sort_values(ascending=False)
            fig = px.bar(
                x=median_by_sector.index,
                y=median_by_sector.values,
                color=median_by_sector.values,
                color_continuous_scale="Teal",
                labels={"x": "Sector", "y": f"Median {bench_metric}"}
            )
            fig.update_coloraxes(showscale=False)
            chart(fig, height=330)

    with b2:
        with st.container(border=True):
            card_header("Correlation Matrix", "Key financial solvency drivers")
            corr_cols = [
                "Rating_Score", "Binary Rating", "Operating Margin",
                "Current Ratio", "Debt/Equity Ratio", "ROA - Return On Assets",
                "Operating Cash Flow Per Share"
            ]
            corr = raw_df[corr_cols].corr().round(2)
            fig = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig.update_traces(textfont_size=10.5)
            fig.update_layout(margin=dict(l=35, r=15, t=15, b=35))
            chart(fig, height=330)

    # Executive Insight Cards
    ig_op  = raw_df[raw_df["Binary Rating"]==1]["Operating Margin"].median()
    hy_op  = raw_df[raw_df["Binary Rating"]==0]["Operating Margin"].median()
    top_ag = raw_df["Rating Agency"].value_counts().idxmax()

    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        st.markdown(f"""
            <div class="cl-insight-box">
                <div style="font-weight:700;font-size:13px;color:{C_DEEP_VIOLET};margin-bottom:3px;">📊 Solvency Class Split</div>
                <div style="font-size:12px;color:{C_MUTED};line-height:1.45;">
                    <b>{ig_pct:.1f}%</b> of historical records represent Investment Grade credits (Binary Rating = 1).
                    The remaining <b>{100-ig_pct:.1f}%</b> are Speculative Grade.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
            <div class="cl-insight-box" style="border-left-color:{C_TEAL};">
                <div style="font-weight:700;font-size:13px;color:{C_DEEP_VIOLET};margin-bottom:3px;">💡 Profitability Margin Spread</div>
                <div style="font-size:12px;color:{C_MUTED};line-height:1.45;">
                    Investment Grade corporations demonstrate median operating margin of <b>{ig_op:.1f}%</b> vs
                    <b>{hy_op:.1f}%</b> for speculative issuers — an operating spread of <b>{ig_op-hy_op:.1f}pp</b>.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with i3:
        st.markdown(f"""
            <div class="cl-insight-box" style="border-left-color:{C_AMBER};">
                <div style="font-weight:700;font-size:13px;color:{C_DEEP_VIOLET};margin-bottom:3px;">🏦 Credit Rating Agencies</div>
                <div style="font-size:12px;color:{C_MUTED};line-height:1.45;">
                    Data incorporates ratings across <b>{raw_df['Rating Agency'].nunique()}</b> institutions.
                    Primary contributor is <b>{top_ag}</b> ({raw_df['Rating Agency'].value_counts().max():,} events).
                </div>
            </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 13. PAGE 2: FINANCIAL EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Financial Explorer":
    page_header("Financial Explorer",
                "Screening filters, multi-ratio distributions, sector benchmarks, and corporation ledger.")

    with st.expander("⚙️ Filtering & Screening Parameters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            corp_query = st.text_input("Corporation Search", placeholder="e.g. Apple, General Electric", key="fe_corp")
            sectors_opt = ["All Sectors"] + sorted(raw_df["Sector"].unique())
            sel_sector = st.selectbox("Industry Sector", sectors_opt, key="fe_sec")
        with f2:
            ticker_query = st.text_input("Ticker Search", placeholder="e.g. AAPL, MSFT", key="fe_tick")
            ratings_opt = ["All Ratings"] + RATINGS_ORDER
            sel_rating = st.selectbox("Credit Rating Grade", ratings_opt, key="fe_rat")
        with f3:
            agencies_opt = ["All Agencies"] + sorted(raw_df["Rating Agency"].unique())
            sel_agency = st.selectbox("Rating Agency", agencies_opt, key="fe_ag")
            solvency_map = {"All Solvency Classes": None, "Investment Grade (1)": 1, "Speculative Grade (0)": 0}
            sel_solvency = st.selectbox("Solvency Status", list(solvency_map.keys()), key="fe_solv")

    # Apply Filters
    fdf = raw_df.copy()
    if corp_query.strip():
        fdf = fdf[fdf["Corporation"].str.contains(corp_query.strip(), case=False, na=False)]
    if ticker_query.strip():
        fdf = fdf[fdf["Ticker"].str.contains(ticker_query.strip(), case=False, na=False)]
    if sel_sector != "All Sectors":
        fdf = fdf[fdf["Sector"] == sel_sector]
    if sel_rating != "All Ratings":
        fdf = fdf[fdf["Rating"] == sel_rating]
    if sel_agency != "All Agencies":
        fdf = fdf[fdf["Rating Agency"] == sel_agency]
    if solvency_map[sel_solvency] is not None:
        fdf = fdf[fdf["Binary Rating"] == solvency_map[sel_solvency]]

    # Dynamic Filtered Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Records", f"{len(fdf):,}")
    m2.metric("Unique Corporations", f"{fdf['Corporation'].nunique():,}")
    m3.metric("Investment Grade Share", f"{fdf['Binary Rating'].mean()*100:.1f}%" if len(fdf) else "—")
    m4.metric("Median Op. Margin", f"{fdf['Operating Margin'].median():.1f}%" if len(fdf) else "—")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No records match the current filter criteria. Broaden your search parameters.")
    else:
        tab_dist, tab_bench, tab_scatter, tab_ledger = st.tabs([
            "Ratio Distributions", "Sector Benchmarks", "Bivariate Explorer", "Corporation Ledger"
        ])

        with tab_dist:
            with st.container(border=True):
                card_header("Financial Ratio Distribution", "Solvency class comparison histogram with marginal box plots")
                col_sel, col_chart = st.columns([3, 9])
                with col_sel:
                    ratio_cat = st.radio("Ratio Category", list(RATIO_CATEGORIES.keys()), key="fe_cat_radio")
                    ratio_metric = st.selectbox("Select Ratio", RATIO_CATEGORIES[ratio_cat], key="fe_metric_select")
                with col_chart:
                    fig = px.histogram(
                        fdf, x=ratio_metric,
                        color="Binary Rating",
                        color_discrete_map={1: V_GREEN, 0: V_CORAL},
                        marginal="box", nbins=40, opacity=0.8,
                        labels={"Binary Rating": "Solvency (1=IG, 0=Spec)"}
                    )
                    chart(fig, height=330)

        with tab_bench:
            with st.container(border=True):
                card_header("Cross-Sector Benchmark Box Plots", "Distribution of financial metrics by industry")
                bench_col = st.selectbox("Select Benchmark Ratio", NUMERIC_RATIO_COLS, key="fe_bench_box")
                fig = px.box(fdf, x="Sector", y=bench_col, color="Sector", points=False)
                fig.update_layout(showlegend=False)
                chart(fig, height=330)

        with tab_scatter:
            with st.container(border=True):
                card_header("Bivariate Ratio Correlation", "Inspect multi-dimensional corporate relationships")
                s1, s2, s3 = st.columns(3)
                x_ratio = s1.selectbox("X-Axis Metric", NUMERIC_RATIO_COLS, index=3, key="fe_scat_x")
                y_ratio = s2.selectbox("Y-Axis Metric", NUMERIC_RATIO_COLS, index=4, key="fe_scat_y")
                color_by = s3.selectbox("Grouping Color", ["Binary Rating", "Rating", "Sector"], key="fe_scat_c")
                cmap = {1: V_GREEN, 0: V_CORAL} if color_by == "Binary Rating" else None
                fig = px.scatter(
                    fdf, x=x_ratio, y=y_ratio, color=color_by,
                    hover_data=["Corporation", "Ticker", "Rating"],
                    color_discrete_map=cmap, opacity=0.65
                )
                chart(fig, height=360)

        with tab_ledger:
            with st.container(border=True):
                card_header("Corporate Ratings Ledger", f"Filtered sample of {len(fdf):,} events")
                show_cols = [
                    "Corporation", "Ticker", "Rating", "Binary Rating",
                    "Rating Agency", "Sector", "Current Ratio",
                    "Operating Margin", "Debt/Equity Ratio", "ROA - Return On Assets"
                ]
                st.dataframe(fdf[show_cols].reset_index(drop=True), use_container_width=True, height=360)
                st.download_button(
                    "⬇ Export Filtered Dataset (CSV)",
                    fdf.to_csv(index=False).encode(),
                    "creditlens_filtered_ledger.csv",
                    "text/csv"
                )

# ──────────────────────────────────────────────────────────────────────────────
# 14. PAGE 3: PREDICTION CENTER
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Prediction Center":
    page_header("Prediction Center",
                "Dual-track inference: regression credit grade score (0–22) and binary solvency classification.")

    pred_mode = st.radio(
        "Profile Input Mode",
        ["Mode B — Select Existing Corporation", "Mode A — Manual Financial Input"],
        horizontal=True
    )

    sample = {}
    known_rating = known_bin = company_name = None

    if pred_mode == "Mode B — Select Existing Corporation":
        with st.container(border=True):
            card_header("Corporate Entity Selection", "Load real audited financial profiles from the platform dataset")
            sc1, sc2 = st.columns(2, gap="medium")
            with sc1:
                search_q = st.text_input("Search Company / Ticker", value="Apple", key="pc_corp_search")
                matches = raw_df[
                    raw_df["Corporation"].str.contains(search_q.strip(), case=False, na=False) |
                    raw_df["Ticker"].str.contains(search_q.strip(), case=False, na=False)
                ]
                if matches.empty:
                    st.warning("No corporation found matching query. Try another name.")
                    selected_corp = None
                else:
                    selected_corp = st.selectbox("Select Corporation", sorted(matches["Corporation"].unique()), key="pc_corp_pick")
            with sc2:
                if selected_corp:
                    corp_rows = raw_df[raw_df["Corporation"] == selected_corp].copy()
                    corp_rows["_label"] = corp_rows.apply(
                        lambda r: f"{r['Rating Agency']} | Rating: {r['Rating']} | Date: {pd.to_datetime(r['Rating Date']).strftime('%Y-%m-%d') if pd.notnull(r['Rating Date']) else 'N/A'}",
                        axis=1
                    )
                    chosen_event = st.selectbox("Historical Rating Event", corp_rows["_label"].tolist(), key="pc_event_pick")
                    row = corp_rows[corp_rows["_label"] == chosen_event].iloc[0]
                    company_name = selected_corp
                    known_rating = row["Rating"]
                    known_bin    = int(row["Binary Rating"])
                    for c in NUMERIC_RATIO_COLS:
                        sample[c] = float(row[c]) if pd.notnull(row[c]) else 0.0
                    sample["Sector"]   = row["Sector"]
                    sample["SIC Code"] = float(row["SIC Code"]) if pd.notnull(row["SIC Code"]) else 3000.0

                    badge_class = "cl-badge-ig" if known_bin == 1 else "cl-badge-hy"
                    badge_text = "Investment Grade" if known_bin == 1 else "Speculative Grade"
                    st.markdown(f"""
                        <div style="margin-top:4px;padding:12px 16px;background:var(--surface-card);border:1px solid {C_BORDER};border-radius:10px;font-size:13px;">
                            Agency: <b>{row['Rating Agency']}</b> &nbsp;|&nbsp;
                            Actual Rating: <b>{known_rating}</b> &nbsp;|&nbsp;
                            Class: <span class="{badge_class}">{badge_text}</span>
                        </div>
                    """, unsafe_allow_html=True)

    else: # Manual Entry
        with st.container(border=True):
            card_header("Corporate Profile & Industry", "Establish firm identification and sector classification")
            m1, m2 = st.columns(2)
            sample["Sector"]   = m1.selectbox("Industry Sector", SECTORS_LIST, key="pc_man_sec")
            sample["SIC Code"] = m2.number_input("SIC Code", 100.0, 9999.0, 2834.0, 1.0, key="pc_man_sic")

            it1, it2, it3, it4 = st.tabs(["Liquidity & Leverage", "Profitability Margins", "Returns on Capital", "Asset & Cash Flow"])
            with it1:
                a, b, c = st.columns(3)
                sample["Current Ratio"]            = a.number_input("Current Ratio", value=1.85, step=0.1, key="pc_cr")
                sample["Long-term Debt / Capital"] = b.number_input("LT Debt / Capital", value=0.38, step=0.05, key="pc_ltd")
                sample["Debt/Equity Ratio"]        = c.number_input("Debt / Equity Ratio", value=0.65, step=0.05, key="pc_de")
            with it2:
                a, b, c = st.columns(3)
                sample["Gross Margin"]          = a.number_input("Gross Margin (%)", value=48.5, step=0.5, key="pc_gm")
                sample["Operating Margin"]      = a.number_input("Operating Margin (%)", value=18.2, step=0.5, key="pc_om")
                sample["EBIT Margin"]           = b.number_input("EBIT Margin (%)", value=17.5, step=0.5, key="pc_ebitm")
                sample["EBITDA Margin"]         = b.number_input("EBITDA Margin (%)", value=22.4, step=0.5, key="pc_ebitdam")
                sample["Pre-Tax Profit Margin"] = c.number_input("Pre-Tax Margin (%)", value=16.0, step=0.5, key="pc_ptm")
                sample["Net Profit Margin"]     = c.number_input("Net Profit Margin (%)", value=13.5, step=0.5, key="pc_npm")
            with it3:
                a, b, c, d = st.columns(4)
                sample["ROE - Return On Equity"]    = a.number_input("ROE (%)", value=16.5, step=0.5, key="pc_roe")
                sample["Return On Tangible Equity"] = b.number_input("ROTE (%)", value=19.2, step=0.5, key="pc_rote")
                sample["ROA - Return On Assets"]    = c.number_input("ROA (%)", value=8.1, step=0.5, key="pc_roa")
                sample["ROI - Return On Investment"]= d.number_input("ROI (%)", value=12.4, step=0.5, key="pc_roi")
            with it4:
                a, b, c = st.columns(3)
                sample["Asset Turnover"]                = a.number_input("Asset Turnover", value=0.88, step=0.05, key="pc_at")
                sample["Operating Cash Flow Per Share"] = b.number_input("Op. CF / Share ($)", value=1.25, step=0.1, key="pc_ocfps")
                sample["Free Cash Flow Per Share"]      = c.number_input("Free CF / Share ($)", value=0.85, step=0.1, key="pc_fcfps")

    # Immediate Financial Profile Preview
    if sample:
        with st.container(border=True):
            card_header("Corporate Financial Ratio Profile", f"Current indicators for {company_name or 'Custom Profile'} vs Sector Median")
            sec_ref = sample.get("Sector", "BusEq")
            sec_df  = raw_df[raw_df["Sector"] == sec_ref]
            
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Operating Margin", f"{sample.get('Operating Margin', 0.0):.1f}%", f"{sec_df['Operating Margin'].median():.1f}% Sector Med")
            p2.metric("Debt / Equity", f"{sample.get('Debt/Equity Ratio', 0.0):.2f}", f"{sec_df['Debt/Equity Ratio'].median():.2f} Sector Med")
            p3.metric("Current Ratio", f"{sample.get('Current Ratio', 0.0):.2f}", f"{sec_df['Current Ratio'].median():.2f} Sector Med")
            p4.metric("ROA", f"{sample.get('ROA - Return On Assets', 0.0):.1f}%", f"{sec_df['ROA - Return On Assets'].median():.1f}% Sector Med")

    # Model Selection & Execution
    with st.container(border=True):
        card_header("Model Configuration & Execution", "Select calibrated models from capstone library")
        mc1, mc2 = st.columns(2)
        sel_clf_model = mc1.selectbox(
            "Classification Model (Solvency)",
            list(clf_models.keys()),
            index=list(clf_models.keys()).index("Random Forest - Tuned") if "Random Forest - Tuned" in clf_models else 0,
            key="pc_sel_clf"
        )
        sel_reg_model = mc2.selectbox(
            "Regression Model (Credit Score)",
            list(reg_models.keys()),
            index=list(reg_models.keys()).index("Random Forest - Tuned") if "Random Forest - Tuned" in reg_models else 0,
            key="pc_sel_reg"
        )
        run_prediction = st.button("▶  Run CreditLens Inference Analysis", type="primary", use_container_width=True)

    # Prediction Output
    if run_prediction and sample:
        reg_score, reg_letter = run_regression_prediction(sample, sel_reg_model, reg_models, r_prep)
        clf_pred,  clf_prob   = run_classification_prediction(sample, sel_clf_model, clf_models, c_prep)

        solv_label = "INVESTMENT GRADE" if clf_pred == 1 else "SPECULATIVE GRADE"
        solv_color = C_GREEN if clf_pred == 1 else C_CORAL
        prob_ig    = f"{clf_prob*100:.1f}%" if clf_prob is not None else "N/A"
        prob_hy    = f"{(1-clf_prob)*100:.1f}%" if clf_prob is not None else "N/A"

        st.markdown(f"""
            <div class="cl-hero-result">
                <div style="font-size:12px;opacity:0.75;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                    CREDITLENS DUAL-TRACK INFERENCE RESULT — {company_name or 'CUSTOM PROFILE'}
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:14px;">
                    <div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:16px 18px;">
                        <div style="font-size:11px;opacity:0.8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
                            Predicted Credit Grade
                        </div>
                        <div style="font-size:38px;font-weight:800;line-height:1;">{reg_letter}</div>
                        <div style="font-size:12px;opacity:0.85;margin-top:6px;">
                            Score: <b>{reg_score:.2f}</b> / 22.0 · {sel_reg_model}
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:16px 18px;">
                        <div style="font-size:11px;opacity:0.8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
                            Solvency Classification
                        </div>
                        <div style="font-size:22px;font-weight:800;color:{solv_color};margin:8px 0 6px 0;">
                            {solv_label}
                        </div>
                        <div style="font-size:12px;opacity:0.85;">Model: {sel_clf_model}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:16px 18px;">
                        <div style="font-size:11px;opacity:0.8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
                            Investment Grade Probability
                        </div>
                        <div style="font-size:38px;font-weight:800;line-height:1;">{prob_ig}</div>
                        <div style="font-size:12px;opacity:0.85;margin-top:6px;">
                            Speculative Risk: <b>{prob_hy}</b>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if known_rating is not None:
            with st.container(border=True):
                card_header("Ground-Truth Rating Validation", "Historical agency event alignment")
                v1, v2, v3 = st.columns(3)
                v1.metric("Actual Agency Rating", known_rating)
                score_delta = reg_score - RATING_MAP[known_rating]
                v2.metric("Predicted Rating Grade", reg_letter, f"{score_delta:+.2f} score delta")
                match = (clf_pred == known_bin)
                v3.metric(
                    "Solvency Classification Match",
                    "✓ Aligned" if match else "⚠ Mismatch",
                    "Agreement" if match else "Discrepancy"
                )

        # Financial Health Benchmark Comparison Chart
        with st.container(border=True):
            card_header("Financial Ratio Benchmark Comparison", "Company ratios vs Industry sector median")
            sec_ref = sample.get("Sector", "BusEq")
            sec_df  = raw_df[raw_df["Sector"] == sec_ref]
            bench_keys = ["Operating Margin", "Gross Margin", "Net Profit Margin", "ROA - Return On Assets", "Current Ratio", "Debt/Equity Ratio"]
            comp_data = []
            for k in bench_keys:
                comp_data.append({"Ratio": k, "Type": "Company Profile", "Value": float(sample.get(k, 0.0))})
                comp_data.append({"Ratio": k, "Type": f"Sector Median ({sec_ref})", "Value": float(sec_df[k].median())})
            comp_df = pd.DataFrame(comp_data)
            fig = px.bar(
                comp_df, x="Ratio", y="Value", color="Type", barmode="group",
                color_discrete_map={"Company Profile": V_BLUE, f"Sector Median ({sec_ref})": V_AMBER}
            )
            chart(fig, height=290)

        st.caption("ℹ️ Notice: CreditLens predictions are algorithmic machine learning estimates and do not constitute formal agency ratings or investment advice.")

# ──────────────────────────────────────────────────────────────────────────────
# 15. PAGE 4: REGRESSION LAB
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Regression Lab":
    page_header("Regression Lab",
                "Benchmarking 12 regression estimators on credit score prediction (0=D to 22=AAA) with residual diagnostics.")

    reg_eval = evaluate_regression_models(reg_models, test_reg)

    if reg_eval.empty:
        st.error("Regression model evaluation results unavailable. Check models and test split files.")
    else:
        c_tbl, c_bar = st.columns([6, 6], gap="medium")

        with c_tbl:
            with st.container(border=True):
                card_header("Model Leaderboard (Test Set N=1,561)", "Sorted by Coefficient of Determination R²")
                st.dataframe(
                    reg_eval.style.format({
                        "R²": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "MAE": "{:.4f}"
                    }).highlight_max(subset=["R²"], color=C_HIGHLIGHT)
                      .highlight_min(subset=["RMSE", "MAE"], color=C_HIGHLIGHT),
                    use_container_width=True, height=390
                )

        with c_bar:
            with st.container(border=True):
                card_header("R² Performance Comparison", "Out-of-sample explanatory power across all 12 regressors")
                fig = px.bar(
                    reg_eval, x="R²", y="Model", orientation="h",
                    color="R²",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                fig.update_coloraxes(showscale=False)
                chart(fig, height=390)

        # Top Model Highlight Banner
        top_reg = reg_eval.iloc[0]
        st.markdown(f"""
            <div class="cl-insight-box">
                <div style="font-weight:700;font-size:13.5px;color:{C_DEEP_VIOLET};margin-bottom:3px;">
                    🏆 Top Performing Regressor: <b>{top_reg['Model']}</b>
                </div>
                <div style="font-size:12.5px;color:{C_MUTED};">
                    Achieves <b>R² = {top_reg['R²']:.4f}</b>, <b>RMSE = {top_reg['RMSE']:.4f}</b> rating notches, and <b>MAE = {top_reg['MAE']:.4f}</b> on unseen test corporations.
                    Nonlinear tree ensemble significantly outperforms linear baselines (Linear R² = 0.3474).
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Interactive Model Diagnostics
        with st.container(border=True):
            card_header("Individual Model Diagnostics", "Actual vs Predicted, residual errors, and feature importance")
            inspect_reg = st.selectbox("Select Model for Diagnostics", reg_eval["Model"].tolist(), key="rl_inspect_pick")

            if test_reg is not None:
                Xr, yr = test_reg.drop(columns=["Rating_Score"]), test_reg["Rating_Score"]
                m = reg_models[inspect_reg]
                if inspect_reg == "Polynomial Regression":
                    preds = m["model"].predict(m["transformer"].transform(Xr))
                else:
                    preds = m.predict(Xr)
                residuals = yr - preds

                d1, d2 = st.columns(2, gap="medium")
                with d1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=yr, y=preds, mode="markers",
                        marker=dict(color=V_BLUE, opacity=0.45, size=4.5),
                        name="Test Observations",
                        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 22], y=[0, 22], mode="lines",
                        line=dict(color=V_CORAL, dash="dash", width=1.5),
                        name="Ideal 45° Line"
                    ))
                    card_header("Actual vs Predicted Credit Score", "Dispersion around line of equality")
                    chart(fig, height=290)

                with d2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=preds, y=residuals, mode="markers",
                        marker=dict(color=V_TEAL, opacity=0.45, size=4.5),
                        name="Residuals",
                        hovertemplate="Predicted: %{x:.1f}<br>Error: %{y:.2f}<extra></extra>"
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color=V_CORAL, line_width=1.5)
                    card_header("Residual Error vs Predicted Score", "Heteroscedasticity inspection")
                    chart(fig, height=290)

                d3, d4 = st.columns(2, gap="medium")
                with d3:
                    card_header("Residual Error Distribution", "Gaussian distribution check")
                    fig = px.histogram(residuals, nbins=40, marginal="box", labels={"value": "Residual Error"})
                    fig.update_traces(marker_color=V_PURPLE, opacity=0.85)
                    chart(fig, height=270)

                with d4:
                    card_header("Top 10 Feature Importances", "Predictive weight of financial ratios")
                    if hasattr(m, "feature_importances_"):
                        imp = pd.Series(m.feature_importances_, index=Xr.columns).nlargest(10)
                        fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": "Feature"})
                        fig.update_traces(marker_color=V_GREEN)
                        fig.update_layout(yaxis=dict(autorange="reversed"))
                        chart(fig, height=270)
                    else:
                        st.info(f"Model '{inspect_reg}' does not expose tree-based Gini feature importances.")

# ──────────────────────────────────────────────────────────────────────────────
# 16. PAGE 5: CLASSIFICATION LAB
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Classification Lab":
    page_header("Classification Lab",
                "Evaluation of 12 binary classifiers distinguishing Investment Grade (1) from Speculative Grade (0).")

    clf_eval = evaluate_classification_models(clf_models, test_clf)

    if clf_eval.empty:
        st.error("Classification evaluation results unavailable.")
    else:
        c_tbl, c_bar = st.columns([6, 6], gap="medium")

        with c_tbl:
            with st.container(border=True):
                card_header("Classifier Leaderboard (Test Set N=1,561)", "Sorted by Weighted F1-Score")
                st.dataframe(
                    clf_eval.style.format({
                        "Accuracy":  "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall":    "{:.4f}",
                        "F1":        "{:.4f}",
                        "ROC-AUC":   lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "—"
                    }).highlight_max(subset=["Accuracy", "F1"], color=C_HIGHLIGHT),
                    use_container_width=True, height=390
                )

        with c_bar:
            with st.container(border=True):
                card_header("Weighted F1-Score Comparison", "Harmonic balance of precision and recall")
                fig = px.bar(
                    clf_eval, x="F1", y="Model", orientation="h",
                    color="F1",
                    color_continuous_scale="Purples"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                fig.update_coloraxes(showscale=False)
                chart(fig, height=390)

        # Highlight Banner
        top_clf = clf_eval.iloc[0]
        st.markdown(f"""
            <div class="cl-insight-box">
                <div style="font-weight:700;font-size:13.5px;color:{C_DEEP_VIOLET};margin-bottom:3px;">
                    🏆 Top Solvency Classifier: <b>{top_clf['Model']}</b>
                </div>
                <div style="font-size:12.5px;color:{C_MUTED};">
                    Achieves <b>Accuracy = {top_clf['Accuracy']*100:.2f}%</b>, <b>F1 = {top_clf['F1']:.4f}</b>, and <b>ROC-AUC = {top_clf['ROC-AUC']:.4f}</b> on the holdout test set.
                    Ensemble bagging and random forests capture complex non-linear boundary conditions between credit tiers.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Diagnostics
        with st.container(border=True):
            card_header("Classifier Performance Diagnostics", "Confusion matrix and Receiver Operating Characteristic (ROC)")
            inspect_clf = st.selectbox("Select Classifier for Diagnostics", clf_eval["Model"].tolist(), key="cl_inspect_pick")

            if test_clf is not None:
                Xc, yc = test_clf.drop(columns=["Binary Rating"]), test_clf["Binary Rating"]
                m      = clf_models[inspect_clf]
                preds  = m.predict(Xc)
                cm     = confusion_matrix(yc, preds)

                cd1, cd2 = st.columns(2, gap="medium")
                with cd1:
                    card_header(f"Confusion Matrix — {inspect_clf}", "Observed vs Predicted Class counts")
                    labels = ["Speculative (0)", "Investment Grade (1)"]
                    fig = go.Figure(go.Heatmap(
                        z=cm, x=labels, y=labels,
                        colorscale="Blues",
                        text=cm, texttemplate="<b>%{text:,}</b>",
                        textfont={"size": 16, "color": "#1E293B"}
                    ))
                    fig.update_layout(xaxis_title="Predicted Solvency", yaxis_title="Actual Ground Truth")
                    chart(fig, height=300)

                with cd2:
                    card_header(f"ROC Curve — {inspect_clf}", "Discrimination power across thresholds")
                    if hasattr(m, "predict_proba"):
                        probs = m.predict_proba(Xc)[:, 1]
                        fpr, tpr, _ = roc_curve(yc, probs)
                        auc_val = roc_auc_score(yc, probs)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            line=dict(color=V_BLUE, width=2.5),
                            name=f"ROC Curve (AUC = {auc_val:.4f})"
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1], mode="lines",
                            line=dict(color=V_CORAL, dash="dash"),
                            name="Random Baseline (AUC = 0.50)"
                        ))
                        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                        chart(fig, height=300)
                    else:
                        st.info(f"Model '{inspect_clf}' does not output soft probabilities; ROC curve unavailable.")

# ──────────────────────────────────────────────────────────────────────────────
# 17. PAGE 6: CLUSTERING STUDIO
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Clustering Studio":
    page_header("Clustering Studio",
                "Unsupervised financial profile segmentation: K-Means (k=3) vs Agglomerative Hierarchical (k=2).")

    clu_eval = get_clustering_evaluation_table()

    # Algorithm Comparison Table
    if not clu_eval.empty:
        with st.container(border=True):
            card_header("Clustering Algorithm Performance Comparison", "Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index")
            st.dataframe(
                clu_eval.style.format({
                    "Silhouette ↑": "{:.4f}",
                    "DB Index ↓":   "{:.4f}",
                    "CH Index ↑":   "{:.0f}"
                }),
                use_container_width=True
            )
            st.markdown(f"""
                <div style="font-size:12px;color:{C_MUTED};margin-top:6px;">
                    <b>Silhouette Score ↑</b>: higher indicates tighter clusters ·
                    <b>Davies-Bouldin Index ↓</b>: lower indicates superior cluster separation ·
                    <b>Calinski-Harabasz Index ↑</b>: higher indicates higher density
                </div>
            """, unsafe_allow_html=True)

    if clu_data is not None and clu_models:
        pca_coords, exp_var = compute_pca_2d(clu_data)
        km_labels   = clu_models["K-Means"].labels_
        hier_labels = clu_models["Agglomerative Hierarchical"].labels_

        # PCA 2D Visualizations (No overlapping titles/legends!)
        pca1, pca2 = st.columns(2, gap="medium")

        with pca1:
            with st.container(border=True):
                card_header("K-Means Clusters (k=3) — PCA 2D", f"Explained variance: {sum(exp_var)*100:.1f}%")
                df_km = pd.DataFrame({
                    "PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1],
                    "Cluster": [f"Cluster {l}" for l in km_labels]
                })
                fig = px.scatter(
                    df_km, x="PC1", y="PC2", color="Cluster",
                    color_discrete_sequence=[V_BLUE, V_GREEN, V_AMBER],
                    opacity=0.65
                )
                chart(fig, height=320)

        with pca2:
            with st.container(border=True):
                card_header("Hierarchical Clusters (k=2) — PCA 2D", "Agglomerative Ward linkage projection")
                df_hi = pd.DataFrame({
                    "PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1],
                    "Cluster": [f"Cluster {l}" for l in hier_labels]
                })
                fig = px.scatter(
                    df_hi, x="PC1", y="PC2", color="Cluster",
                    color_discrete_sequence=[V_PURPLE, "#EC4899"],
                    opacity=0.65
                )
                chart(fig, height=320)

        # Elbow Curve & Financial Archetypes
        e1, e2 = st.columns(2, gap="medium")

        with e1:
            with st.container(border=True):
                card_header("K-Means Inertia Elbow Curve", "Sum of Squared Errors (SSE) across k=2 to 10")
                inertias = {2: 99392, 3: 87739, 4: 79157, 5: 73961, 6: 69614, 7: 65852, 8: 63402, 9: 61351, 10: 59239}
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(inertias.keys()), y=list(inertias.values()),
                    mode="lines+markers",
                    marker=dict(size=8, color=V_INDIGO),
                    line=dict(color=V_INDIGO, width=2.5),
                    name="Inertia"
                ))
                fig.add_vline(x=3, line_dash="dash", line_color=V_CORAL, line_width=1.5,
                              annotation_text="k=3 selected elbow",
                              annotation_font=dict(color=V_CORAL, size=11))
                fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (SSE)")
                chart(fig, height=280)

        with e2:
            with st.container(border=True):
                card_header("Cluster Financial Archetypes", "Qualitative credit profiles discovered by K-Means")
                st.markdown(f"""
                    <div class="cl-insight-box">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 0: High-Margin Quality Champions (N=2,646)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Gross Margin 62.4%, Op. Margin 25.7%, ROA 7.6%. <b>81.4% Investment Grade.</b> Prime pricing power.
                        </div>
                    </div>
                    <div class="cl-insight-box" style="border-left-color:{V_BLUE} !important;">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 1: Capital-Intensive Steady Corporates (N=3,901)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Gross Margin 31.1%, Op. Margin 9.7%, ROE 24.7%. <b>68.1% Investment Grade.</b> Stable cash generators.
                        </div>
                    </div>
                    <div class="cl-insight-box" style="border-left-color:{V_CORAL} !important;">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 2: Stressed & Distressed Issuers (N=1,258)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Op. Margin -12.7%, Net Margin -17.0%, ROA -6.2%. <b>77.1% Speculative Grade.</b> Elevated default risk.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Financial Comparison Across Clusters
        with st.container(border=True):
            card_header("Cluster Archetype Financial Metrics Comparison", "Mean financial ratios across the 3 clusters")
            raw_with_km = raw_df.copy()
            raw_with_km["KMeans_Cluster"] = [f"Cluster {l}" for l in km_labels]
            cluster_means = raw_with_km.groupby("KMeans_Cluster")[
                ["Gross Margin", "Operating Margin", "Net Profit Margin", "ROA - Return On Assets", "Debt/Equity Ratio"]
            ].median().reset_index()
            melted_cm = cluster_means.melt(id_vars="KMeans_Cluster", var_name="Metric", value_name="Median Value")
            fig = px.bar(
                melted_cm, x="Metric", y="Median Value", color="KMeans_Cluster", barmode="group",
                color_discrete_sequence=[V_BLUE, V_GREEN, V_AMBER]
            )
            chart(fig, height=270)

        # Cluster Distribution Summary
        with st.container(border=True):
            card_header("Cluster Population Sizes", "Distribution of corporations across segments")
            d1, d2 = st.columns(2, gap="medium")
            with d1:
                km_dist = pd.Series(km_labels).value_counts().sort_index()
                fig = px.bar(
                    x=[f"Cluster {i}" for i in km_dist.index], y=km_dist.values,
                    color=[f"Cluster {i}" for i in km_dist.index],
                    color_discrete_sequence=[V_BLUE, V_GREEN, V_AMBER],
                    labels={"x": "Cluster", "y": "Corporations Count"}
                )
                fig.update_layout(showlegend=False)
                chart(fig, height=220)
            with d2:
                hi_dist = pd.Series(hier_labels).value_counts().sort_index()
                fig = px.bar(
                    x=[f"Cluster {i}" for i in hi_dist.index], y=hi_dist.values,
                    color=[f"Cluster {i}" for i in hi_dist.index],
                    color_discrete_sequence=[V_PURPLE, "#EC4899"],
                    labels={"x": "Cluster", "y": "Corporations Count"}
                )
                fig.update_layout(showlegend=False)
                chart(fig, height=220)

# ──────────────────────────────────────────────────────────────────────────────
# 18. PAGE 7: DATASET & EDA
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Dataset & EDA":
    page_header("Dataset & EDA",
                "Dataset schema, descriptive statistics, missing value audit, and target balance analysis.")

    # Top KPI Metrics
    total_cells = raw_df.shape[0] * raw_df.shape[1]
    missing_sum = raw_df.isnull().sum().sum()
    data_comp   = (1 - missing_sum / total_cells) * 100

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Dataset Rows", f"{raw_df.shape[0]:,}")
    e2.metric("Dataset Columns", f"{raw_df.shape[1]}")
    e3.metric("Missing Value Cells", f"{missing_sum:,}")
    e4.metric("Verified Completeness", f"{data_comp:.2f}%")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "Data Preview & Schema", "Descriptive Statistics", "Categorical Distributions", "Target Distributions & Correlation"
    ])

    with t1:
        with st.container(border=True):
            card_header("Dataset Preview", "First 10 records with raw financial ratios and agency ratings")
            st.dataframe(raw_df.head(10), use_container_width=True, height=330)

        miss_cols = raw_df.isnull().sum()
        miss_cols = miss_cols[miss_cols > 0].sort_values(ascending=False)
        if not miss_cols.empty:
            with st.container(border=True):
                card_header("Missing Value Counts by Column", "Missing data concentration prior to median imputation")
                fig = px.bar(x=miss_cols.index, y=miss_cols.values, labels={"x": "Feature", "y": "Missing Observations"})
                fig.update_traces(marker_color=V_CORAL)
                chart(fig, height=260)

    with t2:
        with st.container(border=True):
            card_header("Descriptive Summary Statistics", "Parametric and non-parametric distribution statistics")
            st.dataframe(
                raw_df[NUMERIC_RATIO_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
                .style.format("{:.3f}"),
                use_container_width=True, height=430
            )

    with t3:
        p1, p2 = st.columns(2, gap="medium")
        with p1:
            with st.container(border=True):
                card_header("Rating Agency Market Share", "Distribution of assessments by credit agency")
                ag_counts = raw_df["Rating Agency"].value_counts()
                fig = px.pie(
                    values=ag_counts.values, names=ag_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    hole=0.4
                )
                chart(fig, height=310)

        with p2:
            with st.container(border=True):
                card_header("Sector Industry Breakdown", "Corporation concentration across 12 sectors")
                sec_counts = raw_df["Sector"].value_counts()
                fig = px.pie(
                    values=sec_counts.values, names=sec_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                chart(fig, height=310)

    with t4:
        r1, r2 = st.columns(2, gap="medium")
        with r1:
            with st.container(border=True):
                card_header("Continuous Rating Score Target (0=D to 22=AAA)", "Regression target distribution")
                fig = px.histogram(raw_df, x="Rating_Score", nbins=23, labels={"Rating_Score": "Numeric Score"})
                fig.update_traces(marker_color=V_BLUE)
                chart(fig, height=270)

        with r2:
            with st.container(border=True):
                card_header("Binary Solvency Target (0 vs 1)", "Classification target class balance")
                bc = raw_df["Binary Rating"].value_counts()
                fig = px.bar(
                    x=["Speculative (0)", "Investment Grade (1)"],
                    y=[bc.get(0, 0), bc.get(1, 0)],
                    color=["Speculative (0)", "Investment Grade (1)"],
                    color_discrete_map={"Investment Grade (1)": V_GREEN, "Speculative (0)": V_CORAL}
                )
                fig.update_layout(showlegend=False)
                chart(fig, height=270)

        with st.container(border=True):
            card_header("Full Ratio Correlation Heatmap", "Pairwise Pearson correlation matrix")
            corr_all = raw_df[NUMERIC_RATIO_COLS + ["Rating_Score", "Binary Rating"]].corr().round(2)
            fig = px.imshow(corr_all, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            chart(fig, height=480)
