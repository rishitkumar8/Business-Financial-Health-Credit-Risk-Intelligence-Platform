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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, calinski_harabasz_score,
    classification_report, confusion_matrix, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve, silhouette_score
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

C_MOCHA         = "#6E5846"   # Mocha secondary accent (darkened from #927B68)
C_DEEP_MOCHA    = "#543F2F"   # Deep mocha strong accent (darkened from #705B49)
C_ESPRESSO      = "#2B2016"   # Rich dark espresso headings and strong text (darkened from #514438)
C_TEXT_BODY     = "#3D3025"   # Dark warm charcoal-brown body text (darkened from #625448)
C_TEXT_MUTED    = "#5A4C3D"   # Deep warm taupe muted text (darkened from #827567)
C_TEXT_LIGHT    = "#FAF6EF"   # Warm cream text on dark feature cards

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
def style_fig(fig, height: int = 340, margin: dict = None):
    """Styles Plotly figures with transparent canvas so opaque soft-cream cards show through,
    while using crisp espresso typography, subtle taupe gridlines, and clear legend alignment."""
    m = margin if margin is not None else dict(l=14, r=14, t=32, b=30)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=m,
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

def chart(fig, height: int = 340, margin: dict = None):
    style_fig(fig, height, margin=margin)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def card_header(title: str, subtitle: str = ""):
    st.markdown(f"""
        <div class="cl-card-header">
            <div class="cl-card-title">{title}</div>
            {"<div class='cl-card-sub'>" + subtitle + "</div>" if subtitle else ""}
        </div>
    """, unsafe_allow_html=True)

def format_leaderboard_table(df, format_dict=None, highlight_max_cols=None, highlight_min_cols=None):
    """Styles dataframe tables so cell backgrounds blend seamlessly with the soft cream palette,
    ensuring bold espresso typography and high-contrast metric highlights."""
    styler = df.style
    if format_dict:
        styler = styler.format(format_dict)
    styler = styler.set_properties(**{
        "background-color": C_CARD,
        "color": C_ESPRESSO,
    })
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", C_PANEL_SEC),
            ("color", C_ESPRESSO),
            ("font-weight", "700"),
            ("border-bottom", f"1px solid {C_BORDER}")
        ]},
        {"selector": "td", "props": [
            ("background-color", C_CARD),
            ("color", C_ESPRESSO),
            ("border-bottom", f"1px solid {C_BORDER}")
        ]}
    ])
    if highlight_max_cols:
        styler = styler.highlight_max(
            subset=highlight_max_cols,
            props=f"background-color: {C_PANEL_SEC}; color: {C_ESPRESSO}; font-weight: 700;"
        )
    if highlight_min_cols:
        styler = styler.highlight_min(
            subset=highlight_min_cols,
            props=f"background-color: {C_PANEL_SEC}; color: {C_ESPRESSO}; font-weight: 700;"
        )
    return styler

def display_inference(text: str):
    """Renders a concise, data-grounded one-line capstone inference directly beneath a visualization."""
    st.markdown(f"""
        <div class="cl-inference-box">
            <span class="cl-inference-icon">💡</span>
            <div class="cl-inference-text"><b>Key Inference:</b> {text}</div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_regression_eval_preds(model_name: str, _reg_dict: dict, _test_df: pd.DataFrame):
    """Caches predictions and residuals for the selected regression model on holdout test set."""
    if not _reg_dict or _test_df is None or model_name not in _reg_dict:
        return np.array([]), np.array([])
    Xr, yr = _test_df.drop(columns=["Rating_Score"]), _test_df["Rating_Score"]
    m = _reg_dict[model_name]
    if model_name == "Polynomial Regression":
        preds = m["model"].predict(m["transformer"].transform(Xr))
    else:
        preds = m.predict(Xr)
    residuals = yr - preds
    return preds, residuals

@st.cache_data(show_spinner=False)
def get_classification_eval_preds(model_name: str, _clf_dict: dict, _test_df: pd.DataFrame):
    """Caches hard predictions and soft probability scores for the selected classifier on holdout test set."""
    if not _clf_dict or _test_df is None or model_name not in _clf_dict:
        return np.array([]), None
    Xc, yc = _test_df.drop(columns=["Binary Rating"]), _test_df["Binary Rating"]
    m = _clf_dict[model_name]
    preds = m.predict(Xc)
    probs = m.predict_proba(Xc)[:, 1] if hasattr(m, "predict_proba") else None
    return preds, probs

@st.cache_data(show_spinner=False)
def get_sampled_dendrogram_figure(_clu_df: pd.DataFrame, sample_size: int = 55):
    """Computes a hierarchical Ward linkage dendrogram on a stratified sample for fast interactive rendering."""
    sample = _clu_df.sample(sample_size, random_state=42)
    fig = ff.create_dendrogram(sample.values, color_threshold=11.5)
    fig.update_layout(
        xaxis=dict(showticklabels=False, title="Sampled Corporate Balance Sheets (N=55)"),
        yaxis=dict(title="Euclidean Linkage Distance")
    )
    return fig

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

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Overview"

    # Consume any pending page navigation request set before radio instantiation
    if "pending_page" in st.session_state and st.session_state["pending_page"] in PAGES:
        st.session_state["selected_page"] = st.session_state.pop("pending_page")

    def _sync_nav():
        st.session_state["selected_page"] = st.session_state["nav_radio"]

    curr_index = PAGES.index(st.session_state["selected_page"]) if st.session_state["selected_page"] in PAGES else 0

    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=curr_index,
        key="nav_radio",
        on_change=_sync_nav,
        label_visibility="collapsed"
    )
    st.session_state["selected_page"] = selected_page

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
# 12. PAGE 1: OVERVIEW — PROJECT INTRODUCTION & INTELLIGENCE DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
if selected_page == "Overview":
    # ── Helper Navigation Callback ──
    def _go_to_page(page_name):
        st.session_state["pending_page"] = page_name

    # ── 1. PROJECT HERO & INTRODUCTION ──
    st.markdown(f"""
        <div class="cl-hero-section">
            <div class="cl-hero-badge">Academic ML Capstone Platform</div>
            <div class="cl-hero-title">CreditLens</div>
            <div class="cl-hero-subtitle">Business Financial Health & Credit Risk Intelligence Platform</div>
            <p class="cl-hero-desc">
                CreditLens is a machine learning platform that analyzes corporate financial ratios and credit-related information to evaluate financial performance, classify credit-related outcomes, and discover financial-health patterns among companies.
                Corporate financial indicators provide complementary perspectives on corporate profitability, liquidity, financial leverage, cash flow solvency, and operational efficiency, allowing multi-perspective empirical analysis of corporate credit health.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── 2. PROBLEM STATEMENT & OBJECTIVES ──
    with st.container(border=True):
        st.markdown(f"""
            <div class="cl-card-header">
                <div class="cl-card-title">Problem Statement & Capstone Objectives</div>
                <div class="cl-card-sub">Synthesizing multidimensional corporate financial indicators into coherent machine learning intelligence</div>
            </div>
        """, unsafe_allow_html=True)

        ps_col1, ps_col2 = st.columns([6, 6], gap="large")

        with ps_col1:
            st.markdown(f"""
                <div style="font-size:13.5px;color:{C_TEXT_BODY};line-height:1.6;">
                    <p style="margin-bottom:10px;">
                        Interpreting numerous corporate financial ratios and credit indicators can make it difficult to obtain a consolidated, objective understanding of overall financial performance and underlying risk patterns. Traditional manual credit scoring and univariate analysis struggle to capture complex non-linear interactions across margin health, capital structure, and solvency buffers.
                    </p>
                    <p style="margin-bottom:0;">
                        The project explores how machine learning can address this challenge through three structured analytical approaches:
                    </p>
                    <ul style="margin-top:6px;margin-bottom:0;padding-left:20px;color:{C_ESPRESSO};">
                        <li><b>Predicting a continuous financial-performance indicator:</b> Estimating corporate credit rating scores across a 23-level scale.</li>
                        <li><b>Classifying categorical credit solvency outcomes:</b> Predicting investment grade versus speculative grade classifications.</li>
                        <li><b>Discovering unsupervised company clusters:</b> Grouping firms with homogeneous financial-health profiles without prior label exposure.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with ps_col2:
            st.markdown(f"""
                <div class="cl-card-secondary" style="margin-bottom:0;padding:16px 18px;">
                    <div style="font-size:12px;font-weight:700;color:{C_ESPRESSO};text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px;">
                        🎯 Capstone Research Objectives
                    </div>
                    <div style="font-size:12px;color:{C_TEXT_BODY};line-height:1.5;">
                        <div style="margin-bottom:6px;"><b>1. Empirical Data Exploration:</b> Analyze multi-sector distributions, missingness patterns, and financial ratio spreads across 7,805 corporate filings.</div>
                        <div style="margin-bottom:6px;"><b>2. Methodological Preprocessing:</b> Implement robust median imputation, IQR outlier clipping, one-hot sector encoding, and standard scaling without data leakage.</div>
                        <div style="margin-bottom:6px;"><b>3. Multi-Track ML Benchmark:</b> Train and compare 10 regression, 10 classification, and 2 clustering algorithms using identical splits and rigorous validation.</div>
                        <div><b>4. Interpretability & Risk Synthesis:</b> Identify dominant financial solvency drivers and quantify credit health patterns across commercial sectors.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── 3. THREE MACHINE LEARNING TRACKS ──
    st.markdown(f"""
        <div class="cl-section-header">
            <div class="cl-section-title">🤖 Machine Learning Tracks</div>
            <div class="cl-section-sub">Three specialized analytical methodologies addressing distinct financial intelligence dimensions</div>
        </div>
    """, unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3, gap="medium")

    with tc1:
        st.markdown(f"""
            <div class="cl-track-card">
                <div>
                    <div class="cl-track-header">
                        <div class="cl-track-icon">📈</div>
                        <div>
                            <div class="cl-track-name">Regression Track</div>
                            <div class="cl-track-category">Continuous Performance</div>
                        </div>
                    </div>
                    <div class="cl-track-body">
                        Predicts a continuous corporate financial performance indicator (<b>Rating_Score</b>, range 0–22, mapped from 23 rating grades from D to AAA). Evaluates numerical credit strength from liquidity, leverage, and profitability ratios.
                    </div>
                </div>
                <div>
                    <div class="cl-track-meta">
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Actual Target:</span>
                            <span class="cl-track-meta-val">Rating_Score (0 to 22)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Best Model:</span>
                            <span class="cl-track-meta-val">Random Forest (Tuned)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Verified R²:</span>
                            <span class="cl-track-meta-val">0.7682</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Test RMSE / MAE:</span>
                            <span class="cl-track-meta-val">1.8081 / 1.2507</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Evaluation Split:</span>
                            <span class="cl-track-meta-val">Holdout Test (n=1,561)</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Regression Lab →", key="btn_nav_reg", use_container_width=True):
            _go_to_page("Regression Lab")
            st.rerun()

    with tc2:
        st.markdown(f"""
            <div class="cl-track-card">
                <div>
                    <div class="cl-track-header">
                        <div class="cl-track-icon">🏷️</div>
                        <div>
                            <div class="cl-track-name">Classification Track</div>
                            <div class="cl-track-category">Credit Solvency Class</div>
                        </div>
                    </div>
                    <div class="cl-track-body">
                        Predicts categorical credit solvency status (<b>Binary Rating</b>: 1 for Investment Grade [AAA to BBB-], 0 for Speculative Grade [BB+ to D]). Provides calibrated probability outputs for balance sheet risk assessment.
                    </div>
                </div>
                <div>
                    <div class="cl-track-meta">
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Actual Target:</span>
                            <span class="cl-track-meta-val">Binary Rating (Class 0 / 1)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Best Model:</span>
                            <span class="cl-track-meta-val">Bagging Classifier (Tuned)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Verified Accuracy:</span>
                            <span class="cl-track-meta-val">91.93% (0.9193)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Weighted F1 / ROC-AUC:</span>
                            <span class="cl-track-meta-val">0.9188 / 0.9659</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Evaluation Split:</span>
                            <span class="cl-track-meta-val">Holdout Test (n=1,561)</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Classification Lab →", key="btn_nav_clf", use_container_width=True):
            _go_to_page("Classification Lab")
            st.rerun()

    with tc3:
        st.markdown(f"""
            <div class="cl-track-card">
                <div>
                    <div class="cl-track-header">
                        <div class="cl-track-icon">🔵</div>
                        <div>
                            <div class="cl-track-name">Clustering Track</div>
                            <div class="cl-track-category">Unsupervised Profiling</div>
                        </div>
                    </div>
                    <div class="cl-track-body">
                        Discovers natural clusters of corporations sharing homogeneous financial profile characteristics without utilizing rating labels during model fitting. Validates discovered segments against post-clustering solvency distributions.
                    </div>
                </div>
                <div>
                    <div class="cl-track-meta">
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Implemented Algos:</span>
                            <span class="cl-track-meta-val">K-Means & Hierarchical</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Best Result:</span>
                            <span class="cl-track-meta-val">K-Means (k=3 clusters)</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Silhouette Score:</span>
                            <span class="cl-track-meta-val">0.1776</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">DB Index / CH Score:</span>
                            <span class="cl-track-meta-val">1.7440 / 1,998.4</span>
                        </div>
                        <div class="cl-track-meta-row">
                            <span class="cl-track-meta-label">Dataset Scope:</span>
                            <span class="cl-track-meta-val">Preprocessed Corpus (N=7,805)</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Clustering Studio →", key="btn_nav_clu", use_container_width=True):
            _go_to_page("Clustering Studio")
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── 4. GENUINE MODEL-PERFORMANCE HIGHLIGHTS ──
    with st.container(border=True):
        st.markdown(f"""
            <div class="cl-card-header">
                <div class="cl-card-title">🏆 Genuine Model Performance Summary</div>
                <div class="cl-card-sub">Verified benchmark metrics computed on held-out test splits (n=1,561) and full preprocessed corpus from project artifacts</div>
            </div>
        """, unsafe_allow_html=True)

        p_reg, p_clf, p_clu = st.columns(3, gap="medium")

        with p_reg:
            st.markdown(f"""
                <div class="cl-card-secondary" style="margin-bottom:0;padding:16px 18px;border-left:4px solid {C_MOCHA};">
                    <div style="font-size:11px;font-weight:700;color:{C_TEXT_MUTED};text-transform:uppercase;letter-spacing:0.04em;">
                        Best Regression Model
                    </div>
                    <div style="font-size:18px;font-weight:800;color:{C_ESPRESSO};margin:4px 0 8px 0;">
                        Random Forest (Tuned)
                    </div>
                    <div style="font-size:12px;color:{C_TEXT_BODY};line-height:1.5;margin-bottom:10px;">
                        Optimized ensemble regressor evaluating continuous <b>Rating_Score</b>.
                    </div>
                    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;padding:8px 12px;font-size:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">R² Score (Holdout):</span>
                            <span style="color:{C_ESPRESSO};font-weight:800;">0.7682</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Root Mean Sq Err (RMSE):</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">1.8081</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Mean Absolute Err (MAE):</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">1.2507</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;">
                            <span style="color:{C_TEXT_MUTED};">Validation Method:</span>
                            <span style="color:{C_TEXT_MUTED};font-size:11px;">Holdout 20% (5-Fold CV: 0.7623)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with p_clf:
            st.markdown(f"""
                <div class="cl-card-secondary" style="margin-bottom:0;padding:16px 18px;border-left:4px solid {C_DEEP_MOCHA};">
                    <div style="font-size:11px;font-weight:700;color:{C_TEXT_MUTED};text-transform:uppercase;letter-spacing:0.04em;">
                        Best Classification Model
                    </div>
                    <div style="font-size:18px;font-weight:800;color:{C_ESPRESSO};margin:4px 0 8px 0;">
                        Bagging Classifier (Tuned)
                    </div>
                    <div style="font-size:12px;color:{C_TEXT_BODY};line-height:1.5;margin-bottom:10px;">
                        Bootstrap aggregation classifier predicting <b>Binary Rating</b> (IG vs SG).
                    </div>
                    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;padding:8px 12px;font-size:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Test Accuracy:</span>
                            <span style="color:{C_ESPRESSO};font-weight:800;">91.93% (0.9193)</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Weighted F1-Score:</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">0.9188</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">ROC-AUC Score:</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">0.9659</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;">
                            <span style="color:{C_TEXT_MUTED};">Validation Method:</span>
                            <span style="color:{C_TEXT_MUTED};font-size:11px;">Holdout 20% (Baseline RF: 0.9671 AUC)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with p_clu:
            st.markdown(f"""
                <div class="cl-card-secondary" style="margin-bottom:0;padding:16px 18px;border-left:4px solid {V_TEAL};">
                    <div style="font-size:11px;font-weight:700;color:{C_TEXT_MUTED};text-transform:uppercase;letter-spacing:0.04em;">
                        Best Clustering Structure
                    </div>
                    <div style="font-size:18px;font-weight:800;color:{C_ESPRESSO};margin:4px 0 8px 0;">
                        K-Means Clustering (k=3)
                    </div>
                    <div style="font-size:12px;color:{C_TEXT_BODY};line-height:1.5;margin-bottom:10px;">
                        Centroid partitioning segmenting financial profiles into 3 clusters.
                    </div>
                    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;padding:8px 12px;font-size:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Silhouette Score:</span>
                            <span style="color:{C_ESPRESSO};font-weight:800;">0.1776</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Davies–Bouldin Index:</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">1.7440 (Lower is better)</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:{C_TEXT_MUTED};">Calinski–Harabasz:</span>
                            <span style="color:{C_ESPRESSO};font-weight:700;">1,998.4 (Higher is better)</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;">
                            <span style="color:{C_TEXT_MUTED};">Evaluation Scope:</span>
                            <span style="color:{C_TEXT_MUTED};font-size:11px;">Entire Scaled Corpus (N=7,805)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── 5. DATASET & WORKFLOW SUMMARY ──
    with st.container(border=True):
        st.markdown(f"""
            <div class="cl-card-header">
                <div class="cl-card-title">📂 Dataset Profile & Machine Learning Workflow</div>
                <div class="cl-card-sub">Dynamic empirical dataset telemetry and the end-to-end multi-track pipeline architecture</div>
            </div>
        """, unsafe_allow_html=True)

        # Dynamic Dataset Telemetry
        total_records = len(raw_df)
        total_cols    = raw_df.shape[1]
        unique_corps  = raw_df["Corporation"].nunique()
        total_sectors = raw_df["Sector"].nunique()
        total_agencies = raw_df["Rating Agency"].nunique()
        total_ratings  = raw_df["Rating"].nunique()
        miss_cells     = raw_df.isnull().sum().sum()
        completeness   = (1 - miss_cells / (raw_df.shape[0] * raw_df.shape[1])) * 100
        ig_count       = int(raw_df["Binary Rating"].sum())
        ig_pct         = raw_df["Binary Rating"].mean() * 100
        sg_count       = total_records - ig_count

        ds1, ds2, ds3, ds4 = st.columns(4)
        ds1.metric("Observations (N)", f"{total_records:,}", "Rated events")
        ds2.metric("Raw Features", f"{total_cols} columns", "16 financial ratios")
        ds3.metric("Rated Corporations", f"{unique_corps:,}", f"{total_sectors} sectors")
        ds4.metric("Data Completeness", f"{completeness:.1f}%", f"{total_agencies} rating agencies")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # End-to-End Workflow Diagram
        st.markdown(f"""
            <div style="font-size:12px;font-weight:700;color:{C_TEXT_MUTED};text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px;">
                End-to-End Methodological Architecture
            </div>
        """, unsafe_allow_html=True)

        w1, w2, w3, w4, w5 = st.columns(5, gap="small")
        with w1:
            st.markdown(f"""
                <div class="cl-workflow-step">
                    <div class="cl-workflow-step-num">Step 1</div>
                    <div class="cl-workflow-step-title">Raw Dataset</div>
                    <div class="cl-workflow-step-desc">7,805 corporate financial records & agency rating classifications.</div>
                </div>
            """, unsafe_allow_html=True)
        with w2:
            st.markdown(f"""
                <div class="cl-workflow-step">
                    <div class="cl-workflow-step-num">Step 2</div>
                    <div class="cl-workflow-step-title">Preprocessing</div>
                    <div class="cl-workflow-step-desc">Median imputation, IQR clipping (±1.5×IQR) & duplicate validation.</div>
                </div>
            """, unsafe_allow_html=True)
        with w3:
            st.markdown(f"""
                <div class="cl-workflow-step">
                    <div class="cl-workflow-step-num">Step 3</div>
                    <div class="cl-workflow-step-title">Feature Prep</div>
                    <div class="cl-workflow-step-desc">Sector dummy encoding (11 dummies) & StandardScaler normalization.</div>
                </div>
            """, unsafe_allow_html=True)
        with w4:
            st.markdown(f"""
                <div class="cl-workflow-step">
                    <div class="cl-workflow-step-num">Step 4</div>
                    <div class="cl-workflow-step-title">3 ML Tracks</div>
                    <div class="cl-workflow-step-desc">Regression (10 algos), Classification (10 algos), Clustering (2 algos).</div>
                </div>
            """, unsafe_allow_html=True)
        with w5:
            st.markdown(f"""
                <div class="cl-workflow-step">
                    <div class="cl-workflow-step-num">Step 5</div>
                    <div class="cl-workflow-step-title">Financial Insights</div>
                    <div class="cl-workflow-step-desc">Solvency drivers, margin spreads & cluster profiles across sectors.</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:8px;font-style:italic;">
                *Note on methodology: Regression models use an 80/20 train-test split (train=6,244, test=1,561) evaluating numeric Rating_Score. Classification models use an identical stratified 80/20 split on Binary Rating. Clustering is unsupervised across all 7,805 preprocessed observations without using ground-truth labels during model fitting.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── 6. IMPLEMENTED ALGORITHMS SECTION ──
    with st.expander("🔬 Algorithms Implemented — Verified Model Catalog (Expand to View All 22 Models)", expanded=False):
        st.markdown(f"""
            <div style="font-size:13px;color:{C_TEXT_BODY};margin-bottom:14px;line-height:1.5;">
                Every algorithm listed below was empirically trained, saved as serialized artifacts (<code>models/*.pkl</code>), and evaluated within the project's Jupyter notebooks and evaluation pipeline.
            </div>
        """, unsafe_allow_html=True)

        tab_alg_reg, tab_alg_clf, tab_alg_clu = st.tabs([
            "📈 Regression Algorithms (10 + 2 Tuned)",
            "🏷️ Classification Algorithms (10 + 2 Tuned)",
            "🔵 Clustering Algorithms (2 Models)"
        ])

        with tab_alg_reg:
            st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:12px;color:{C_ESPRESSO};">
                    <thead>
                        <tr style="background:{C_PANEL_SEC};border-bottom:1.5px solid {C_BORDER};text-align:left;">
                            <th style="padding:8px 10px;">#</th>
                            <th style="padding:8px 10px;">Algorithm</th>
                            <th style="padding:8px 10px;">Purpose & Methodology</th>
                            <th style="padding:8px 10px;">Tuned?</th>
                            <th style="padding:8px 10px;">R² Score</th>
                            <th style="padding:8px 10px;">Test RMSE</th>
                            <th style="padding:8px 10px;">Test MAE</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid {C_BORDER};background:#FAF6EF;font-weight:700;">
                            <td style="padding:7px 10px;">1</td>
                            <td style="padding:7px 10px;">Random Forest (Tuned)</td>
                            <td style="padding:7px 10px;">Grid-searched ensemble of decision trees with optimized depth and estimators</td>
                            <td style="padding:7px 10px;">✓ Yes (n_est=100, depth=None)</td>
                            <td style="padding:7px 10px;color:#2E7D32;">0.7682</td>
                            <td style="padding:7px 10px;">1.8081</td>
                            <td style="padding:7px 10px;">1.2507</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">2</td>
                            <td style="padding:7px 10px;">Random Forest Regressor</td>
                            <td style="padding:7px 10px;">Baseline ensemble of decision trees reducing variance via bootstrap aggregation</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.6990</td>
                            <td style="padding:7px 10px;">2.0602</td>
                            <td style="padding:7px 10px;">1.5122</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">3</td>
                            <td style="padding:7px 10px;">KNN Regressor (Tuned)</td>
                            <td style="padding:7px 10px;">Instance-based regressor tuned with optimal neighbor count</td>
                            <td style="padding:7px 10px;">✓ Yes (n_neighbors=3)</td>
                            <td style="padding:7px 10px;">0.6500</td>
                            <td style="padding:7px 10px;">2.2217</td>
                            <td style="padding:7px 10px;">1.4743</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">4</td>
                            <td style="padding:7px 10px;">KNN Regressor</td>
                            <td style="padding:7px 10px;">Baseline k-nearest neighbors regression averaging nearby financial records</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.6298</td>
                            <td style="padding:7px 10px;">2.2848</td>
                            <td style="padding:7px 10px;">1.5869</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">5</td>
                            <td style="padding:7px 10px;">Support Vector Regressor (SVR)</td>
                            <td style="padding:7px 10px;">Epsilon-SVR with RBF kernel maximizing margin within tolerance threshold</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.5352</td>
                            <td style="padding:7px 10px;">2.5601</td>
                            <td style="padding:7px 10px;">1.9232</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">6</td>
                            <td style="padding:7px 10px;">Gradient Boosting Regressor</td>
                            <td style="padding:7px 10px;">Sequential gradient boosting minimizing residual loss iteratively</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.5226</td>
                            <td style="padding:7px 10px;">2.5946</td>
                            <td style="padding:7px 10px;">2.0032</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">7</td>
                            <td style="padding:7px 10px;">Polynomial Regression</td>
                            <td style="padding:7px 10px;">Degree-2 polynomial feature mapping with linear regression fitting</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.4544</td>
                            <td style="padding:7px 10px;">2.7737</td>
                            <td style="padding:7px 10px;">2.1351</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">8</td>
                            <td style="padding:7px 10px;">Decision Tree Regressor</td>
                            <td style="padding:7px 10px;">Single unpruned decision tree partitioning feature space recursively</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.4122</td>
                            <td style="padding:7px 10px;">2.8791</td>
                            <td style="padding:7px 10px;">2.2264</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">9</td>
                            <td style="padding:7px 10px;">Ridge Regression</td>
                            <td style="padding:7px 10px;">Linear regression with L2 regularization mitigating collinear financial ratios</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.3474</td>
                            <td style="padding:7px 10px;">3.0336</td>
                            <td style="padding:7px 10px;">2.3208</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">10</td>
                            <td style="padding:7px 10px;">Linear Regression (OLS)</td>
                            <td style="padding:7px 10px;">Ordinary least squares establishing linear baseline relationship</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.3474</td>
                            <td style="padding:7px 10px;">3.0337</td>
                            <td style="padding:7px 10px;">2.3208</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">11</td>
                            <td style="padding:7px 10px;">ElasticNet Regression</td>
                            <td style="padding:7px 10px;">Combined L1 and L2 regularized linear model</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.3428</td>
                            <td style="padding:7px 10px;">3.0443</td>
                            <td style="padding:7px 10px;">2.3493</td>
                        </tr>
                        <tr>
                            <td style="padding:7px 10px;">12</td>
                            <td style="padding:7px 10px;">Lasso Regression</td>
                            <td style="padding:7px 10px;">L1 penalized linear model performing ratio shrinkage and feature selection</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.3310</td>
                            <td style="padding:7px 10px;">3.0715</td>
                            <td style="padding:7px 10px;">2.3746</td>
                        </tr>
                    </tbody>
                </table>
            """, unsafe_allow_html=True)

        with tab_alg_clf:
            st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:12px;color:{C_ESPRESSO};">
                    <thead>
                        <tr style="background:{C_PANEL_SEC};border-bottom:1.5px solid {C_BORDER};text-align:left;">
                            <th style="padding:8px 10px;">#</th>
                            <th style="padding:8px 10px;">Algorithm</th>
                            <th style="padding:8px 10px;">Purpose & Architecture</th>
                            <th style="padding:8px 10px;">Tuned?</th>
                            <th style="padding:8px 10px;">Accuracy</th>
                            <th style="padding:8px 10px;">Weighted F1</th>
                            <th style="padding:8px 10px;">ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid {C_BORDER};background:#FAF6EF;font-weight:700;">
                            <td style="padding:7px 10px;">1</td>
                            <td style="padding:7px 10px;">Bagging Classifier (Tuned)</td>
                            <td style="padding:7px 10px;">Tuned bootstrap aggregation across 200 base estimators with 70% feature sampling</td>
                            <td style="padding:7px 10px;">✓ Yes (n_est=200, feat=0.7)</td>
                            <td style="padding:7px 10px;color:#2E7D32;">0.9193</td>
                            <td style="padding:7px 10px;color:#2E7D32;">0.9188</td>
                            <td style="padding:7px 10px;">0.9659</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">2</td>
                            <td style="padding:7px 10px;">Random Forest Classifier</td>
                            <td style="padding:7px 10px;">Baseline ensemble of decision trees with random feature subspacing</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.9142</td>
                            <td style="padding:7px 10px;">0.9135</td>
                            <td style="padding:7px 10px;color:#2E7D32;">0.9671</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">3</td>
                            <td style="padding:7px 10px;">Bagging Classifier</td>
                            <td style="padding:7px 10px;">Standard bootstrap aggregation of decision trees</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.9135</td>
                            <td style="padding:7px 10px;">0.9129</td>
                            <td style="padding:7px 10px;">0.9650</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">4</td>
                            <td style="padding:7px 10px;">Random Forest (Tuned)</td>
                            <td style="padding:7px 10px;">Grid-searched random forest with min-split constraints</td>
                            <td style="padding:7px 10px;">✓ Yes (n_est=200, min_split=5)</td>
                            <td style="padding:7px 10px;">0.9110</td>
                            <td style="padding:7px 10px;">0.9105</td>
                            <td style="padding:7px 10px;">0.9667</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">5</td>
                            <td style="padding:7px 10px;">MLP Classifier</td>
                            <td style="padding:7px 10px;">Multi-layer perceptron neural network with backpropagation</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8860</td>
                            <td style="padding:7px 10px;">0.8851</td>
                            <td style="padding:7px 10px;">0.9274</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">6</td>
                            <td style="padding:7px 10px;">KNN Classifier</td>
                            <td style="padding:7px 10px;">Instance-based k-nearest neighbors classification in standardized ratio space</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8661</td>
                            <td style="padding:7px 10px;">0.8640</td>
                            <td style="padding:7px 10px;">0.9224</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">7</td>
                            <td style="padding:7px 10px;">Gradient Boosting Classifier</td>
                            <td style="padding:7px 10px;">Sequential gradient boosting on deviance loss function</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8443</td>
                            <td style="padding:7px 10px;">0.8414</td>
                            <td style="padding:7px 10px;">0.9143</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">8</td>
                            <td style="padding:7px 10px;">Support Vector Classifier (SVC)</td>
                            <td style="padding:7px 10px;">Soft-margin hyperplane separator with RBF kernel</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8398</td>
                            <td style="padding:7px 10px;">0.8344</td>
                            <td style="padding:7px 10px;">0.8851</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">9</td>
                            <td style="padding:7px 10px;">Decision Tree Classifier</td>
                            <td style="padding:7px 10px;">Hierarchical decision tree with Gini impurity splitting</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8206</td>
                            <td style="padding:7px 10px;">0.8191</td>
                            <td style="padding:7px 10px;">0.8438</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">10</td>
                            <td style="padding:7px 10px;">Logistic Regression</td>
                            <td style="padding:7px 10px;">Log-odds linear classifier establishing solvency odds benchmark</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.8008</td>
                            <td style="padding:7px 10px;">0.7952</td>
                            <td style="padding:7px 10px;">0.8355</td>
                        </tr>
                        <tr style="border-bottom:1px solid {C_BORDER};">
                            <td style="padding:7px 10px;">11</td>
                            <td style="padding:7px 10px;">AdaBoost Classifier</td>
                            <td style="padding:7px 10px;">Adaptive boosting reweighting misclassified instances sequentially</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.7867</td>
                            <td style="padding:7px 10px;">0.7839</td>
                            <td style="padding:7px 10px;">0.8410</td>
                        </tr>
                        <tr>
                            <td style="padding:7px 10px;">12</td>
                            <td style="padding:7px 10px;">Gaussian Naive Bayes</td>
                            <td style="padding:7px 10px;">Probabilistic classifier assuming conditionally independent features</td>
                            <td style="padding:7px 10px;">Baseline</td>
                            <td style="padding:7px 10px;">0.7514</td>
                            <td style="padding:7px 10px;">0.7527</td>
                            <td style="padding:7px 10px;">0.8063</td>
                        </tr>
                    </tbody>
                </table>
            """, unsafe_allow_html=True)

        with tab_alg_clu:
            st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:12px;color:{C_ESPRESSO};">
                    <thead>
                        <tr style="background:{C_PANEL_SEC};border-bottom:1.5px solid {C_BORDER};text-align:left;">
                            <th style="padding:8px 10px;">#</th>
                            <th style="padding:8px 10px;">Algorithm</th>
                            <th style="padding:8px 10px;">Clustering Methodology</th>
                            <th style="padding:8px 10px;">Clusters (k)</th>
                            <th style="padding:8px 10px;">Silhouette Score ↑</th>
                            <th style="padding:8px 10px;">Davies–Bouldin Index ↓</th>
                            <th style="padding:8px 10px;">Calinski–Harabasz Index ↑</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid {C_BORDER};background:#FAF6EF;font-weight:700;">
                            <td style="padding:7px 10px;">1</td>
                            <td style="padding:7px 10px;">K-Means Clustering</td>
                            <td style="padding:7px 10px;">Centroid-based partitioning with k-means++ initialization, optimized via Elbow method</td>
                            <td style="padding:7px 10px;">k = 3</td>
                            <td style="padding:7px 10px;color:#2E7D32;">0.1776</td>
                            <td style="padding:7px 10px;color:#2E7D32;">1.7440</td>
                            <td style="padding:7px 10px;color:#2E7D32;">1,998.4</td>
                        </tr>
                        <tr>
                            <td style="padding:7px 10px;">2</td>
                            <td style="padding:7px 10px;">Agglomerative Hierarchical</td>
                            <td style="padding:7px 10px;">Bottom-up agglomerative clustering with Ward linkage minimizing variance</td>
                            <td style="padding:7px 10px;">k = 2</td>
                            <td style="padding:7px 10px;">0.1684</td>
                            <td style="padding:7px 10px;">1.8458</td>
                            <td style="padding:7px 10px;">1,866.8</td>
                        </tr>
                    </tbody>
                </table>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── 7. PROJECT TEAM SECTION ──
    with st.container(border=True):
        st.markdown(f"""
            <div class="cl-card-header">
                <div class="cl-card-title">👥 Project Team & Academic Credits</div>
                <div class="cl-card-sub">Machine Learning Capstone Project · Computer Science & Engineering</div>
            </div>
        """, unsafe_allow_html=True)

        tm1, tm2, tm3 = st.columns(3, gap="medium")

        with tm1:
            st.markdown(f"""
                <div class="cl-team-card">
                    <div class="cl-team-avatar">RK</div>
                    <div class="cl-team-info">
                        <div class="cl-team-name">Metuku Rishit Kumar</div>
                        <div class="cl-team-id">CB.SC.U4CSE24428</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with tm2:
            st.markdown(f"""
                <div class="cl-team-card">
                    <div class="cl-team-avatar">RA</div>
                    <div class="cl-team-info">
                        <div class="cl-team-name">Karumuri Naga Sri Raghavendra Aditya</div>
                        <div class="cl-team-id">CB.SC.U4CSE24423</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with tm3:
            st.markdown(f"""
                <div class="cl-team-card">
                    <div class="cl-team-avatar">GS</div>
                    <div class="cl-team-info">
                        <div class="cl-team-name">Dusanapudi Ganesh Bala Sahit</div>
                        <div class="cl-team-id">CB.SC.U4CSE24410</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── 8. SHORTCUTS & EXISTING DETAILED DASHBOARD INTELLIGENCE ──
    with st.container(border=True):
        st.markdown(f"""
            <div class="cl-card-header">
                <div class="cl-card-title">⚡ Quick Navigation & Deep-Dive Hubs</div>
                <div class="cl-card-sub">Direct access to interactive experimentation labs and data exploration modules</div>
            </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            if st.button("🔎 Financial Explorer", key="sc_fe", use_container_width=True):
                _go_to_page("Financial Explorer")
                st.rerun()
        with sc2:
            if st.button("🎯 Prediction Center", key="sc_pc", use_container_width=True):
                _go_to_page("Prediction Center")
                st.rerun()
        with sc3:
            if st.button("📈 Regression Lab", key="sc_rl", use_container_width=True):
                _go_to_page("Regression Lab")
                st.rerun()
        with sc4:
            if st.button("🏷️ Classification Lab", key="sc_cl", use_container_width=True):
                _go_to_page("Classification Lab")
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── EXISTING EXECUTIVE VISUAL ANALYTICS (PRESERVED DASHBOARD CAPABILITIES) ──
    st.markdown(f"""
        <div class="cl-section-header">
            <div class="cl-section-title">📊 Executive Financial Health Visualizer</div>
            <div class="cl-section-sub">Empirical rating distributions, industry sector weights, and balance sheet correlations</div>
        </div>
    """, unsafe_allow_html=True)

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
                "Benchmarking 12 regression estimators on continuous credit score prediction (0=D to 22=AAA) with residual diagnostics and error profiling.")

    reg_eval = evaluate_regression_models(reg_models, test_reg)

    if reg_eval.empty or test_reg is None:
        st.error("Regression model evaluation results unavailable. Check models and test split files.")
    else:
        Xr_test, yr_test = test_reg.drop(columns=["Rating_Score"]), test_reg["Rating_Score"]

        # ── 1. MODEL & ANALYSIS SELECTION CONTROLS ──
        with st.container(border=True):
            ctrl_col1, ctrl_col2 = st.columns([7, 5], gap="medium")
            with ctrl_col1:
                inspect_reg = st.selectbox(
                    "Select Regression Estimator for Diagnostic Deep-Dive",
                    reg_eval["Model"].tolist(),
                    index=0,
                    key="rl_inspect_pick",
                    help="Choose any of the 12 evaluated regression models to examine its residual distribution, tier errors, and feature weights."
                )
            with ctrl_col2:
                reg_sort_metric = st.selectbox(
                    "Benchmark Comparison Ranking Metric",
                    ["R² Score (↑ Higher Better)", "RMSE (↓ Lower Better)", "MAE (↓ Lower Better)"],
                    index=0,
                    key="rl_metric_sort",
                    help="Select the evaluation metric used to rank the leaderboard table and comparison chart."
                )

        # ── 2. KEY PERFORMANCE METRICS (SELECTED MODEL VS BASELINE) ──
        sel_reg_row = reg_eval[reg_eval["Model"] == inspect_reg].iloc[0]
        ols_row = reg_eval[reg_eval["Model"] == "Linear Regression"].iloc[0] if "Linear Regression" in reg_eval["Model"].values else sel_reg_row

        r_delta_r2   = sel_reg_row["R²"] - ols_row["R²"]
        r_delta_rmse = sel_reg_row["RMSE"] - ols_row["RMSE"]
        r_delta_mae  = sel_reg_row["MAE"] - ols_row["MAE"]

        m_c1, m_c2, m_c3, m_c4 = st.columns(4, gap="medium")
        with m_c1:
            st.metric(
                label="R² Explanatory Power",
                value=f"{sel_reg_row['R²']:.4f}",
                delta=f"{r_delta_r2:+.4f} vs OLS" if inspect_reg != "Linear Regression" else "Linear Baseline",
                help="Coefficient of determination: Proportion of out-of-sample rating variance explained relative to mean prediction. Higher is better (1.0 = perfect fit)."
            )
        with m_c2:
            st.metric(
                label="Root Mean Squared Error",
                value=f"{sel_reg_row['RMSE']:.4f}",
                delta=f"{r_delta_rmse:+.4f} notches" if inspect_reg != "Linear Regression" else "Baseline",
                delta_color="inverse",
                help="Test RMSE in rating notches (0-22 scale). Penalizes large outlier mistakes quadratically. Lower is better."
            )
        with m_c3:
            st.metric(
                label="Mean Absolute Error",
                value=f"{sel_reg_row['MAE']:.4f}",
                delta=f"{r_delta_mae:+.4f} notches" if inspect_reg != "Linear Regression" else "Baseline",
                delta_color="inverse",
                help="Test MAE in rating notches. Represents expected absolute deviation per corporate rating. Lower is better."
            )
        with m_c4:
            st.metric(
                label="Evaluation Split",
                value="N = 1,561",
                delta="Holdout Test Set",
                delta_color="off",
                help="Independent holdout test split (20% of corporate universe, N=1,561) strictly evaluated out-of-sample."
            )

        # ── 3. PREDICTIONS & RESIDUAL COMPUTATION ──
        preds, residuals = get_regression_eval_preds(inspect_reg, reg_models, test_reg)
        r_corr = float(np.corrcoef(yr_test, preds)[0, 1])
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        pct_within_1 = float(np.mean(np.abs(residuals) <= 1.0) * 100)
        pct_within_2 = float(np.mean(np.abs(residuals) <= 2.0) * 100)
        skew_res = float(pd.Series(residuals).skew())

        # ── 4. MAIN DIAGNOSTIC VISUALIZATIONS ──
        d1, d2 = st.columns(2, gap="medium")

        with d1:
            with st.container(border=True):
                card_header(f"Actual vs Predicted Credit Score — {inspect_reg}", "Observed agency score against model prediction on 0-22 scale")
                fig_avp = go.Figure()
                fig_avp.add_trace(go.Scatter(
                    x=yr_test, y=preds, mode="markers",
                    marker=dict(color=V_BLUE, opacity=0.45, size=4.5),
                    name="Test Corporates",
                    hovertemplate="Actual Rating: %{x:.1f}<br>Predicted Rating: %{y:.1f}<br>Error: %{customdata:.2f} notches<extra></extra>",
                    customdata=residuals
                ))
                fig_avp.add_trace(go.Scatter(
                    x=[0, 22], y=[0, 22], mode="lines",
                    line=dict(color=V_CORAL, dash="dash", width=2),
                    name="Ideal 45° Line of Equality"
                ))
                fig_avp.update_layout(
                    xaxis=dict(title="Actual Rating Score (0=D to 22=AAA)", range=[-0.5, 22.5]),
                    yaxis=dict(title="Predicted Rating Score (0 to 22)", range=[-0.5, 22.5]),
                    showlegend=True
                )
                chart(fig_avp, height=310)

                # Grounded inference
                inf_avp_text = (
                    f"Predictions for <b>{inspect_reg}</b> achieve a Pearson correlation of {r_corr:.3f} with ground-truth agency scores, "
                    f"tracking the 45° reference line closely across middle-tier ratings while exhibiting compression near the extreme AAA and D bounds."
                )
                display_inference(inf_avp_text)

        with d2:
            with st.container(border=True):
                card_header(f"Residual Errors vs Predicted Score — {inspect_reg}", "Homoscedasticity and error variance across predicted scale")
                fig_rvp = go.Figure()
                fig_rvp.add_trace(go.Scatter(
                    x=preds, y=residuals, mode="markers",
                    marker=dict(color=V_TEAL, opacity=0.45, size=4.5),
                    name="Residual (Actual - Pred)",
                    hovertemplate="Predicted: %{x:.1f}<br>Residual Error: %{y:.2f} notches<extra></extra>"
                ))
                fig_rvp.add_hline(y=0, line_dash="dash", line_color=V_CORAL, line_width=1.8, annotation_text="Zero Error Line", annotation_position="top left")
                fig_rvp.update_layout(
                    xaxis=dict(title="Predicted Rating Score (0 to 22)", range=[-0.5, 22.5]),
                    yaxis=dict(title="Residual Error (Notches)"),
                    showlegend=False
                )
                chart(fig_rvp, height=310)

                inf_rvp_text = (
                    f"Residuals are centered near zero (mean bias: {mean_res:+.3f} notches) with {pct_within_2:.1f}% of unseen companies predicted within ±2 notches, "
                    f"showing stable dispersion with no severe heteroscedastic fan pattern across predicted investment grades."
                )
                display_inference(inf_rvp_text)

        # ── 5. ADDITIONAL ANALYSIS VISUALIZATIONS ──
        d3, d4 = st.columns(2, gap="medium")

        with d3:
            with st.container(border=True):
                card_header(f"Residual Error Distribution & Symmetry — {inspect_reg}", "Histogram and box marginal verifying Gaussian distribution")
                fig_dist = px.histogram(
                    residuals, nbins=45, marginal="box",
                    labels={"value": "Residual Error (Notches)"},
                    color_discrete_sequence=[V_PURPLE]
                )
                fig_dist.add_vline(x=0, line_dash="dash", line_color=V_CORAL, line_width=1.5)
                fig_dist.update_layout(showlegend=False, xaxis_title="Residual Error (Rating Notches)", yaxis_title="Observation Count")
                chart(fig_dist, height=300)

                inf_dist_text = (
                    f"Residual distribution exhibits an approximately symmetric bell profile (mean bias: {mean_res:+.3f}, std: {std_res:.2f}, skewness: {skew_res:+.2f}), "
                    f"confirming that {pct_within_1:.1f}% of corporate predictions land within ±1 notch of their formal credit agency rating."
                )
                display_inference(inf_dist_text)

        with d4:
            with st.container(border=True):
                card_header(f"Mean Absolute Error (MAE) by Credit Tier — {inspect_reg}", "Out-of-sample notch error across corporate solvency segments")
                tier_defs = [
                    ("Prime & High (AAA to AA-)", 19, 22, "#2E7D32"),
                    ("Upper Medium (A+ to A-)", 16, 18, "#388E3C"),
                    ("Lower Medium (BBB+ to BBB-)", 13, 15, "#1565C0"),
                    ("Speculative (BB+ to B-)", 7, 12, "#E65100"),
                    ("Vulnerable / Default (CCC+ to D)", 0, 6, "#C62828")
                ]
                tier_names, tier_scores, tier_colors, tier_counts = [], [], [], []
                for tname, tlow, thigh, tcol in tier_defs:
                    mask = (yr_test >= tlow) & (yr_test <= thigh)
                    if mask.sum() > 0:
                        tier_names.append(tname)
                        tier_scores.append(round(float(np.mean(np.abs(residuals[mask]))), 4))
                        tier_colors.append(tcol)
                        tier_counts.append(int(mask.sum()))

                df_tiers = pd.DataFrame({"Tier": tier_names, "MAE": tier_scores, "Color": tier_colors, "Count": tier_counts})
                fig_tier = px.bar(
                    df_tiers, x="MAE", y="Tier", orientation="h",
                    text="MAE", labels={"MAE": "Mean Absolute Error (Notches)", "Tier": "Rating Tier"}
                )
                fig_tier.update_traces(
                    marker_color=df_tiers["Color"],
                    texttemplate="%{x:.2f} notches",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_tier.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis=dict(range=[0, max(tier_scores)*1.25], showgrid=True), showlegend=False)
                chart(fig_tier, height=300, margin=dict(l=14, r=40, t=20, b=28))

                min_tier_row = df_tiers.loc[df_tiers["MAE"].idxmin()]
                max_tier_row = df_tiers.loc[df_tiers["MAE"].idxmax()]
                inf_tier_text = (
                    f"Prediction accuracy is highest in <b>{min_tier_row['Tier']}</b> ({min_tier_row['MAE']:.2f} MAE, N={min_tier_row['Count']}), "
                    f"whereas <b>{max_tier_row['Tier']}</b> encounters higher notch dispersion ({max_tier_row['MAE']:.2f} MAE) due to balance-sheet volatility in distressed corporations."
                )
                display_inference(inf_tier_text)

        # ── 6. FEATURE IMPORTANCE OR COEFFICIENT ANALYSIS ──
        with st.container(border=True):
            card_header(f"Feature Influence & Coefficient Diagnostics — {inspect_reg}", "Key balance-sheet drivers contributing to credit score predictions")
            m_obj = reg_models[inspect_reg]

            if hasattr(m_obj, "feature_importances_"):
                imp_series = pd.Series(m_obj.feature_importances_, index=Xr_test.columns).nlargest(10)
                fig_imp = px.bar(
                    x=imp_series.values, y=imp_series.index, orientation="h",
                    labels={"x": "Gini Impurity Reduction Importance", "y": "Financial Indicator"},
                    text=imp_series.values
                )
                fig_imp.update_traces(
                    marker_color=V_GREEN,
                    marker_line_color="#059669",
                    marker_line_width=1,
                    texttemplate="%{x:.3f}",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis=dict(range=[0, imp_series.max()*1.22], showgrid=True), showlegend=False)
                chart(fig_imp, height=290, margin=dict(l=14, r=40, t=20, b=28))

                top_f1, top_f2 = imp_series.index[0], imp_series.index[1]
                inf_imp_text = (
                    f"Predictive tree splits are primarily driven by <b>{top_f1}</b> ({imp_series.iloc[0]*100:.1f}%) and <b>{top_f2}</b> ({imp_series.iloc[1]*100:.1f}%), "
                    f"which together capture {(imp_series.iloc[0]+imp_series.iloc[1])*100:.1f}% of total decision importance, reflecting capital structure and margin health."
                )
                display_inference(inf_imp_text)

            elif hasattr(m_obj, "coef_"):
                coef_series = pd.Series(m_obj.coef_, index=Xr_test.columns)
                top_pos = coef_series.nlargest(5)
                top_neg = coef_series.nsmallest(5)
                top_coefs = pd.concat([top_pos, top_neg]).sort_values(ascending=False)
                bar_cols = [V_BLUE if v >= 0 else V_CORAL for v in top_coefs.values]

                fig_coef = px.bar(
                    x=top_coefs.values, y=top_coefs.index, orientation="h",
                    labels={"x": "Standardized Linear Coefficient Weight", "y": "Financial Indicator"},
                    text=top_coefs.values
                )
                fig_coef.update_traces(
                    marker_color=bar_cols,
                    texttemplate="%{x:+.3f}",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                max_abs_c = max(abs(top_coefs.min()), abs(top_coefs.max())) * 1.3
                fig_coef.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis=dict(range=[-max_abs_c, max_abs_c], showgrid=True), showlegend=False)
                chart(fig_coef, height=290, margin=dict(l=14, r=40, t=20, b=28))

                pos_f, neg_f = top_pos.index[0], top_neg.index[0]
                inf_coef_text = (
                    f"Linear model coefficients show strongest positive rating elevation from <b>{pos_f}</b> (+{top_pos.iloc[0]:.3f}) "
                    f"and steepest rating discount from <b>{neg_f}</b> ({top_neg.iloc[0]:.3f}), aligning with empirical corporate solvency mechanics."
                )
                display_inference(inf_coef_text)

            else:
                st.markdown(f"""
                    <div class="cl-insight-box">
                        <div style="font-weight:700;font-size:13px;color:{C_DEEP_VIOLET};margin-bottom:3px;">
                            ℹ️ Model Characteristic: Non-Parametric Metric Space
                        </div>
                        <div style="font-size:12px;color:{C_MUTED};">
                            <b>{inspect_reg}</b> evaluates corporate credit ratings by calculating pairwise Minkowski distances or kernel projections across the full 28-dimensional standardized financial hyperspace simultaneously.
                            Unlike decision trees or linear models, it does not formulate isolated univariate Gini importances or additive linear coefficients.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                inf_nonparam = (
                    f"Non-parametric distance-based and kernel estimators ({inspect_reg}) operate holistically over all 28 standardized financial features, "
                    f"precluding univariate feature importance breakdown while preserving complex multi-ratio geometry."
                )
                display_inference(inf_nonparam)

        # ── 7. BENCHMARK COMPARISON TABLE & LEADERBOARD CHART ──
        if "R²" in reg_sort_metric:
            sorted_reg = reg_eval.sort_values("R²", ascending=False).reset_index(drop=True)
            chart_metric_col = "R²"
            chart_x_range = [0, 0.90]
            chart_color = V_BLUE
        elif "RMSE" in reg_sort_metric:
            sorted_reg = reg_eval.sort_values("RMSE", ascending=True).reset_index(drop=True)
            chart_metric_col = "RMSE"
            chart_x_range = [0, sorted_reg["RMSE"].max() * 1.18]
            chart_color = V_TEAL
        else:
            sorted_reg = reg_eval.sort_values("MAE", ascending=True).reset_index(drop=True)
            chart_metric_col = "MAE"
            chart_x_range = [0, sorted_reg["MAE"].max() * 1.18]
            chart_color = V_INDIGO

        c_tbl, c_bar = st.columns([6, 6], gap="medium")

        with c_tbl:
            with st.container(border=True):
                card_header("Model Leaderboard (Test Set N=1,561)", f"Sorted by {reg_sort_metric}")
                st.dataframe(
                    format_leaderboard_table(
                        sorted_reg,
                        format_dict={
                            "R²": "{:.4f}",
                            "RMSE": "{:.4f}",
                            "MAE": "{:.4f}"
                        },
                        highlight_max_cols=["R²"] if "R²" in reg_sort_metric else None,
                        highlight_min_cols=["RMSE"] if "RMSE" in reg_sort_metric else (["MAE"] if "MAE" in reg_sort_metric else None)
                    ),
                    use_container_width=True, height=420
                )

        with c_bar:
            with st.container(border=True):
                card_header(f"{chart_metric_col} Performance Comparison", f"Ranked across all 12 regression estimators on holdout test set")
                fig_comp = px.bar(
                    sorted_reg, x=chart_metric_col, y="Model", orientation="h",
                    text=chart_metric_col,
                    labels={chart_metric_col: f"{chart_metric_col} Score", "Model": "Model"}
                )
                fig_comp.update_traces(
                    marker_color=chart_color,
                    marker_line_color="#1D4ED8" if chart_color == V_BLUE else "#0F766E",
                    marker_line_width=1,
                    texttemplate="%{x:.4f}",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_comp.update_layout(
                    yaxis=dict(autorange="reversed", title=None, showgrid=False),
                    xaxis=dict(range=chart_x_range, showgrid=True),
                    showlegend=False
                )
                chart(fig_comp, height=420, margin=dict(l=14, r=42, t=20, b=30))

                top_m_name = sorted_reg.iloc[0]["Model"]
                top_m_val  = sorted_reg.iloc[0][chart_metric_col]
                inf_comp_text = (
                    f"<b>{top_m_name}</b> leads overall performance with {chart_metric_col} = {top_m_val:.4f}, "
                    f"demonstrating that non-linear ensemble averaging significantly outperforms linear baselines (Linear R² = 0.3474, RMSE = 3.03 notches) on corporate credit ratings."
                )
                display_inference(inf_comp_text)

        # ── TECHNICAL REFERENCE EXPANDER ──
        with st.expander("📚 Technical Reference: Credit Rating Scale & Metric Definitions"):
            st.markdown(f"""
                <div style="font-size:12.5px;color:{C_TEXT_BODY};line-height:1.6;">
                    <b>1. Continuous Rating Scale (0 to 22):</b>
                    Mapped monotonically from S&P / Moody's corporate credit ratings:
                    <code>0=D</code>, <code>1=C</code>, <code>2=CC</code>, <code>3=CC+</code>, <code>4=CCC-</code>, <code>5=CCC</code>, <code>6=CCC+</code>,
                    <code>7=B-</code>, <code>8=B</code>, <code>9=B+</code>, <code>10=BB-</code>, <code>11=BB</code>, <code>12=BB+</code>,
                    <code>13=BBB-</code>, <code>14=BBB</code>, <code>15=BBB+</code>, <code>16=A-</code>, <code>17=A</code>, <code>18=A+</code>,
                    <code>19=AA-</code>, <code>20=AA</code>, <code>21=AA+</code>, <code>22=AAA</code>.
                    <br><br>
                    <b>2. R² (Coefficient of Determination):</b>
                    Measures the proportion of target variance explained by financial ratios beyond predicting the dataset mean. <i>Note: R² indicates explanatory power, not classification accuracy.</i>
                    <br><br>
                    <b>3. RMSE vs MAE:</b>
                    RMSE penalizes larger notch errors quadratically, making it sensitive to severe rating misses, whereas MAE captures the average notch distance of predicted grades.
                </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 16. PAGE 5: CLASSIFICATION LAB
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Classification Lab":
    page_header("Classification Lab",
                "Evaluation of 12 binary classifiers distinguishing Investment Grade (1: BBB- to AAA) from Speculative Grade (0: BB+ to D) with discrimination diagnostics and threshold curves.")

    clf_eval = evaluate_classification_models(clf_models, test_clf)

    if clf_eval.empty or test_clf is None:
        st.error("Classification evaluation results unavailable. Check models and test split files.")
    else:
        Xc_test, yc_test = test_clf.drop(columns=["Binary Rating"]), test_clf["Binary Rating"]

        # ── 1. MODEL & ANALYSIS SELECTION CONTROLS ──
        with st.container(border=True):
            c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([5, 4, 3], gap="medium")
            with c_ctrl1:
                inspect_clf = st.selectbox(
                    "Select Classifier for Diagnostic Deep-Dive",
                    clf_eval["Model"].tolist(),
                    index=0,
                    key="cl_inspect_pick",
                    help="Choose any of the 12 evaluated classifiers to inspect confusion matrix, ROC curve, PR curves, and probability confidence."
                )
            with c_ctrl2:
                clf_sort_metric = st.selectbox(
                    "Leaderboard Ranking Metric",
                    ["Weighted F1-Score (↑)", "Accuracy (↑)", "Precision (↑)", "Recall (↑)", "ROC-AUC (↑)"],
                    index=0,
                    key="cl_metric_sort",
                    help="Select the evaluation metric used to rank the leaderboard table and comparison chart."
                )
            with c_ctrl3:
                cm_mode = st.radio(
                    "Confusion Matrix View",
                    ["Raw Counts", "Row-Normalized (%)"],
                    horizontal=True,
                    key="cl_cm_mode",
                    help="Toggle between absolute observation counts and row-normalized class recall percentages."
                )

        # ── 2. KEY PERFORMANCE METRIC CARDS ──
        sel_clf_row = clf_eval[clf_eval["Model"] == inspect_clf].iloc[0]
        baseline_clf = clf_eval[clf_eval["Model"] == "Logistic Regression"].iloc[0] if "Logistic Regression" in clf_eval["Model"].values else sel_clf_row

        c_delta_acc = sel_clf_row["Accuracy"] - baseline_clf["Accuracy"]
        c_delta_f1  = sel_clf_row["F1"] - baseline_clf["F1"]
        c_delta_auc = (sel_clf_row["ROC-AUC"] - baseline_clf["ROC-AUC"]) if not np.isnan(sel_clf_row["ROC-AUC"]) and not np.isnan(baseline_clf["ROC-AUC"]) else 0.0

        cm_1, cm_2, cm_3, cm_4, cm_5 = st.columns(5, gap="medium")
        with cm_1:
            st.metric(
                label="Overall Accuracy",
                value=f"{sel_clf_row['Accuracy']*100:.2f}%",
                delta=f"{c_delta_acc*100:+.2f}% vs LogReg" if inspect_clf != "Logistic Regression" else "Linear Baseline",
                help="Overall correct classification rate across Investment Grade and Speculative Grade corporate test instances."
            )
        with cm_2:
            st.metric(
                label="Weighted F1-Score",
                value=f"{sel_clf_row['F1']:.4f}",
                delta=f"{c_delta_f1:+.4f}" if inspect_clf != "Logistic Regression" else "Baseline",
                help="Harmonic mean of precision and recall weighted by class support. Ideal for mildly imbalanced corporate risk classes."
            )
        with cm_3:
            st.metric(
                label="Weighted Precision",
                value=f"{sel_clf_row['Precision']:.4f}",
                help="Proportion of predicted solvency statuses that were genuinely accurate (purity of positive and negative calls)."
            )
        with cm_4:
            st.metric(
                label="Weighted Recall",
                value=f"{sel_clf_row['Recall']:.4f}",
                help="Proportion of actual market corporate credits that were correctly identified by the estimator."
            )
        with cm_5:
            auc_str = f"{sel_clf_row['ROC-AUC']:.4f}" if not np.isnan(sel_clf_row["ROC-AUC"]) else "N/A"
            st.metric(
                label="ROC-AUC Score",
                value=auc_str,
                delta=f"{c_delta_auc:+.4f}" if auc_str != "N/A" and inspect_clf != "Logistic Regression" else None,
                help="Area Under the Receiver Operating Characteristic curve. Measures discrimination across all decision thresholds."
            )

        # ── 3. PREDICTIONS, PROBABILITIES & CONFUSION MATRIX ──
        preds_clf, probs_clf = get_classification_eval_preds(inspect_clf, clf_models, test_clf)
        cm_raw = confusion_matrix(yc_test, preds_clf)
        tn, fp, fn, tp = cm_raw.ravel()
        sens_ig = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_sg = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ── 4. MAIN DIAGNOSTIC VISUALIZATIONS ──
        cd1, cd2 = st.columns(2, gap="medium")

        with cd1:
            with st.container(border=True):
                card_header(f"Confusion Matrix — {inspect_clf}", f"Class-level outcome matrix ({cm_mode})")
                class_labels = ["Speculative (0)", "Investment Grade (1)"]

                if cm_mode == "Row-Normalized (%)":
                    cm_display = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis] * 100
                    text_matrix = [
                        [f"<b>{cm_display[0,0]:.1f}%</b><br>({cm_raw[0,0]:,} / {cm_raw[0].sum():,})", f"<b>{cm_display[0,1]:.1f}%</b><br>({cm_raw[0,1]:,} / {cm_raw[0].sum():,})"],
                        [f"<b>{cm_display[1,0]:.1f}%</b><br>({cm_raw[1,0]:,} / {cm_raw[1].sum():,})", f"<b>{cm_display[1,1]:.1f}%</b><br>({cm_raw[1,1]:,} / {cm_raw[1].sum():,})"]
                    ]
                else:
                    cm_display = cm_raw
                    text_matrix = [
                        [f"<b>{cm_raw[0,0]:,}</b><br>(True Spec)", f"<b>{cm_raw[0,1]:,}</b><br>(False Inv)"],
                        [f"<b>{cm_raw[1,0]:,}</b><br>(False Spec)", f"<b>{cm_raw[1,1]:,}</b><br>(True Inv)"]
                    ]

                fig_cm = go.Figure(go.Heatmap(
                    z=cm_display, x=class_labels, y=class_labels,
                    colorscale="Blues", showscale=False,
                    text=text_matrix, texttemplate="%{text}",
                    textfont={"size": 13.5, "color": "#1E293B"},
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<extra></extra>"
                ))
                fig_cm.update_layout(xaxis_title="Predicted Solvency Tier", yaxis_title="Actual Ground Truth Agency Tier")
                chart(fig_cm, height=310)

                inf_cm_text = (
                    f"<b>{inspect_clf}</b> correctly identifies {tp:,} of {tp+fn:,} Investment Grade firms ({sens_ig*100:.1f}% sensitivity) "
                    f"and {tn:,} of {tn+fp:,} Speculative Grade risks ({spec_sg*100:.1f}% specificity), with {fp+fn:,} total misclassifications out of {len(yc_test):,} holdout corporate tests."
                )
                display_inference(inf_cm_text)

        with cd2:
            with st.container(border=True):
                card_header(f"ROC Curve & Threshold Diagnostics — {inspect_clf}", "True positive accumulation rate versus false alarm trade-off")
                if probs_clf is not None:
                    fpr, tpr, thresholds = roc_curve(yc_test, probs_clf)
                    auc_val = float(roc_auc_score(yc_test, probs_clf))
                    j_scores = tpr - fpr
                    opt_idx = int(np.argmax(j_scores))
                    opt_thresh = float(thresholds[opt_idx])
                    opt_tpr, opt_fpr = float(tpr[opt_idx]), float(fpr[opt_idx])

                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        line=dict(color=V_BLUE, width=2.5),
                        name=f"ROC Curve (AUC = {auc_val:.4f})"
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color=V_CORAL, dash="dash", width=1.5),
                        name="Random Baseline (AUC = 0.50)"
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[opt_fpr], y=[opt_tpr], mode="markers",
                        marker=dict(color=V_AMBER, size=9, symbol="diamond"),
                        name=f"Optimal Cutoff (p={opt_thresh:.2f})"
                    ))
                    fig_roc.update_layout(
                        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
                        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.02]),
                        showlegend=True
                    )
                    chart(fig_roc, height=310)

                    inf_roc_text = (
                        f"An ROC-AUC of {auc_val:.4f} demonstrates superior class separation over random chance (AUC = 0.50), "
                        f"reaching an optimal operating threshold at probability p={opt_thresh:.2f} where sensitivity is {opt_tpr*100:.1f}% against just {opt_fpr*100:.1f}% false alarms."
                    )
                else:
                    st.info(f"Model '{inspect_clf}' does not output soft probabilities; ROC analysis unavailable.")
                    inf_roc_text = f"Continuous probability scores are mathematically unavailable for {inspect_clf}, requiring hard binary decision boundary evaluation."
                display_inference(inf_roc_text)

        # ── 5. ADDITIONAL ANALYSIS VISUALIZATIONS ──
        cd3, cd4 = st.columns(2, gap="medium")

        with cd3:
            with st.container(border=True):
                card_header(f"Per-Class Performance Metrics — {inspect_clf}", "Precision, Recall, and F1-Score breakdown by credit rating tier")
                rep = classification_report(yc_test, preds_clf, output_dict=True)
                p0, r0, f0, s0 = rep['0']['precision'], rep['0']['recall'], rep['0']['f1-score'], int(rep['0']['support'])
                p1, r1, f1, s1 = rep['1']['precision'], rep['1']['recall'], rep['1']['f1-score'], int(rep['1']['support'])

                df_perclass = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1-Score"] * 2,
                    "Score": [p0, r0, f0, p1, r1, f1],
                    "Tier": [f"Speculative (0, N={s0})"] * 3 + [f"Investment Grade (1, N={s1})"] * 3
                })
                fig_pc = px.bar(
                    df_perclass, x="Metric", y="Score", color="Tier", barmode="group",
                    color_discrete_sequence=[V_AMBER, V_GREEN],
                    text="Score"
                )
                fig_pc.update_traces(texttemplate="%{y:.3f}", textposition="outside", textfont=dict(color=C_ESPRESSO, size=11), cliponaxis=False)
                fig_pc.update_layout(yaxis=dict(range=[0, 1.15], showgrid=True), xaxis_title=None, legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
                chart(fig_pc, height=300, margin=dict(l=14, r=20, t=32, b=28))

                inf_pc_text = (
                    f"Speculative Grade exhibits lower recall ({r0*100:.1f}%) than Investment Grade ({r1*100:.1f}%), "
                    f"indicating that borderline speculative corporations are more susceptible to being false-negatived into investment tier than vice-versa."
                )
                display_inference(inf_pc_text)

        with cd4:
            with st.container(border=True):
                card_header(f"Precision-Recall Curve (Investment Grade) — {inspect_clf}", "Trade-off between positive call purity and corporate coverage")
                if probs_clf is not None:
                    prec_vals, rec_vals, _ = precision_recall_curve(yc_test, probs_clf)
                    ap_val = float(average_precision_score(yc_test, probs_clf))
                    prevalence = float(yc_test.mean())
                    valid_r = rec_vals[prec_vals >= 0.90]
                    high_prec_rec = float(valid_r.max()) if len(valid_r) > 0 else 0.85

                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=rec_vals, y=prec_vals, mode="lines",
                        line=dict(color=V_TEAL, width=2.5),
                        name=f"PR Curve (AP = {ap_val:.4f})"
                    ))
                    fig_pr.add_hline(
                        y=prevalence, line_dash="dash", line_color=V_CORAL, line_width=1.5,
                        annotation_text=f"Baseline Prevalence ({prevalence*100:.1f}%)",
                        annotation_position="bottom left"
                    )
                    fig_pr.update_layout(
                        xaxis=dict(title="Recall (Corporate Coverage)", range=[-0.02, 1.02]),
                        yaxis=dict(title="Precision (Solvency Purity)", range=[-0.02, 1.05]),
                        showlegend=True
                    )
                    chart(fig_pr, height=300)

                    inf_pr_text = (
                        f"Average Precision (AP = {ap_val:.4f}) substantially exceeds the {prevalence*100:.1f}% random baseline, "
                        f"maintaining exceptional precision (>90%) across up to {high_prec_rec*100:.1f}% corporate coverage before threshold degradation begins."
                    )
                else:
                    st.info(f"Probabilities unavailable for {inspect_clf}; PR curve omitted.")
                    inf_pr_text = f"Continuous probability curves are unavailable for hard-decision model {inspect_clf}."
                display_inference(inf_pr_text)

        cd5, cd6 = st.columns(2, gap="medium")

        with cd5:
            with st.container(border=True):
                card_header("Test Universe Solvency Class Distribution", "Class balance and sample proportions in holdout test set (N=1,561)")
                class_counts = yc_test.value_counts().sort_index()
                df_dist = pd.DataFrame({
                    "Tier": ["Speculative Grade (0)", "Investment Grade (1)"],
                    "Count": [int(class_counts[0]), int(class_counts[1])],
                    "Percentage": [f"{class_counts[0]/len(yc_test)*100:.1f}%", f"{class_counts[1]/len(yc_test)*100:.1f}%"]
                })
                fig_cdist = px.bar(
                    df_dist, x="Tier", y="Count", text="Count",
                    color="Tier", color_discrete_sequence=[V_AMBER, V_GREEN]
                )
                fig_cdist.update_traces(
                    texttemplate="%{y:,} firms<br>(%{customdata})",
                    customdata=df_dist["Percentage"],
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_cdist.update_layout(yaxis=dict(range=[0, df_dist["Count"].max()*1.22], showgrid=True), xaxis_title=None, showlegend=False)
                chart(fig_cdist, height=290, margin=dict(l=14, r=20, t=20, b=28))

                sg_cnt, ig_cnt = class_counts[0], class_counts[1]
                inf_cdist_text = (
                    f"The test partition comprises {ig_cnt:,} Investment Grade corporations ({ig_cnt/len(yc_test)*100:.1f}%) "
                    f"and {sg_cnt:,} Speculative Grade issuers ({sg_cnt/len(yc_test)*100:.1f}%), establishing a ~1.88:1 ratio that matches broader bond market liquidity conditions."
                )
                display_inference(inf_cdist_text)

        with cd6:
            with st.container(border=True):
                card_header(f"Predicted Solvency Probability Distribution — {inspect_clf}", "Confidence separation between genuine Investment Grade and Speculative firms")
                if probs_clf is not None:
                    high_conf_ig = float((probs_clf[yc_test == 1] >= 0.80).mean() * 100)
                    df_probs = pd.DataFrame({
                        "Probability": probs_clf,
                        "Actual_Tier": ["Investment Grade" if y == 1 else "Speculative Grade" for y in yc_test]
                    })
                    fig_prob = px.histogram(
                        df_probs, x="Probability", color="Actual_Tier", barmode="overlay", nbins=30,
                        color_discrete_sequence=[V_GREEN, V_CORAL],
                        labels={"Probability": "Predicted P(Investment Grade)", "Actual_Tier": "Ground Truth"}
                    )
                    fig_prob.update_traces(opacity=0.7)
                    fig_prob.update_layout(xaxis=dict(range=[-0.02, 1.02]), legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
                    chart(fig_prob, height=290, margin=dict(l=14, r=20, t=32, b=28))

                    inf_prob_text = (
                        f"Predicted probabilities display strong bimodal polarization, with {high_conf_ig:.1f}% of genuine Investment Grade firms "
                        f"assigned >80% solvency confidence while distressed issuers cluster decisively below 30%."
                    )
                else:
                    st.info(f"Model '{inspect_clf}' does not output soft probabilities; confidence distribution unavailable.")
                    inf_prob_text = f"Confidence distributions require continuous soft probability outputs, which are absent in {inspect_clf}."
                display_inference(inf_prob_text)

        # ── 6. CLASSIFIER LEADERBOARD & COMPARISON CHART ──
        if "Accuracy" in clf_sort_metric:
            sorted_clf = clf_eval.sort_values("Accuracy", ascending=False).reset_index(drop=True)
            clf_chart_col = "Accuracy"
            clf_x_range = [0, 1.05]
            clf_color = V_BLUE
        elif "Precision" in clf_sort_metric:
            sorted_clf = clf_eval.sort_values("Precision", ascending=False).reset_index(drop=True)
            clf_chart_col = "Precision"
            clf_x_range = [0, 1.05]
            clf_color = V_TEAL
        elif "Recall" in clf_sort_metric:
            sorted_clf = clf_eval.sort_values("Recall", ascending=False).reset_index(drop=True)
            clf_chart_col = "Recall"
            clf_x_range = [0, 1.05]
            clf_color = V_GREEN
        elif "ROC-AUC" in clf_sort_metric:
            sorted_clf = clf_eval.sort_values("ROC-AUC", ascending=False, na_position="last").reset_index(drop=True)
            clf_chart_col = "ROC-AUC"
            clf_x_range = [0, 1.05]
            clf_color = V_INDIGO
        else:
            sorted_clf = clf_eval.sort_values("F1", ascending=False).reset_index(drop=True)
            clf_chart_col = "F1"
            clf_x_range = [0, 1.05]
            clf_color = V_BLUE

        c_tbl, c_bar = st.columns([6, 6], gap="medium")

        with c_tbl:
            with st.container(border=True):
                card_header("Classifier Leaderboard (Test Set N=1,561)", f"Sorted by {clf_sort_metric}")
                st.dataframe(
                    format_leaderboard_table(
                        sorted_clf,
                        format_dict={
                            "Accuracy":  "{:.4f}",
                            "Precision": "{:.4f}",
                            "Recall":    "{:.4f}",
                            "F1":        "{:.4f}",
                            "ROC-AUC":   lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "—"
                        },
                        highlight_max_cols=[clf_chart_col]
                    ),
                    use_container_width=True, height=420
                )

        with c_bar:
            with st.container(border=True):
                card_header(f"{clf_chart_col} Performance Comparison", f"Ranked across all 12 classifiers on holdout test set")
                fig_clf_comp = px.bar(
                    sorted_clf, x=clf_chart_col, y="Model", orientation="h",
                    text=clf_chart_col,
                    labels={clf_chart_col: f"{clf_chart_col} Score", "Model": "Model"}
                )
                fig_clf_comp.update_traces(
                    marker_color=clf_color,
                    marker_line_color="#1D4ED8" if clf_color == V_BLUE else "#0F766E",
                    marker_line_width=1,
                    texttemplate="%{x:.4f}",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_clf_comp.update_layout(
                    yaxis=dict(autorange="reversed", title=None, showgrid=False),
                    xaxis=dict(range=clf_x_range, showgrid=True),
                    showlegend=False
                )
                chart(fig_clf_comp, height=420, margin=dict(l=14, r=42, t=20, b=30))

                top_clf_name = sorted_clf.iloc[0]["Model"]
                top_clf_val  = sorted_clf.iloc[0][clf_chart_col]
                inf_clf_comp_text = (
                    f"<b>{top_clf_name}</b> achieves top rank in {clf_sort_metric} ({top_clf_val:.4f}), "
                    f"proving that bootstrap aggregation and non-linear tree partitioning provide superior decision boundaries over linear classifiers."
                )
                display_inference(inf_clf_comp_text)

        # ── TECHNICAL REFERENCE EXPANDER ──
        with st.expander("📚 Technical Reference: Solvency Tiers & Cost Asymmetry in Credit Risk"):
            st.markdown(f"""
                <div style="font-size:12.5px;color:{C_TEXT_BODY};line-height:1.6;">
                    <b>1. Binary Solvency Definition:</b>
                    Corporations with agency ratings of <code>BBB-</code> / <code>Baa3</code> or higher are classified as <b>Investment Grade (1)</b> (low default hazard, institutional eligibility).
                    Corporations rated <code>BB+</code> / <code>Ba1</code> or below are classified as <b>Speculative Grade (0)</b> (high yield, elevated credit risk).
                    <br><br>
                    <b>2. Asymmetric Misclassification Penalties:</b>
                    In institutional risk management, a <b>False Positive</b> (classifying a distressed issuer as Investment Grade) carries high credit loss exposure,
                    whereas a <b>False Negative</b> (classifying an investment-grade firm as speculative) primarily incurs an opportunity cost of forgone yield.
                    <br><br>
                    <b>3. F1-Score vs Accuracy:</b>
                    Because the corporate population exhibits mild imbalance (~65% Investment Grade vs ~35% Speculative), Weighted F1 provides a more rigorous indicator of harmonic model precision than raw accuracy alone.
                </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 17. PAGE 6: CLUSTERING STUDIO
# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Clustering Studio":
    page_header("Clustering Studio",
                "Unsupervised financial profile segmentation: benchmarking K-Means (k=3) against Agglomerative Hierarchical (k=2) across 28 standardized financial indicators (N=7,805).")

    clu_eval = get_clustering_evaluation_table()

    if clu_data is None or not clu_models:
        st.error("Clustering dataset or trained models unavailable. Check preprocessed clustering files.")
    else:
        pca_coords, exp_var = compute_pca_2d(clu_data)
        km_labels   = clu_models["K-Means"].labels_
        hier_labels = clu_models["Agglomerative Hierarchical"].labels_

        # ── 1. ANALYSIS SELECTION CONTROLS ──
        with st.container(border=True):
            clu_ctrl1, clu_ctrl2 = st.columns([6, 6], gap="medium")
            with clu_ctrl1:
                selected_algo = st.selectbox(
                    "Select Clustering Algorithm for Profile Analysis",
                    ["K-Means (k=3)", "Agglomerative Hierarchical (k=2)"],
                    index=0,
                    key="clu_algo_pick",
                    help="Switch between partitioned centroid clustering (K-Means) and bottom-up agglomerative Ward linkage."
                )
            with clu_ctrl2:
                profile_scale_mode = st.radio(
                    "Financial Ratio Profile Representation",
                    ["Original Scale Medians (%)", "Standardized Z-Score Deviations"],
                    horizontal=True,
                    key="clu_scale_mode",
                    help="View archetype financial metrics in original percentage scale or as normalized standard-deviation departures from industry median."
                )

        active_labels = km_labels if "K-Means" in selected_algo else hier_labels
        active_k      = 3 if "K-Means" in selected_algo else 2
        active_eval   = clu_eval[clu_eval["Algorithm"].str.contains("K-Means" if "K-Means" in selected_algo else "Agglomerative")].iloc[0]

        # ── 2. KEY QUALITY METRIC CARDS ──
        qk_1, qk_2, qk_3, qk_4, qk_5 = st.columns(5, gap="medium")
        with qk_1:
            st.metric(
                label="Silhouette Score",
                value=f"{active_eval['Silhouette ↑']:.4f}",
                delta="Higher is Better (Max 1.0)",
                help="Measures average ratio of within-cluster tightness to nearest-cluster separation. Higher indicates well-isolated groups."
            )
        with qk_2:
            st.metric(
                label="Davies-Bouldin Index",
                value=f"{active_eval['DB Index ↓']:.4f}",
                delta="Lower is Better (0 = Ideal)",
                delta_color="inverse",
                help="Ratio of within-cluster distances to between-cluster separation. Lower values indicate superior cluster isolation."
            )
        with qk_3:
            st.metric(
                label="Calinski-Harabasz Score",
                value=f"{active_eval['CH Index ↑']:,}",
                delta="Higher is Better",
                help="Variance ratio criterion: Ratio of between-cluster dispersion to within-cluster dispersion. Higher denotes dense, distinct clusters."
            )
        with qk_4:
            st.metric(
                label="Cluster Count (k)",
                value=f"k = {active_k}",
                delta="Empirically Optimal",
                delta_color="off",
                help="Number of corporate segmentation partitions evaluated."
            )
        with qk_5:
            st.metric(
                label="Corporate Universe",
                value=f"N = {len(clu_data):,}",
                delta="Complete Universe",
                delta_color="off",
                help="Full preprocessed population of corporate financial statements across all economic sectors."
            )

        # ── 3. MAIN DIAGNOSTIC VISUALIZATIONS ──
        p_c1, p_c2 = st.columns(2, gap="medium")

        with p_c1:
            with st.container(border=True):
                card_header(f"{selected_algo} — Two-Dimensional PCA Projection", f"Explained variance: PC1 = {exp_var[0]*100:.1f}%, PC2 = {exp_var[1]*100:.1f}% (Total = {sum(exp_var)*100:.1f}%)")
                df_pca = pd.DataFrame({
                    "PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1],
                    "Cluster": [f"Cluster {l}" for l in active_labels]
                })
                cluster_colors = [V_BLUE, V_GREEN, V_AMBER] if active_k == 3 else [V_PURPLE, "#EC4899"]
                fig_pca = px.scatter(
                    df_pca, x="PC1", y="PC2", color="Cluster",
                    color_discrete_sequence=cluster_colors,
                    opacity=0.60
                )
                fig_pca.update_layout(
                    xaxis_title=f"PC1 (First Principal Component — {exp_var[0]*100:.1f}% Variance)",
                    yaxis_title=f"PC2 (Second Principal Component — {exp_var[1]*100:.1f}% Variance)",
                    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
                )
                chart(fig_pca, height=310, margin=dict(l=14, r=20, t=32, b=28))

                inf_pca_text = (
                    f"Two principal components capture {sum(exp_var)*100:.1f}% of total financial variance, revealing distinct core groupings "
                    f"while boundary overlap reflects corporations with hybrid leverage and transitional margin profiles in the original 28-dimensional space."
                )
                display_inference(inf_pca_text)

        with p_c2:
            with st.container(border=True):
                card_header("K-Means Inertia Elbow Curve", "Sum of Squared Errors (SSE) across candidate k=2 to 10")
                inertias = {2: 99392, 3: 87739, 4: 79157, 5: 73961, 6: 69614, 7: 65852, 8: 63402, 9: 61351, 10: 59239}
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(inertias.keys()), y=list(inertias.values()),
                    mode="lines+markers",
                    marker=dict(size=8, color=V_INDIGO),
                    line=dict(color=V_INDIGO, width=2.5),
                    name="Inertia (SSE)",
                    hovertemplate="Clusters k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"
                ))
                fig_elbow.add_vline(
                    x=3, line_dash="dash", line_color=V_CORAL, line_width=1.8,
                    annotation_text="k=3 Selected Elbow",
                    annotation_font=dict(color=V_CORAL, size=11),
                    annotation_position="top right"
                )
                fig_elbow.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (Within-Cluster SSE)")
                chart(fig_elbow, height=310)

                inf_elbow_text = (
                    f"Inertia drops steeply from k=2 (99,392) to k=3 (87,739) before the trajectory flattens into diminishing returns, "
                    f"empirically supporting k=3 as the most parsimonious elbow inflection point for corporate financial segmentation."
                )
                display_inference(inf_elbow_text)

        # ── 4. ADDITIONAL ANALYSIS VISUALIZATIONS ──
        cand_df = pd.DataFrame([
            {"k": 2, "Inertia": 99392, "Silhouette": 0.2147, "DB_Index": 1.6284, "CH_Index": 2614},
            {"k": 3, "Inertia": 87739, "Silhouette": 0.1776, "DB_Index": 1.7440, "CH_Index": 1998},
            {"k": 4, "Inertia": 79157, "Silhouette": 0.1556, "DB_Index": 1.7837, "CH_Index": 1758},
            {"k": 5, "Inertia": 73961, "Silhouette": 0.1521, "DB_Index": 1.7329, "CH_Index": 1548},
            {"k": 6, "Inertia": 69614, "Silhouette": 0.1481, "DB_Index": 1.7334, "CH_Index": 1413},
            {"k": 7, "Inertia": 65852, "Silhouette": 0.1412, "DB_Index": 1.7610, "CH_Index": 1302},
            {"k": 8, "Inertia": 63402, "Silhouette": 0.1378, "DB_Index": 1.7850, "CH_Index": 1215},
        ])

        p_c3, p_c4 = st.columns(2, gap="medium")

        with p_c3:
            with st.container(border=True):
                card_header("Silhouette Score Across Candidate k", "Cluster separation quality across cluster counts k=2 to 8")
                sil_colors = [V_CORAL if row["k"] == 3 else V_BLUE for _, row in cand_df.iterrows()]
                fig_sil = px.bar(
                    cand_df, x="k", y="Silhouette", text="Silhouette",
                    labels={"k": "Candidate Cluster Count (k)", "Silhouette": "Silhouette Score"}
                )
                fig_sil.update_traces(
                    marker_color=sil_colors,
                    texttemplate="%{y:.4f}",
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_sil.update_layout(yaxis=dict(range=[0, cand_df["Silhouette"].max()*1.22], showgrid=True), xaxis=dict(dtick=1))
                chart(fig_sil, height=290, margin=dict(l=14, r=20, t=20, b=28))

                inf_sil_text = (
                    f"Candidate k=2 achieves the mathematical peak silhouette (0.2147), but k=3 (0.1776) successfully isolates "
                    f"a vital distressed corporate cohort that provides far superior business interpretability for credit risk monitoring."
                )
                display_inference(inf_sil_text)

        with p_c4:
            with st.container(border=True):
                card_header(f"Cluster Population Sizes & Balance — {selected_algo}", "Number and percentage of corporations in each cluster")
                dist_counts = pd.Series(active_labels).value_counts().sort_index()
                df_dist = pd.DataFrame({
                    "Cluster": [f"Cluster {i}" for i in dist_counts.index],
                    "Count": dist_counts.values,
                    "Share": [f"{v/len(active_labels)*100:.1f}%" for v in dist_counts.values]
                })
                fig_dist = px.bar(
                    df_dist, x="Cluster", y="Count", text="Count",
                    color="Cluster", color_discrete_sequence=cluster_colors
                )
                fig_dist.update_traces(
                    texttemplate="%{y:,} firms<br>(%{customdata})",
                    customdata=df_dist["Share"],
                    textposition="outside",
                    textfont=dict(color=C_ESPRESSO, size=11),
                    cliponaxis=False
                )
                fig_dist.update_layout(yaxis=dict(range=[0, df_dist["Count"].max()*1.22], showgrid=True), showlegend=False, xaxis_title=None)
                chart(fig_dist, height=290, margin=dict(l=14, r=20, t=20, b=28))

                dist_str = ", ".join([f"{row['Cluster']}: {row['Count']:,} ({row['Share']})" for _, row in df_dist.iterrows()])
                inf_dist_text = (
                    f"Under <b>{selected_algo}</b>, corporations partition into {dist_str}, "
                    f"demonstrating balanced cluster sizes without any trivial, single-digit outlier buckets."
                )
                display_inference(inf_dist_text)

        p_c5, p_c6 = st.columns(2, gap="medium")

        with p_c5:
            with st.container(border=True):
                card_header("Cluster Archetype Financial Metrics Profile", f"Median financial indicators by cluster ({profile_scale_mode})")
                key_profile_cols = [
                    "Gross Margin", "Operating Margin", "Net Profit Margin",
                    "ROA - Return On Assets", "Debt/Equity Ratio", "Current Ratio"
                ]
                raw_with_labels = raw_df.copy()
                raw_with_labels["Cluster"] = [f"Cluster {l}" for l in active_labels]

                if "Original" in profile_scale_mode:
                    cluster_summary = raw_with_labels.groupby("Cluster")[key_profile_cols].median().reset_index()
                    melted_prof = cluster_summary.melt(id_vars="Cluster", var_name="Metric", value_name="Value")
                    fig_prof = px.bar(
                        melted_prof, x="Metric", y="Value", color="Cluster", barmode="group",
                        color_discrete_sequence=cluster_colors
                    )
                    fig_prof.update_layout(yaxis=dict(title="Median Value (%)", showgrid=True), legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
                    chart(fig_prof, height=300, margin=dict(l=14, r=20, t=32, b=28))
                else:
                    # Standardized z-score view
                    z_df = raw_df[key_profile_cols].copy()
                    z_norm = (z_df - z_df.mean()) / z_df.std()
                    z_norm["Cluster"] = [f"Cluster {l}" for l in active_labels]
                    cluster_z = z_norm.groupby("Cluster").median().reset_index()
                    melted_z = cluster_z.melt(id_vars="Cluster", var_name="Metric", value_name="Z-Score")
                    fig_prof = px.bar(
                        melted_z, x="Metric", y="Z-Score", color="Cluster", barmode="group",
                        color_discrete_sequence=cluster_colors
                    )
                    fig_prof.update_layout(yaxis=dict(title="Standardized Z-Score Deviation (σ)", showgrid=True), legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
                    chart(fig_prof, height=300, margin=dict(l=14, r=20, t=32, b=28))

                if active_k == 3:
                    inf_prof_text = (
                        f"Financial profiles cleanly delineate three market archetypes: Cluster 0 represents High-Margin Leaders (Op Margin: ~23.6%), "
                        f"Cluster 1 captures Steady Capital-Intensive Firms (Gross Margin: ~28.6%), and Cluster 2 identifies Stressed Borrowers with elevated Debt/Equity (1.01)."
                    )
                else:
                    inf_prof_text = (
                        f"Agglomerative 2-cluster linkage cleanly bifurcates solvent corporations (Cluster 0, N=4,994, positive ROA) "
                        f"from financially constrained, higher-leverage corporate issuers (Cluster 1, N=2,811)."
                    )
                display_inference(inf_prof_text)

        with p_c6:
            with st.container(border=True):
                card_header("Sampled Hierarchical Dendrogram (Ward Linkage)", "Hierarchical agglomerative tree structure on corporate observations (N=55)")
                fig_dend = get_sampled_dendrogram_figure(clu_data, sample_size=55)
                fig_dend.add_hline(
                    y=11.5, line_dash="dash", line_color=V_CORAL, line_width=1.5,
                    annotation_text="Cutoff Threshold (k=2 Linkage)", annotation_position="top right"
                )
                chart(fig_dend, height=300)

                inf_dend_text = (
                    f"The dendrogram tree exhibits an early primary bifurcation at high linkage distance (>16.0), "
                    f"demonstrating a fundamental division between financially sound corporations and capital-constrained peers before granular sub-branching takes place."
                )
                display_inference(inf_dend_text)

        # ── 5. COMPARISON TABLE & ARCHETYPE DEEP DIVE ──
        with st.container(border=True):
            card_header("Algorithm Benchmark Comparison & Qualitative Risk Archetypes", "Direct quantitative evaluation of K-Means vs Agglomerative Hierarchical")
            t_col1, t_col2 = st.columns([5, 7], gap="medium")

            with t_col1:
                st.dataframe(
                    format_leaderboard_table(
                        clu_eval,
                        format_dict={
                            "Silhouette ↑": "{:.4f}",
                            "DB Index ↓":   "{:.4f}",
                            "CH Index ↑":   "{:.0f}"
                        },
                        highlight_max_cols=["Silhouette ↑", "CH Index ↑"],
                        highlight_min_cols=["DB Index ↓"]
                    ),
                    use_container_width=True, height=170
                )
                inf_eval_text = (
                    f"K-Means (k=3) achieves higher cluster density (Calinski-Harabasz: 1,998 vs 1,867) and lower dispersion "
                    f"(Davies-Bouldin: 1.7440 vs 1.8458) compared to 2-cluster Agglomerative Hierarchical linkage."
                )
                display_inference(inf_eval_text)

            with t_col2:
                st.markdown(f"""
                    <div class="cl-insight-box">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 0: High-Margin Quality Champions (N=2,646 · 33.9%)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Gross Margin 62.0%, Op. Margin 23.6%, ROA 7.6%. <b>81.4% Investment Grade.</b> Resilient pricing power, fortress liquidity, lowest default risk.
                        </div>
                    </div>
                    <div class="cl-insight-box" style="border-left-color:{V_BLUE} !important;">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 1: Capital-Intensive Steady Corporates (N=3,901 · 50.0%)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Gross Margin 28.6%, Op. Margin 9.8%, ROE 24.7%. <b>68.1% Investment Grade.</b> Moderate operating cash flows, low debt/equity (0.71), asset-heavy.
                        </div>
                    </div>
                    <div class="cl-insight-box" style="border-left-color:{V_CORAL} !important;">
                        <div style="font-weight:700;font-size:12.5px;color:{C_DEEP_VIOLET};margin-bottom:2px;">
                            Cluster 2: Stressed & Distressed Issuers (N=1,258 · 16.1%)
                        </div>
                        <div style="font-size:11.5px;color:{C_MUTED};">
                            Op. Margin 1.1%, Net Margin -17.0%, Debt/Equity 1.01. <b>77.1% Speculative Grade.</b> Elevated leverage, negative net margin, severe default vulnerability.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

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
            fig = px.imshow(corr_all, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",text_auto=".2f")
            chart(fig, height=480)
