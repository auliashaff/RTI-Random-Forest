import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import time
import io
import os
import base64
from PIL import Image

from sklearn.preprocessing  import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

warnings.filterwarnings('ignore')

# ── BUNDLED DEFAULT DATASET (so the deployed link works without manual upload) ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_PATH = os.path.join(APP_DIR, "data", "UNSW_NB15_training-set.csv")
DEFAULT_TEST_PATH  = os.path.join(APP_DIR, "data", "UNSW_NB15_testing-set.csv")
HAS_DEFAULT_DATA = os.path.exists(DEFAULT_TRAIN_PATH) and os.path.exists(DEFAULT_TEST_PATH)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Feature Selection Study",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLE ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f1629 0%, #1a2744 50%, #0f2444 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    font-size: clamp(22px, 3vw, 36px);
    font-weight: 700;
    line-height: 1.2;
    color: #f1f5f9;
    margin-bottom: 12px;
    max-width: 760px;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    font-size: 15px;
    color: #94a3b8;
    max-width: 620px;
    line-height: 1.6;
}

/* Metric cards */
.metric-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.metric-card {
    flex: 1; min-width: 150px;
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 18px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-value.blue  { color: #38bdf8; }
.metric-value.green { color: #34d399; }
.metric-value.amber { color: #fbbf24; }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 18px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

/* Method badge */
.method-badge {
    display: inline-block;
    background: #0f2444;
    border: 1px solid #1e4a7a;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #38bdf8;
    margin-bottom: 12px;
}

/* Feature pill */
.feature-pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
.feature-pill {
    background: #0f2444;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #7dd3fc;
}

/* Comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    border-radius: 10px;
    overflow: hidden;
}
.compare-table th {
    background: #0f2444;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 12px 16px;
    text-align: left;
}
.compare-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #1e2d4a;
    color: #cbd5e1;
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table tr:hover td { background: #0f1a2e; }
.best { color: #34d399; font-weight: 600; }

/* Info box */
.info-box {
    background: #0a1628;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
}
.info-box b { color: #e2e8f0; }

/* Spinner override */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Plot container */
.plot-container {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: #38bdf8 !important;
    color: #0a0e1a !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Divider */
hr { border-color: #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)

# ── CACHED DATA PIPELINE ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(train_bytes, test_bytes):
    train = pd.read_csv(io.BytesIO(train_bytes))
    test  = pd.read_csv(io.BytesIO(test_bytes))
    df = pd.concat([train, test], axis=0, ignore_index=True)

    for c in ['id', 'attack_cat']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    X = df.drop('label', axis=1)
    y = df['label']
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode
    le_enc = {}
    X_tr_enc = X_train.copy()
    X_te_enc = X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_tr_enc[col] = le.fit_transform(X_train[col].astype(str))
        X_te_enc[col] = X_test[col].astype(str).map(
            lambda s, le=le: le.transform([s])[0] if s in le.classes_ else -1
        )
        le_enc[col] = le

    X_tr_enc = X_tr_enc.astype(float)
    X_te_enc  = X_te_enc.astype(float)
    feature_names = list(X_tr_enc.columns)

    # StandardScaler
    ss = StandardScaler()
    X_tr_std = ss.fit_transform(X_tr_enc)
    X_te_std  = ss.transform(X_te_enc)

    # MinMaxScaler (for chi2)
    mm = MinMaxScaler()
    X_tr_mm = mm.fit_transform(X_tr_enc)
    X_te_mm  = mm.transform(X_te_enc)

    return (X_tr_std, X_te_std, X_tr_mm, X_te_mm,
            y_train, y_test, feature_names,
            df.shape, len(train), len(test))


@st.cache_data(show_spinner=False)
def run_full(_Xtr, _Xte, _ytr, _yte, feature_names):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time(); clf.fit(_Xtr, _ytr); t1 = time.time()
    yp = clf.predict(_Xte); t2 = time.time()
    return _metrics(yp, _yte, t0, t1, t2), yp, clf.feature_importances_, feature_names

@st.cache_data(show_spinner=False)
def run_filter(_Xtr_mm, _Xte_mm, _Xtr_std, _Xte_std, _ytr, _yte, feature_names, k):
    sel = SelectKBest(score_func=chi2, k=k)
    Xtr = sel.fit_transform(_Xtr_mm, _ytr)
    Xte = sel.transform(_Xte_mm)
    idx = sel.get_support(indices=True)
    chosen = [feature_names[i] for i in idx]
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time(); clf.fit(Xtr, _ytr); t1 = time.time()
    yp = clf.predict(Xte); t2 = time.time()
    return _metrics(yp, _yte, t0, t1, t2), yp, clf.feature_importances_, chosen

@st.cache_data(show_spinner=False)
def run_wrapper(_Xtr, _Xte, _ytr, _yte, feature_names, n):
    base = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe  = RFE(estimator=base, n_features_to_select=n, step=5)
    rfe.fit(_Xtr, _ytr)
    Xtr = rfe.transform(_Xtr)
    Xte = rfe.transform(_Xte)
    idx = rfe.get_support(indices=True)
    chosen = [feature_names[i] for i in idx]
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time(); clf.fit(Xtr, _ytr); t1 = time.time()
    yp = clf.predict(Xte); t2 = time.time()
    return _metrics(yp, _yte, t0, t1, t2), yp, clf.feature_importances_, chosen

def _metrics(yp, yte, t0, t1, t2):
    return {
        'accuracy':  accuracy_score(yte, yp),
        'recall':    recall_score(yte, yp, average='weighted'),
        'precision': precision_score(yte, yp, average='weighted'),
        'f1':        f1_score(yte, yp, average='weighted'),
        'train_t':   t1 - t0,
        'pred_t':    t2 - t1,
        'report':    classification_report(yte, yp, output_dict=True)
    }

def make_cm_fig(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0f1629')
    ax.set_facecolor('#0f1629')
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='#1e2d4a',
                annot_kws={'size': 13, 'color': 'white'},
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
    ax.set_xlabel('Predicted', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Actual',    color='#94a3b8', fontsize=11)
    ax.set_title(title, color='#e2e8f0', fontsize=12, pad=12)
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    return fig

def make_fi_fig(importances, names, title):
    s = pd.Series(importances, index=names).nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f1629')
    ax.set_facecolor('#0f1629')
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(s)))
    s.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(title, color='#e2e8f0', fontsize=12)
    ax.set_xlabel('Importance', color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.spines[:].set_color('#1e2d4a')
    plt.tight_layout()
    return fig

def make_compare_fig(results):
    labels = list(results.keys())
    metrics = ['accuracy','recall','precision','f1']
    x = np.arange(len(labels))
    width = 0.2
    colors = ['#38bdf8','#34d399','#a78bfa','#fb923c']
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1629')
    ax.set_facecolor('#0f1629')
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [results[l][m] for l in labels]
        bars = ax.bar(x + i*width, vals, width, label=m.capitalize(), color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, color='#cbd5e1')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(labels, color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e2d4a')
    ax.set_ylim(0.75, 1.03)
    ax.set_title('Model Performance Comparison', color='#e2e8f0', fontsize=13, pad=14)
    ax.set_ylabel('Score', color='#94a3b8')
    ax.legend(facecolor='#0f1629', edgecolor='#1e2d4a', labelcolor='#cbd5e1', fontsize=10)
    plt.tight_layout()
    return fig

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ IDS Evaluation")
    st.markdown("---")
    st.markdown("#### Upload Dataset")
    if HAS_DEFAULT_DATA:
        st.caption("✅ Bundled UNSW-NB15 dataset loaded by default. Upload your own files below to override it.")
    train_file = st.file_uploader("Training Set (.csv)", type="csv", key="train")
    test_file  = st.file_uploader("Testing Set (.csv)",  type="csv", key="test")
    st.markdown("---")
    st.markdown("#### Parameters")
    k_best   = st.slider("Filter Method – K features",  5, 30, 10)
    n_rfe    = st.slider("Wrapper Method – N features", 5, 30, 10)
    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#475569; line-height:1.7'>
    <b style='color:#64748b'>Dataset</b><br>UNSW-NB15<br><br>
    <b style='color:#64748b'>Methods</b><br>
    • All Features (Baseline)<br>
    • Filter — SelectKBest χ²<br>
    • Wrapper — RFE<br><br>
    <b style='color:#64748b'>Classifier</b><br>Random Forest (n=100)
    </div>
    """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-eyebrow">🛡️ Intrusion Detection · Feature Selection · Machine Learning</div>
  <div class="hero-title">Comparative Evaluation of<br><span>Feature Selection Methods</span><br>on Random Forest for Anomaly-Based<br>Intrusion Detection in Network Traffic</div>
  <div class="hero-sub">Benchmarking Filter (SelectKBest χ²) and Wrapper (RFE) feature selection strategies against a full-feature baseline using the UNSW-NB15 dataset.</div>
</div>
""", unsafe_allow_html=True)

# ── MAIN LOGIC ─────────────────────────────────────────────────────────────────
using_default_data = False
if train_file and test_file:
    train_bytes = train_file.read()
    test_bytes  = test_file.read()
elif HAS_DEFAULT_DATA:
    with open(DEFAULT_TRAIN_PATH, "rb") as f:
        train_bytes = f.read()
    with open(DEFAULT_TEST_PATH, "rb") as f:
        test_bytes = f.read()
    using_default_data = True
else:
    train_bytes = test_bytes = None

if train_bytes is None or test_bytes is None:
    st.markdown("""
    <div class="info-box">
        <b>👈 Upload your dataset</b> to begin — load the <code>UNSW_NB15_training-set.csv</code> and
        <code>UNSW_NB15_testing-set.csv</code> files using the sidebar, then click <b>Run Analysis</b>.
    </div>
    """, unsafe_allow_html=True)

    # Static result preview (from prior run)
    st.markdown('<div class="section-header">Preview — Results from Prior Run</div>', unsafe_allow_html=True)
    prev = {
        "All Features":  {"accuracy":0.9525,"recall":0.9525,"precision":0.9525,"f1":0.9525,"train_t":36.0,"pred_t":0.74},
        "Filter (k=10)": {"accuracy":0.9085,"recall":0.9085,"precision":0.9094,"f1":0.9071,"train_t":23.2,"pred_t":0.59},
        "Wrapper (n=10)":{"accuracy":0.9416,"recall":0.9416,"precision":0.9415,"f1":0.9415,"train_t":23.7,"pred_t":0.56},
    }
    cols = st.columns(3)
    colors_ = ['blue','green','amber']
    for (name, m), col, clr in zip(prev.items(), cols, colors_):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="method-badge">{name}</div>
              <div class="metric-label">Accuracy</div>
              <div class="metric-value {clr}">{m['accuracy']:.2%}</div>
              <div style="margin-top:12px">
                <div class="metric-label">F1-Score</div>
                <div style="font-size:20px;font-weight:600;color:#e2e8f0">{m['f1']:.4f}</div>
              </div>
              <div style="margin-top:8px">
                <div class="metric-label">Train Time</div>
                <div style="font-family:'JetBrains Mono';font-size:13px;color:#64748b">{m['train_t']:.1f}s</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

if run_btn:
    st.cache_data.clear()

if using_default_data:
    st.markdown("""
    <div class="info-box" style="border-left-color:#34d399">
        <b>ℹ️ Using bundled UNSW-NB15 dataset</b> (shipped with this app). Upload your own
        training/testing CSV in the sidebar to analyze different data instead.
    </div>
    """, unsafe_allow_html=True)

with st.spinner("Preprocessing data…"):
    (X_tr_std, X_te_std, X_tr_mm, X_te_mm,
     y_train, y_test, feature_names,
     dshape, n_train, n_test) = load_and_preprocess(train_bytes, test_bytes)

# Dataset stats
col1, col2, col3, col4 = st.columns(4)
for col, label, val, clr in [
    (col1, "Total Samples",  f"{dshape[0]:,}",        "blue"),
    (col2, "Features",       f"{dshape[1]-2}",         "blue"),
    (col3, "Training Set",   f"{n_train:,}",            "green"),
    (col4, "Testing Set",    f"{n_test:,}",             "amber"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value {clr}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Comparison",
    "🌲 All Features",
    "🔵 Filter Method",
    "🔷 Wrapper Method",
])

# ── RUN MODELS ─────────────────────────────────────────────────────────────────
with st.spinner("Training Random Forest — All Features…"):
    m_full, yp_full, fi_full, fn_full = run_full(
        X_tr_std, X_te_std, y_train, y_test, tuple(feature_names))

with st.spinner(f"Training Filter Method — SelectKBest k={k_best}…"):
    m_filt, yp_filt, fi_filt, fn_filt = run_filter(
        X_tr_mm, X_te_mm, X_tr_std, X_te_std, y_train, y_test, tuple(feature_names), k_best)

with st.spinner(f"Training Wrapper Method — RFE n={n_rfe}…"):
    m_wrap, yp_wrap, fi_wrap, fn_wrap = run_wrapper(
        X_tr_std, X_te_std, y_train, y_test, tuple(feature_names), n_rfe)

all_results = {
    "All Features":  m_full,
    f"Filter (k={k_best})":  m_filt,
    f"Wrapper (n={n_rfe})": m_wrap,
}

# ── TAB 1 — COMPARISON ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)

    # Metric cards
    cols = st.columns(3)
    card_colors = ['blue','green','amber']
    for (name, m), col, clr in zip(all_results.items(), cols, card_colors):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="method-badge">{name}</div>
              <div class="metric-label">Accuracy</div>
              <div class="metric-value {clr}">{m['accuracy']:.2%}</div>
              <div style="margin-top:12px;display:flex;gap:24px">
                <div>
                  <div class="metric-label">Recall</div>
                  <div style="font-size:18px;font-weight:600;color:#e2e8f0">{m['recall']:.4f}</div>
                </div>
                <div>
                  <div class="metric-label">Precision</div>
                  <div style="font-size:18px;font-weight:600;color:#e2e8f0">{m['precision']:.4f}</div>
                </div>
              </div>
              <div style="margin-top:10px">
                <div class="metric-label">F1-Score</div>
                <div style="font-size:22px;font-weight:700;color:#e2e8f0">{m['f1']:.4f}</div>
              </div>
              <div style="margin-top:8px;display:flex;gap:16px">
                <div>
                  <div class="metric-label">Train Time</div>
                  <div style="font-family:'JetBrains Mono';font-size:13px;color:#64748b">{m['train_t']:.2f}s</div>
                </div>
                <div>
                  <div class="metric-label">Predict Time</div>
                  <div style="font-family:'JetBrains Mono';font-size:13px;color:#64748b">{m['pred_t']:.2f}s</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="section-header">Score Comparison Chart</div>', unsafe_allow_html=True)
    fig_cmp = make_compare_fig(all_results)
    st.pyplot(fig_cmp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary table
    st.markdown('<div class="section-header">Detailed Metrics Table</div>', unsafe_allow_html=True)
    rows = ""
    metrics_order = ['accuracy','recall','precision','f1']
    for name, m in all_results.items():
        best_acc = max(v['accuracy'] for v in all_results.values())
        acc_cls = 'best' if m['accuracy'] == best_acc else ''
        rows += f"""<tr>
            <td><b>{name}</b></td>
            <td class="{acc_cls}">{m['accuracy']:.4%}</td>
            <td>{m['recall']:.4%}</td>
            <td>{m['precision']:.4%}</td>
            <td>{m['f1']:.4f}</td>
            <td>{m['train_t']:.2f}s</td>
            <td>{m['pred_t']:.3f}s</td>
        </tr>"""
    st.markdown(f"""
    <table class="compare-table">
      <thead><tr>
        <th>Method</th><th>Accuracy</th><th>Recall</th>
        <th>Precision</th><th>F1-Score</th><th>Train Time</th><th>Pred Time</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:20px">
        <b>Interpretation:</b> The Wrapper Method (RFE) achieves near-baseline accuracy using only
        a subset of features, demonstrating a strong trade-off between model complexity and performance.
        The Filter Method is the fastest to train but sacrifices some recall on normal traffic.
        Highlighted cells in green indicate the best score per metric.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2 — ALL FEATURES ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Baseline — Random Forest with All Features</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>All {len(feature_names)} features</b> are used with no selection step. StandardScaler is applied before training.
        This serves as the performance ceiling for the other methods.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(make_cm_fig(y_test, yp_full, "Random Forest — All Features"), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Top-10 Feature Importances</div>', unsafe_allow_html=True)
        st.pyplot(make_fi_fig(fi_full, list(fn_full), "All Features"), use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:20px">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(m_full['report']).transpose().round(4)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']),
                 use_container_width=True)

# ── TAB 3 — FILTER ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Filter Method — SelectKBest (χ² Score)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>SelectKBest</b> ranks features independently using the chi-squared (χ²) statistic between
        each feature and the target label. MinMaxScaler is applied first (chi² requires non-negative values).
        Top <b>k = {k_best}</b> features are selected.
    </div>""", unsafe_allow_html=True)

    st.markdown("**Selected Features:**")
    pills = "".join(f'<span class="feature-pill">{f}</span>' for f in fn_filt)
    st.markdown(f'<div class="feature-pills">{pills}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(make_cm_fig(y_test, yp_filt, f"Filter Method (k={k_best})"), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        st.pyplot(make_fi_fig(fi_filt, list(fn_filt), f"Filter Method (k={k_best})"), use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:20px">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(m_filt['report']).transpose().round(4)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']),
                 use_container_width=True)

# ── TAB 4 — WRAPPER ───────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Wrapper Method — Recursive Feature Elimination (RFE)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>RFE</b> iteratively trains a Random Forest estimator and prunes the least important features
        each round (step = 5), until <b>n = {n_rfe}</b> features remain. StandardScaler is applied.
        RFE is computationally heavier but produces feature sets that are explicitly optimised for the classifier.
    </div>""", unsafe_allow_html=True)

    st.markdown("**Selected Features:**")
    pills = "".join(f'<span class="feature-pill">{f}</span>' for f in fn_wrap)
    st.markdown(f'<div class="feature-pills">{pills}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(make_cm_fig(y_test, yp_wrap, f"Wrapper Method (n={n_rfe})"), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        st.pyplot(make_fi_fig(fi_wrap, list(fn_wrap), f"Wrapper Method (n={n_rfe})"), use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:20px">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(m_wrap['report']).transpose().round(4)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']),
                 use_container_width=True)
