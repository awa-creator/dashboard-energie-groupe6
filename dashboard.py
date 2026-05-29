"""
Dashboard interactif — Prediction de la Consommation Energetique
Dataset : UCI Appliances Energy Prediction + Donnees Senelec 2022-2024
Lancer avec : streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from fpdf import FPDF
import io, datetime, warnings
import shap
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PALETTE
# ═══════════════════════════════════════════════════════════════════════════
VIOLET   = "#7B2FBE"
MAGENTA  = "#D4227A"
ORANGE   = "#F15A24"
GRAD_CSS = "linear-gradient(135deg, #7B2FBE 0%, #D4227A 50%, #F15A24 100%)"
CARD_BG  = "#FFFFFF"
CARD_BG2 = "#F8F4FF"
BORDER   = "rgba(123,47,190,0.18)"
TEXT_SEC = "#6B5B8A"
TEXT_PRI = "#1A0A2E"
WHITE    = "#FFFFFF"

SENELEC_SCALE = [
    [0.0,  "#F8F4FF"],
    [0.25, "#7B2FBE"],
    [0.6,  "#D4227A"],
    [1.0,  "#F15A24"],
]

_AXIS_BASE = dict(
    gridcolor="rgba(123,47,190,0.14)", showline=True,
    linecolor="rgba(123,47,190,0.28)", color="#000000",
    tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"),
    title_font=dict(color="#000000", size=14, family="Plus Jakarta Sans", weight="bold"),
)

def _merge_axis(custom=None):
    base = _AXIS_BASE.copy()
    if custom:
        base.update(custom)
        if "tickfont" not in custom:
            base["tickfont"] = dict(color="#000000", size=11, family="Plus Jakarta Sans")
        if "title_font" not in custom:
            base["title_font"] = dict(color="#000000", size=13, family="Plus Jakarta Sans")
    return base

def apply_plotly_layout(fig, **kwargs):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="Plus Jakarta Sans, sans-serif", size=12),
        title_font=dict(size=15, color=TEXT_PRI, family="Plus Jakarta Sans"),
        xaxis=_merge_axis(kwargs.pop("xaxis", None)),
        yaxis=_merge_axis(kwargs.pop("yaxis", None)),
        margin=kwargs.pop("margin", dict(l=20, r=20, t=50, b=20)),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=BORDER,
                    borderwidth=1, orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1,
                    font=dict(color=TEXT_PRI, size=12, family="Plus Jakarta Sans"),
                    title_font=dict(color=TEXT_PRI)),
        showlegend=True,
        hoverlabel=dict(bgcolor=WHITE, font_color=TEXT_PRI, bordercolor=MAGENTA),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    # Force noir sur TOUS les axes sans exception
    fig.update_xaxes(
        tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"),
        title_font=dict(color="#000000", size=14, family="Plus Jakarta Sans"),
        color="#000000",
    )
    fig.update_yaxes(
        tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"),
        title_font=dict(color="#000000", size=14, family="Plus Jakarta Sans"),
        color="#000000",
    )

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard | Intelligence Energetique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #FAFAFA;
    color: {TEXT_PRI};
    -webkit-font-smoothing: antialiased;
}}
.stApp, .main {{ background-color: #FAFAFA !important; }}
.block-container {{ background-color: #FAFAFA !important; padding-top: 2rem !important; max-width: 1400px; }}

section[data-testid="stSidebar"] {{
    background: #FFFFFF;
    border-right: 1px solid #EDE8F8;
    box-shadow: 4px 0 24px rgba(123,47,190,0.06);
}}
section[data-testid="stSidebar"] > div {{ padding: 1.5rem 1rem; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{ color: {TEXT_PRI} !important; font-size: 0.88rem; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    background: {GRAD_CSS};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em;
}}

.header-wrap {{
    background: {GRAD_CSS};
    border-radius: 24px; padding: 32px 44px; margin-bottom: 32px;
    display: flex; align-items: center; gap: 28px;
    box-shadow: 0 8px 40px rgba(123,47,190,0.22);
}}
.header-title {{
    font-size: 1.9rem; font-weight: 800; color: #FFFFFF;
    letter-spacing: -0.03em; margin: 0; line-height: 1.2;
}}
.header-sub {{
    font-size: 0.85rem; color: rgba(255,255,255,0.78);
    font-weight: 400; margin: 6px 0 0 0; letter-spacing: 0.02em;
}}
.header-badge {{
    background: rgba(255,255,255,0.15); border-radius: 50px;
    padding: 6px 16px; font-size: 0.75rem; color: #FFFFFF;
    font-weight: 600; display: inline-block; margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.25);
}}

.kpi-row {{ display: flex; gap: 16px; margin: 16px 0 24px 0; flex-wrap: wrap; }}
.kpi-card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 18px; padding: 20px 24px; flex: 1; min-width: 160px;
    box-shadow: 0 2px 16px rgba(123,47,190,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default; position: relative;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(123,47,190,0.13);
}}
.kpi-label {{
    font-size: 0.75rem; font-weight: 600; color: {TEXT_SEC};
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.85rem; font-weight: 800; color: {TEXT_PRI};
    letter-spacing: -0.04em; line-height: 1;
}}
.kpi-sub {{ font-size: 0.72rem; color: {TEXT_SEC}; margin-top: 6px; font-weight: 400; }}
.kpi-accent {{ background: {GRAD_CSS}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

.stitle {{
    font-size: 1.1rem; font-weight: 700; color: {TEXT_PRI};
    letter-spacing: -0.02em; display: block; margin: 20px 0 10px 0;
    padding-left: 12px;
    border-left: 3px solid {MAGENTA};
}}

.stTabs [data-baseweb="tab-list"] {{
    background: #FFFFFF; border-radius: 14px; padding: 6px;
    border: 1px solid {BORDER}; gap: 4px; margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(123,47,190,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px; padding: 8px 18px; font-weight: 600;
    font-size: 0.83rem; color: {TEXT_SEC}; letter-spacing: 0.01em;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: {GRAD_CSS} !important; color: #FFFFFF !important;
    box-shadow: 0 3px 12px rgba(212,34,122,0.28);
}}

.alert-card {{
    background: rgba(241,90,36,0.06); border-left: 3px solid {ORANGE};
    border-radius: 10px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.85rem; color: {TEXT_PRI};
}}
.alert-ok {{
    background: rgba(123,47,190,0.05); border-left: 3px solid {VIOLET};
    border-radius: 10px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.85rem; color: {TEXT_PRI};
}}

.stTextInput input, .stNumberInput input, .stSelectbox select {{
    background-color: #FFFFFF !important; color: {TEXT_PRI} !important;
    border-radius: 10px !important; border: 1.5px solid #E2D9F3 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 0.88rem !important;
}}
div[data-testid="stNumberInput"] > div {{
    background-color: #FFFFFF !important; border-radius: 10px !important;
    border: 1.5px solid #E2D9F3 !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: #F0EBF8 !important; color: {VIOLET} !important;
    border: none !important; font-weight: 700 !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    background-color: {VIOLET} !important; color: #FFFFFF !important;
}}
div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div {{
    background-color: #FFFFFF !important; border: 1.5px solid #E2D9F3 !important;
    border-radius: 10px !important; color: {TEXT_PRI} !important;
}}
div[data-baseweb="select"] span {{ color: {TEXT_PRI} !important; }}
div[data-baseweb="popover"] ul {{ background-color: #FFFFFF !important; border: 1px solid #E2D9F3 !important; }}
div[data-baseweb="popover"] li {{ background-color: #FFFFFF !important; color: {TEXT_PRI} !important; }}
div[data-baseweb="popover"] li:hover {{ background-color: #F0EBF8 !important; }}
div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] > div {{
    background-color: #FFFFFF !important; color: {TEXT_PRI} !important;
    border: 1.5px solid #E2D9F3 !important; border-radius: 10px !important;
}}
div[data-baseweb="input"] {{ background-color: #FFFFFF !important; border-radius: 10px !important; }}
div[data-baseweb="input"] input {{ background-color: #FFFFFF !important; color: {TEXT_PRI} !important; }}
div[data-testid="stButton"] > button, .stButton > button {{
    background: {GRAD_CSS} !important; color: #FFFFFF !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    padding: 12px 28px !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(123,47,190,0.25) !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; transform: translateY(-1px) !important; }}

hr {{
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, #E2D9F3, transparent); margin: 28px 0;
}}

.sidebar-logo {{ text-align: center; font-size: 3.2rem; margin: 4px 0 8px 0; line-height: 1; }}
.sidebar-title {{
    text-align: center; font-size: 1.15rem; font-weight: 800;
    background: {GRAD_CSS}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin-bottom: 4px;
}}
.sidebar-sub {{
    text-align: center; font-size: 0.72rem; color: {TEXT_SEC};
    letter-spacing: 0.06em; text-transform: uppercase; font-weight: 500;
}}
.sidebar-stat {{
    background: #F8F4FF; border: 1px solid #EDE8F8; border-radius: 12px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: {TEXT_PRI};
}}
.sidebar-stat strong {{
    background: {GRAD_CSS}; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 700;
}}

.footer {{
    text-align: center; color: rgba(107,91,138,0.5); font-size: 0.75rem;
    margin-top: 48px; padding-top: 16px; border-top: 1px solid {BORDER};
}}

@keyframes senelec-spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
@keyframes senelec-pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.6; transform: scale(0.92); }} }}
.senelec-loader {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; padding: 40px 0; }}
.senelec-ring {{
    width: 56px; height: 56px; border-radius: 50%; border: 5px solid transparent;
    border-top-color: {VIOLET}; border-right-color: {MAGENTA}; border-bottom-color: {ORANGE};
    animation: senelec-spin 1s linear infinite; box-shadow: 0 0 18px rgba(123,47,190,0.25);
}}
.senelec-loader-text {{
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.9rem;
    background: {GRAD_CSS}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: senelec-pulse 1.4s ease-in-out infinite; letter-spacing: 0.04em;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_train():
    df = pd.read_csv("energydata_complete.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["rv1", "rv2", "lights"], errors="ignore")

    # Feature engineering
    df["hour"]       = df["date"].dt.hour
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    df["conso_H_1"]  = df["Appliances"].shift(1)
    df["conso_H_2"]  = df["Appliances"].shift(2)
    df["conso_H_6"]  = df["Appliances"].shift(6)
    df["conso_J_1"]  = df["Appliances"].shift(144)
    df["conso_J_7"]  = df["Appliances"].shift(1008)
    df["rolling_6h"] = df["Appliances"].shift(1).rolling(36).mean()
    df["rolling_24h"]= df["Appliances"].shift(1).rolling(144).mean()
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "T_out","RH_out","Windspeed","Tdewpoint","Press_mm_hg","Visibility",
        "T1","T2","T3","T6","T9","RH_1","RH_2","RH_6","RH_9",
        "hour","dayofweek","month","is_weekend",
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "conso_H_1","conso_H_2","conso_H_6","conso_J_1","conso_J_7",
        "rolling_6h","rolling_24h",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["Appliances"]

    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # 5 modeles
    models = {
        "Regression Lineaire": LinearRegression(),
        "Ridge (L2)"         : Ridge(alpha=100),
        "Lasso (L1)"         : Lasso(alpha=0.1),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            subsample=0.8, min_samples_leaf=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        use_scale = name in ("Regression Lineaire","Ridge (L2)","Lasso (L1)")
        Xtr = X_tr_sc if use_scale else X_train.values
        Xte = X_te_sc if use_scale else X_test.values
        model.fit(Xtr, y_train)
        pred = np.maximum(model.predict(Xte), 0)
        results[name] = {
            "model": model, "pred": pred,
            "r2":   round(r2_score(y_test, pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "mae":  round(mean_absolute_error(y_test, pred), 2),
            "mape": round(np.mean(np.abs((y_test.values - pred) / (y_test.values + 1))) * 100, 2),
        }

    # Meilleur modele
    best_name = max(results, key=lambda k: results[k]["r2"])
    best = results[best_name]

    df_test = df.iloc[split:].copy().reset_index(drop=True)
    df_test["Prediction"] = best["pred"]
    df_test["Residu"]     = df_test["Appliances"] - df_test["Prediction"]

    # SHAP — sur 500 observations du jeu de test pour la vitesse
    best_model = results[best_name]["model"]
    is_linear  = best_name in ("Regression Lineaire", "Ridge (L2)", "Lasso (L1)")
    Xte_for_shap = X_te_sc if is_linear else X_test.values
    sample_n = min(500, len(Xte_for_shap))
    rng = np.random.default_rng(42)
    idx_s = rng.choice(len(Xte_for_shap), sample_n, replace=False)
    X_shap = Xte_for_shap[idx_s]
    if is_linear:
        explainer = shap.LinearExplainer(best_model, X_shap)
    else:
        explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)   # (sample_n, n_features)
    shap_mean   = np.abs(shap_values).mean(axis=0) # mean |SHAP| per feature
    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": shap_mean,
        "shap_vals":  [shap_values[:, i] for i in range(len(feature_cols))],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df, df_test, results, best_name, feature_cols, X_test, y_test, scaler, shap_df, X_shap


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════
_loader = st.empty()
_loader.markdown("""
<div class="senelec-loader">
  <div class="senelec-ring"></div>
  <div class="senelec-loader-text">Chargement Intelligence Energetique...</div>
</div>
""", unsafe_allow_html=True)

df, df_test, results, best_name, features, X_test, y_test, _scaler, shap_df, X_shap = load_and_train()
_loader.empty()

best_r2   = results[best_name]["r2"]
best_rmse = results[best_name]["rmse"]
best_mae  = results[best_name]["mae"]
best_mape = results[best_name]["mape"]
best_pred = results[best_name]["pred"]

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-wrap">
  <div>
    <h1 class="header-title">Dashboard — Intelligence Energetique</h1>
    <p class="header-sub">UCI Appliances Energy Prediction · 19 735 observations · 10 min · Jan–Mai 2016</p>
    <span class="header-badge">Meilleur modele : {best_name} · R² = {best_r2:.4f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Conso. moyenne", f"{df['Appliances'].mean():.1f}", "Wh / 10 min"),
    (k2, "Pic maximum",    f"{df['Appliances'].max():.0f}", "Wh enregistre"),
    (k3, "R² meilleur",    f"{best_r2:.4f}", best_name),
    (k4, "RMSE",           f"{best_rmse:.1f}", "Wh d'erreur"),
    (k5, "Observations",   f"{len(df):,}", "apres feature eng."),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value kpi-accent">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">⚡</div>
    <div class="sidebar-title">Dashboard</div>
    <div class="sidebar-sub">Intelligence Energetique</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Filtres")

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    date_range = st.date_input("Plage de dates", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start = pd.Timestamp(date_range[0])
        d_end   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        d_start = pd.Timestamp(date_min)
        d_end   = pd.Timestamp(date_max) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    n_h          = st.slider("Points de prediction affiches", 144, 2016, 288)
    seuil_alerte = st.slider("Seuil d'alerte (Wh)", 100, 1000, 300)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Stats globales")
    st.markdown(f"""
    <div class="sidebar-stat">Conso. moyenne : <strong>{df['Appliances'].mean():.1f} Wh</strong></div>
    <div class="sidebar-stat">Pic maximum : <strong>{df['Appliances'].max():.0f} Wh</strong></div>
    <div class="sidebar-stat">Periode : <strong>Jan - Mai 2016</strong></div>
    <div class="sidebar-stat">Frequence : <strong>10 minutes</strong></div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Groupe 6 · Data Analyst 2025 - 2026")

n_jours = max(1, (d_end.date() - d_start.date()).days + 1)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Temporel",
    "Predictions",
    "Anomalies",
    "Clustering",
    "Modeles",
    "Simulateur",
    "Rapport PDF",
    "Contexte Senelec",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — SERIE TEMPORELLE
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<span class="stitle">Consommation energetique des appareils (Wh)</span>', unsafe_allow_html=True)

    df_plot = df[(df["date"] >= d_start) & (df["date"] <= d_end)].copy()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_plot["date"], y=df_plot["Appliances"],
        mode="lines", name="Consommation",
        line=dict(color=MAGENTA, width=1.2),
        fill="tozeroy",
        fillgradient=dict(type="vertical",
                          colorscale=[[0, "rgba(212,34,122,0)"],
                                      [1, "rgba(212,34,122,0.18)"]]),
    ))
    # Moyenne mobile journaliere
    df_daily = df_plot.set_index("date")["Appliances"].resample("D").mean().reset_index()
    fig_ts.add_trace(go.Scatter(
        x=df_daily["date"], y=df_daily["Appliances"],
        mode="lines", name="Moyenne journaliere",
        line=dict(color=VIOLET, width=2.5),
    ))
    fig_ts.add_hline(y=df_plot["Appliances"].mean(),
                     line_dash="dot", line_color=ORANGE, line_width=1.5,
                     annotation_text="Moyenne globale", annotation_font_color=ORANGE)
    apply_plotly_layout(fig_ts,
        title=f"Consommation horaire — {n_jours} jours",
        xaxis_title="Date", yaxis_title="Wh")
    st.plotly_chart(fig_ts, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<span class="stitle">Profil moyen par heure</span>', unsafe_allow_html=True)
        profil = df.groupby("hour")["Appliances"].mean().reset_index()
        fig_h = go.Figure(go.Scatter(
            x=profil["hour"], y=profil["Appliances"],
            mode="lines+markers",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(color=MAGENTA, size=8, line=dict(color=WHITE, width=1.5)),
            fill="tozeroy",
            fillgradient=dict(type="vertical",
                              colorscale=[[0,"rgba(241,90,36,0)"],[1,"rgba(241,90,36,0.15)"]]),
        ))
        apply_plotly_layout(fig_h, title="Profil journalier moyen",
                            xaxis_title="Heure", yaxis_title="Wh moyen")
        st.plotly_chart(fig_h, use_container_width=True)

    with cb:
        st.markdown('<span class="stitle">Consommation par jour de la semaine</span>', unsafe_allow_html=True)
        jours = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        sem = df.groupby("dayofweek")["Appliances"].mean().reset_index()
        sem["jour"] = sem["dayofweek"].map(lambda x: jours[x])
        colors_bar = [VIOLET if i < 5 else ORANGE for i in range(len(sem))]
        fig_s = go.Figure(go.Bar(
            x=sem["jour"], y=sem["Appliances"],
            marker=dict(color=colors_bar, opacity=0.85, line=dict(width=0)),
            text=sem["Appliances"].round(0), textposition="outside",
        ))
        apply_plotly_layout(fig_s, title="Conso. moyenne par jour", yaxis_title="Wh")
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<span class="stitle">Heatmap heure x jour de la semaine</span>', unsafe_allow_html=True)
    hm = df.groupby(["hour","dayofweek"])["Appliances"].mean().reset_index()
    hm_pivot = hm.pivot(index="dayofweek", columns="hour", values="Appliances")
    hm_pivot.index = [jours[i] for i in hm_pivot.index]
    fig_heat = go.Figure(go.Heatmap(
        z=hm_pivot.values,
        x=[f"{h}h" for h in hm_pivot.columns],
        y=hm_pivot.index,
        colorscale=SENELEC_SCALE,
        colorbar=dict(title=dict(text="Wh", font=dict(color=TEXT_PRI)),
                      tickfont=dict(color=TEXT_PRI)),
        hoverongaps=False,
    ))
    apply_plotly_layout(fig_heat, title="Heatmap consommation moyenne (heure x jour)", height=320)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Matrice de correlation
    st.markdown('<span class="stitle">Matrice de correlation des variables</span>', unsafe_allow_html=True)
    corr_cols = ["Appliances","T_out","RH_out","Windspeed","T1","T2","T3","RH_1","RH_2","Tdewpoint","Press_mm_hg"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_mat  = df[corr_cols].corr()
    text_mat  = [[f"{corr_mat.iloc[i,j]:.2f}" for j in range(len(corr_cols))] for i in range(len(corr_cols))]

    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values, x=corr_cols, y=corr_cols,
        text=text_mat, texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT_PRI),
        colorscale=[[0,"#F15A24"],[0.5,"#FFFFFF"],[1,"#7B2FBE"]],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="r", font=dict(color=TEXT_PRI)),
                      tickfont=dict(color=TEXT_PRI), tickvals=[-1,-0.5,0,0.5,1]),
        hovertemplate="<b>%{y}</b> x <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    apply_plotly_layout(fig_corr, title="Correlation de Pearson — variables numeriques", height=450,
                        xaxis=dict(tickangle=-35, tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
                        yaxis=dict(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_corr, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTIONS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<span class="stitle">Predictions vs Valeurs reelles</span>', unsafe_allow_html=True)

    df_show = df_test.head(n_h)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_show["date"], y=df_show["Appliances"],
        mode="lines", name="Reel",
        line=dict(color=VIOLET, width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=df_show["date"], y=df_show["Prediction"],
        mode="lines", name=f"Prediction ({best_name})",
        line=dict(color=MAGENTA, width=2, dash="dot"),
        fill="tonexty", fillcolor="rgba(212,34,122,0.07)"
    ))
    apply_plotly_layout(fig_pred, title=f"Predictions vs Reel — {n_h} points",
                        xaxis_title="Date", yaxis_title="Wh")
    st.plotly_chart(fig_pred, use_container_width=True)

    pr1, pr2 = st.columns(2)
    with pr1:
        st.markdown('<span class="stitle">Distribution des residus</span>', unsafe_allow_html=True)
        residus = df_test["Residu"].values
        fig_res = go.Figure(go.Histogram(
            x=residus, nbinsx=60,
            marker=dict(color=MAGENTA, opacity=0.75)
        ))
        fig_res.add_vline(x=0, line_color=VIOLET, line_width=2)
        apply_plotly_layout(fig_res, title="Distribution des residus",
                            xaxis_title="Residu (Wh)", yaxis_title="Frequence")
        st.plotly_chart(fig_res, use_container_width=True)

    with pr2:
        st.markdown('<span class="stitle">Predit vs Reel</span>', unsafe_allow_html=True)
        lim = float(max(df_test["Appliances"].max(), df_test["Prediction"].max()))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_test["Appliances"].values[:2000], y=df_test["Prediction"].values[:2000],
            mode="markers", name="Points de test",
            marker=dict(color=MAGENTA, opacity=0.3, size=4)
        ))
        fig_sc.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines", name="Parfait",
            line=dict(color=VIOLET, width=1.5, dash="dash")
        ))
        apply_plotly_layout(fig_sc, title=f"Predit vs Reel (R²={best_r2:.4f})",
                            xaxis_title="Reel (Wh)", yaxis_title="Predit (Wh)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Importance des variables
    best_model_obj = results[best_name]["model"]
    if hasattr(best_model_obj, "feature_importances_"):
        st.markdown('<span class="stitle">Importance des variables</span>', unsafe_allow_html=True)
        imp = pd.Series(best_model_obj.feature_importances_, index=features).sort_values(ascending=False).head(15)
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation="h",
            marker=dict(
                color=[VIOLET if i < 5 else MAGENTA if i < 10 else ORANGE for i in range(len(imp))],
                opacity=0.85, line=dict(width=0)
            )
        ))
        apply_plotly_layout(fig_imp, title="Top 15 variables les plus importantes",
                            xaxis_title="Importance", height=420,
                            margin=dict(l=160, r=30, t=50, b=20),
                            yaxis=dict(tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"),
                                       title_font=dict(color="#3D2A5A"),
                                       automargin=True))
        st.plotly_chart(fig_imp, use_container_width=True)

    # CSV download
    csv_bytes = df_test[["date","Appliances","Prediction","Residu"]].head(n_h).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Telecharger les predictions (CSV)",
        data=csv_bytes,
        file_name="predictions_dashboard.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOMALIES
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<span class="stitle">Detection automatique d\'anomalies</span>', unsafe_allow_html=True)

    residus_all = df_test["Residu"].values
    seuil_sigma = 2 * np.std(residus_all)
    df_test["Anomalie"] = np.abs(residus_all) > seuil_sigma
    n_anomalies = df_test["Anomalie"].sum()
    pct = n_anomalies / len(df_test) * 100

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Anomalies detectees</div>
          <div class="kpi-value kpi-accent">{n_anomalies}</div>
          <div class="kpi-sub">{pct:.1f}% des observations</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Seuil 2 sigma</div>
          <div class="kpi-value kpi-accent">{seuil_sigma:.1f}</div>
          <div class="kpi-sub">Wh d'ecart admis</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        err_anom = np.abs(residus_all[df_test["Anomalie"].values]).mean()
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Erreur moy. anomalies</div>
          <div class="kpi-value kpi-accent">{err_anom:.1f}</div>
          <div class="kpi-sub">Wh</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    df_norm = df_test[~df_test["Anomalie"]]
    df_anom = df_test[df_test["Anomalie"]]

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=df_test["date"], y=df_test["Appliances"],
        mode="lines", name="Reel", line=dict(color=VIOLET, width=1.2)
    ))
    fig_anom.add_trace(go.Scatter(
        x=df_test["date"], y=df_test["Prediction"],
        mode="lines", name="Prediction", line=dict(color=MAGENTA, width=1.2, dash="dot")
    ))
    fig_anom.add_trace(go.Scatter(
        x=df_anom["date"], y=df_anom["Appliances"],
        mode="markers", name=f"Anomalie ({n_anomalies})",
        marker=dict(color=ORANGE, size=8, symbol="x", line=dict(color=WHITE, width=1.5))
    ))
    apply_plotly_layout(fig_anom, title=f"Detection des anomalies (seuil = 2sigma = {seuil_sigma:.1f} Wh)",
                        xaxis_title="Date", yaxis_title="Wh")
    st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown('<span class="stitle">Alertes les plus critiques</span>', unsafe_allow_html=True)
    top_anom = (df_anom[["date","Appliances","Prediction","Residu"]]
                .assign(Ecart_abs=lambda x: x["Residu"].abs())
                .sort_values("Ecart_abs", ascending=False).head(8))

    for _, row in top_anom.iterrows():
        sens = "Sur-consommation" if row["Residu"] > 0 else "Sous-consommation"
        css  = "alert-card" if row["Residu"] > 0 else "alert-ok"
        st.markdown(f"""
        <div class="{css}">
          <strong>{row['date'].strftime('%d/%m/%Y %Hh%M')}</strong> &nbsp;·&nbsp;
          {sens} &nbsp;·&nbsp;
          Reel : <strong>{row['Appliances']:.1f} Wh</strong> &nbsp;|&nbsp;
          Predit : <strong>{row['Prediction']:.1f} Wh</strong> &nbsp;|&nbsp;
          Ecart : <strong>{row['Residu']:+.1f} Wh</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown('<span class="stitle">Depassements du seuil personnalise</span>', unsafe_allow_html=True)
    df_seuil = df_test[df_test["Appliances"] >= seuil_alerte]
    if len(df_seuil) > 0:
        st.markdown(f"""<div class="alert-card">
        {len(df_seuil)} heures depassent le seuil de <strong>{seuil_alerte} Wh</strong>.
        </div>""", unsafe_allow_html=True)
        fig_seuil = go.Figure()
        fig_seuil.add_trace(go.Scatter(
            x=df_test["date"], y=df_test["Appliances"],
            mode="lines", line=dict(color=VIOLET, width=1.2), name="Consommation"))
        fig_seuil.add_hline(y=seuil_alerte, line_color=ORANGE, line_width=2, line_dash="dash",
                            annotation_text=f"Seuil {seuil_alerte} Wh",
                            annotation_font_color=ORANGE)
        apply_plotly_layout(fig_seuil, title=f"Depassements du seuil {seuil_alerte} Wh")
        st.plotly_chart(fig_seuil, use_container_width=True)
    else:
        st.markdown(f'<div class="alert-ok">Aucun depassement du seuil {seuil_alerte} Wh.</div>',
                    unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTERING
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<span class="stitle">Clustering des profils journaliers (K-Means)</span>', unsafe_allow_html=True)

    @st.cache_data
    def compute_clustering(_df):
        pivot = (_df.set_index("date")["Appliances"]
                 .resample("10min").mean()
                 .reset_index())
        pivot["hour_slot"] = pivot["date"].dt.hour * 6 + pivot["date"].dt.minute // 10
        pivot["day"] = pivot["date"].dt.date
        pt = pivot.pivot_table(index="day", columns="hour_slot", values="Appliances")
        pt = pt.dropna()
        k = 4
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pt)
        return pt, labels, km, k

    with st.spinner("Calcul du clustering..."):
        pt, labels, km, k = compute_clustering(df)

    fig_cl = go.Figure()
    colors_cl = [VIOLET, MAGENTA, ORANGE, "#A855F7"]
    slots = [f"{h//6:02d}h{(h%6)*10:02d}" for h in range(144)]
    for c in range(k):
        mask = labels == c
        mean_prof = pt[mask].mean()
        fig_cl.add_trace(go.Scatter(
            x=list(range(144)), y=mean_prof.values,
            mode="lines", name=f"Cluster {c+1} ({mask.sum()} jours)",
            line=dict(color=colors_cl[c % len(colors_cl)], width=2.5),
        ))
    apply_plotly_layout(fig_cl, title="Profils moyens par cluster (consommation journaliere)",
                        xaxis_title="Tranche horaire (10 min)", yaxis_title="Wh moyen",
                        xaxis=dict(tickvals=list(range(0,144,12)),
                                   ticktext=[f"{h}h" for h in range(0,24,2)]))
    st.plotly_chart(fig_cl, use_container_width=True)

    # Distribution des clusters
    cl_dist = pd.Series(labels).value_counts().sort_index()
    fig_dist = go.Figure(go.Bar(
        x=[f"Cluster {i+1}" for i in cl_dist.index],
        y=cl_dist.values,
        marker=dict(color=colors_cl[:k], opacity=0.85, line=dict(width=0)),
        text=cl_dist.values, textposition="outside",
    ))
    apply_plotly_layout(fig_dist, title="Nombre de jours par cluster", yaxis_title="Jours")
    st.plotly_chart(fig_dist, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — MODELES
# ───────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<span class="stitle">Comparaison des modeles</span>', unsafe_allow_html=True)

    df_mod = pd.DataFrame([
        {"Modele": name, "R²": v["r2"], "RMSE": v["rmse"],
         "MAE": v["mae"], "MAPE (%)": v["mape"]}
        for name, v in results.items()
    ]).sort_values("R²", ascending=False).reset_index(drop=True)

    best_r2_idx   = df_mod["R²"].idxmax()
    best_rmse_idx = df_mod["RMSE"].idxmin()

    # Tableau HTML
    rows_html = ""
    for i, row in df_mod.iterrows():
        is_best = i == 0
        bg = f"background:linear-gradient(90deg,rgba(123,47,190,0.08),rgba(212,34,122,0.05));" if is_best else ""
        r2_cell   = f'<td style="color:{VIOLET};font-weight:800;">{row["R²"]:.4f}</td>' if i == best_r2_idx else f'<td>{row["R²"]:.4f}</td>'
        rmse_cell = f'<td style="color:#16a34a;font-weight:800;">{row["RMSE"]:.2f}</td>' if i == best_rmse_idx else f'<td>{row["RMSE"]:.2f}</td>'
        rows_html += f"""<tr style="{bg}font-weight:{'700' if is_best else '400'};">
            <td>{'★ ' if is_best else ''}{row['Modele']}</td>
            {r2_cell}{rmse_cell}
            <td>{row['MAE']:.2f}</td><td>{row['MAPE (%)']:.2f}%</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;font-family:'Plus Jakarta Sans',sans-serif;">
      <thead>
        <tr style="background:{GRAD_CSS};color:#fff;">
          <th style="padding:12px 16px;text-align:left;border-radius:10px 0 0 0;">Modele</th>
          <th style="padding:12px 16px;">R²</th>
          <th style="padding:12px 16px;">RMSE (Wh)</th>
          <th style="padding:12px 16px;">MAE (Wh)</th>
          <th style="padding:12px 16px;border-radius:0 10px 0 0;">MAPE (%)</th>
        </tr>
      </thead>
      <tbody style="background:#fff;">{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        fig_r2 = go.Figure(go.Bar(
            x=df_mod["R²"], y=df_mod["Modele"], orientation="h",
            marker=dict(color=[VIOLET if i==0 else MAGENTA if i==1 else ORANGE for i in range(len(df_mod))],
                        opacity=0.85, line=dict(width=0)),
            text=df_mod["R²"].round(4), textposition="outside",
        ))
        apply_plotly_layout(fig_r2, title="R² par modele (plus haut = meilleur)",
                            xaxis_title="R²", height=320,
                            margin=dict(l=180, r=30, t=50, b=20),
                            yaxis=dict(tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"), automargin=True))
        st.plotly_chart(fig_r2, use_container_width=True)

    with m2:
        fig_rmse = go.Figure(go.Bar(
            x=df_mod["RMSE"], y=df_mod["Modele"], orientation="h",
            marker=dict(color=[VIOLET if i==best_rmse_idx else ORANGE for i in range(len(df_mod))],
                        opacity=0.85, line=dict(width=0)),
            text=df_mod["RMSE"].round(2), textposition="outside",
        ))
        apply_plotly_layout(fig_rmse, title="RMSE par modele (plus bas = meilleur)",
                            xaxis_title="RMSE (Wh)", height=320,
                            margin=dict(l=180, r=30, t=50, b=20),
                            yaxis=dict(tickfont=dict(color="#000000", size=12, family="Plus Jakarta Sans"), automargin=True))
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Radar multicritere
    # Encadre explicatif R²
    r2_best_val = results[best_name]["r2"]
    st.markdown(f"""
    <div class="kpi-card" style="padding:18px 24px;margin:4px 0 20px 0;border-left:4px solid {VIOLET};">
      <div class="kpi-label" style="margin-bottom:8px;">Pourquoi R² = {r2_best_val:.2f} est un bon resultat sur ce dataset ?</div>
      <div style="font-size:0.84rem;color:{TEXT_PRI};line-height:1.8;">
        La consommation des appareils electriques (<b>Appliances</b>) depend largement du <b>comportement humain</b>
        (allumer la cuisine, la TV, le chauffage…), qui est par nature <b>aleatoire et non mesure</b> par les capteurs.
        La litterature scientifique sur ce dataset UCI rapporte des R² entre <b>0.45 et 0.60</b> pour les meilleurs modeles.
        Notre {best_name} a R²={r2_best_val:.4f} se situe dans cette plage — ce qui confirme que le modele capture
        correctement la part previsible de la consommation. Le bruit residuel est <i>irreductible</i> avec les features disponibles.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="stitle">Comparaison multicritere (radar)</span>', unsafe_allow_html=True)
    r2_n    = df_mod["R²"] / df_mod["R²"].max()
    rmse_n  = 1 - (df_mod["RMSE"] - df_mod["RMSE"].min()) / (df_mod["RMSE"].max() - df_mod["RMSE"].min() + 1e-9)
    mae_n   = 1 - (df_mod["MAE"] - df_mod["MAE"].min()) / (df_mod["MAE"].max() - df_mod["MAE"].min() + 1e-9)
    mape_n  = 1 - (df_mod["MAPE (%)"] - df_mod["MAPE (%)"].min()) / (df_mod["MAPE (%)"].max() - df_mod["MAPE (%)"].min() + 1e-9)
    speed_n = pd.Series([1.0, 0.9, 0.85, 0.5, 0.4], index=range(len(df_mod)))
    interp_n= pd.Series([1.0, 0.9, 0.8, 0.5, 0.3], index=range(len(df_mod)))

    categories = ["R²","RMSE","MAE","Vitesse","Interpretabilite"]
    colors_r = [VIOLET, MAGENTA, ORANGE, "#A855F7", "#F97316"]

    def hex_to_rgba(h, a=0.08):
        h = h.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    fig_radar = go.Figure()
    for i, (_, row) in enumerate(df_mod.iterrows()):
        vals = [r2_n.iloc[i], rmse_n.iloc[i], mae_n.iloc[i], speed_n.iloc[i], interp_n.iloc[i]]
        vals += [vals[0]]
        c = colors_r[i % len(colors_r)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            name=row["Modele"],
            line=dict(color=c, width=2), fill="toself",
            fillcolor=hex_to_rgba(c, 0.08), opacity=0.9,
        ))
    apply_plotly_layout(fig_radar, title="Comparaison multicritere",
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0,1], color=TEXT_SEC,
                                            gridcolor="rgba(123,47,190,0.12)"),
                            angularaxis=dict(color=TEXT_PRI, gridcolor="rgba(123,47,190,0.12)")
                        ), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.markdown('<span class="stitle">Interpretabilite SHAP — Impact des variables</span>', unsafe_allow_html=True)
    st.markdown(
        f"Analyse SHAP (SHapley Additive exPlanations) sur le modele **{best_name}**. "
        "Chaque valeur SHAP mesure la contribution d'une variable a la prediction, "
        "en valeur absolue moyennee sur 500 observations du jeu de test."
    )

    top_n = st.slider("Nombre de variables a afficher", 5, min(20, len(shap_df)), 15, key="shap_top")
    df_top = shap_df.head(top_n)

    sh1, sh2 = st.columns(2)

    # Graphique 1 : importance SHAP (bar)
    with sh1:
        colors_shap = []
        for i in range(len(df_top)):
            frac = i / max(len(df_top) - 1, 1)
            if frac < 0.5:
                colors_shap.append(VIOLET)
            elif frac < 0.8:
                colors_shap.append(MAGENTA)
            else:
                colors_shap.append(ORANGE)

        fig_shap_bar = go.Figure(go.Bar(
            x=df_top["importance"][::-1],
            y=df_top["feature"][::-1],
            orientation="h",
            marker=dict(color=colors_shap[::-1], opacity=0.88, line=dict(width=0)),
            text=df_top["importance"][::-1].round(3),
            textposition="outside",
        ))
        apply_plotly_layout(fig_shap_bar,
            title="Importance SHAP moyenne |SHAP|",
            xaxis_title="Importance (Wh)",
            height=max(320, top_n * 24),
            margin=dict(l=160, r=40, t=50, b=20),
            yaxis=dict(automargin=True, tickfont=dict(color="#000000", size=11, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True)

    # Graphique 2 : beeswarm SHAP (scatter, valeurs SHAP colorees par valeur de feature)
    with sh2:
        fig_bee = go.Figure()
        feat_labels = df_top["feature"].tolist()
        for rank, (_, row) in enumerate(df_top.iterrows()):
            sv  = row["shap_vals"]
            jit = np.random.default_rng(rank).uniform(-0.3, 0.3, len(sv))
            y_pos = np.full(len(sv), float(rank)) + jit * 0.4
            fig_bee.add_trace(go.Scatter(
                x=sv, y=y_pos,
                mode="markers",
                name=row["feature"],
                marker=dict(
                    color=sv, colorscale=[[0, "#F8F4FF"], [0.5, MAGENTA], [1, ORANGE]],
                    size=4, opacity=0.6,
                    showscale=(rank == 0),
                    colorbar=dict(
                        title=dict(text="Valeur SHAP", font=dict(color="#000000", size=11)),
                        tickfont=dict(color="#000000", size=10),
                        len=0.5, x=1.02,
                    ) if rank == 0 else None,
                ),
                showlegend=False,
            ))
        fig_bee.add_vline(x=0, line_dash="dot", line_color=VIOLET, line_width=1.5)
        apply_plotly_layout(fig_bee,
            title="Distribution des valeurs SHAP (beeswarm)",
            xaxis_title="Valeur SHAP (Wh)",
            height=max(320, top_n * 24),
            margin=dict(l=160, r=60, t=50, b=20),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(feat_labels))),
                ticktext=feat_labels,
                automargin=True,
                tickfont=dict(color="#000000", size=11, family="Plus Jakarta Sans"),
            ),
        )
        st.plotly_chart(fig_bee, use_container_width=True)

    # Encadre pedagogique
    st.markdown(f"""
    <div class="kpi-card" style="padding:18px 24px;margin-top:8px;">
      <div class="kpi-label" style="margin-bottom:8px;">Comment lire ces graphiques ?</div>
      <div style="font-size:0.84rem;color:{TEXT_PRI};line-height:1.75;">
        <b>Importance SHAP</b> : plus la barre est longue, plus la variable influence la prediction en moyenne.<br>
        <b>Beeswarm</b> : chaque point est une observation. Un SHAP positif pousse la prediction a la hausse,
        negatif a la baisse. La couleur indique la valeur de la variable (violet = faible, orange = elevee).
      </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — SIMULATEUR
# ───────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<span class="stitle">Simulateur de consommation</span>', unsafe_allow_html=True)
    st.markdown("Entrez les conditions et obtenez une prediction instantanee de la consommation.")

    _JOURS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    _MOIS_NOM = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                 "Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Conditions meteorologiques**")
        t_out    = st.number_input("Temperature exterieure (°C)", -10.0, 40.0, 6.0)
        rh_out   = st.number_input("Humidite exterieure (%)", 10.0, 100.0, 92.0)
        wind     = st.number_input("Vitesse du vent (m/s)", 0.0, 20.0, 7.0)
        tdew     = st.number_input("Point de rosee (°C)", -20.0, 25.0, 5.3)
        press    = st.number_input("Pression (mm Hg)", 700.0, 800.0, 733.5)
        vis      = st.number_input("Visibilite (km)", 0.0, 100.0, 63.0)

    with s2:
        st.markdown("**Conditions interieures**")
        t1       = st.number_input("Temperature piece 1 (°C)", 15.0, 35.0, 19.9)
        rh1      = st.number_input("Humidite piece 1 (%)", 20.0, 80.0, 47.6)
        conso_h1 = st.number_input("Consommation H-1 (Wh)", 10.0, 1000.0, 60.0)
        conso_j1 = st.number_input("Consommation J-1 (Wh)", 10.0, 1000.0, 60.0)

    with s3:
        st.markdown("**Contexte temporel**")
        heure    = st.slider("Heure de la journee", 0, 23, 17)
        jour_sem = st.selectbox("Jour de la semaine", _JOURS)
        mois_nom = st.selectbox("Mois", _MOIS_NOM, index=0)
        modele_sim = st.selectbox("Modele a utiliser", list(results.keys()), index=list(results.keys()).index(best_name))

    with st.expander("Parametres avances — autres pieces (optionnel)"):
        pa1, pa2 = st.columns(2)
        with pa1:
            t2  = st.number_input("Temperature piece 2 (°C)", 15.0, 35.0, float(round(t1 - 0.3, 1)))
            t3  = st.number_input("Temperature piece 3 (°C)", 15.0, 35.0, float(round(t1 - 0.5, 1)))
            t6  = st.number_input("Temperature piece 6 (°C)", 15.0, 35.0, float(round(t1 - 1.0, 1)))
            t9  = st.number_input("Temperature piece 9 (°C)", 15.0, 35.0, float(round(t1 - 1.5, 1)))
        with pa2:
            rh2 = st.number_input("Humidite piece 2 (%)", 20.0, 80.0, float(round(rh1 + 1.0, 1)))
            rh6 = st.number_input("Humidite piece 6 (%)", 20.0, 80.0, float(round(max(20.0, rh1 - 2.0), 1)))
            rh9 = st.number_input("Humidite piece 9 (%)", 20.0, 80.0, float(round(rh1 + 0.5, 1)))

    if st.button("Lancer la prediction", use_container_width=True):
        jour_idx = _JOURS.index(jour_sem)
        mois     = _MOIS_NOM.index(mois_nom) + 1
        is_we    = 1 if jour_idx >= 5 else 0

        feat_sim = {
            "T_out": t_out, "RH_out": rh_out, "Windspeed": wind,
            "Tdewpoint": tdew, "Press_mm_hg": press, "Visibility": vis,
            "T1": t1, "T2": t2, "T3": t3, "T6": t6, "T9": t9,
            "RH_1": rh1, "RH_2": rh2, "RH_6": rh6, "RH_9": rh9,
            "hour": heure, "dayofweek": jour_idx, "month": mois, "is_weekend": is_we,
            "hour_sin": np.sin(2*np.pi*heure/24), "hour_cos": np.cos(2*np.pi*heure/24),
            "dow_sin": np.sin(2*np.pi*jour_idx/7), "dow_cos": np.cos(2*np.pi*jour_idx/7),
            "month_sin": np.sin(2*np.pi*mois/12), "month_cos": np.cos(2*np.pi*mois/12),
            "conso_H_1": conso_h1, "conso_H_2": conso_h1*0.95,
            "conso_H_6": conso_h1*0.9, "conso_J_1": conso_j1, "conso_J_7": conso_j1*0.98,
            "rolling_6h": conso_h1*0.92, "rolling_24h": conso_j1*0.95,
        }
        row_df = pd.DataFrame([[feat_sim.get(f, 0) for f in features]], columns=features)
        row_sc = _scaler.transform(row_df)

        # Prediction de tous les modeles en une passe
        all_preds = {}
        for nm, res in results.items():
            Xin = row_sc if nm in ("Regression Lineaire","Ridge (L2)","Lasso (L1)") else row_df.values
            all_preds[nm] = float(np.maximum(res["model"].predict(Xin), 0)[0])

        pred_sim = all_preds[modele_sim]
        r2_sim   = results[modele_sim]["r2"]

        # Carte resultat principale
        niveau = "Faible" if pred_sim < 50 else "Moderee" if pred_sim < 150 else "Elevee" if pred_sim < 400 else "Tres elevee"
        couleur_niv = VIOLET if pred_sim < 50 else MAGENTA if pred_sim < 150 else ORANGE if pred_sim < 400 else "#DC2626"
        st.markdown(f"""
        <div style="background:{GRAD_CSS};border-radius:18px;padding:28px 36px;margin:20px 0;text-align:center;">
          <div style="color:rgba(255,255,255,0.7);font-size:0.9rem;font-weight:600;margin-bottom:8px;">
            Consommation predite — {modele_sim}
          </div>
          <div style="color:#FFFFFF;font-size:3.2rem;font-weight:800;letter-spacing:-0.04em;">
            {pred_sim:.1f} <span style="font-size:1.4rem;font-weight:400;">Wh</span>
          </div>
          <div style="color:rgba(255,255,255,0.65);font-size:0.8rem;margin-top:8px;">
            R² = {r2_sim:.4f} &nbsp;·&nbsp; {mois_nom}, {heure:02d}h, {jour_sem}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="alert-card" style="border-color:{couleur_niv};">Niveau de consommation : <strong style="color:{couleur_niv};">{niveau}</strong></div>',
                    unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_sim,
            number=dict(suffix=" Wh", font=dict(size=28, color=TEXT_PRI, family="Plus Jakarta Sans")),
            gauge=dict(
                axis=dict(range=[0, 600], tickcolor="#000000",
                          tickfont=dict(color="#000000", size=11, family="Plus Jakarta Sans")),
                bar=dict(color=couleur_niv),
                steps=[
                    dict(range=[0,   50],  color="rgba(123,47,190,0.10)"),
                    dict(range=[50,  150], color="rgba(212,34,122,0.10)"),
                    dict(range=[150, 400], color="rgba(241,90,36,0.10)"),
                    dict(range=[400, 600], color="rgba(220,38,38,0.10)"),
                ],
                threshold=dict(line=dict(color=ORANGE, width=3), thickness=0.8, value=pred_sim),
            ),
            title=dict(text="Consommation (Wh)", font=dict(color=TEXT_PRI, size=13, family="Plus Jakarta Sans")),
        ))
        apply_plotly_layout(fig_gauge, height=280, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Comparaison tous modeles
        st.markdown('<span class="stitle">Comparaison des 5 modeles</span>', unsafe_allow_html=True)
        mod_names = list(all_preds.keys())
        mod_vals  = list(all_preds.values())
        mod_colors = [couleur_niv if n == modele_sim else "rgba(123,47,190,0.45)" for n in mod_names]
        fig_comp = go.Figure(go.Bar(
            x=mod_names, y=mod_vals,
            marker=dict(color=mod_colors, line=dict(width=0)),
            text=[f"{v:.1f} Wh" for v in mod_vals], textposition="outside",
        ))
        apply_plotly_layout(fig_comp,
            title="Prediction de chaque modele pour ces conditions",
            yaxis_title="Consommation (Wh)", height=320,
            margin=dict(l=20, r=20, t=50, b=80),
            xaxis=dict(tickangle=-20, tickfont=dict(color="#000000", size=11, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 7 — RAPPORT PDF
# ───────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown('<span class="stitle">Generateur de rapport PDF</span>', unsafe_allow_html=True)
    st.markdown("Exportez un rapport complet du projet en un clic.")

    r1, r2c = st.columns([1, 2])
    with r1:
        auteur  = st.text_input("Auteur", "Groupe 6 : Jean Paul MALAN, Guy Roger Junior GNAORE, COULIBALY Fobeh, TOURE Awa")
        promo   = st.text_input("Promotion", "2025-2026 · UCAD Dakar")
    with r2c:
        st.markdown(f"""
        <div class="kpi-card" style="padding:20px;">
          <div class="kpi-label">Contenu du rapport</div>
          <div style="font-size:0.85rem;color:{TEXT_PRI};margin-top:8px;line-height:1.8;">
            Page de garde · Resume executif · Metriques · Tableau comparatif 5 modeles ·
            Variables importantes · Analyse anomalies · Recommandations · Conclusion
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Generer le rapport PDF", use_container_width=True):
        with st.spinner("Generation en cours..."):
            date_pdf = datetime.date.today().strftime("%Y%m%d")

            class SenelecPDF(FPDF):
                def header(self):
                    if self.page_no() > 1:
                        self.set_fill_color(123, 47, 190)
                        self.rect(0, 0, 70, 3, "F")
                        self.set_fill_color(212, 34, 122)
                        self.rect(70, 0, 70, 3, "F")
                        self.set_fill_color(241, 90, 36)
                        self.rect(140, 0, 70, 3, "F")
                        self.ln(6)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(150, 150, 150)
                    self.cell(0, 10, f"Dashboard - Prediction Energetique | Page {self.page_no()}", align="C")

                def chapter_title(self, title):
                    self.set_font("Helvetica", "B", 12)
                    self.set_text_color(123, 47, 190)
                    self.ln(4)
                    self.cell(0, 8, title, ln=True)
                    self.set_draw_color(212, 34, 122)
                    self.set_line_width(0.5)
                    self.line(self.get_x(), self.get_y(), self.get_x() + 170, self.get_y())
                    self.ln(4)

                def body_text(self, txt):
                    self.set_font("Helvetica", "", 10)
                    self.set_text_color(50, 50, 50)
                    self.multi_cell(0, 6, txt)
                    self.ln(3)

            pdf = SenelecPDF()
            pdf.set_margins(18, 18, 18)
            pdf.set_auto_page_break(True, margin=20)
            pdf.add_page()

            # Page de garde
            pdf.set_fill_color(123, 47, 190)
            pdf.rect(0, 0, 70, 297, "F")
            pdf.set_fill_color(212, 34, 122)
            pdf.rect(70, 0, 70, 297, "F")
            pdf.set_fill_color(241, 90, 36)
            pdf.rect(140, 0, 70, 297, "F")
            pdf.set_fill_color(255, 255, 255)
            pdf.rect(20, 60, 170, 180, "F")
            pdf.set_xy(25, 75)
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(26, 10, 46)
            pdf.multi_cell(160, 10, "RAPPORT DE PREDICTION DE CONSOMMATION ENERGETIQUE", align="C")
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(123, 47, 190)
            pdf.cell(0, 8, "Dashboard - Intelligence Energetique", align="C", ln=True)
            pdf.ln(10)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(80, 80, 80)
            for ligne in [
                "Groupe 6",
                "Jean Paul MALAN  |  Guy Roger Junior GNAORE  |  COULIBALY Fobeh  |  TOURE Awa",
                f"Promotion : {promo}",
                f"Date : {datetime.date.today().strftime('%d/%m/%Y')}",
                "Dataset : UCI Appliances Energy Prediction",
                "19 735 observations · Frequence 10 min · Jan-Mai 2016",
            ]:
                pdf.cell(0, 7, ligne, align="C", ln=True)

            # Page 2
            pdf.add_page()
            pdf.chapter_title("1. Resume Executif")
            pdf.body_text(
                f"Ce projet evalue plusieurs modeles de machine learning pour la prediction "
                f"de la consommation energetique des appareils electriques. Le dataset UCI "
                f"contient 19 735 observations collectees toutes les 10 minutes entre "
                f"janvier et mai 2016 dans une maison en Belgique.\n\n"
                f"Le meilleur modele obtenu est {best_name} avec R²={best_r2:.4f}. "
                f"Un R² de 0.45-0.60 est attendu sur ce dataset car la consommation des "
                f"appareils menagers depend fortement du comportement humain - une part "
                f"intrinsèquement aleatoire que nul modele ne peut capturer. "
                f"Le RMSE de {best_rmse:.1f} Wh represente une erreur de "
                f"{best_rmse/97.7*100:.1f}% relativement a la consommation moyenne (97.7 Wh), "
                f"ce qui est satisfaisant pour une application reelle."
            )
            pdf.chapter_title("2. Metriques du Meilleur Modele")
            for label, val in [("Meilleur modele", best_name),
                                ("R²", f"{best_r2:.4f}"),
                                ("RMSE", f"{best_rmse:.2f} Wh"),
                                ("MAE", f"{best_mae:.2f} Wh"),
                                ("MAPE", f"{best_mape:.2f}%")]:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(123, 47, 190)
                pdf.cell(55, 7, f"{label} :", ln=False)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(0, 7, val, ln=True)
            pdf.ln(4)

            pdf.chapter_title("3. Comparaison des Modeles")
            headers = ["Modele", "R2", "RMSE", "MAE", "MAPE"]
            widths  = [65, 25, 25, 25, 25]
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(123, 47, 190)
            pdf.set_text_color(255, 255, 255)
            for h, w in zip(headers, widths):
                pdf.cell(w, 8, h, border=1, fill=True)
            pdf.ln()
            for i, row in df_mod.iterrows():
                pdf.set_font("Helvetica", "B" if i == 0 else "", 9)
                pdf.set_text_color(123, 47, 190 if i == 0 else 50)
                pdf.set_fill_color(240, 235, 248 if i == 0 else 255)
                vals_row = [row["Modele"], str(row["R²"]), str(row["RMSE"]),
                            str(row["MAE"]), f"{row['MAPE (%)']}%"]
                for v, w in zip(vals_row, widths):
                    pdf.cell(w, 7, str(v)[:20], border=1, fill=(i==0))
                pdf.ln()
            pdf.ln(6)

            pdf.chapter_title("4. Interpretabilite SHAP - Variables les plus influentes")
            pdf.body_text(
                "L'analyse SHAP (SHapley Additive exPlanations) identifie les variables "
                "qui contribuent le plus a la prediction, en moyenne sur 500 observations du jeu de test :"
            )
            top5 = shap_df.head(5)
            for rank, (_, srow) in enumerate(top5.iterrows(), 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(123, 47, 190)
                pdf.cell(8, 7, f"{rank}.", ln=False)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(0, 7, f"{srow['feature']}  (importance SHAP moyenne : {srow['importance']:.3f} Wh)", ln=True)
            pdf.ln(4)

            pdf.chapter_title("5. Analyse des Anomalies")
            n_anom_pdf = int(df_test["Anomalie"].sum()) if "Anomalie" in df_test.columns else 0
            pct_anom   = n_anom_pdf / len(df_test) * 100 if len(df_test) > 0 else 0
            pdf.body_text(
                f"Le systeme de detection d'anomalies (seuil 2-sigma sur les residus) a identifie "
                f"{n_anom_pdf} anomalies sur {len(df_test)} observations du jeu de test "
                f"({pct_anom:.1f}% des donnees). Ces points correspondent a des pics de consommation "
                f"inattendus ou des chutes soudaines pouvant indiquer une panne ou un comportement atypique."
            )

            pdf.chapter_title("6. Recommandations")
            recs = [
                "Deployer le modele Ridge comme solution de reference (meilleur compromis R2/interpretabilite).",
                "Enrichir le dataset avec des donnees meteorologiques recentes pour ameliorer la generalisation.",
                "Explorer des modeles de deep learning (LSTM) pour capturer les dependances temporelles longues.",
                "Mettre en place un re-entrainement mensuel pour adapter le modele aux changements saisonniers.",
                "Surveiller les pics de consommation aux heures 18h-22h : erreurs les plus elevees.",
            ]
            for i, rec in enumerate(recs, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(123, 47, 190)
                pdf.cell(8, 7, f"{i}.", ln=False)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 7, rec)
                pdf.ln(1)

            pdf.chapter_title("7. Conclusion")
            pdf.body_text(
                f"Le meilleur modele ({best_name}) atteint R²={best_r2:.4f}, "
                f"RMSE={best_rmse:.2f} Wh sur le jeu de test chronologique. "
                f"Le dashboard interactif developpe permet une exploration complete, "
                f"des predictions en temps reel via le simulateur, et la detection automatique "
                f"des anomalies de consommation."
            )

            pdf_bytes = bytes(pdf.output())

        st.success("Rapport genere avec succes !")
        st.download_button(
            label="Telecharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"rapport_dashboard_{date_pdf}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ───────────────────────────────────────────────────────────────────────────
# TAB 8 — CONTEXTE SENELEC
# ───────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<span class="stitle">Contexte energetique Senelec 2022–2024</span>', unsafe_allow_html=True)
    st.markdown(
        "Donnees journalieres du reseau electrique senegalais (Senelec) sur 3 ans. "
        "Ces indicateurs macroeconomiques complementent l'analyse de consommation du modele UCI."
    )

    @st.cache_data
    def load_senelec():
        ds = pd.read_csv("Donnees_senelec_2022_2024.csv", sep=";", parse_dates=["date"], dayfirst=True)
        ds = ds.sort_values("date").reset_index(drop=True)
        return ds

    ds = load_senelec()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Ventes totales moy.</div>
          <div class="kpi-value">{ds['ventes_totales_MWh'].mean():,.0f}</div>
          <div class="kpi-sub">MWh / jour</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Production moy.</div>
          <div class="kpi-value">{ds['production_totale_MWh'].mean():,.0f}</div>
          <div class="kpi-sub">MWh / jour</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Taux renouvelable moy.</div>
          <div class="kpi-value kpi-accent">{ds['taux_renouvelable_pct'].mean():.1f}%</div>
          <div class="kpi-sub">Part EnR dans le mix</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Prix moyen</div>
          <div class="kpi-value">{ds['prix_moyen_FCFA_kWh'].mean():,.0f}</div>
          <div class="kpi-sub">FCFA / kWh</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphique 1 : ventes totales par annee ─────────────────────────────
    st.markdown('<span class="stitle">Evolution journaliere des ventes</span>', unsafe_allow_html=True)
    year_sel = st.multiselect("Annee(s)", [2022, 2023, 2024], default=[2022, 2023, 2024], key="sn_year")
    ds_f = ds[ds["annee"].isin(year_sel)] if year_sel else ds

    fig_ventes = go.Figure()
    colors_yr = {2022: VIOLET, 2023: MAGENTA, 2024: ORANGE}
    for yr in sorted(ds_f["annee"].unique()):
        sub = ds_f[ds_f["annee"] == yr]
        fig_ventes.add_trace(go.Scatter(
            x=sub["date"], y=sub["ventes_totales_MWh"],
            mode="lines", name=str(yr),
            line=dict(color=colors_yr.get(yr, VIOLET), width=1.8),
        ))
    apply_plotly_layout(fig_ventes,
        title="Ventes totales journalieres (MWh)",
        xaxis_title="Date", yaxis_title="MWh", height=340)
    st.plotly_chart(fig_ventes, use_container_width=True)

    # ── Graphique 2 : mix energetique (thermique vs renouvelable) ──────────
    st.markdown('<span class="stitle">Mix energetique — Thermique vs Renouvelable</span>', unsafe_allow_html=True)
    ds_monthly = ds_f.groupby(["annee", "mois"])[["taux_thermique_pct", "taux_renouvelable_pct"]].mean().reset_index()
    ds_monthly["periode"] = ds_monthly["annee"].astype(str) + "-" + ds_monthly["mois"].astype(str).str.zfill(2)

    fig_mix = go.Figure()
    fig_mix.add_trace(go.Bar(
        x=ds_monthly["periode"], y=ds_monthly["taux_thermique_pct"],
        name="Thermique (%)", marker_color=MAGENTA, opacity=0.85,
    ))
    fig_mix.add_trace(go.Bar(
        x=ds_monthly["periode"], y=ds_monthly["taux_renouvelable_pct"],
        name="Renouvelable (%)", marker_color=VIOLET, opacity=0.85,
    ))
    fig_mix.update_layout(barmode="stack")
    apply_plotly_layout(fig_mix,
        title="Part thermique vs renouvelable par mois (%)",
        xaxis_title="Mois", yaxis_title="%", height=340)
    st.plotly_chart(fig_mix, use_container_width=True)

    # ── Graphique 3 : ventes par segment (BT / MT / HT) ───────────────────
    st.markdown('<span class="stitle">Repartition des ventes par segment</span>', unsafe_allow_html=True)
    seg_cols = {"BT (Basse Tension)": "ventes_BT_MWh",
                "MT (Moyenne Tension)": "ventes_MT_MWh",
                "HT (Haute Tension)": "ventes_HT_MWh"}
    totals = {k: ds_f[v].sum() for k, v in seg_cols.items()}
    fig_seg = go.Figure(go.Pie(
        labels=list(totals.keys()),
        values=list(totals.values()),
        hole=0.52,
        marker=dict(colors=[VIOLET, MAGENTA, ORANGE]),
        textfont=dict(color="#000000", size=12),
    ))
    apply_plotly_layout(fig_seg, title="Part des ventes par segment", height=340, showlegend=True)
    st.plotly_chart(fig_seg, use_container_width=True)

    # ── Graphique 4 : correlation temperature / ventes ─────────────────────
    st.markdown('<span class="stitle">Temperature vs Ventes — correlation</span>', unsafe_allow_html=True)
    fig_corr = go.Figure(go.Scatter(
        x=ds_f["temperature_C"], y=ds_f["ventes_totales_MWh"],
        mode="markers",
        marker=dict(
            color=ds_f["mois"], colorscale=SENELEC_SCALE,
            size=5, opacity=0.55,
            colorbar=dict(title=dict(text="Mois", font=dict(color="#000000", size=11)),
                          tickfont=dict(color="#000000", size=10)),
        ),
        text=ds_f["date"].dt.strftime("%d/%m/%Y"),
        hovertemplate="Date: %{text}<br>Temp: %{x:.1f}°C<br>Ventes: %{y:,.0f} MWh<extra></extra>",
    ))
    apply_plotly_layout(fig_corr,
        title="Correlation temperature / ventes (chaque point = 1 jour)",
        xaxis_title="Temperature (°C)", yaxis_title="Ventes (MWh)", height=360)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tableau mensuel ────────────────────────────────────────────────────
    st.markdown('<span class="stitle">Resume mensuel</span>', unsafe_allow_html=True)
    noms_mois = {1:"Jan",2:"Fev",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                 7:"Jul",8:"Aou",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    ds_tab = ds_f.groupby(["annee","mois"]).agg(
        Ventes_MWh=("ventes_totales_MWh","sum"),
        Production_MWh=("production_totale_MWh","sum"),
        Temp_moy=("temperature_C","mean"),
        EnR_pct=("taux_renouvelable_pct","mean"),
        Prix_FCFA=("prix_moyen_FCFA_kWh","mean"),
    ).reset_index()
    ds_tab["Mois"] = ds_tab["mois"].map(noms_mois) + " " + ds_tab["annee"].astype(str)
    ds_tab = ds_tab[["Mois","Ventes_MWh","Production_MWh","Temp_moy","EnR_pct","Prix_FCFA"]]
    ds_tab.columns = ["Mois","Ventes (MWh)","Production (MWh)","Temp moy (°C)","EnR (%)","Prix (FCFA/kWh)"]
    for col in ["Ventes (MWh)","Production (MWh)"]:
        ds_tab[col] = ds_tab[col].map(lambda x: f"{x:,.0f}")
    for col in ["Temp moy (°C)","EnR (%)","Prix (FCFA/kWh)"]:
        ds_tab[col] = ds_tab[col].map(lambda x: f"{x:.1f}")
    st.dataframe(ds_tab, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  Dashboard — Intelligence Energetique &nbsp;|&nbsp;
  Groupe 6 &nbsp;·&nbsp; Jean Paul MALAN · Guy Roger Junior GNAORE · COULIBALY Fobeh · TOURE Awa &nbsp;·&nbsp; Projet 2 · Data Analyst 2025 - 2026 &nbsp;|&nbsp;
  {best_name} · R² = {best_r2:.4f}
</div>
""", unsafe_allow_html=True)
