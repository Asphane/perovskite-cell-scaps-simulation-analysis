import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution

class JV_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

st.set_page_config(page_title="PSC AI Simulator", page_icon="🌞", layout="wide", initial_sidebar_state="expanded")

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'saved_designs' not in st.session_state:
    st.session_state.saved_designs = []
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = []

# ── THEME-AWARE CSS ────────────────────────────────────────────────────────────
def get_css(theme):
    if theme == 'light':
        bg_main = "linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 50%, #f5f0ff 100%)"
        bg_sidebar = "rgba(255,255,255,0.95)"
        text_color = "#1e293b"
        text_sec = "#64748b"
        metric_bg = "rgba(255,255,255,0.7)"
        metric_border = "rgba(59,130,246,0.2)"
        metric_label = "#64748b"
        metric_val = "#0f172a"
        tab_bg = "rgba(255,255,255,0.6)"
        tab_color = "#64748b"
        tab_active = "linear-gradient(135deg,#2563eb,#7c3aed)"
        btn_bg = "linear-gradient(135deg, #2563eb, #7c3aed)"
        btn_shadow = "rgba(124,58,237,0.3)"
        border_color = "rgba(59,130,246,0.2)"
        hero_bg = "linear-gradient(135deg, rgba(37,99,235,0.1), rgba(124,58,237,0.1))"
        hero_border = "rgba(59,130,246,0.2)"
        header_color = "#0f172a"
        slider_color = "#1e293b"
        table_bg = "rgba(255,255,255,0.5)"
        table_th_bg = "rgba(37,99,235,0.2)"
        table_th_color = "#1e40af"
        table_td_color = "#1e293b"
        hover_metric_border = "rgba(59,130,246,0.5)"
    else:
        bg_main = "linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #1a0a2e 100%)"
        bg_sidebar = "rgba(13,17,38,0.95)"
        text_color = "#e2e8f0"
        text_sec = "#94a3b8"
        metric_bg = "rgba(255,255,255,0.04)"
        metric_border = "rgba(99,179,237,0.2)"
        metric_label = "#94a3b8"
        metric_val = "#f0f9ff"
        tab_bg = "rgba(255,255,255,0.04)"
        tab_color = "#94a3b8"
        tab_active = "linear-gradient(135deg,#1e40af,#7c3aed)"
        btn_bg = "linear-gradient(135deg, #1e40af, #7c3aed)"
        btn_shadow = "rgba(124,58,237,0.35)"
        border_color = "rgba(99,179,237,0.15)"
        hero_bg = "linear-gradient(135deg, rgba(30,64,175,0.25), rgba(124,58,237,0.25))"
        hero_border = "rgba(99,179,237,0.2)"
        header_color = "#e2e8f0"
        slider_color = "#cbd5e1"
        table_bg = "rgba(255,255,255,0.03)"
        table_th_bg = "rgba(30,64,175,0.35)"
        table_th_color = "#93c5fd"
        table_td_color = "#e2e8f0"
        hover_metric_border = "rgba(99,179,237,0.55)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {bg_main}; }}

[data-testid="stSidebar"] {{
    background: {bg_sidebar};
    border-right: 1px solid {border_color};
    padding: 0.5rem 0;
}}
[data-testid="stSidebar"] .stMarkdown p {{ color: {text_sec}; font-size: 0.85rem; }}
[data-testid="stSidebar"] .stSelectbox {{ margin-bottom: 0.6rem; }}
[data-testid="stSidebar"] .stCaption {{ color: {text_sec} !important; font-size: 0.75rem !important; margin-bottom: 0.3rem !important; }}
[data-testid="stSidebar"] .stButton {{ margin-top: 0.2rem; margin-bottom: 0.2rem; }}

[data-testid="metric-container"] {{
    background: {metric_bg};
    border: 1px solid {metric_border};
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}}
[data-testid="metric-container"]:hover {{
    border-color: {hover_metric_border};
    transform: translateY(-3px);
}}
[data-testid="stMetricLabel"] {{ color: {metric_label} !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }}
[data-testid="stMetricValue"] {{ color: {metric_val} !important; font-weight: 700 !important; font-size: 2rem !important; }}

[data-baseweb="tab-list"] {{ background: {tab_bg}; border-radius: 12px; padding: 4px; border: 1px solid {border_color}; }}
[data-baseweb="tab"] {{ border-radius: 8px !important; color: {tab_color} !important; font-weight: 500 !important; }}
[aria-selected="true"] {{ background: {tab_active} !important; color: #fff !important; }}

.stButton > button {{
    background: {btn_bg};
    color: #fff; border: none; border-radius: 10px;
    padding: 0.65rem 1.5rem; font-weight: 600; letter-spacing: 0.03em;
    transition: all 0.3s ease; box-shadow: 0 4px 20px {btn_shadow};
}}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px {btn_shadow}; }}

[data-testid="stDownloadButton"] > button {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid {border_color} !important;
    color: {text_sec} !important; border-radius: 8px !important; width: 100%;
}}

[data-testid="stExpander"] {{ background: {metric_bg}; border: 1px solid {border_color}; border-radius: 12px; padding: 0.2rem; }}
[data-testid="stExpander"] details > div:last-child {{ padding: 0.8rem 1rem 0.5rem !important; }}
[data-testid="stExpander"] .stSlider {{ padding: 0.5rem 0 0.7rem 0 !important; margin-bottom: 0.3rem !important; border-bottom: 1px solid {border_color}; }}
[data-testid="stExpander"] .stSlider:last-child {{ border-bottom: none; }}
[data-testid="stExpander"] .stCaption {{ margin-top: -0.3rem !important; margin-bottom: 0.2rem !important; }}
[data-testid="stExpander"] .stSlider label {{ margin-bottom: 0.3rem !important; }}
[data-testid="stExpander"] .element-container {{ margin-bottom: 0.1rem !important; }}

.section-header {{
    font-size: 1.1rem; font-weight: 700; color: {header_color};
    padding: 0.6rem 0 0.3rem 0; margin-top: 1rem;
    border-bottom: 1px solid {border_color};
}}

.hero-banner {{
    background: {hero_bg};
    border: 1px solid {hero_border};
    border-radius: 20px; padding: 2rem 2.5rem;
    backdrop-filter: blur(12px); margin-bottom: 1.5rem;
}}
.hero-title {{ font-size: 2.1rem; font-weight: 900; color: {metric_val};
    background: linear-gradient(135deg, #93c5fd, #c4b5fd, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.hero-sub {{ color: {text_sec}; font-size: 0.97rem; margin-top: 0.6rem; max-width: 680px; line-height: 1.7; }}
.hero-badge {{
    display: inline-block; background: rgba(124,58,237,0.25);
    border: 1px solid rgba(167,139,250,0.35); border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; color: #c4b5fd;
    padding: 0.25rem 0.75rem; margin-right: 0.5rem; margin-top: 0.8rem;
    letter-spacing: 0.05em; text-transform: uppercase;
}}

.layer-stack {{ display: flex; flex-direction: column; gap: 3px; width: 100%; }}
.layer-box {{
    border-radius: 6px; text-align: center; padding: 7px 0;
    font-size: 0.73rem; font-weight: 600; letter-spacing: 0.04em;
}}

[data-testid="stTable"] table {{ background: {table_bg}; border-radius: 10px; }}
[data-testid="stTable"] th {{ background: {table_th_bg}; color: {table_th_color} !important; }}
[data-testid="stTable"] td {{ color: {table_td_color} !important; }}

.stSlider label {{ color: {slider_color} !important; font-size: 0.82rem !important; font-weight: 500 !important; }}

/* ── Responsive layout ─────────────────────────────────────────── */
@media screen and (max-width: 768px) {{
    .hero-title {{ font-size: 1.4rem !important; }}
    .hero-banner {{ padding: 1rem 1.2rem !important; }}
    .hero-sub {{ font-size: 0.85rem !important; }}
    [data-testid="column"] {{ flex: 0 0 100% !important; min-width: 100% !important; width: 100% !important; }}
    [data-baseweb="tab"] {{ font-size: 0.65rem !important; padding: 0.3rem 0.5rem !important; white-space: normal !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
    [data-testid="metric-container"] {{ padding: 0.8rem 1rem !important; }}
    .stApp > header {{ height: 2.5rem !important; }}
}}
@media screen and (max-width: 480px) {{
    .hero-title {{ font-size: 1.1rem !important; }}
    .section-header {{ font-size: 0.85rem !important; }}
    [data-testid="stSidebar"] {{ min-width: 100% !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1rem !important; }}
    .stButton > button {{ font-size: 0.75rem !important; padding: 0.4rem 0.8rem !important; }}
}}
</style>"""

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'psc_surrogate_model.pkl')
FEATURES_PATH  = os.path.join(BASE_DIR, 'feature_names.pkl')
DATASET_PATH   = os.path.join(BASE_DIR, 'ml_dataset.csv')
PYTORCH_DIR    = os.path.join(BASE_DIR, 'pytorch_assets')

@st.cache_resource
def load_all():
    required = [
        MODEL_PATH,
        FEATURES_PATH,
        DATASET_PATH,
        os.path.join(PYTORCH_DIR, 'scaler.pkl'),
        os.path.join(PYTORCH_DIR, 'phys_features.pkl'),
        os.path.join(PYTORCH_DIR, 'jv_mlp.pth'),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s):\n" + "\n".join(missing)
        )

    rf = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATASET_PATH)

    global BASELINE_PCE
    ranges = {}
    pce_values = []
    for f in feats:
        series = pd.to_numeric(df[f], errors='coerce').dropna() if f in df.columns else pd.Series(dtype=float)
        if series.empty:
            ranges[f] = {'min': 0.0, 'max': 1.0, 'default': 0.5}
            continue
        mn = float(series.min())
        mx = float(series.max())
        md = float(series.median())
        if not np.isfinite(mn) or not np.isfinite(mx):
            ranges[f] = {'min': 0.0, 'max': 1.0, 'default': 0.5}
            continue
        if mn == mx:
            ranges[f] = {'min': mn, 'max': mx, 'default': mn}
        else:
            if not np.isfinite(md):
                md = (mn + mx) / 2
            ranges[f] = {'min': mn, 'max': mx, 'default': min(max(md, mn), mx)}

    if 'PCE_%' in df.columns:
        pce_values = pd.to_numeric(df['PCE_%'], errors='coerce').dropna()
        if not pce_values.empty:
            BASELINE_PCE = float(pce_values.median())

    scaler = joblib.load(os.path.join(PYTORCH_DIR, 'scaler.pkl'))
    pfeats = joblib.load(os.path.join(PYTORCH_DIR, 'phys_features.pkl'))
    jv_m = JV_Model(input_dim=len(pfeats) + 1)

    checkpoint_path = os.path.join(PYTORCH_DIR, 'jv_mlp.pth')
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    jv_m.load_state_dict(state_dict)
    jv_m.eval()
    return rf, feats, ranges, jv_m, scaler, pfeats

BASELINE_PCE = 15.0

with st.spinner("🔄 Loading ML models and SCAPS-1D data..."):
    try:
        model, feature_names, feature_ranges, jv_model, scaler, phys_features = load_all()
        model_loaded = True
    except Exception as exc:
        model_loaded = False

if not model_loaded:
    st.error("App initialization failed. Please verify model files and environment dependencies.")
    st.exception(exc)
    st.stop()
else:
    st.success("✅ Models loaded successfully! 1,181 SCAPS simulations ready.")

FEATURE_META = {
    'Temperature_K':                                              {'label':'Operating Temperature (K)',    'group':'🌡️ Environment',             'fmt':'%d',    'help':'Higher temp → lower efficiency due to thermal recombination.'},
    'MaPbI3_xClx_PAL1___L2___thickness__m':                     {'label':'PAL1 Thickness (μm)',           'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.3f',  'help':'PAL1 light-absorbing layer thickness.'},
    'MaPbI3_xClx_PAL1___L2___shallow_acceptor_density_1_cm':    {'label':'PAL1 Acceptor Density (cm⁻³)', 'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.1e',  'help':'p-type doping concentration in PAL1.'},
    'MaPbI3_xClx_PAL1___L2___defect_1__total_defect_density__1_cm': {'label':'PAL1 Defect Density (cm⁻³)',  'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.1e',  'help':'Trap states → recombination centres. Lower is better.'},
    'MaPbI3_Ti3C2_PAL2___L3___thickness__m':                    {'label':'PAL2 Thickness (μm)',           'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.3f',  'help':'PAL2 MXene-doped light-absorbing layer thickness.'},
    'MaPbI3_Ti3C2_PAL2___L3___shallow_acceptor_density_1_cm':   {'label':'PAL2 Acceptor Density (cm⁻³)', 'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.1e',  'help':'p-type doping concentration in PAL2.'},
    'MaPbI3_Ti3C2_PAL2___L3___defect_1__total_defect_density__1_cm': {'label':'PAL2 Defect Density (cm⁻³)',  'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.1e',  'help':'Trap states in PAL2. Lower → better charge transport.'},
}

PRESETS = {
    "🔬 High Efficiency Lab Cell": {
        f: feature_ranges[f]['default'] for f in feature_names
    },
    "⚡ Optimized for Voc": {
        **{f: feature_ranges[f]['default'] for f in feature_names},
        'MaPbI3_xClx_PAL1___L2___shallow_acceptor_density_1_cm': 1e18,
        'MaPbI3_Ti3C2_PAL2___L3___shallow_acceptor_density_1_cm': 1e18,
    },
    "🛡️ Low Defect Design": {
        **{f: feature_ranges[f]['default'] for f in feature_names},
        'MaPbI3_xClx_PAL1___L2___defect_1__total_defect_density__1_cm': 1e13,
        'MaPbI3_Ti3C2_PAL2___L3___defect_1__total_defect_density__1_cm': 1e13,
    },
    "🌡️ Room Temperature (300K)": {
        **{f: feature_ranges[f]['default'] for f in feature_names},
        'Temperature_K': 300,
    },
}

def clean_label(f): return FEATURE_META.get(f,{}).get('label', f.replace('_',' '))

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">☀️ Perovskite Solar Cell AI Simulator</div>
  <div class="hero-sub">
    A Machine Learning surrogate of SCAPS-1D physical simulations.
    Instantly predict J-V characteristics, PCE, and optimal device design — no simulation software needed.
  </div>
  <span class="hero-badge">Random Forest Regressor</span>
  <span class="hero-badge">PyTorch MLP</span>
  <span class="hero-badge">Genetic Optimizer</span>
  <span class="hero-badge">SCAPS-1D Data</span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <div style="font-size:1.6rem">🔬</div>
  <div style="font-weight:700;color:#f0f9ff;font-size:1rem;">Device Designer</div>
  <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Adjust parameters below</div>
</div>
""", unsafe_allow_html=True)

theme_icon = "🌙" if st.session_state.theme == 'dark' else "☀️"
theme_label = "Light Mode" if st.session_state.theme == 'dark' else "Dark Mode"
if st.sidebar.button(f"{theme_icon} {theme_label}", width='stretch'):
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.rerun()

st.sidebar.markdown(f'<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">💡 Quick Presets</div>', unsafe_allow_html=True)
preset_options = ["Custom"] + list(PRESETS.keys())
selected_preset = st.sidebar.selectbox("Load preset configuration", preset_options, label_visibility="collapsed")

if selected_preset != "Custom":
    st.sidebar.info(f"📋 Loaded: {selected_preset}")
else:
    st.sidebar.caption("Adjust sliders to customize")

preset_defaults = PRESETS.get(selected_preset, {})

user_inputs = {}
groups = {}
for f in feature_names:
    meta = FEATURE_META.get(f, {'label': f, 'group': 'Other', 'fmt': '%g', 'help': ''})
    groups.setdefault(meta['group'], []).append((f, meta))

for group_name, feats_in_group in groups.items():
    with st.sidebar.expander(group_name, expanded=True):
        for feature, meta in feats_in_group:
            rng = feature_ranges[feature]
            mn, mx = rng['min'], rng['max']
            df_val = preset_defaults.get(feature, rng['default'])
            df_val = max(mn, min(mx, df_val))

            is_density = "density" in feature.lower()
            is_large_range = mx > 1e6

            if mn == mx:
                user_inputs[feature] = mn
                st.caption(f"{meta['label']}: {mn:.2e} (fixed)")
            elif is_density or is_large_range:
                lo = np.log10(mn) if mn > 0 else 0
                hi = np.log10(mx) if mx > 0 else 1
                dv = np.log10(df_val) if df_val > 0 else lo
                ev = st.slider(f"{meta['label']} · 10^x", float(lo), float(hi), float(dv), 0.1, help=meta['help'], key=feature)
                av = 10**ev
                st.caption(f"**{av:.2e}** cm⁻³")
                user_inputs[feature] = av
            else:
                step = (mx-mn)/100 or 0.01
                user_inputs[feature] = st.slider(meta['label'], float(mn), float(mx), float(df_val), float(step), format=meta['fmt'], help=meta['help'], key=feature)

            threshold_tip = ""
            if is_density and user_inputs[feature] > 1e17:
                threshold_tip = "⚠️ High – may increase recombination"
            elif is_density and user_inputs[feature] < 1e14:
                threshold_tip = "✅ Low – good for carrier transport"
            elif "defect" in feature.lower() and user_inputs[feature] > 1e16:
                threshold_tip = "⚠️ High defect density"
            elif "defect" in feature.lower() and user_inputs[feature] < 1e14:
                threshold_tip = "✅ Low defects – optimal"
            if threshold_tip:
                warn_type = "warn" if "⚠️" in threshold_tip else "ok"
                bg_warn = "rgba(239,68,68,0.12)" if warn_type == "warn" else "rgba(52,211,153,0.12)"
                border_warn = "rgba(239,68,68,0.25)" if warn_type == "warn" else "rgba(52,211,153,0.25)"
                st.markdown(f"<div style='background:{bg_warn};border:1px solid {border_warn};border-radius:6px;padding:3px 8px;font-size:0.7rem;color:{'#f87171' if warn_type == 'warn' else '#34d399'};margin:2px 0 6px 0'>{threshold_tip}</div>", unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
inp = pd.DataFrame([user_inputs])[feature_names]
preds = model.predict(inp)[0]
pce, voc, jsc, ff = preds[0], preds[1], preds[2], preds[3]

# ── KEY METRICS ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Predicted Performance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

pce_delta = pce - BASELINE_PCE
voc_delta = voc - 1.1
jsc_delta = jsc - 25.0
ff_delta = ff - 75.0

c1.metric(
    "Power Conversion Efficiency", f"{pce:.2f} %",
    delta=f"{pce_delta:+.2f} vs median",
    help=f"Baseline median PCE: {BASELINE_PCE:.2f}%"
)
c2.metric(
    "Open-Circuit Voltage (Voc)", f"{voc:.3f} V",
    delta=f"{voc_delta:+.3f} V",
    help="Typical Voc range: 0.9 - 1.2 V"
)
c3.metric(
    "Short-Circuit Current (Jsc)", f"{jsc:.2f} mA/cm²",
    delta=f"{jsc_delta:+.2f} mA/cm²",
    help="Typical Jsc range: 20 - 30 mA/cm²"
)
c4.metric(
    "Fill Factor (FF)", f"{ff:.2f} %",
    delta=f"{ff_delta:+.2f} %",
    help="Typical FF range: 70 - 85 %"
)

st.markdown("<br>", unsafe_allow_html=True)

# ── SAVE DESIGN FOR COMPARISON ─────────────────────────────────────────────────
save_col1, save_col2 = st.columns([1, 5])
with save_col1:
    if st.button("💾 Save This Design", width='stretch', type="primary"):
        design_entry = {
            'name': f"Design {len(st.session_state.saved_designs) + 1}",
            'PCE': pce, 'Voc': voc, 'Jsc': jsc, 'FF': ff,
            'params': dict(user_inputs)
        }
        st.session_state.saved_designs.append(design_entry)
        st.success(f"Design {len(st.session_state.saved_designs)} saved!")
with save_col2:
    if st.session_state.saved_designs:
        st.info(f"📁 {len(st.session_state.saved_designs)} design(s) saved. Go to the **🔄 Design Comparison** tab to compare.")
    else:
        st.caption("Save this device configuration for later comparison in the Design Comparison tab.")


st.markdown("<br>", unsafe_allow_html=True)

# ── LAYER ARCHITECTURE DIAGRAM ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🏗️ Device Architecture</div>', unsafe_allow_html=True)
layers = [
    ("#94a3b8","Silver (Ag) — Back Contact"),
    ("#a78bfa","Spiro-OMeTAD — HTL"),
    ("#60a5fa","MAPbI₃-ₓClₓ — PAL 1"),
    ("#34d399","MAPbI₃ + Ti₃C₂ MXene — PAL 2"),
    ("#f97316","ZnO — ETL"),
    ("#e2e8f0","ITO — Front Contact"),
    ("#6b7280","Glass Substrate"),
]
cols_diag = st.columns([1,2,1])
with cols_diag[1]:
    html_layers = '<div class="layer-stack">'
    for color, label in layers:
        html_layers += f'<div class="layer-box" style="background:{color}22;border:1px solid {color}55;color:{color}">{label}</div>'
    html_layers += '</div>'
    st.markdown(html_layers, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.4rem'>↑ Illumination (AM 1.5G, 100 mW/cm²)</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PLOT HELPERS (theme-aware) ──────────────────────────────────────────────────
def get_plot_layout(theme):
    if theme == 'light':
        return dict(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)',
            font=dict(color='#475569', family='Inter'),
            margin=dict(l=50, r=30, t=40, b=50)
        )
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=50, r=30, t=40, b=50)
    )

def get_plot_axis(theme):
    if theme == 'light':
        return dict(gridcolor='rgba(0,0,0,0.08)', showgrid=True, zeroline=False, linecolor='rgba(0,0,0,0.12)')
    return dict(gridcolor='rgba(255,255,255,0.07)', showgrid=True, zeroline=False, linecolor='rgba(255,255,255,0.1)')

DARK_LAYOUT = get_plot_layout(st.session_state.theme)
DARK_AXIS = get_plot_axis(st.session_state.theme)

# ── TABS ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Analysis Dashboard</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["⚡ J-V Curve (PyTorch ANN)", "🕸️ Radar Balance", "📌 Feature Importance", "🏆 Efficiency Benchmark", "📈 Sensitivity Analysis", "🔄 Design Comparison", "🗺️ Thickness Contour", "🌡️ Temperature Animator"])

with tab1:
    st.markdown("**Predicted J-V curve** — generated dynamically by the trained PyTorch MLP (ANN) for this exact parameter combination.")
    max_v = float(voc) + 0.1
    v_pts = np.linspace(0, max_v, 100)
    base = np.array([user_inputs.get(pf, 0.0) for pf in phys_features])
    j_pts = []
    with torch.no_grad():
        for v in v_pts:
            row = scaler.transform(np.append(base, v).reshape(1,-1))
            j_pts.append(-jv_model(torch.tensor(row, dtype=torch.float32)).item())
    j_pts = [max(j, 0) for j in j_pts]
    j_max = max(j_pts) if any(j > 0 for j in j_pts) else 40

    if 'op_voltage' not in st.session_state:
        st.session_state.op_voltage = max_v * 0.5

    op_v = st.slider(
        "🎯 Operating Voltage (V)",
        0.0, max_v, min(st.session_state.op_voltage, max_v), 0.01,
        key='op_voltage_slider'
    )
    st.session_state.op_voltage = op_v

    op_idx = np.argmin(np.abs(v_pts - op_v))
    op_j = j_pts[op_idx]
    power_mw = op_v * op_j

    col_jv, col_power = st.columns([3, 1])
    with col_jv:
        jv_fig = go.Figure()
        jv_fig.add_trace(go.Scatter(
            x=v_pts, y=j_pts, mode='lines', name='J-V (ANN)',
            line=dict(color='#60a5fa', width=3),
            fill='tozeroy', fillcolor='rgba(96,165,250,0.1)'
        ))
        jv_fig.add_shape(type='line', x0=voc, x1=voc, y0=0, y1=j_max*1.1,
                         line=dict(color='#f472b6', width=1.5, dash='dash'))
        jv_fig.add_annotation(x=voc*0.85, y=j_max*0.55, text=f"Voc = {voc:.3f} V",
                              showarrow=False, font=dict(color='#f472b6', size=12))
        jv_fig.add_annotation(x=max_v*0.05, y=j_max*1.05, text=f"Jsc ≈ {jsc:.2f} mA/cm²",
                              showarrow=False, font=dict(color='#34d399', size=12), xanchor='left')

        jv_fig.add_trace(go.Scatter(
            x=[op_v], y=[op_j], mode='markers',
            name='Operating Point',
            marker=dict(color='#f59e0b', size=12, line=dict(color='#fff', width=2)),
            hovertemplate=f'V = {op_v:.3f} V<br>J = {op_j:.2f} mA/cm²<br>Power = {power_mw:.2f} mW/cm²'
        ))
        jv_fig.add_shape(type='line', x0=op_v, x1=op_v, y0=0, y1=op_j,
                         line=dict(color='#f59e0b', width=1.5, dash='dot'))

        jv_fig.update_layout(**DARK_LAYOUT, height=450, xaxis_title="Voltage (V)", yaxis_title="Current Density (mA/cm²)")
        jv_fig.update_xaxes(**DARK_AXIS, range=[0, max_v])
        jv_fig.update_yaxes(**DARK_AXIS, range=[0, j_max * 1.2])
        st.plotly_chart(jv_fig, width='stretch')

    with col_power:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:12px;padding:1rem;text-align:center">
            <div style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em">Power Output</div>
            <div style="color:#f59e0b;font-size:2rem;font-weight:800;padding:0.3rem 0">{:.2f}</div>
            <div style="color:#94a3b8;font-size:0.8rem">mW/cm²</div>
        </div>
        """.format(power_mw), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);border-radius:10px;padding:0.6rem;margin-top:0.5rem;text-align:center">
            <div style="color:#94a3b8;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em">Operating J</div>
            <div style="color:#60a5fa;font-size:1.2rem;font-weight:700">{op_j:.2f} mA/cm²</div>
        </div>
        <div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:0.6rem;margin-top:0.5rem;text-align:center">
            <div style="color:#94a3b8;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em">Max Power Point</div>
            <div style="color:#34d399;font-size:1.2rem;font-weight:700">{jsc * voc * ff / 100:.2f} mW/cm²</div>
        </div>
        """, unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download J-V Data (CSV)",
            data=pd.DataFrame({'Voltage_V': v_pts, 'Current_mA_per_cm2': j_pts}).to_csv(index=False),
            file_name="jv_curve.csv", mime="text/csv",
            width='stretch'
        )
    with col_dl2:
        img_bytes = pio.to_image(jv_fig, format='png', width=900, height=500, scale=2)
        st.download_button(
            "🖼️ Export J-V Curve as PNG",
            data=img_bytes,
            file_name="jv_curve.png", mime="image/png",
            width='stretch'
        )

with tab2:
    st.markdown("How well-rounded is this design? A wider polygon = better balanced performance.")
    cats = ['PCE', 'Voc', 'Jsc', 'FF']
    nvals = [min(max(pce/30*100,0),100), min(max(voc/1.5*100,0),100),
             min(max(jsc/40*100,0),100), min(max(ff/85*100,0),100)]
    rf_fig = go.Figure(go.Scatterpolar(
        r=nvals + [nvals[0]], theta=cats + [cats[0]], fill='toself',
        line=dict(color='#a78bfa', width=2.5),
        fillcolor='rgba(167,139,250,0.25)'
    ))
    rf_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', height=420,
        polar=dict(
            bgcolor='rgba(255,255,255,0.03)',
            radialaxis=dict(visible=True, range=[0,100], gridcolor='rgba(255,255,255,0.1)',
                           tickcolor='#64748b', tickfont=dict(color='#64748b', size=10)),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickcolor='#94a3b8',
                            tickfont=dict(color='#cbd5e1', size=12))
        ),
        showlegend=False, margin=dict(l=60,r=60,t=40,b=40)
    )
    st.plotly_chart(rf_fig, width='stretch')

with tab3:
    st.markdown("Which parameters matter most? Extracted directly from the trained **Random Forest model**.")
    rf_reg = model.named_steps['regressor'] if hasattr(model, 'named_steps') else model
    imp = rf_reg.feature_importances_
    c_names = [FEATURE_META.get(f,{}).get('label', f) for f in feature_names]
    imp_df = pd.DataFrame({'Feature': c_names, 'Importance': imp}).sort_values('Importance')
    imp_fig = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        text=[f"{v*100:.1f}%" for v in imp_df['Importance']],
        textposition='outside', textfont=dict(color='#cbd5e1', size=11),
        marker=dict(
            color=imp_df['Importance'],
            colorscale=[[0,'#1e3a5f'],[0.5,'#3b82f6'],[1,'#a78bfa']],
            line=dict(width=0)
        )
    ))
    imp_fig.update_layout(**DARK_LAYOUT, xaxis_title="Relative Importance", yaxis_title="", height=420)
    imp_fig.update_xaxes(**DARK_AXIS)
    imp_fig.update_yaxes(**DARK_AXIS)
    st.plotly_chart(imp_fig, width='stretch')

with tab4:
    st.markdown("How does your design compare against real-world PSC benchmarks?")
    bench_names = ['Your Design', 'World Record\nPSC (2024)', 'NREL Certified', 'Commercial Target']
    bench_vals  = [pce, 26.1, 25.7, 20.0]
    bench_colors= ['#3b82f6','#10b981','#a78bfa','#f59e0b']
    bf = go.Figure(go.Bar(
        x=bench_names, y=bench_vals,
        marker=dict(color=bench_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in bench_vals],
        textposition='outside', textfont=dict(color='#e2e8f0', size=13)
    ))
    bf.update_layout(**DARK_LAYOUT, yaxis_title="Power Conversion Efficiency (%)", height=430)
    bf.update_xaxes(**DARK_AXIS)
    bf.update_yaxes(**DARK_AXIS, range=[0, 30])
    st.plotly_chart(bf, width='stretch')

with tab5:
    st.markdown("**Sensitivity Analysis** — see how changing a single parameter affects Efficiency (PCE), holding all other parameters constant at their current slider values.")
    
    selected_feat = st.selectbox(
        "Select parameter to analyze:",
        feature_names,
        format_func=lambda x: FEATURE_META.get(x, {}).get('label', x),
        key='sens_select'
    )
    
    # Generate 50 points across the range of the selected feature
    rng = feature_ranges[selected_feat]
    x_vals = np.linspace(rng['min'], rng['max'], 50)
    y_vals = []
    
    # Copy current inputs to avoid mutating them
    sim_inputs = user_inputs.copy()
    
    for val in x_vals:
        sim_inputs[selected_feat] = val
        inp_df = pd.DataFrame([sim_inputs])[feature_names]
        # Predict using RF model (first target is PCE)
        y_vals.append(model.predict(inp_df)[0][0])
        
    sens_fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals, mode='lines',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'
    ))
    
    # Add a marker for the current value
    current_val = user_inputs[selected_feat]
    current_pce = pce # Current predicted PCE
    
    sens_fig.add_trace(go.Scatter(
        x=[current_val], y=[current_pce], mode='markers',
        name='Current Value',
        marker=dict(color='#f472b6', size=10, line=dict(color='#fff', width=1))
    ))
    
    sens_fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=FEATURE_META.get(selected_feat, {}).get('label', selected_feat),
        yaxis_title="Efficiency (PCE %)", height=430,
        showlegend=False
    )
    sens_fig.update_xaxes(**DARK_AXIS)
    sens_fig.update_yaxes(**DARK_AXIS)
    
    st.plotly_chart(sens_fig, width='stretch')

with tab6:
    st.markdown("**Design Comparison** — compare multiple saved device configurations side by side.")

    if not st.session_state.saved_designs:
        st.info("💡 No saved designs yet. Use the **Save This Design** button above to add designs for comparison.")
    else:
        rename_cols = st.columns([2] * len(st.session_state.saved_designs))
        for i, d in enumerate(st.session_state.saved_designs):
            with rename_cols[i]:
                new_name = st.text_input(f"Design {i+1} name", d['name'], key=f"name_{i}", label_visibility="collapsed")
                st.session_state.saved_designs[i]['name'] = new_name

        comp_df = pd.DataFrame([
            {'Design': d['name'], 'PCE (%)': d['PCE'], 'Voc (V)': d['Voc'],
             'Jsc (mA/cm²)': d['Jsc'], 'FF (%)': d['FF']}
            for d in st.session_state.saved_designs
        ])
        st.dataframe(comp_df.style.highlight_max(color='rgba(52,211,153,0.25)').highlight_min(color='rgba(239,68,68,0.15)'),
                     width='stretch', hide_index=True)

        comp_fig = go.Figure()
        colors = ['#3b82f6', '#10b981', '#a78bfa', '#f59e0b', '#ef4444', '#ec4899']
        for i, d in enumerate(st.session_state.saved_designs):
            comp_fig.add_trace(go.Bar(
                name=d['name'],
                x=['PCE (%)', 'Voc (V)×20', 'Jsc (mA/cm²)', 'FF (%)'],
                y=[d['PCE'], d['Voc']*20, d['Jsc'], d['FF']],
                marker_color=colors[i % len(colors)],
                text=[f"{d['PCE']:.1f}", f"{d['Voc']:.2f}", f"{d['Jsc']:.1f}", f"{d['FF']:.1f}"],
                textposition='inside', textfont=dict(color='#fff', size=11)
            ))

        comp_fig.update_layout(
            **DARK_LAYOUT, barmode='group', height=430,
            yaxis_title="Value", legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        comp_fig.update_xaxes(**DARK_AXIS)
        comp_fig.update_yaxes(**DARK_AXIS)
        st.plotly_chart(comp_fig, width='stretch')

        if st.button("🗑️ Clear All Designs", type="secondary"):
            st.session_state.saved_designs = []
            st.rerun()

with tab7:
    st.markdown("**Thickness Contour** — visualize how two parameters jointly affect PCE.")

    thickness_feats = [f for f in feature_names if 'thickness' in f.lower()]
    default_x = thickness_feats[0] if len(thickness_feats) > 0 else feature_names[0]
    default_y = thickness_feats[1] if len(thickness_feats) > 1 else feature_names[1]

    col_x, col_y = st.columns(2)
    with col_x:
        x_feat = st.selectbox("X-axis parameter", feature_names,
                              index=feature_names.index(default_x),
                              format_func=lambda x: FEATURE_META.get(x,{}).get('label',x),
                              key='contour_x')
    with col_y:
        y_feat = st.selectbox("Y-axis parameter", feature_names,
                              index=feature_names.index(default_y),
                              format_func=lambda x: FEATURE_META.get(x,{}).get('label',x),
                              key='contour_y')

    res = st.slider("Grid resolution", 15, 50, 30, key='contour_res')

    x_rng = feature_ranges[x_feat]
    y_rng = feature_ranges[y_feat]
    x_vals = np.linspace(x_rng['min'], x_rng['max'], res)
    y_vals = np.linspace(y_rng['min'], y_rng['max'], res)
    X, Y = np.meshgrid(x_vals, y_vals)

    sim_inputs = user_inputs.copy()

    with st.spinner("Generating contour..."):
        Z = np.zeros_like(X)
        for i in range(res):
            row_batch = []
            for j in range(res):
                sim_inputs[x_feat] = X[i, j]
                sim_inputs[y_feat] = Y[i, j]
                inp_df = pd.DataFrame([sim_inputs])[feature_names]
                row_batch.append(model.predict(inp_df)[0][0])
            Z[i, :] = row_batch

    x_label = FEATURE_META.get(x_feat,{}).get('label', x_feat)
    y_label = FEATURE_META.get(y_feat,{}).get('label', y_feat)

    contour_fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='#fff')),
            colorbar=dict(title=dict(text='PCE (%)', font=dict(color='#94a3b8')),
                          tickfont=dict(color='#94a3b8'),
                          outlinewidth=0)
        )
    )
    contour_fig.add_trace(go.Scatter(
        x=[user_inputs[x_feat]], y=[user_inputs[y_feat]],
        mode='markers',
        name='Current Design',
        marker=dict(color='#f472b6', size=10, symbol='star',
                    line=dict(color='#fff', width=2))
    ))
    contour_fig.update_layout(
        **DARK_LAYOUT, height=500,
        xaxis_title=x_label, yaxis_title=y_label,
    )
    contour_fig.update_xaxes(**DARK_AXIS)
    contour_fig.update_yaxes(**DARK_AXIS)
    st.plotly_chart(contour_fig, width='stretch')

    st.caption("⭐ The pink star marks your current design's position.")

with tab8:
    st.markdown("**Temperature Animator** — visualize how J-V characteristics and performance metrics change with temperature.")

    temp_feat = 'Temperature_K'
    t_rng = feature_ranges.get(temp_feat, {'min': 260, 'max': 400, 'default': 300})
    t_min, t_max = float(t_rng['min']), float(t_rng['max'])

    if 'temp_anim_speed' not in st.session_state:
        st.session_state.temp_anim_speed = 0.5
    if 'temp_anim_playing' not in st.session_state:
        st.session_state.temp_anim_playing = False
    if 'temp_anim_t' not in st.session_state:
        st.session_state.temp_anim_t = user_inputs.get(temp_feat, t_min)
    if 'temp_anim_last_update' not in st.session_state:
        st.session_state.temp_anim_last_update = 0

    if st.session_state.temp_anim_playing:
        now = time.time()
        last = st.session_state.temp_anim_last_update
        frame_gap = 0.35 / st.session_state.temp_anim_speed
        step = 2.0 * st.session_state.temp_anim_speed

        if now - last >= frame_gap:
            new_t = st.session_state.temp_anim_t + step
            if new_t > t_max:
                new_t = t_min
            st.session_state.temp_anim_t = new_t
            st.session_state.temp_anim_last_update = now
            time.sleep(min(frame_gap * 0.5, 0.2))
            st.rerun()
        else:
            time.sleep(0.05)
            st.rerun()

    t_col1, t_col2, t_col3 = st.columns([2, 1, 1])
    with t_col1:
        temp_val = st.slider("Sweep Temperature (K)", t_min, t_max,
                             st.session_state.temp_anim_t, 1.0, key='temp_anim_slider')
        if st.session_state.temp_anim_playing:
            temp_val = st.session_state.temp_anim_t
        st.session_state.temp_anim_t = temp_val
    with t_col2:
        play_label = "⏸️ Pause" if st.session_state.temp_anim_playing else "▶️ Play"
        if st.button(play_label, width='stretch'):
            st.session_state.temp_anim_playing = not st.session_state.temp_anim_playing
            st.session_state.temp_anim_last_update = 0 if st.session_state.temp_anim_playing else time.time()
            st.rerun()
    with t_col3:
        st.slider("Speed", 0.1, 2.0, st.session_state.temp_anim_speed, 0.1,
                  key='temp_anim_speed_slider')
        st.session_state.temp_anim_speed = st.session_state.temp_anim_speed_slider

    num_curves = 6
    temp_curves = np.linspace(t_min, t_max, num_curves)

    sim_base = user_inputs.copy()
    jv_data_by_temp = {}
    metrics_by_temp = []

    max_j = 0
    max_v_all = 0

    for t in temp_curves:
        sim_base[temp_feat] = t
        inp_df = pd.DataFrame([sim_base])[feature_names]
        p = model.predict(inp_df)[0]
        pce_t, voc_t, jsc_t, ff_t = p[0], p[1], p[2], p[3]
        metrics_by_temp.append({'T (K)': t, 'PCE (%)': pce_t, 'Voc (V)': voc_t,
                                'Jsc (mA/cm²)': jsc_t, 'FF (%)': ff_t})

        mv = float(voc_t) + 0.05
        vp = np.linspace(0, mv, 60)
        base_arr = np.array([sim_base.get(pf, 0.0) for pf in phys_features])
        jp = []
        with torch.no_grad():
            for v in vp:
                row = scaler.transform(np.append(base_arr, v).reshape(1, -1))
                jp.append(max(-jv_model(torch.tensor(row, dtype=torch.float32)).item(), 0))
        jv_data_by_temp[t] = (vp, jp)
        max_j = max(max_j, max(jp) if jp else 0)
        max_v_all = max(max_v_all, mv)

    temps_sorted = sorted(jv_data_by_temp.keys())
    temp_colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#a78bfa']

    col_jvs, col_metrics = st.columns([3, 2])

    with col_jvs:
        anim_fig = go.Figure()
        for i, t in enumerate(temps_sorted):
            vp, jp = jv_data_by_temp[t]
            is_current = abs(t - temp_val) < 2
            width = 4 if is_current else 1.5
            dash = 'solid' if is_current else 'dot'
            anim_fig.add_trace(go.Scatter(
                x=vp, y=jp, mode='lines',
                name=f'{t:.0f} K',
                line=dict(color=temp_colors[i % len(temp_colors)], width=width, dash=dash),
                hovertemplate=f'T = {t:.0f} K<br>J = %{{y:.2f}} mA/cm²<br>V = %{{x:.3f}} V'
            ))

        if temp_val in jv_data_by_temp:
            vp, jp = jv_data_by_temp[temp_val]
            anim_fig.add_trace(go.Scatter(
                x=vp, y=jp, mode='lines',
                name=f'▶ {temp_val:.0f} K',
                line=dict(color='#f472b6', width=5),
                hovertemplate=f'T = {temp_val:.0f} K<br>J = %{{y:.2f}} mA/cm²<br>V = %{{x:.3f}} V'
            ))

        anim_fig.update_layout(
            **DARK_LAYOUT, height=420,
            xaxis_title="Voltage (V)", yaxis_title="Current Density (mA/cm²)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(size=9))
        )
        anim_fig.update_xaxes(**DARK_AXIS, range=[0, max_v_all * 1.05])
        anim_fig.update_yaxes(**DARK_AXIS, range=[0, max_j * 1.2])
        st.plotly_chart(anim_fig, width='stretch')

    with col_metrics:
        mdf = pd.DataFrame(metrics_by_temp)
        curr_row = mdf.iloc[(mdf['T (K)'] - temp_val).abs().argmin()]

        st.markdown(f"""
        <div style="background:rgba(244,114,182,0.1);border:1px solid rgba(244,114,182,0.3);border-radius:12px;padding:0.8rem;text-align:center;margin-bottom:0.8rem">
            <div style="color:#94a3b8;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em">At {temp_val:.0f} K</div>
            <div style="display:flex;justify-content:space-around;margin-top:0.4rem">
                <div><div style="color:#f472b6;font-size:1.1rem;font-weight:700">{curr_row['PCE (%)']:.2f}%</div><div style="color:#94a3b8;font-size:0.6rem">PCE</div></div>
                <div><div style="color:#60a5fa;font-size:1.1rem;font-weight:700">{curr_row['Voc (V)']:.3f}V</div><div style="color:#94a3b8;font-size:0.6rem">Voc</div></div>
                <div><div style="color:#34d399;font-size:1.1rem;font-weight:700">{curr_row['Jsc (mA/cm²)']:.2f}</div><div style="color:#94a3b8;font-size:0.6rem">Jsc</div></div>
                <div><div style="color:#f59e0b;font-size:1.1rem;font-weight:700">{curr_row['FF (%)']:.2f}%</div><div style="color:#94a3b8;font-size:0.6rem">FF</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        trend_fig = go.Figure()
        for col, color, axis in [('Voc (V)', '#60a5fa', 'y'), ('Jsc (mA/cm²)', '#34d399', 'y2')]:
            trend_fig.add_trace(go.Scatter(
                x=mdf['T (K)'], y=mdf[col], mode='lines+markers',
                name=col, line=dict(color=color, width=2.5),
                marker=dict(size=6), yaxis=axis
            ))

        trend_fig.add_vline(x=temp_val, line=dict(color='#f472b6', width=1.5, dash='dash'))

        trend_fig.update_layout(
            **DARK_LAYOUT, height=280,
            xaxis_title="Temperature (K)",
            yaxis=dict(title=dict(text='Voc (V)', font=dict(color='#60a5fa')), tickfont=dict(color='#60a5fa')),
            yaxis2=dict(title=dict(text='Jsc (mA/cm²)', font=dict(color='#34d399')),
                        tickfont=dict(color='#34d399'), overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
        )
        trend_fig.update_xaxes(**DARK_AXIS)
        trend_fig.update_yaxes(**DARK_AXIS)
        st.plotly_chart(trend_fig, width='stretch')

    st.caption(f"Drag the temperature slider or press ▶️ Play to animate. Showing {num_curves} temperature snapshots overlaid.")

st.markdown("<br>", unsafe_allow_html=True)

# ── INVERSE DESIGN OPTIMIZER ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Inverse Design Optimizer</div>', unsafe_allow_html=True)
st.markdown("Use a **Genetic Algorithm** to automatically find the combination of physical parameters that yields the highest possible PCE — no manual trial and error required.")

col_btn, col_info = st.columns([1,3])
with col_btn:
    run_opt = st.button("🚀 Find Optimal Design")
with col_info:
    st.info("This may take ~10–20 seconds. The optimizer tests thousands of combinations through the ML model to find the global maximum.")

if run_opt:
    progress_bar = st.progress(0, text="Initializing genetic algorithm...")

    with st.spinner("🧬 Genetic Algorithm running..."):
        def objective(x):
            return -model.predict(pd.DataFrame([x], columns=feature_names))[0][0]
        bounds = [(feature_ranges[f]['min'], feature_ranges[f]['max']) for f in feature_names]

        def callback(x, convergence=0.0):
            progress_bar.progress(min(convergence / 30, 0.95), text="Converging to optimal solution...")

        res = differential_evolution(objective, bounds, maxiter=30, popsize=15, seed=42, callback=callback, updating='deferred')

    progress_bar.progress(1.0, text="Optimization complete!")
    st.info("🎉 Genetic algorithm converged!")

    best_pce = -res.fun
    st.success(f"✅ Optimization complete! Maximum theoretical PCE: **{best_pce:.2f} %**")
    opt_rows = [{"Parameter": FEATURE_META.get(f,{}).get('label',f),
                 "Optimal Value": f"{v:.2e}" if v>1e6 else f"{v:.4f}"}
                for f, v in zip(feature_names, res.x)]
    st.table(pd.DataFrame(opt_rows))

# ── SIDEBAR EXPORT ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em">💾 Config Manager</div>', unsafe_allow_html=True)

st.sidebar.caption("Save current slider values to restore later.")
config_name = st.sidebar.text_input("Config name", placeholder="e.g. High Voc run", label_visibility="collapsed", key="config_name_input")
if st.sidebar.button("💾 Save Config", width='stretch'):
    if config_name.strip():
        cfg = {'name': config_name.strip(), 'params': dict(user_inputs)}
        st.session_state.saved_configs.append(cfg)
        st.sidebar.success(f"Saved '{config_name}'!")
    else:
        st.sidebar.warning("Enter a name first.")

if st.session_state.saved_configs:
    config_options = [f"{i+1}. {c['name']}" for i, c in enumerate(st.session_state.saved_configs)]
    selected_config = st.sidebar.selectbox("Load saved config", [""] + config_options, format_func=lambda x: x if x else "— Select —", label_visibility="collapsed")
    if selected_config:
        idx = config_options.index(selected_config)
        loaded = st.session_state.saved_configs[idx]
        if st.sidebar.button("📂 Load Config", width='stretch'):
            for k, v in loaded['params'].items():
                if k in st.session_state:
                    st.session_state[k] = v
                else:
                    st.session_state[k] = v
            st.sidebar.success(f"Loaded '{loaded['name']}'!")
            st.rerun()

    del_idx_str = st.sidebar.selectbox("Delete saved config", [""] + config_options, format_func=lambda x: x if x else "— Select —", key="del_config")
    if del_idx_str:
        del_idx = config_options.index(del_idx_str)
        if st.sidebar.button("🗑️ Delete Config", width='stretch'):
            st.session_state.saved_configs.pop(del_idx)
            st.sidebar.info("Config deleted.")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em">💾 Export</div>', unsafe_allow_html=True)
export_d = {**user_inputs, 'Predicted_PCE': pce, 'Predicted_Voc': voc, 'Predicted_Jsc': jsc, 'Predicted_FF': ff}
st.sidebar.download_button(
    "⬇ Download Configuration (CSV)",
    data=pd.DataFrame([export_d]).to_csv(index=False),
    file_name="psc_config.csv", mime="text/csv"
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:0.8rem;font-size:0.75rem;color:#64748b;text-align:center">
  Model: Random Forest + PyTorch ANN<br>
  Training data: 1,181 SCAPS simulations<br>
  R² (PCE): 0.9873
</div>
""", unsafe_allow_html=True)
