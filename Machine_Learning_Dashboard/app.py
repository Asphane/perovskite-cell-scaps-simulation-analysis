import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
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

# ── PREMIUM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #1a0a2e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(13,17,38,0.95);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; font-size: 0.85rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(99,179,237,0.55);
    transform: translateY(-3px);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #f0f9ff !important; font-weight: 700 !important; font-size: 2rem !important; }

/* Tab styling */
[data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; border: 1px solid rgba(99,179,237,0.15); }
[data-baseweb="tab"] { border-radius: 8px !important; color: #94a3b8 !important; font-weight: 500 !important; }
[aria-selected="true"] { background: linear-gradient(135deg,#1e40af,#7c3aed) !important; color: #fff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
    color: #fff; border: none; border-radius: 10px;
    padding: 0.65rem 1.5rem; font-weight: 600; letter-spacing: 0.03em;
    transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(124,58,237,0.35);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(124,58,237,0.5); }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    color: #93c5fd !important; border-radius: 8px !important; width: 100%;
}

/* Expander */
[data-testid="stExpander"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(99,179,237,0.12); border-radius: 12px; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    padding: 0.6rem 0 0.3rem 0; margin-top: 1rem;
    border-bottom: 1px solid rgba(99,179,237,0.15);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30,64,175,0.25) 0%, rgba(124,58,237,0.25) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px; padding: 2rem 2.5rem;
    backdrop-filter: blur(12px); margin-bottom: 1.5rem;
}
.hero-title { font-size: 2.1rem; font-weight: 900; color: #f0f9ff; line-height: 1.2;
    background: linear-gradient(135deg, #93c5fd, #c4b5fd, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { color: #94a3b8; font-size: 0.97rem; margin-top: 0.6rem; max-width: 680px; line-height: 1.7; }
.hero-badge {
    display: inline-block; background: rgba(124,58,237,0.25);
    border: 1px solid rgba(167,139,250,0.35); border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; color: #c4b5fd;
    padding: 0.25rem 0.75rem; margin-right: 0.5rem; margin-top: 0.8rem;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* Layer diagram */
.layer-stack { display: flex; flex-direction: column; gap: 3px; width: 100%; }
.layer-box {
    border-radius: 6px; text-align: center; padding: 7px 0;
    font-size: 0.73rem; font-weight: 600; letter-spacing: 0.04em;
}

/* Table */
[data-testid="stTable"] table { background: rgba(255,255,255,0.03); border-radius: 10px; }
[data-testid="stTable"] th { background: rgba(30,64,175,0.35); color: #93c5fd !important; }
[data-testid="stTable"] td { color: #e2e8f0 !important; }

/* Slider labels */
.stSlider label { color: #cbd5e1 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

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
    ranges = {}
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

try:
    model, feature_names, feature_ranges, jv_model, scaler, phys_features = load_all()
except Exception as exc:
    st.error(
        "App initialization failed. Please verify model files and environment dependencies."
    )
    st.exception(exc)
    st.stop()

FEATURE_META = {
    'Temperature_K':                                              {'label':'Operating Temperature (K)',    'group':'🌡️ Environment',             'fmt':'%d',    'help':'Higher temp → lower efficiency due to thermal recombination.'},
    'MaPbI3_xClx_PAL1___L2___thickness__m':                     {'label':'PAL1 Thickness (μm)',           'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.3f',  'help':'PAL1 light-absorbing layer thickness.'},
    'MaPbI3_xClx_PAL1___L2___shallow_acceptor_density_1_cm':    {'label':'PAL1 Acceptor Density (cm⁻³)', 'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.1e',  'help':'p-type doping concentration in PAL1.'},
    'MaPbI3_xClx_PAL1___L2___defect_1__total_defect_density__1_cm': {'label':'PAL1 Defect Density (cm⁻³)',  'group':'🔵 Active Layer 1 (MAPbI3-xClx)', 'fmt':'%.1e',  'help':'Trap states → recombination centres. Lower is better.'},
    'MaPbI3_Ti3C2_PAL2___L3___thickness__m':                    {'label':'PAL2 Thickness (μm)',           'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.3f',  'help':'PAL2 MXene-doped light-absorbing layer thickness.'},
    'MaPbI3_Ti3C2_PAL2___L3___shallow_acceptor_density_1_cm':   {'label':'PAL2 Acceptor Density (cm⁻³)', 'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.1e',  'help':'p-type doping concentration in PAL2.'},
    'MaPbI3_Ti3C2_PAL2___L3___defect_1__total_defect_density__1_cm': {'label':'PAL2 Defect Density (cm⁻³)',  'group':'🟣 Active Layer 2 (MAPbI3+Ti3C2)','fmt':'%.1e',  'help':'Trap states in PAL2. Lower → better charge transport.'},
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

user_inputs = {}
groups = {}
for f in feature_names:
    meta = FEATURE_META.get(f, {'label': f, 'group': 'Other', 'fmt': '%g', 'help': ''})
    groups.setdefault(meta['group'], []).append((f, meta))

for group_name, feats_in_group in groups.items():
    with st.sidebar.expander(group_name, expanded=True):
        for feature, meta in feats_in_group:
            rng = feature_ranges[feature]
            mn, mx, df_val = rng['min'], rng['max'], rng['default']
            if mn == mx:
                user_inputs[feature] = mn
                st.caption(f"{meta['label']}: {mn:.2e} (fixed)")
            elif "density" in feature.lower() or mx > 1e6:
                lo, hi, dv = np.log10(mn) if mn>0 else 0, np.log10(mx) if mx>0 else 1, np.log10(df_val) if df_val>0 else 0
                ev = st.slider(f"{meta['label']} · 10^x", float(lo), float(hi), float(dv), 0.1, help=meta['help'], key=feature)
                av = 10**ev
                st.caption(f"**Value:** {av:.2e} cm⁻³")
                user_inputs[feature] = av
            else:
                step = (mx-mn)/100 or 0.01
                user_inputs[feature] = st.slider(meta['label'], float(mn), float(mx), float(df_val), float(step), format=meta['fmt'], help=meta['help'], key=feature)

# ── PREDICT ───────────────────────────────────────────────────────────────────
inp = pd.DataFrame([user_inputs])[feature_names]
preds = model.predict(inp)[0]
pce, voc, jsc, ff = preds[0], preds[1], preds[2], preds[3]

# ── KEY METRICS ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Predicted Performance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Power Conversion Efficiency", f"{pce:.2f} %")
c2.metric("Open-Circuit Voltage  Voc", f"{voc:.3f} V")
c3.metric("Short-Circuit Current  Jsc", f"{jsc:.2f} mA/cm²")
c4.metric("Fill Factor  FF", f"{ff:.2f} %")

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

# ── PLOT HELPERS (dark theme) ──────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
    font=dict(color='#94a3b8', family='Inter'),
    margin=dict(l=50, r=30, t=40, b=50)
)
DARK_AXIS = dict(gridcolor='rgba(255,255,255,0.07)', showgrid=True, zeroline=False, linecolor='rgba(255,255,255,0.1)')

# ── TABS ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Analysis Dashboard</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ J-V Curve (PyTorch ANN)", "🕸️ Radar Balance", "📌 Feature Importance", "🏆 Efficiency Benchmark", "📈 Sensitivity Analysis"])

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
    # Clip to illuminated region only (J >= 0)
    j_pts = [max(j, 0) for j in j_pts]
    j_max = max(j_pts) if any(j > 0 for j in j_pts) else 40
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
    jv_fig.update_layout(**DARK_LAYOUT, height=450, xaxis_title="Voltage (V)", yaxis_title="Current Density (mA/cm²)")
    jv_fig.update_xaxes(**DARK_AXIS, range=[0, max_v])
    jv_fig.update_yaxes(**DARK_AXIS, range=[0, j_max * 1.2])
    st.plotly_chart(jv_fig, use_container_width=True)

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
    st.plotly_chart(rf_fig, use_container_width=True)

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
    st.plotly_chart(imp_fig, use_container_width=True)

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
    st.plotly_chart(bf, use_container_width=True)

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
    
    st.plotly_chart(sens_fig, use_container_width=True)

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
    with st.spinner("🧬 Genetic Algorithm running..."):
        def objective(x):
            return -model.predict(pd.DataFrame([x], columns=feature_names))[0][0]
        bounds = [(feature_ranges[f]['min'], feature_ranges[f]['max']) for f in feature_names]
        res = differential_evolution(objective, bounds, maxiter=30, popsize=15, seed=42)

    best_pce = -res.fun
    st.success(f"✅ Optimization complete! Maximum theoretical PCE: **{best_pce:.2f} %**")
    opt_rows = [{"Parameter": FEATURE_META.get(f,{}).get('label',f),
                 "Optimal Value": f"{v:.2e}" if v>1e6 else f"{v:.4f}"}
                for f, v in zip(feature_names, res.x)]
    st.table(pd.DataFrame(opt_rows))

# ── SIDEBAR EXPORT ────────────────────────────────────────────────────────────
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
