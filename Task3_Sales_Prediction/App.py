"""
Sales Prediction App  |  CodSoft Data Science Internship
Predicts sales from TV, Radio and Newspaper advertising spend
using a Random Forest model trained in Google Colab.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "sales_model.pkl"
FEATURES = ["TV", "Radio", "Newspaper"]

# Ranges seen in the training data (the model is reliable inside these)
RANGES = {"TV": (0.0, 300.0), "Radio": (0.0, 50.0), "Newspaper": (0.0, 120.0)}

# Scores measured on the 20% test split in the notebook
MODEL_SCORES = {"R² score": "0.982", "MAE": "0.63", "RMSE": "0.76"}

# Palette
OLIVE = "#8A9A3F"
MOSS = "#A9BA5B"
SAGE = "#D3DCA3"
CREAM = "#F3F5E6"
DEEP = "#1E2610"

# ------------------------------------------------------------------
# Styling (olive green + glass)
# ------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Outfit', sans-serif;
    color: {CREAM};
}}

/* Background */
.stApp {{
    background:
        radial-gradient(900px 500px at 8% 0%, rgba(138,154,63,.45), transparent 60%),
        radial-gradient(800px 600px at 100% 100%, rgba(169,186,91,.28), transparent 60%),
        linear-gradient(160deg, #141a0a 0%, #26321455 45%, #1E2610 100%),
        #1E2610;
    background-attachment: fixed;
}}

#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: rgba(255,255,255,.06);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(255,255,255,.14);
}}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-weight: 600; letter-spacing: .2px;
}}

/* Glass card */
.glass {{
    background: linear-gradient(135deg, rgba(255,255,255,.14), rgba(255,255,255,.05));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,.20);
    border-radius: 22px;
    box-shadow: 0 10px 40px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.18);
    padding: 1.6rem 1.8rem;
}}
.glass.tight {{ padding: 1.1rem 1.3rem; }}

.hero-title {{ font-size: 2.5rem; font-weight: 700; line-height: 1.1; margin: 0; }}
.hero-sub {{ color: {SAGE}; font-weight: 300; font-size: 1.05rem; margin-top: .5rem; max-width: 640px; }}

.result-label {{ color: {SAGE}; font-size: .95rem; font-weight: 400; }}
.result-value {{ font-size: 4.2rem; font-weight: 600; line-height: 1; margin: .35rem 0 .2rem; color: {CREAM}; }}
.result-unit {{ font-size: 1.2rem; font-weight: 300; color: {SAGE}; margin-left: .4rem; }}
.result-range {{ color: {CREAM}; opacity: .8; font-size: .95rem; }}

.stat-label {{ color: {SAGE}; font-size: .85rem; }}
.stat-value {{ font-size: 1.7rem; font-weight: 600; margin-top: .15rem; }}
.stat-note {{ font-size: .8rem; opacity: .7; margin-top: .1rem; }}

.pill {{
    display: inline-block; padding: .25rem .8rem; border-radius: 999px;
    background: rgba(169,186,91,.22); border: 1px solid rgba(169,186,91,.45);
    color: {SAGE}; font-size: .8rem; margin-right: .4rem;
}}

.warn {{
    background: rgba(230,190,90,.14); border: 1px solid rgba(230,190,90,.4);
    border-radius: 14px; padding: .7rem 1rem; font-size: .9rem; color: #F3E4B5;
}}

/* Tabs */
button[data-baseweb="tab"] {{ font-family: 'Outfit', sans-serif; font-size: 1rem; }}
div[data-baseweb="tab-list"] {{ gap: .4rem; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {MOSS}; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {OLIVE}, #6B7A2F);
    color: {CREAM}; border: 1px solid rgba(255,255,255,.25);
    border-radius: 14px; padding: .6rem 1.4rem; font-weight: 500;
    transition: transform .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px); box-shadow: 0 8px 22px rgba(0,0,0,.35);
    border-color: rgba(255,255,255,.45); color: {CREAM};
}}

/* Inputs */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,.08) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.18) !important;
}}
div[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
@media (max-width: 700px) {{ .hero-title {{ font-size: 1.8rem; }} .result-value {{ font-size: 3rem; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def to_frame(tv, radio, news):
    return pd.DataFrame([[tv, radio, news]], columns=FEATURES)


def predict_with_range(model, row: pd.DataFrame):
    """Return the forest's prediction plus a 10th to 90th percentile band
    taken from the individual trees (a simple measure of uncertainty)."""
    tree_preds = np.array([t.predict(row.values)[0] for t in model.estimators_])
    return float(model.predict(row)[0]), float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90))


@st.cache_data(show_spinner=False)
def best_split(_model, budget: float):
    """Search budget splits (step 5 for TV, 2.5 for Radio) and return the best one."""
    rows = []
    for tv in np.arange(0, RANGES["TV"][1] + 1, 5.0):
        for radio in np.arange(0, RANGES["Radio"][1] + 1, 2.5):
            news = budget - tv - radio
            if 0 <= news <= RANGES["Newspaper"][1]:
                rows.append((tv, radio, news))
    if not rows:
        return None
    grid = pd.DataFrame(rows, columns=FEATURES)
    grid["Sales"] = _model.predict(grid)
    return grid.sort_values("Sales", ascending=False).iloc[0]


def style_fig(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color=CREAM),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.10)", zeroline=False)
    return fig


try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(
        "Could not load `sales_model.pkl`. Keep it in the same folder as `app.py` "
        "and install the scikit-learn version from `requirements.txt`."
    )
    st.exception(e)
    st.stop()

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 Ad budget")
    st.caption("Set your spend for each channel.")

    tv = st.slider("TV", *RANGES["TV"], 150.0, step=0.5)
    radio = st.slider("Radio", *RANGES["Radio"], 25.0, step=0.1)
    news = st.slider("Newspaper", *RANGES["Newspaper"], 30.0, step=0.1)

    st.markdown("---")
    st.caption(
        "Sliders cover the spend ranges in the training data. The model is most "
        "reliable inside these ranges."
    )

total = tv + radio + news
row = to_frame(tv, radio, news)
pred, low, high = predict_with_range(model, row)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="glass" style="margin-bottom:1.4rem;">
        <p class="hero-title">Sales Predictor</p>
        <p class="hero-sub">See how your TV, Radio and Newspaper advertising spend
        translates into expected sales, and find the split that works best.</p>
        <div style="margin-top:1rem;">
            <span class="pill">Random Forest</span>
            <span class="pill">R² {MODEL_SCORES['R² score']}</span>
            <span class="pill">200 campaigns</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_whatif, tab_opt, tab_about = st.tabs(
    ["Prediction", "What-if analysis", "Budget optimizer", "About the model"]
)

# ------------------------------------------------------------------
# Tab 1: prediction
# ------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="glass">
                <div class="result-label">Expected sales</div>
                <div class="result-value">{pred:.2f}<span class="result-unit">units</span></div>
                <div class="result-range">Likely range {low:.2f} to {high:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="glass tight" style="margin-top:1rem;">
                <div class="stat-label">Total ad spend</div>
                <div class="stat-value">{total:.1f}</div>
                <div class="stat-note">TV + Radio + Newspaper</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            eff = pred / total if total > 0 else 0
            st.markdown(
                f"""<div class="glass tight" style="margin-top:1rem;">
                <div class="stat-label">Sales per unit of spend</div>
                <div class="stat-value">{eff:.2f}</div>
                <div class="stat-note">Higher means more efficient</div></div>""",
                unsafe_allow_html=True,
            )

        if total == 0:
            st.markdown(
                '<div class="warn" style="margin-top:1rem;">Add some spend in the sidebar to see a meaningful prediction.</div>',
                unsafe_allow_html=True,
            )
        elif news > radio and news > 0:
            st.markdown(
                '<div class="warn" style="margin-top:1rem;">Newspaper spend is higher than Radio. '
                'In this data Newspaper has the weakest link to sales, so moving some of it to TV or Radio usually helps.</div>',
                unsafe_allow_html=True,
            )

    with right:
        share = go.Figure(
            go.Pie(
                labels=FEATURES,
                values=[tv, radio, news] if total > 0 else [1, 1, 1],
                hole=0.62,
                marker=dict(colors=[OLIVE, SAGE, "#5B6A2A"], line=dict(color=DEEP, width=2)),
                textinfo="percent",
                sort=False,
            )
        )
        share.update_layout(title="Where your budget goes", showlegend=True)
        st.plotly_chart(style_fig(share, 360), width="stretch")

    st.markdown("#### Your inputs")
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": FEATURES,
                "Spend": [tv, radio, news],
                "Share of budget": [f"{(v / total * 100) if total else 0:.1f}%" for v in (tv, radio, news)],
            }
        ),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------------------
# Tab 2: what-if
# ------------------------------------------------------------------
with tab_whatif:
    st.markdown("#### How sales change as one channel moves")
    st.caption("Each chart changes one channel and keeps the other two at your sidebar values. The dot marks your current spend.")

    fig = make_subplots(rows=1, cols=3, subplot_titles=FEATURES, horizontal_spacing=0.08)
    current = {"TV": tv, "Radio": radio, "Newspaper": news}
    for i, ch in enumerate(FEATURES, start=1):
        xs = np.linspace(*RANGES[ch], 60)
        grid = pd.DataFrame({f: [current[f]] * len(xs) for f in FEATURES})
        grid[ch] = xs
        ys = model.predict(grid)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=MOSS, width=3), showlegend=False), row=1, col=i)
        fig.add_trace(
            go.Scatter(x=[current[ch]], y=[pred], mode="markers", marker=dict(color=CREAM, size=11, line=dict(color=OLIVE, width=2)), showlegend=False),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="Spend", row=1, col=i)
    fig.update_yaxes(title_text="Predicted sales", row=1, col=1)
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    st.markdown(
        '<div class="glass tight">TV and Radio lines rise steadily. The Newspaper line stays close to flat, '
        "which matches its low importance in the model.</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Tab 3: budget optimizer
# ------------------------------------------------------------------
with tab_opt:
    st.markdown("#### Find the best split for a fixed budget")
    st.caption("Enter a total budget. The app tests many TV, Radio and Newspaper splits and shows the one with the highest predicted sales.")

    budget = st.slider("Total budget", 20.0, 400.0, float(round(total / 5) * 5) if 20 <= total <= 400 else 200.0, step=5.0)
    best = best_split(model, budget)

    if best is None:
        st.markdown('<div class="warn">No valid split exists for this budget within the data ranges. Try a lower budget.</div>', unsafe_allow_html=True)
    else:
        current_split_pred = None
        c1, c2, c3 = st.columns(3)
        for col, ch in zip((c1, c2, c3), FEATURES):
            with col:
                st.markdown(
                    f"""<div class="glass tight"><div class="stat-label">{ch}</div>
                    <div class="stat-value">{best[ch]:.1f}</div>
                    <div class="stat-note">{best[ch] / budget * 100:.0f}% of budget</div></div>""",
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"""<div class="glass" style="margin-top:1rem;">
            <div class="result-label">Predicted sales with this split</div>
            <div class="result-value" style="font-size:3rem;">{best['Sales']:.2f}<span class="result-unit">units</span></div></div>""",
            unsafe_allow_html=True,
        )

        bar = go.Figure(
            go.Bar(
                x=FEATURES, y=[best[f] for f in FEATURES],
                marker_color=[OLIVE, SAGE, "#5B6A2A"],
                text=[f"{best[f]:.1f}" for f in FEATURES], textposition="outside",
            )
        )
        bar.update_layout(title="Recommended spend per channel", yaxis_title="Spend")
        st.plotly_chart(style_fig(bar, 320), width="stretch")

# ------------------------------------------------------------------
# Tab 4: about
# ------------------------------------------------------------------
with tab_about:
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("#### What drives sales")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig = go.Figure(
            go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color=[SAGE, MOSS, OLIVE],
                text=[f"{v * 100:.1f}%" for v in imp.values], textposition="outside",
            )
        )
        fig.update_layout(title="Feature importance", xaxis_range=[0, max(imp.values) * 1.2])
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    with b:
        st.markdown("#### Test performance")
        for k, v in MODEL_SCORES.items():
            st.markdown(
                f"""<div class="glass tight" style="margin-bottom:.7rem;">
                <div class="stat-label">{k}</div><div class="stat-value">{v}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="glass" style="margin-top:1rem;">
        <b>How it works.</b> A Random Forest with 200 trees was trained on 200 past campaigns,
        using TV, Radio and Newspaper spend to predict sales. It was compared with Linear Regression
        (R² 0.90) and scored higher (R² 0.98) on 20% of data it had not seen.<br><br>
        <b>Limits.</b> A Random Forest cannot predict beyond the spend ranges it was trained on,
        and it shows relationships in past data, not guaranteed results.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Built for the CodSoft Data Science Internship")
