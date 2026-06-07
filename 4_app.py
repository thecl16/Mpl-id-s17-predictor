"""
=============================================================
BAGIAN 4: DEPLOYMENT — Streamlit Web Application
=============================================================
Run: streamlit run scripts/4_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="MPL ID S17 — Match Predictor",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #e8b923 0%, #ff4444 50%, #e8b923 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
}

.vs-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ff4444;
    text-align: center;
    padding: 1rem 0;
}

.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e8b923;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.winner-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    color: #e8b923;
    font-weight: 700;
    letter-spacing: 1px;
}

.winner-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.3rem 0;
}

.confidence-text {
    color: #aaa;
    font-size: 0.9rem;
}

.stSelectbox > label { color: #ccc !important; font-size: 0.9rem; }
.metric-card {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ─────────────────────────────────────
MODEL_PATH   = "models/model_mpl_s17.pkl"
ENCODER_PATH = "data/team_encoder.json"
DATASET_PATH = "data/dataset_siap_ml.csv"
REPORT_PATH  = "models/training_report.json"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]

@st.cache_data
def load_encoder():
    with open(ENCODER_PATH) as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

@st.cache_data
def load_report():
    if Path(REPORT_PATH).exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


def check_files_exist():
    missing = []
    for p in [MODEL_PATH, ENCODER_PATH, DATASET_PATH]:
        if not Path(p).exists():
            missing.append(p)
    return missing


def compute_team_features(team: str, opponent: str,
                           df: pd.DataFrame, encoder: dict) -> dict:
    """Compute ML features for a given matchup."""

    def get_history(t, df):
        mask = (df["team_a"] == t) | (df["team_b"] == t)
        sub = df[mask].copy()
        sub["win"] = np.where(
            sub["team_a"] == t, (sub["label"] == 1).astype(int),
            (sub["label"] == 0).astype(int)
        )
        return sub

    hist_team = get_history(team, df)
    hist_opp  = get_history(opponent, df)

    # Rolling win rate (last 5)
    roll_team = hist_team["win"].tail(5).mean() if len(hist_team) else 0.5
    roll_opp  = hist_opp["win"].tail(5).mean()  if len(hist_opp)  else 0.5

    # H2H
    h2h_mask = ((df["team_a"] == team) & (df["team_b"] == opponent)) | \
               ((df["team_a"] == opponent) & (df["team_b"] == team))
    h2h_df = df[h2h_mask]
    if len(h2h_df):
        team_wins = ((h2h_df["team_a"] == team) & (h2h_df["label"] == 1)).sum() + \
                    ((h2h_df["team_b"] == team) & (h2h_df["label"] == 0)).sum()
        h2h_team = team_wins / len(h2h_df)
        h2h_opp  = 1 - h2h_team
    else:
        h2h_team = h2h_opp = 0.5

    # Score diff
    if "a_avg_score_diff" in df.columns:
        avg_diff_team = df[df["team_a"] == team]["a_avg_score_diff"].mean() or 0
        avg_diff_opp  = df[df["team_a"] == opponent]["a_avg_score_diff"].mean() or 0
    else:
        avg_diff_team = avg_diff_opp = 0

    # Streak
    streak_team = 0
    for w in hist_team["win"].tail(5).values[::-1]:
        if w == 1: streak_team += 1
        else: break
    streak_opp = 0
    for w in hist_opp["win"].tail(5).values[::-1]:
        if w == 1: streak_opp += 1
        else: break

    enc_team = encoder.get(team, 0)
    enc_opp  = encoder.get(opponent, 0)

    return {
        "a_rolling_win_rate": roll_team,
        "b_rolling_win_rate": roll_opp,
        "diff_rolling_wr": roll_team - roll_opp,
        "a_h2h_win_rate": h2h_team,
        "b_h2h_win_rate": h2h_opp,
        "a_avg_score_diff": avg_diff_team,
        "b_avg_score_diff": avg_diff_opp,
        "diff_score": avg_diff_team - avg_diff_opp,
        "a_win_streak": streak_team,
        "b_win_streak": streak_opp,
        "diff_streak": streak_team - streak_opp,
        "a_team_encoded": enc_team,
        "b_team_encoded": enc_opp,
    }


def make_proba_chart(team_a: str, team_b: str, prob_a: float, prob_b: float):
    fig = go.Figure()

    # Background bars (gray)
    fig.add_trace(go.Bar(
        x=[team_a, team_b],
        y=[1, 1],
        marker_color=["rgba(255,255,255,0.05)", "rgba(255,255,255,0.05)"],
        showlegend=False,
        hoverinfo="skip",
    ))

    # Probability bars
    colors_a = "#e8b923" if prob_a >= prob_b else "#555"
    colors_b = "#e8b923" if prob_b > prob_a  else "#555"

    fig.add_trace(go.Bar(
        x=[team_a, team_b],
        y=[prob_a, prob_b],
        marker_color=[colors_a, colors_b],
        marker_line=dict(width=0),
        text=[f"{prob_a*100:.1f}%", f"{prob_b*100:.1f}%"],
        textposition="outside",
        textfont=dict(size=16, color="white", family="Rajdhani"),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        name="Win Probability",
    ))

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(
            range=[0, 1.2],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.05)",
            title="Win Probability",
        ),
        xaxis=dict(tickfont=dict(size=14, family="Rajdhani")),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        showlegend=False,
    )

    # Add 50% reference line
    fig.add_hline(y=0.5, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="50% line",
                  annotation_font_color="gray")
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════
def main():
    # ── Header ─────────────────────────────────────────────
    st.markdown('<div class="main-title">⚔️ MPL ID S17</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine Learning Match Predictor · Playoffs Edition</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── File checks ────────────────────────────────────────
    missing = check_files_exist()
    if missing:
        st.error("⚠️ **Model files not found!**\n\n"
                 "Jalankan scripts berikut terlebih dahulu:\n"
                 "```\npython scripts/1_scrape_data.py\n"
                 "python scripts/2_feature_engineering.py\n"
                 "python scripts/3_train_model.py\n```")
        st.info(f"Missing: {', '.join(missing)}")

        # Demo mode with fake data
        st.markdown("---")
        st.markdown("### 🎮 Demo Mode (tanpa model)")
        demo_mode = True
    else:
        demo_mode = False
        model, feature_cols = load_model()
        encoder = load_encoder()
        df      = load_dataset()
        report  = load_report()

    # ── Model Metrics (sidebar) ─────────────────────────────
    teams = list(load_encoder().keys()) if not demo_mode else [
        "EVOS Legends", "RRQ Hoshi", "ONIC Esports", "Alter Ego",
        "Bigetron Alpha", "Geek Fam", "Aura Fire", "MDH Esports",
        "Rebellion Zion", "Dewa United"
    ]

    if not demo_mode and report:
        with st.expander("📊 Model Performance Metrics", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Test Accuracy", f"{report['test_accuracy']*100:.1f}%")
            c2.metric("ROC-AUC", f"{report['roc_auc']:.3f}")
            c3.metric("CV Score", f"{report['cv_mean']*100:.1f}% ± {report['cv_std']*100:.1f}%")

    # ── Team Selection ─────────────────────────────────────
    st.markdown("### 🎯 Pilih Tim Bertanding")
    col1, col_vs, col2 = st.columns([5, 1, 5])

    with col1:
        team_a = st.selectbox("🔵 Team A", options=teams, key="team_a")
    with col_vs:
        st.markdown('<div class="vs-badge" style="margin-top:1.8rem">VS</div>',
                    unsafe_allow_html=True)
    with col2:
        default_b = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("🔴 Team B", options=teams, index=default_b, key="team_b")

    # ── Predict Button ─────────────────────────────────────
    st.markdown("")
    col_btn = st.columns([1, 2, 1])[1]
    predict_clicked = col_btn.button("⚡ PREDICT OUTCOME", use_container_width=True,
                                      type="primary")

    # ── Prediction Logic ───────────────────────────────────
    if predict_clicked:
        if team_a == team_b:
            st.warning("⚠️ Pilih dua tim yang **berbeda**!")
            return

        with st.spinner("Menghitung prediksi..."):
            if demo_mode:
                # Random demo prediction
                import random
                prob_a = random.uniform(0.35, 0.75)
                prob_b = 1 - prob_a
                winner = team_a if prob_a > 0.5 else team_b
                prob_winner = max(prob_a, prob_b)
            else:
                feats = compute_team_features(team_a, team_b, df, encoder)
                X_input = pd.DataFrame([feats])[feature_cols]
                proba   = model.predict_proba(X_input)[0]
                prob_a, prob_b = proba[1], proba[0]
                winner = team_a if prob_a > prob_b else team_b
                prob_winner = max(prob_a, prob_b)

        st.markdown("---")

        # ── Result Card ───────────────────────────────────
        confidence_label = (
            "🔥 Sangat Yakin" if prob_winner > 0.75 else
            "✅ Cukup Yakin"  if prob_winner > 0.60 else
            "⚖️ Pertandingan Ketat"
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="winner-text">🏆 PREDIKSI PEMENANG</div>
            <div class="winner-name">{winner}</div>
            <div style="font-size:2rem; margin: 0.5rem 0">{prob_winner*100:.1f}%</div>
            <div class="confidence-text">{confidence_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Chart ─────────────────────────────
        st.markdown("### 📊 Perbandingan Probabilitas Kemenangan")
        fig = make_proba_chart(team_a, team_b, prob_a, prob_b)
        st.plotly_chart(fig, use_container_width=True)

        # ── Breakdown ─────────────────────────────────────
        if not demo_mode:
            feats_display = compute_team_features(team_a, team_b, df, encoder)
            st.markdown("### 🔍 Breakdown Statistik")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rolling Win Rate",
                       f"{feats_display['a_rolling_win_rate']*100:.0f}%",
                       f"{(feats_display['a_rolling_win_rate']-feats_display['b_rolling_win_rate'])*100:+.0f}%")
            c2.metric("H2H Win Rate",
                       f"{feats_display['a_h2h_win_rate']*100:.0f}%",
                       "Team A vs B")
            c3.metric("Win Streak",
                       f"{int(feats_display['a_win_streak'])}",
                       f"vs {int(feats_display['b_win_streak'])}")

        # ── Prediction Summary ────────────────────────────
        st.info(f"**{team_a}** {prob_a*100:.1f}%  ·  **{team_b}** {prob_b*100:.1f}%  "
                f"→  Model memprediksi **{winner}** menang dengan keyakinan **{prob_winner*100:.1f}%**")

    # ── Footer ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.8rem'>"
        "MPL ID S17 Predictor · Powered by XGBoost + Streamlit · "
        "Prediksi bersifat probabilistik, bukan jaminan hasil."
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
