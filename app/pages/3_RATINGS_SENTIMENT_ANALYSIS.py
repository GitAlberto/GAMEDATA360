# -*- coding: utf-8 -*-
"""
GameData360 - Page Ratings & Sentiment Analysis
================================================
Analyse de sentiment basée sur Metacritic (scores critiques) et reviews Steam.
Insights: Excellence par Genre, Prix-Qualité, ROI, Sentiment vs Critique.

Auteur: GameData360 Team
Version: 3.1 (Metacritic Focus Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Ajout du chemin utils au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    COLORS, 
    PLOTLY_LAYOUT,
    FILE_PATH,
    NON_GAME_GENRES
)
from utils.data_helpers import (
    load_game_data,
    format_number
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — Ratings & Sentiment",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le thème gaming
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    
    .stMetric {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(255,0,255,0.1) 100%);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 15px;
    }
    
    .stMetric label {
        color: #00ff88 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700;
    }
    
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,255,136,0.1);
        border-radius: 8px;
        border: 1px solid rgba(0,255,136,0.3);
        padding: 10px 20px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,255,136,0.3), rgba(0,255,255,0.3));
        border-color: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("# ⭐ RATINGS & SENTIMENT ANALYSIS")
st.markdown("##### Analyse critique Metacritic & sentiment communauté Steam")

# ============================================================
# 2. CHARGEMENT DES DONNÉES (CACHE OPTIMISÉ)
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_filter_data():
    """Charge les données et filtre logiciels."""
    df = load_game_data(str(FILE_PATH))
    
    # Filtrage genres non-jeux
    def is_game(genres_list):
        if not isinstance(genres_list, list):
            return True
        genres_lower = [g.lower() for g in genres_list]
        return not any(genre in genres_lower for genre in NON_GAME_GENRES)
    
    initial_count = len(df)
    df = df[df["Genres"].apply(is_game)].copy()
    excluded_software = initial_count - len(df)
    
    # Calculs dérivés
    df["Total Reviews"] = df["Positive"] + df["Negative"]
    df["Positive Ratio"] = df["Positive"] / df["Total Reviews"].replace(0, 1)
    
    # ROI Qualité (Metacritic par dollar)
    df["Quality ROI"] = df["Metacritic score"] / df["Price"].replace(0, 1)
    
    return df, excluded_software

# Chargement
try:
    with st.spinner('⚡ Chargement des données...'):
        df_analyse, excluded_software = load_and_filter_data()
        
        if excluded_software > 0:
            st.sidebar.success(f"🎮 {excluded_software:,} logiciels exclus")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. TOP-LEVEL KPIs
# ============================================================
st.markdown("### 📊 Indicateurs Globaux")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Filtrer jeux avec Metacritic
df_meta = df_analyse[df_analyse["Metacritic score"] > 0]

# KPI 1: Score Metacritic Médian
if len(df_meta) > 0:
    median_meta = df_meta["Metacritic score"].median()
    avg_meta = df_meta["Metacritic score"].mean()
    
    with kpi1:
        st.metric(
            "⭐ Metacritic Médian",
            f"{median_meta:.0f}",
            delta=f"Moyenne: {avg_meta:.1f}",
            delta_color="off",
            help="Score critique médian"
        )

# KPI 2: Taux Positif Global
df_reviews = df_analyse[df_analyse["Total Reviews"] > 0]
if len(df_reviews) > 0:
    total_pos = df_reviews["Positive"].sum()
    total_neg = df_reviews["Negative"].sum()
    pos_rate = (total_pos / (total_pos + total_neg)) * 100 if (total_pos + total_neg) > 0 else 0
    
    with kpi2:
        st.metric(
            "👍 Taux Positif Global",
            f"{pos_rate:.0f}%",
            help="% reviews positives Steam"
        )

# KPI 3: Jeux Exceptionnels
exceptional = df_meta[df_meta["Metacritic score"] >= 85]

with kpi3:
    st.metric(
        "🏆 Jeux Exceptionnels",
        len(exceptional),
        delta=f"{(len(exceptional)/len(df_meta)*100):.1f}% du total",
        delta_color="off",
        help="Jeux avec Metacritic ≥85"
    )

# KPI 4: Corrélation Sentiment-Critique
df_corr = df_analyse[(df_analyse["Metacritic score"] > 0) & (df_analyse["Total Reviews"] > 10)]
if len(df_corr) > 0:
    correlation = df_corr["Positive Ratio"].corr(df_corr["Metacritic score"])
    
    with kpi4:
        st.metric(
            "🎯 Corrélation Sentiment",
            f"{correlation:.2f}",
            help="Corrélation % positif vs Metacritic"
        )

st.divider()

# ============================================================
# 4. INSIGHTS AUTOMATIQUES PERTINENTS
# ============================================================
st.markdown("### 🎯 Insights Automatiques")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    # Insight 1: Excellence Critique par Genre
    df_genre_meta = df_meta.explode("Genres").dropna(subset=["Genres"])
    
    if len(df_genre_meta) > 0:
        genre_scores = df_genre_meta.groupby("Genres")["Metacritic score"].agg(['median', 'count']).reset_index()
        genre_scores = genre_scores[genre_scores['count'] >= 10]  # Au moins 10 jeux
        
        if len(genre_scores) > 0:
            best_genre = genre_scores.loc[genre_scores['median'].idxmax()]
            
            st.success(
                f"🏆 **Excellence Critique**: Le genre **{best_genre['Genres']}** domine "
                f"avec un Metacritic médian de **{best_genre['median']:.0f}** "
                f"({int(best_genre['count'])} jeux)"
            )
    
    # Insight 2: Corrélation Prix-Qualité
    df_price_quality = df_meta[df_meta["Price"] > 0]
    
    if len(df_price_quality) > 0:
        price_corr = df_price_quality["Price"].corr(df_price_quality["Metacritic score"])
        
        if price_corr > 0.3:
            st.info(
                f"💎 **Prix = Qualité**: Corrélation de **{price_corr:.2f}** — "
                f"les jeux chers sont généralement mieux notés par la critique"
            )
        elif price_corr < 0.1:
            st.success(
                f"💰 **Prix ≠ Qualité**: Corrélation faible ({price_corr:.2f}) — "
                f"on trouve d'excellents jeux à tous les prix !"
            )
    
    # Insight 3: Best ROI par Tranche
    if len(df_price_quality) > 0:
        # Tranche <$10
        budget = df_price_quality[df_price_quality["Price"] < 10]
        
        if len(budget) > 0:
            best_budget = budget.nlargest(1, "Metacritic score").iloc[0]
            
            st.success(
                f"💰 **Best Value <$10**: **{best_budget['Name']}** "
                f"(Metacritic: {best_budget['Metacritic score']:.0f}, ${best_budget['Price']:.2f})"
            )

with col_insight2:
    # Insight 4: Sentiment vs Critique (alignement)
    if len(df_corr) > 0 and correlation > 0.5:
        st.info(
            f"📊 **Alignement Communauté-Critique**: Corrélation **{correlation:.2f}** — "
            f"la communauté Steam et les critiques sont généralement d'accord"
        )
    elif len(df_corr) > 0 and correlation < 0.3:
        st.warning(
            f"⚠️ **Divergence Communauté-Critique**: Corrélation **{correlation:.2f}** — "
            f"la communauté et les critiques ont des opinions différentes"
        )
    
    # Insight 5: Jeux Polarisants (Reviews abondantes mais 50/50)
    polarizing = df_reviews[
        (df_reviews["Total Reviews"] > 1000) &
        (df_reviews["Positive Ratio"] > 0.45) &
        (df_reviews["Positive Ratio"] < 0.55)
    ]
    
    if len(polarizing) > 0:
        st.warning(
            f"⚡ **{len(polarizing)} Jeux Polarisants Détectés**: "
            f"Très reviewés (>1000) mais ratio 50/50 — opinion divisée"
        )
    
    # Insight 6: Saturation Qualité par Genre
    if len(df_genre_meta) > 0:
        genre_exceptional = df_genre_meta[df_genre_meta["Metacritic score"] >= 85]
        genre_exc_counts = genre_exceptional.groupby("Genres").size().sort_values(ascending=False)
        
        if len(genre_exc_counts) > 0:
            top_exc_genre = genre_exc_counts.index[0]
            count_exc = genre_exc_counts.iloc[0]
            
            st.success(
                f"🎮 **Excellence en Volume**: Le genre **{top_exc_genre}** compte "
                f"**{count_exc}** jeux exceptionnels (≥85) — marché de qualité"
            )

st.divider()

# ============================================================
# 5. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Qualité Critique",
    "💬 Sentiment Communauté",
    "💰 Prix vs Qualité",
    "🏆 Excellence par Genre"
])

# ============================================================
# TAB 1: QUALITÉ CRITIQUE
# ============================================================
with tab1:
    st.markdown("### ⭐ Distribution Metacritic")
    
    col_dist1, col_dist2 = st.columns([2, 1])
    
    with col_dist1:
        # Histogramme Metacritic
        fig_hist_meta = px.histogram(
            df_meta,
            x="Metacritic score",
            nbins=40,
            color_discrete_sequence=[COLORS['primary']]
        )
        
        # Lignes verticales pour les seuils
        fig_hist_meta.add_vline(x=85, line_dash="dash", line_color=COLORS['primary'], 
                                annotation_text="Exceptionnel (85)")
        fig_hist_meta.add_vline(x=70, line_dash="dash", line_color=COLORS['warning'], 
                                annotation_text="Bon (70)")
        
        fig_hist_meta.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Score Metacritic",
            yaxis_title="Nombre de jeux",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist_meta, use_container_width=True)
    
    with col_dist2:
        st.markdown("#### 📋 Statistiques")
        
        stats_meta = pd.DataFrame({
            'Métrique': ['Médiane', 'Moyenne', 'Min', 'Max', 'Écart-type'],
            'Valeur': [
                f"{df_meta['Metacritic score'].median():.1f}",
                f"{df_meta['Metacritic score'].mean():.1f}",
                f"{df_meta['Metacritic score'].min():.0f}",
                f"{df_meta['Metacritic score'].max():.0f}",
                f"{df_meta['Metacritic score'].std():.1f}"
            ]
        })
        
        st.dataframe(stats_meta, hide_index=True, use_container_width=True)
        
        # Distribution par tranche
        st.markdown("#### 📊 Par Tranche")
        
        exceptional_pct = (len(df_meta[df_meta["Metacritic score"] >= 85]) / len(df_meta)) * 100
        good_pct = (len(df_meta[(df_meta["Metacritic score"] >= 70) & (df_meta["Metacritic score"] < 85)]) / len(df_meta)) * 100
        average_pct = (len(df_meta[(df_meta["Metacritic score"] >= 50) & (df_meta["Metacritic score"] < 70)]) / len(df_meta)) * 100
        poor_pct = (len(df_meta[df_meta["Metacritic score"] < 50]) / len(df_meta)) * 100
        
        tranches = pd.DataFrame({
            'Tranche': ['≥85 (Exceptionnel)', '70-84 (Bon)', '50-69 (Moyen)', '<50 (Faible)'],
            '%': [f"{exceptional_pct:.1f}%", f"{good_pct:.1f}%", f"{average_pct:.1f}%", f"{poor_pct:.1f}%"]
        })
        
        st.dataframe(tranches, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Top & Bottom 10
    col_top, col_bottom = st.columns(2)
    
    with col_top:
        st.markdown("### 🏆 Top 10 Metacritic")
        
        top_10 = df_meta.nlargest(10, "Metacritic score")[["Name", "Metacritic score", "Positive Ratio", "Price"]]
        top_10["Positive Ratio"] = (top_10["Positive Ratio"] * 100).round(1)
        top_10.columns = ["Nom", "Metacritic", "% Positif", "Prix ($)"]
        
        st.dataframe(top_10, hide_index=True, use_container_width=True)
    
    with col_bottom:
        st.markdown("### 📉 Bottom 10 Metacritic")
        
        bottom_10 = df_meta.nsmallest(10, "Metacritic score")[["Name", "Metacritic score", "Positive Ratio", "Price"]]
        bottom_10["Positive Ratio"] = (bottom_10["Positive Ratio"] * 100).round(1)
        bottom_10.columns = ["Nom", "Metacritic", "% Positif", "Prix ($)"]
        
        st.dataframe(bottom_10, hide_index=True, use_container_width=True)

# ============================================================
# TAB 2: SENTIMENT COMMUNAUTÉ
# ============================================================
with tab2:
    st.markdown("### 💬 Sentiment Steam par Genre")
    
    # Agrégation par genre
    df_sentiment = df_reviews.explode("Genres").dropna(subset=["Genres"])
    
    sentiment_by_genre = df_sentiment.groupby("Genres").agg({
        "Positive": "sum",
        "Negative": "sum",
        "Total Reviews": "sum"
    }).reset_index()
    
    sentiment_by_genre = sentiment_by_genre[sentiment_by_genre["Total Reviews"] >= 100]  # Minimum reviews
    sentiment_by_genre["Positive Ratio"] = (sentiment_by_genre["Positive"] / sentiment_by_genre["Total Reviews"]) * 100
    sentiment_by_genre = sentiment_by_genre.sort_values("Positive Ratio", ascending=False).head(15)
    
    fig_sentiment = px.bar(
        sentiment_by_genre,
        x="Positive Ratio",
        y="Genres",
        orientation="h",
        color="Positive Ratio",
        color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]]
    )
    
    fig_sentiment.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="% Reviews Positives",
        height=500,
        showlegend=False,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.divider()
    
    # Jeux Polarisants (table)
    st.markdown("### ⚡ Jeux Polarisants (Ratio ~50/50)")
    
    if len(polarizing) > 0:
        polarizing_display = polarizing.nlargest(10, "Total Reviews")[
            ["Name", "Positive Ratio", "Total Reviews", "Metacritic score"]
        ].copy()
        
        polarizing_display["Positive Ratio"] = (polarizing_display["Positive Ratio"] * 100).round(1)
        polarizing_display.columns = ["Nom", "% Positif", "Total Reviews", "Metacritic"]
        
        st.dataframe(polarizing_display, hide_index=True, use_container_width=True)
    else:
        st.info("Aucun jeu polarisant détecté")

# ============================================================
# TAB 3: PRIX VS QUALITÉ
# ============================================================
with tab3:
    st.markdown("### 💰 Corrélation Prix vs Metacritic")
    
    df_price_scatter = df_meta[df_meta["Price"] > 0]
    
    if len(df_price_scatter) > 0:
        # Échantillonnage
        if len(df_price_scatter) > 2000:
            df_price_scatter = df_price_scatter.sample(2000, random_state=42)
        
        # Catégories de prix
        df_price_scatter["Price Category"] = pd.cut(
            df_price_scatter["Price"],
            bins=[0, 10, 30, 100],
            labels=["Budget (<$10)", "Standard ($10-30)", "Premium (>$30)"]
        )
        
        fig_price_scatter = px.scatter(
            df_price_scatter,
            x="Price",
            y="Metacritic score",
            color="Price Category",
            size=np.log1p(df_price_scatter["Total Reviews"]) * 3,
            hover_data=["Name"],
            color_discrete_map={
                "Budget (<$10)": COLORS['chart'][0],
                "Standard ($10-30)": COLORS['chart'][2],
                "Premium (>$30)": COLORS['chart'][4]
            },
            trendline="ols"
        )
        
        fig_price_scatter.update_traces(marker=dict(opacity=0.6))
        
        fig_price_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            xaxis_title="Prix ($)",
            yaxis_title="Metacritic Score"
        )
        
        st.plotly_chart(fig_price_scatter, use_container_width=True)
        
        st.metric("📊 Corrélation Prix-Qualité", f"{price_corr:.3f}")
    
    st.divider()
    
    # Best ROI par tranche
    st.markdown("### 💎 Best Value for Money par Tranche")
    
    col_roi1, col_roi2, col_roi3 = st.columns(3)
    
    with col_roi1:
        st.markdown("#### Budget (<$10)")
        
        budget_games = df_price_quality[df_price_quality["Price"] < 10]
        
        if len(budget_games) > 0:
            top_budget = budget_games.nlargest(5, "Metacritic score")[["Name", "Price", "Metacritic score"]]
            top_budget.columns = ["Nom", "Prix ($)", "Metacritic"]
            
            st.dataframe(top_budget, hide_index=True, use_container_width=True)
    
    with col_roi2:
        st.markdown("#### Standard ($10-30)")
        
        standard_games = df_price_quality[(df_price_quality["Price"] >= 10) & (df_price_quality["Price"] <= 30)]
        
        if len(standard_games) > 0:
            top_standard = standard_games.nlargest(5, "Metacritic score")[["Name", "Price", "Metacritic score"]]
            top_standard.columns = ["Nom", "Prix ($)", "Metacritic"]
            
            st.dataframe(top_standard, hide_index=True, use_container_width=True)
    
    with col_roi3:
        st.markdown("#### Premium (>$30)")
        
        premium_games = df_price_quality[df_price_quality["Price"] > 30]
        
        if len(premium_games) > 0:
            top_premium = premium_games.nlargest(5, "Metacritic score")[["Name", "Price", "Metacritic score"]]
            top_premium.columns = ["Nom", "Prix ($)", "Metacritic"]
            
            st.dataframe(top_premium, hide_index=True, use_container_width=True)

# ============================================================
# TAB 4: EXCELLENCE PAR GENRE
# ============================================================
with tab4:
    st.markdown("### 🏆 Excellence Critique par Genre")
    
    # Boxplot Metacritic par genre
    df_boxplot_genre = df_meta.explode("Genres").dropna(subset=["Genres"])
    top_genres_meta = df_boxplot_genre["Genres"].value_counts().head(15).index.tolist()
    df_boxplot_genre = df_boxplot_genre[df_boxplot_genre["Genres"].isin(top_genres_meta)]
    
    fig_box_genre = px.box(
        df_boxplot_genre,
        x="Genres",
        y="Metacritic score",
        color="Genres",
        color_discrete_sequence=COLORS['chart']
    )
    
    fig_box_genre.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_tickangle=-45,
        showlegend=False,
        height=450
    )
    
    st.plotly_chart(fig_box_genre, use_container_width=True)
    
    st.divider()
    
    # Table statistiques par genre
    st.markdown("### 📊 Statistiques par Genre (Top 10)")
    
    genre_stats = df_boxplot_genre.groupby("Genres")["Metacritic score"].agg([
        ('Médiane', 'median'),
        ('Moyenne', 'mean'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Nb Jeux', 'count')
    ]).reset_index()
    
    genre_stats = genre_stats.sort_values("Médiane", ascending=False).head(10)
    
    # Arrondir
    for col in ['Médiane', 'Moyenne', 'Min', 'Max']:
        genre_stats[col] = genre_stats[col].round(1)
    
    st.dataframe(genre_stats, hide_index=True, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "⭐ **GameData360 — Ratings & Sentiment Analysis** | "
    f"Analyse Metacritic sur {len(df_meta):,} jeux notés | "
    f"Sentiment Steam sur {len(df_reviews):,} jeux reviewés"
)
