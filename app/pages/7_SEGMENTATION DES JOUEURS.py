# -*- coding: utf-8 -*-
"""
GameData360 - Page Segmentation des Joueurs
============================================
Segmentation comportementale basée sur engagement, préférences et budget.
Segments: Casual, Hardcore, Multiplayer Fans, Budget Gamers, Quality Seekers.

Auteur: GameData360 Team
Version: 3.0 (Behavioral Segmentation Edition)
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
    page_title="GameData360 — Segmentation",
    page_icon="👥",
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
st.markdown("# 👥 SEGMENTATION DES JOUEURS")
st.markdown("##### Analyse comportementale : Engagement, Budget, Préférences Sociales")

# ============================================================
# 2. CHARGEMENT DES DONNÉES
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
# 3. SEGMENTATION AUTOMATIQUE
# ============================================================
@st.cache_data(show_spinner=False)
def create_segments(_df):
    """Crée les segments comportementaux."""
    df = _df.copy()
    
    # Segment 1: Par Engagement (Playtime)
    df["Engagement"] = pd.cut(
        df["Median playtime forever"],
        bins=[0, 300, 1200, float('inf')],  # 0-5h, 5-20h, 20h+
        labels=["Casual", "Regular", "Hardcore"]
    )
    
    # Segment 2: Par Budget (Prix)
    df["Budget Segment"] = pd.cut(
        df["Price"],
        bins=[-1, 0, 15, 40, float('inf')],
        labels=["F2P", "Budget (<$15)", "Standard ($15-40)", "Premium (>$40)"]
    )
    
    # Segment 3: Par Préférence Sociale (Categories)
    def get_social_preference(categories):
        if not isinstance(categories, list):
            return "Solo"
        cats_lower = [c.lower() for c in categories]
        
        if any('multi-player' in c or 'multiplayer' in c for c in cats_lower):
            return "Multiplayer"
        elif any('co-op' in c for c in cats_lower):
            return "Co-op"
        else:
            return "Solo"
    
    df["Social Preference"] = df["Categories"].apply(get_social_preference)
    
    # Segment 4: Par Qualité (Metacritic)
    df_with_meta = df[df["Metacritic score"] > 0].copy()
    df["Quality Seeker"] = pd.cut(
        df["Metacritic score"],
        bins=[0, 60, 75, 85, 100],
        labels=["Low", "Medium", "High", "Exceptional"]
    )
    
    return df

df_segmented = create_segments(df_analyse)

# ============================================================
# 4. TOP-LEVEL KPIs
# ============================================================
st.markdown("### 📊 Segments Dominants")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1: Segment Engagement Dominant
engagement_counts = df_segmented["Engagement"].value_counts()
if len(engagement_counts) > 0:
    dominant_engagement = engagement_counts.index[0]
    pct_engagement = (engagement_counts.iloc[0] / len(df_segmented)) * 100
    
    with kpi1:
        st.metric(
            "⏱️ Engagement Dominant",
            dominant_engagement,
            delta=f"{pct_engagement:.0f}% des jeux",
            delta_color="off",
            help="Segment d'engagement le plus représenté"
        )

# KPI 2: Segment Budget Dominant
budget_counts = df_segmented["Budget Segment"].value_counts()
if len(budget_counts) > 0:
    dominant_budget = budget_counts.index[0]
    pct_budget = (budget_counts.iloc[0] / len(df_segmented)) * 100
    
    with kpi2:
        st.metric(
            "💰 Budget Dominant",
            str(dominant_budget),
            delta=f"{pct_budget:.0f}% des jeux",
            delta_color="off",
            help="Segment de prix le plus représenté"
        )

# KPI 3: Préférence Sociale Dominante
social_counts = df_segmented["Social Preference"].value_counts()
if len(social_counts) > 0:
    dominant_social = social_counts.index[0]
    pct_social = (social_counts.iloc[0] / len(df_segmented)) * 100
    
    with kpi3:
        st.metric(
            "👥 Préférence Sociale",
            dominant_social,
            delta=f"{pct_social:.0f}% des jeux",
            delta_color="off",
            help="Type de jeu le plus répandu"
        )

# KPI 4: Quality Seekers
quality_high = len(df_segmented[(df_segmented["Quality Seeker"] == "High") | (df_segmented["Quality Seeker"] == "Exceptional")])
pct_quality = (quality_high / len(df_segmented[df_segmented["Quality Seeker"].notna()])) * 100 if len(df_segmented[df_segmented["Quality Seeker"].notna()]) > 0 else 0

with kpi4:
    st.metric(
        "⭐ Qualité Haute",
        f"{pct_quality:.0f}%",
        help="% de jeux avec Metacritic ≥75"
    )

st.divider()

# ============================================================
# 5. INSIGHTS AUTOMATIQUES
# ============================================================
st.markdown("### 🎯 Insights de Segmentation")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    # Insight 1: Domination Casual vs Hardcore
    casual_count = len(df_segmented[df_segmented["Engagement"] == "Casual"])
    hardcore_count = len(df_segmented[df_segmented["Engagement"] == "Hardcore"])
    
    if casual_count > hardcore_count * 2:
        st.info(
            f"🎮 **Marché Casual Dominant**: {casual_count:,} jeux Casual vs "
            f"{hardcore_count:,} Hardcore — le marché favorise les expériences courtes"
        )
    elif hardcore_count > casual_count:
        st.success(
            f"⚡ **Opportunité Hardcore**: Plus de jeux Hardcore ({hardcore_count:,}) que Casual — "
            f"segment engagé bien servi"
        )
    
    # Insight 2: F2P vs Payant
    f2p_count = len(df_segmented[df_segmented["Budget Segment"] == "F2P"])
    payant_count = len(df_segmented[df_segmented["Budget Segment"] != "F2P"])
    f2p_pct = (f2p_count / len(df_segmented)) * 100
    
    if f2p_pct > 30:
        st.warning(
            f"🆓 **Forte Présence F2P**: {f2p_pct:.0f}% du marché est gratuit — "
            f"pression sur les jeux payants"
        )
    
    # Insight 3: Segment Budget Sweet Spot
    budget_sweet = df_segmented[df_segmented["Budget Segment"] == "Budget (<$15)"]
    
    if len(budget_sweet) > 0:
        avg_meta_sweet = budget_sweet[budget_sweet["Metacritic score"] > 0]["Metacritic score"].mean()
        
        if avg_meta_sweet > 70:
            st.success(
                f"💎 **Sweet Spot Budget**: Les jeux <$15 ont un Metacritic moyen de "
                f"{avg_meta_sweet:.1f} — excellent rapport qualité/prix"
            )

with col_insight2:
    # Insight 4: Solo vs Multiplayer
    solo_count = len(df_segmented[df_segmented["Social Preference"] == "Solo"])
    multi_count = len(df_segmented[df_segmented["Social Preference"] == "Multiplayer"])
    
    if solo_count > multi_count * 2:
        st.info(
            f"🎮 **Domination Solo**: {solo_count:,} jeux Solo vs {multi_count:,} Multiplayer — "
            f"le solo-player reste roi"
        )
    elif multi_count > solo_count:
        st.success(
            f"👥 **Boom Multiplayer**: Plus de jeux Multiplayer ({multi_count:,}) que Solo — "
            f"social gaming en hausse"
        )
    
    # Insight 5: Qualité par Segment d'Engagement
    hardcore_high_quality = len(df_segmented[
        (df_segmented["Engagement"] == "Hardcore") &
        ((df_segmented["Quality Seeker"] == "High") | (df_segmented["Quality Seeker"] == "Exceptional"))
    ])
    
    casual_high_quality = len(df_segmented[
        (df_segmented["Engagement"] == "Casual") &
        ((df_segmented["Quality Seeker"] == "High") | (df_segmented["Quality Seeker"] == "Exceptional"))
    ])
    
    if hardcore_count > 0:
        hardcore_quality_pct = (hardcore_high_quality / hardcore_count) * 100
        
        if hardcore_quality_pct > 40:
            st.success(
                f"⭐ **Hardcore = Qualité**: {hardcore_quality_pct:.0f}% des jeux Hardcore "
                f"ont une haute qualité — investissement récompensé"
            )

st.divider()

# ============================================================
# 6. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⏱️ Engagement",
    "💰 Budget & Prix",
    "👥 Préférences Sociales",
    "📊 Matrices Croisées"
])

# ============================================================
# TAB 1: ENGAGEMENT
# ============================================================
with tab1:
    st.markdown("### ⏱️ Segmentation par Engagement (Playtime)")
    
    col_eng1, col_eng2 = st.columns([2, 1])
    
    with col_eng1:
        # Distribution
        eng_dist = df_segmented["Engagement"].value_counts().reset_index()
        eng_dist.columns = ["Segment", "Count"]
        
        fig_eng = px.bar(
            eng_dist,
            x="Segment",
            y="Count",
            color="Segment",
            color_discrete_map={
                "Casual": COLORS['chart'][0],
                "Regular": COLORS['chart'][2],
                "Hardcore": COLORS['chart'][4]
            }
        )
        
        fig_eng.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_eng, use_container_width=True)
    
    with col_eng2:
        st.markdown("#### 📋 Définitions")
        
        definitions = pd.DataFrame({
            'Segment': ['Casual', 'Regular', 'Hardcore'],
            'Playtime': ['0-5h', '5-20h', '>20h'],
            'Count': [
                f"{len(df_segmented[df_segmented['Engagement'] == 'Casual']):,}",
                f"{len(df_segmented[df_segmented['Engagement'] == 'Regular']):,}",
                f"{len(df_segmented[df_segmented['Engagement'] == 'Hardcore']):,}"
            ]
        })
        
        st.dataframe(definitions, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Qualité par Engagement
    st.markdown("### ⭐ Qualité Moyenne par Segment d'Engagement")
    
    eng_quality = df_segmented[df_segmented["Metacritic score"] > 0].groupby("Engagement")["Metacritic score"].mean().reset_index()
    eng_quality.columns = ["Segment", "Metacritic Moyen"]
    
    fig_eng_quality = px.bar(
        eng_quality,
        x="Segment",
        y="Metacritic Moyen",
        color="Metacritic Moyen",
        color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]]
    )
    
    fig_eng_quality.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        showlegend=False,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_eng_quality, use_container_width=True)

# ============================================================
# TAB 2: BUDGET & PRIX
# ============================================================
with tab2:
    st.markdown("### 💰 Segmentation par Budget")
    
    # Distribution
    budget_dist = df_segmented["Budget Segment"].value_counts().reset_index()
    budget_dist.columns = ["Segment", "Count"]
    
    fig_budget = px.pie(
        budget_dist,
        values="Count",
        names="Segment",
        hole=0.4,
        color_discrete_sequence=COLORS['chart']
    )
    
    fig_budget.update_layout(
        **PLOTLY_LAYOUT,
        height=450
    )
    
    st.plotly_chart(fig_budget, use_container_width=True)
    
    st.divider()
    
    # Qualité par Segment Budget
    st.markdown("### ⭐ Qualité par Segment de Prix")
    
    budget_quality = df_segmented[df_segmented["Metacritic score"] > 0].groupby("Budget Segment")["Metacritic score"].agg(['mean', 'median', 'count']).reset_index()
    budget_quality.columns = ["Segment", "Moyenne", "Médiane", "Nb Jeux"]
    
    budget_quality["Moyenne"] = budget_quality["Moyenne"].round(1)
    budget_quality["Médiane"] = budget_quality["Médiane"].round(1)
    
    st.dataframe(budget_quality, hide_index=True, use_container_width=True)

# ============================================================
# TAB 3: PRÉFÉRENCES SOCIALES
# ============================================================
with tab3:
    st.markdown("### 👥 Segmentation par Préférences Sociales")
    
    col_social1, col_social2 = st.columns(2)
    
    with col_social1:
        social_dist = df_segmented["Social Preference"].value_counts().reset_index()
        social_dist.columns = ["Préférence", "Count"]
        
        fig_social = px.bar(
            social_dist,
            x="Préférence",
            y="Count",
            color="Préférence",
            color_discrete_map={
                "Solo": COLORS['solo'],
                "Multiplayer": COLORS['multi'],
                "Co-op": COLORS['tertiary']
            }
        )
        
        fig_social.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_social, use_container_width=True)
    
    with col_social2:
        st.markdown("#### 📊 Statistiques")
        
        social_stats = social_dist.copy()
        social_stats["%"] = (social_stats["Count"] / social_stats["Count"].sum() * 100).round(1)
        
        st.dataframe(social_stats, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # CCU par préférence sociale
    st.markdown("### 👥 Popularité (Peak CCU) par Type Social")
    
    social_ccu = df_segmented[df_segmented["Peak CCU"] > 0].groupby("Social Preference")["Peak CCU"].median().reset_index()
    social_ccu.columns = ["Préférence", "CCU Médian"]
    
    fig_social_ccu = px.bar(
        social_ccu,
        x="Préférence",
        y="CCU Médian",
        color="Préférence",
        color_discrete_map={
            "Solo": COLORS['solo'],
            "Multiplayer": COLORS['multi'],
            "Co-op": COLORS['tertiary']
        }
    )
    
    fig_social_ccu.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
        height=350
    )
    
    st.plotly_chart(fig_social_ccu, use_container_width=True)

# ============================================================
# TAB 4: MATRICES CROISÉES
# ============================================================
with tab4:
    st.markdown("### 📊 Matrice Engagement × Budget")
    
    # Heatmap Engagement × Budget
    cross_eng_budget = pd.crosstab(df_segmented["Engagement"], df_segmented["Budget Segment"])
    
    fig_heat_eng_budget = px.imshow(
        cross_eng_budget,
        labels=dict(x="Budget", y="Engagement", color="Nombre de jeux"),
        x=cross_eng_budget.columns,
        y=cross_eng_budget.index,
        color_continuous_scale=[[0, COLORS['background']], [1, COLORS['primary']]],
        text_auto=True
    )
    
    fig_heat_eng_budget.update_layout(
        **PLOTLY_LAYOUT,
        height=350
    )
    
    st.plotly_chart(fig_heat_eng_budget, use_container_width=True)
    
    st.divider()
    
    # Heatmap Engagement × Social
    st.markdown("### 📊 Matrice Engagement × Préférence Sociale")
    
    cross_eng_social = pd.crosstab(df_segmented["Engagement"], df_segmented["Social Preference"])
    
    fig_heat_eng_social = px.imshow(
        cross_eng_social,
        labels=dict(x="Préférence Sociale", y="Engagement", color="Nombre de jeux"),
        x=cross_eng_social.columns,
        y=cross_eng_social.index,
        color_continuous_scale=[[0, COLORS['background']], [1, COLORS['secondary']]],
        text_auto=True
    )
    
    fig_heat_eng_social.update_layout(
        **PLOTLY_LAYOUT,
        height=350
    )
    
    st.plotly_chart(fig_heat_eng_social, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "👥 **GameData360 — Segmentation Comportementale** | "
    f"Analyse sur {len(df_segmented):,} jeux"
)