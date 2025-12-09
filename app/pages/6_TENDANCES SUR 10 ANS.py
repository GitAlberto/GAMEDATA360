# -*- coding: utf-8 -*-
"""
GameData360 - Page Tendances sur 10 ans
========================================
Analyse temporelle avec insights automatiques et détection de tendances.
Insights: Croissance, Saturation, F2P Explosion, Indie Boom, Genre Shift.

Auteur: GameData360 Team
Version: 3.0 (Temporal Intelligence Edition)
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
    page_title="GameData360 — Tendances",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le thème gaming
st.markdown("""
<style>
    /* Import de la police gaming */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    
    /* Thème néon gaming */
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
    
    /* Titres stylisés */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Onglets gaming */
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
st.markdown("# 📈 TENDANCES SUR 15 ANS")
st.markdown("##### Analyse temporelle 2010-2024 : Évolution, Insights & Prédictions")

# ============================================================
# 2. CHARGEMENT DES DONNÉES (CACHE OPTIMISÉ)
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_filter_data():
    """Charge les données et filtre les logiciels et jeux bêtas."""
    df = load_game_data(str(FILE_PATH))
    
    # Filtrage des genres non-jeux
    def is_game(genres_list):
        if not isinstance(genres_list, list):
            return True
        genres_lower = [g.lower() for g in genres_list]
        return not any(genre in genres_lower for genre in NON_GAME_GENRES)
    
    initial_count = len(df)
    df = df[df["Genres"].apply(is_game)].copy()
    excluded_software = initial_count - len(df)
    
    # Filtrage des jeux Early Access / Beta
    def is_not_beta(categories):
        """Retourne False si le jeu est en Early Access."""
        if not isinstance(categories, list):
            return True
        cats_lower = [c.lower() for c in categories]
        beta_keywords = ['early access', 'beta']
        return not any(keyword in cat for cat in cats_lower for keyword in beta_keywords)
    
    initial_after_software = len(df)
    df = df[df["Categories"].apply(is_not_beta)].copy()
    excluded_beta = initial_after_software - len(df)
    
    # Filtrage période 2010-2024
    df = df[(df["Release Year"] >= 2010) & (df["Release Year"] <= 2024)].copy()
    
    return df, excluded_software, excluded_beta

# Chargement avec indicateur
try:
    with st.spinner('⚡ Chargement des données temporelles...'):
        df_trends, excluded_software, excluded_beta = load_and_filter_data()
        
        total_excluded = excluded_software + excluded_beta
        if total_excluded > 0:
            st.sidebar.success(f"🎮 {excluded_software:,} logiciels + {excluded_beta:,} jeux bêta exclus")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. CALCULS STATISTIQUES (CACHE)
# ============================================================
@st.cache_data(show_spinner=False)
def calculate_yearly_stats(_df):
    """Calcule les statistiques annuelles."""
    # Stats générales
    yearly = _df.groupby('Release Year').agg({
        'AppID': 'count',
        'Price': 'mean',
        'Peak CCU': 'mean',
        'User score': 'mean',
        'Recommendations': 'mean'
    }).reset_index()
    
    yearly.columns = ['Year', 'Volume', 'Avg_Price', 'Avg_CCU', 'Avg_UserScore', 'Avg_Reco']
    
    # Metacritic (filtré > 0)
    df_meta = _df[_df['Metacritic score'] > 0]
    meta_stats = df_meta.groupby('Release Year')['Metacritic score'].median().reset_index()
    meta_stats.columns = ['Year', 'Metacritic_Median']
    
    yearly = pd.merge(yearly, meta_stats, on='Year', how='left')
    
    # F2P percentage
    f2p_pct = _df.groupby('Release Year').apply(
        lambda x: (x['Price'] == 0).sum() / len(x) * 100
    ).reset_index(name='F2P_Pct')
    f2p_pct.columns = ['Year', 'F2P_Pct']
    
    yearly = pd.merge(yearly, f2p_pct, on='Year', how='left')
    
    # Revenus totaux
    revenue_stats = _df.groupby('Release Year')['Estimated revenue'].sum().reset_index()
    revenue_stats.columns = ['Year', 'Total_Revenue']
    
    yearly = pd.merge(yearly, revenue_stats, on='Year', how='left')
    
    return yearly

yearly_stats = calculate_yearly_stats(df_trends)

# ============================================================
# 4. SIDEBAR - FILTRES
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Info Dataset")
    st.caption(f"**Période:** 2010-2024")
    st.caption(f"**Total jeux:** {len(df_trends):,}")
    st.caption(f"**Années:** {len(yearly_stats)}")

# ============================================================
# 5. TOP-LEVEL KPIs (SYNTHÈSE DÉCENNALE)
# ============================================================
st.markdown("### 📊 Synthèse Décennale (2010 → 2024)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1: Croissance Volume
vol_2010 = yearly_stats[yearly_stats['Year'] == 2010]['Volume'].iloc[0] if 2010 in yearly_stats['Year'].values else 1
vol_2024 = yearly_stats[yearly_stats['Year'] == 2024]['Volume'].iloc[0] if 2024 in yearly_stats['Year'].values else vol_2010
growth_rate = ((vol_2024 - vol_2010) / vol_2010) * 100 if vol_2010 > 0 else 0

with kpi1:
    st.metric(
        "📈 Croissance Volume",
        f"+{growth_rate:.0f}%",
        delta=f"{vol_2024:,} jeux en 2024",
        help="Croissance du nombre de jeux publiés (2010-2024)"
    )

# KPI 2: Évolution Prix
price_2010 = yearly_stats[yearly_stats['Year'] == 2010]['Avg_Price'].iloc[0] if 2010 in yearly_stats['Year'].values else 0
price_2024 = yearly_stats[yearly_stats['Year'] == 2024]['Avg_Price'].iloc[0] if 2024 in yearly_stats['Year'].values else price_2010
price_change = ((price_2024 - price_2010) / price_2010) * 100 if price_2010 > 0 else 0

with kpi2:
    st.metric(
        "💰 Évolution Prix",
        f"{price_change:+.0f}%",
        delta=f"${price_2024:.2f} en 2024",
        delta_color="inverse",
        help="Évolution du prix moyen"
    )

# KPI 3: Part F2P
f2p_2010 = yearly_stats[yearly_stats['Year'] == 2010]['F2P_Pct'].iloc[0] if 2010 in yearly_stats['Year'].values else 0
f2p_2024 = yearly_stats[yearly_stats['Year'] == 2024]['F2P_Pct'].iloc[0] if 2024 in yearly_stats['Year'].values else f2p_2010
f2p_delta = f2p_2024 - f2p_2010

with kpi3:
    st.metric(
        "🆓 Part F2P",
        f"{f2p_2024:.0f}%",
        delta=f"{f2p_delta:+.0f} pts vs 2010",
        help="Pourcentage de jeux gratuits"
    )

# KPI 4: Évolution Qualité (Metacritic)
meta_2010 = yearly_stats[yearly_stats['Year'] == 2010]['Metacritic_Median'].iloc[0] if 2010 in yearly_stats['Year'].values and not pd.isna(yearly_stats[yearly_stats['Year'] == 2010]['Metacritic_Median'].iloc[0]) else 0
meta_2024 = yearly_stats[yearly_stats['Year'] == 2024]['Metacritic_Median'].iloc[0] if 2024 in yearly_stats['Year'].values and not pd.isna(yearly_stats[yearly_stats['Year'] == 2024]['Metacritic_Median'].iloc[0]) else meta_2010
meta_delta = meta_2024 - meta_2010 if meta_2010 > 0 else 0

with kpi4:
    st.metric(
        "⭐ Qualité (Metacritic)",
        f"{meta_2024:.0f}" if meta_2024 > 0 else "N/A",
        delta=f"{meta_delta:+.1f} pts" if meta_2010 > 0 else None,
        help="Score Metacritic médian"
    )

st.divider()

# ============================================================
# 6. INSIGHTS AUTOMATIQUES (SECTION DÉDIÉE)
# ============================================================
st.markdown("### 🎯 Insights Automatiques")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    # Insight 1: Croissance & CAGR
    if len(yearly_stats) >= 2:
        years_span = yearly_stats['Year'].max() - yearly_stats['Year'].min()
        cagr = ((vol_2024 / vol_2010) ** (1 / years_span) - 1) * 100 if vol_2010 > 0 and years_span > 0 else 0
        
        st.success(
            f"📈 **Croissance Soutenue**: Le marché a crû de **{growth_rate:.0f}%** en {years_span} ans "
            f"(CAGR: **{cagr:.1f}%/an**). Volume passé de {vol_2010:,} à {vol_2024:,} jeux."
        )
    
    # Insight 2: F2P Explosion
    if f2p_delta > 5:
        st.warning(
            f"🆓 **Explosion du F2P**: La part des jeux gratuits a explosé de **{f2p_delta:.0f} points** "
            f"({f2p_2010:.0f}% → {f2p_2024:.0f}%) — bouleversement du modèle économique."
        )
    elif f2p_delta < -5:
        st.info(
            f"💰 **Retour du Payant**: La part des jeux payants remonte (+{abs(f2p_delta):.0f} pts)."
        )
    
    # Insight 3: Quality Squeeze
    if meta_delta < -2 and growth_rate > 50:
        st.warning(
            f"📉 **Quality Squeeze Détecté**: Score Metacritic en baisse de **{abs(meta_delta):.1f} pts** "
            f"malgré **+{growth_rate:.0f}%** de jeux — dilution qualité par volume."
        )
    elif meta_delta > 2:
        st.success(
            f"⭐ **Amélioration Qualité**: Score Metacritic en hausse de **+{meta_delta:.1f} pts** "
            f"— le marché s'améliore qualitativement."
        )

with col_insight2:
    # Insight 4: Pricing vs Inflation
    US_INFLATION_2010_2024 = 35  # Approximation inflation USA sur période
    
    if price_change < US_INFLATION_2010_2024:
        st.success(
            f"💰 **Prix Stables**: Hausse de seulement **{price_change:.0f}%** vs inflation réelle "
            f"~{US_INFLATION_2010_2024}% — le gaming reste accessible !"
        )
    else:
        st.warning(
            f"⚠️ **Augmentation Prix**: +{price_change:.0f}% vs inflation {US_INFLATION_2010_2024}% "
            f"— le gaming devient plus cher."
        )
    
    # Insight 5: Indie Boom
    df_2010 = df_trends[df_trends['Release Year'] == 2010]
    df_2024 = df_trends[df_trends['Release Year'] == 2024]
    
    if len(df_2010) > 0 and len(df_2024) > 0:
        indie_2010_pct = (df_2010['Genres'].apply(
            lambda x: any('indie' in g.lower() for g in x) if isinstance(x, list) else False
        ).sum() / len(df_2010)) * 100
        
        indie_2024_pct = (df_2024['Genres'].apply(
            lambda x: any('indie' in g.lower() for g in x) if isinstance(x, list) else False
        ).sum() / len(df_2024)) * 100
        
        indie_delta = indie_2024_pct - indie_2010_pct
        
        if indie_delta > 5:
            st.info(
                f"🎨 **Indie Boom**: Part des jeux indie passée de **{indie_2010_pct:.0f}%** à "
                f"**{indie_2024_pct:.0f}%** (+{indie_delta:.0f} pts) — démocratisation du dev."
            )
    
    # Insight 6: Genre Shift
    df_2010_exploded = df_2010.explode('Genres').dropna(subset=['Genres'])
    df_2024_exploded = df_2024.explode('Genres').dropna(subset=['Genres'])
    
    if len(df_2010_exploded) > 0 and len(df_2024_exploded) > 0:
        top_2010 = df_2010_exploded['Genres'].value_counts().index[0] if len(df_2010_exploded) > 0 else "N/A"
        top_2024 = df_2024_exploded['Genres'].value_counts().index[0] if len(df_2024_exploded) > 0 else "N/A"
        
        if top_2010 != top_2024 and top_2010 != "N/A" and top_2024 != "N/A":
            st.info(
                f"🎮 **Genre Shift**: Le genre dominant a basculé de **{top_2010}** (2010) "
                f"à **{top_2024}** (2024) — évolution des préférences joueurs."
            )

st.divider()

# ============================================================
# 7. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Volume & Marché",
    "🎮 Genres & Évolution",
    "💰 Pricing & Monétisation",
    "⭐ Qualité & Engagement"
])

# ============================================================
# TAB 1: VOLUME & MARCHÉ
# ============================================================
with tab1:
    st.markdown("### 📈 Évolution du Volume de Jeux Publiés")
    
    col_vol1, col_vol2 = st.columns([2, 1])
    
    with col_vol1:
        # Graphique volume avec trendline
        fig_volume = go.Figure()
        
        # Ligne principale
        fig_volume.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['Volume'],
            mode='lines+markers',
            name='Volume annuel',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=10)
        ))
        
        # Trendline (régression linéaire)
        z = np.polyfit(yearly_stats['Year'], yearly_stats['Volume'], 1)
        p = np.poly1d(z)
        fig_volume.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=p(yearly_stats['Year']),
            mode='lines',
            name='Tendance',
            line=dict(color=COLORS['danger'], width=2, dash='dash')
        ))
        
        fig_volume.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Année",
            yaxis_title="Nombre de jeux publiés",
            height=400
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col_vol2:
        st.markdown("#### 📋 Stats Volume")
        
        # Table statistiques
        stats_vol = pd.DataFrame({
            'Métrique': ['Min', 'Max', 'Moyenne', 'Médiane', 'Écart-type'],
            'Valeur': [
                f"{yearly_stats['Volume'].min():,}",
                f"{yearly_stats['Volume'].max():,}",
                f"{yearly_stats['Volume'].mean():,.0f}",
                f"{yearly_stats['Volume'].median():,.0f}",
                f"{yearly_stats['Volume'].std():,.0f}"
            ]
        })
        
        st.dataframe(stats_vol, hide_index=True, use_container_width=True)
        
        # Détection pic
        peak_year = yearly_stats.loc[yearly_stats['Volume'].idxmax(), 'Year']
        peak_vol = yearly_stats['Volume'].max()
        
        st.metric(
            "🏆 Année Record",
            int(peak_year),
            delta=f"{peak_vol:,} jeux",
            delta_color="off"
        )
    
    st.divider()
    
    # Revenus totaux
    st.markdown("### 💰 Évolution des Revenus Totaux Annuels")
    
    fig_revenue = px.bar(
        yearly_stats,
        x='Year',
        y='Total_Revenue',
        color='Total_Revenue',
        color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['chart'][4]]]
    )
    
    fig_revenue.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Année",
        yaxis_title="Revenus Totaux ($)",
        height=400,
        showlegend=False,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)

# ============================================================
# TAB 2: GENRES & ÉVOLUTION
# ============================================================
with tab2:
    st.markdown("### 🎮 Évolution de la Part de Marché des Genres")
    
    # Préparation données genres
    df_genres_time = df_trends.explode('Genres').dropna(subset=['Genres'])
    
    # Top 8 genres
    top_genres = df_genres_time['Genres'].value_counts().head(8).index.tolist()
    df_genres_top = df_genres_time[df_genres_time['Genres'].isin(top_genres)]
    
    # Calcul parts de marché normalisées
    genre_evolution = df_genres_top.groupby(['Release Year', 'Genres']).size().reset_index(name='Count')
   
    # Stacked area chart
    fig_genres_area = px.area(
        genre_evolution,
        x='Release Year',
        y='Count',
        color='Genres',
        line_group='Genres',
        color_discrete_sequence=COLORS['chart']
    )
    
    fig_genres_area.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Année",
        yaxis_title="Nombre de jeux",
        height=450
    )
    
    st.plotly_chart(fig_genres_area, use_container_width=True)
    
    st.divider()
    
    # Croissance par genre
    st.markdown("### 📊 Taux de Croissance par Genre (2010-2024)")
    
    growth_by_genre = {}
    
    for genre in top_genres:
        vol_2010_genre = len(df_2010_exploded[df_2010_exploded['Genres'] == genre]) if len(df_2010_exploded) > 0 else 0
        vol_2024_genre = len(df_2024_exploded[df_2024_exploded['Genres'] == genre]) if len(df_2024_exploded) > 0 else 0
        
        if vol_2010_genre > 0:
            growth = ((vol_2024_genre - vol_2010_genre) / vol_2010_genre) * 100
            growth_by_genre[genre] = growth
        elif vol_2024_genre > 0:
            growth_by_genre[genre] = 100  # Nouveau genre
    
    if growth_by_genre:
        growth_df = pd.DataFrame(list(growth_by_genre.items()), columns=['Genre', 'Croissance'])
        growth_df = growth_df.sort_values('Croissance', ascending=True)
        
        fig_growth = px.bar(
            growth_df,
            x='Croissance',
            y='Genre',
            orientation='h',
            color='Croissance',
            color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]]
        )
        
        fig_growth.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Croissance (%)",
            yaxis_title="",
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Insight genre en explosion
        fastest = growth_df.iloc[-1]
        if fastest['Croissance'] > 100:
            st.success(
                f"🚀 **Genre en Explosion**: **{fastest['Genre']}** a crû de "
                f"**+{fastest['Croissance']:.0f}%** depuis 2010 !"
            )

# ============================================================
# TAB 3: PRICING & MONÉTISATION
# ============================================================
with tab3:
    st.markdown("### 💵 Évolution du Prix Moyen")
    
    fig_price = go.Figure()
    
    fig_price.add_trace(go.Scatter(
        x=yearly_stats['Year'],
        y=yearly_stats['Avg_Price'],
        mode='lines+markers',
        name='Prix Moyen',
        line=dict(color=COLORS['tertiary'], width=3),
        marker=dict(size=10)
    ))
    
    fig_price.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Année",
        yaxis_title="Prix Moyen ($)",
        height=400
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.divider()
    
    # F2P vs Payant
    st.markdown("### 🆓 Évolution F2P vs Payant")
    
    col_f2p1, col_f2p2 = st.columns(2)
    
    with col_f2p1:
        # Graphique % F2P
        fig_f2p = go.Figure()
        
        fig_f2p.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['F2P_Pct'],
            mode='lines+markers',
            fill='tozeroy',
            name='% F2P',
            line=dict(color=COLORS['multi'], width=3),
            marker=dict(size=10),
            fillcolor=f'rgba({int(COLORS["multi"][1:3], 16)}, {int(COLORS["multi"][3:5], 16)}, {int(COLORS["multi"][5:7], 16)}, 0.3)'
        ))
        
        fig_f2p.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Année",
            yaxis_title="% de jeux F2P",
            height=400
        )
        
        st.plotly_chart(fig_f2p, use_container_width=True)
    
    with col_f2p2:
        # Évolution modèle économique
        df_model = df_trends.copy()
        df_model['Model'] = df_model['Price'].apply(lambda x: 'F2P' if x == 0 else 'Payant')
        
        model_counts = df_model.groupby(['Release Year', 'Model']).size().reset_index(name='Count')
        
        fig_model = px.bar(
            model_counts,
            x='Release Year',
            y='Count',
            color='Model',
            barmode='stack',
            color_discrete_map={'F2P': COLORS['multi'], 'Payant': COLORS['solo']}
        )
        
        fig_model.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Année",
            yaxis_title="Nombre de jeux",
            height=400
        )
        
        st.plotly_chart(fig_model, use_container_width=True)

# ============================================================
# TAB 4: QUALITÉ & ENGAGEMENT
# ============================================================
with tab4:
    st.markdown("### ⭐ Évolution de la Qualité (Metacritic)")
    
    # Filtrer années avec données Metacritic
    yearly_meta = yearly_stats.dropna(subset=['Metacritic_Median'])
    
    if len(yearly_meta) > 0:
        fig_meta = go.Figure()
        
        fig_meta.add_trace(go.Scatter(
            x=yearly_meta['Year'],
            y=yearly_meta['Metacritic_Median'],
            mode='lines+markers',
            name='Score Médian',
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(size=10)
        ))
        
        # Ligne moyenne globale
        avg_meta = yearly_meta['Metacritic_Median'].mean()
        fig_meta.add_hline(
            y=avg_meta,
            line_dash="dash",
            line_color=COLORS['warning'],
            annotation_text=f"Moyenne: {avg_meta:.1f}",
            annotation_position="right"
        )
        
        fig_meta.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Année",
            yaxis_title="Score Metacritic Médian",
            height=400,
            yaxis=dict(range=[60, 85])
        )
        
        st.plotly_chart(fig_meta, use_container_width=True)
    else:
        st.warning("⚠️ Données Metacritic insuffisantes")
    
    st.divider()
    
    # CCU Evolution
    st.markdown("### 👥 Évolution du Peak CCU Moyen")
    
    fig_ccu = go.Figure()
    
    fig_ccu.add_trace(go.Scatter(
        x=yearly_stats['Year'],
        y=yearly_stats['Avg_CCU'],
        mode='lines+markers',
        name='CCU Moyen',
        line=dict(color=COLORS['chart'][2], width=3),
        marker=dict(size=10)
    ))
    
    # Détection boom COVID (2020)
    if 2020 in yearly_stats['Year'].values and 2019 in yearly_stats['Year'].values:
        ccu_2019 = yearly_stats[yearly_stats['Year'] == 2019]['Avg_CCU'].iloc[0]
        ccu_2020 = yearly_stats[yearly_stats['Year'] == 2020]['Avg_CCU'].iloc[0]
        boost_covid = ((ccu_2020 - ccu_2019) / ccu_2019) * 100 if ccu_2019 > 0 else 0
        
        if boost_covid > 20:
            fig_ccu.add_annotation(
                x=2020,
                y=ccu_2020,
                text=f"Boom COVID<br>+{boost_covid:.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLORS['danger'],
                font=dict(color=COLORS['danger'])
            )
    
    fig_ccu.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Année",
        yaxis_title="Peak CCU Moyen",
        height=400
    )
    
    st.plotly_chart(fig_ccu, use_container_width=True)
    
    # Insight COVID
    if 2020 in yearly_stats['Year'].values and 2019 in yearly_stats['Year'].values:
        if boost_covid > 20:
            st.info(
                f"🎮 **Boom Gaming COVID-19**: Le CCU moyen a bondi de **+{boost_covid:.0f}%** "
                f"en 2020 (confinements mondiaux) — explosion de l'audience."
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "📈 **GameData360 — Tendances Temporelles** | "
    f"Analyse 2010-2024 sur {len(df_trends):,} jeux"
)