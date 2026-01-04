# -*- coding: utf-8 -*-
"""
GameData360 - Page Developers & Publishers
===========================================
Analyse des studios et éditeurs avec classification AAA vs Indie.
Performance, concentration marché, spécialisation genres.

Auteur: GameData360 Team
Version: 1.0 (Studio Analysis Edition)
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
    page_title="GameData360 — Developers & Publishers",
    page_icon="🏢",
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
st.markdown("# 🏢 DEVELOPERS & PUBLISHERS")
st.markdown("##### Analyse des Studios — AAA vs Indie, Performance, Concentration Marché")

# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    """Charge et prépare les données avec cache."""
    return load_game_data(str(FILE_PATH))

# Chargement avec indicateur
try:
    with st.spinner('⚡ Chargement des données...'):
        df_analyse = load_data()
        
        # Filtrage des genres non-jeux (logiciels)
        def is_game(genres_list):
            """Retourne False si le jeu contient un genre de logiciel."""
            if not isinstance(genres_list, list):
                return True
            genres_lower = [g.lower() for g in genres_list]
            return not any(genre in genres_lower for genre in NON_GAME_GENRES)
        
        initial_count = len(df_analyse)
        df_analyse = df_analyse[df_analyse["Genres"].apply(is_game)].copy()
        excluded_count = initial_count - len(df_analyse)
        
        if excluded_count > 0:
            st.sidebar.success(f"🎮 {excluded_count:,} logiciels exclus")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. CLASSIFICATION AAA vs INDIE
# ============================================================
@st.cache_data(show_spinner=False)
def classify_games(_df):
    """Classifie les jeux en AAA, Indie, ou AA (mid-tier)."""
    df = _df.copy()
    
    # Publishers majeurs (liste non exhaustive)
    major_publishers = [
        'electronic arts', 'ea', 'ubisoft', 'activision', 'blizzard',
        'take-two', '2k', 'rockstar', 'microsoft', 'xbox', 'bethesda',
        'sony', 'playstation', 'square enix', 'capcom', 'sega',
        'bandai namco', 'warner bros', 'deep silver', 'focus entertainment'
    ]
    
    def has_major_publisher(publishers):
        if not isinstance(publishers, str):
            return False
        publishers_lower = publishers.lower()
        return any(pub in publishers_lower for pub in major_publishers)
    
    # Calcul critères AAA
    df['is_major_publisher'] = df['Publishers'].apply(has_major_publisher)
    df['total_reviews'] = df['Positive'] + df['Negative']
    
    # Classification
    def classify(row):
        aaa_score = 0
        indie_score = 0
        
        # Critères AAA (≥3 pour être AAA)
        if row['Price'] >= 50:
            aaa_score += 1
        if row['Peak CCU'] >= 50000:
            aaa_score += 1
        if row['is_major_publisher']:
            aaa_score += 1
        if row['Metacritic score'] > 0 and row['Metacritic score'] >= 70:
            aaa_score += 1
        if row['total_reviews'] >= 20000:
            aaa_score += 1
        
        # Critères Indie (≥2 pour être Indie)
        if row['Price'] < 20 and row['Price'] > 0:  # Exclure F2P
            indie_score += 1
        if not row['is_major_publisher']:
            indie_score += 1
        if row['Peak CCU'] < 5000:
            indie_score += 1
        
        # Classification finale
        if aaa_score >= 3:
            return 'AAA'
        elif indie_score >= 2:
            return 'Indie'
        else:
            return 'AA (Mid-tier)'
    
    df['Game Tier'] = df.apply(classify, axis=1)
    
    return df

df_classified = classify_games(df_analyse)

# ============================================================
# 4. TOP-LEVEL KPIs
# ============================================================
st.markdown("### 📊 Vue d'Ensemble du Marché")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Distribution par tier
tier_counts = df_classified['Game Tier'].value_counts()

with kpi1:
    aaa_count = tier_counts.get('AAA', 0)
    aaa_pct = (aaa_count / len(df_classified)) * 100
    st.metric(
        "🏆 Jeux AAA",
        f"{aaa_count:,}",
        delta=f"{aaa_pct:.1f}% du marché",
        delta_color="off"
    )

with kpi2:
    indie_count = tier_counts.get('Indie', 0)
    indie_pct = (indie_count / len(df_classified)) * 100
    st.metric(
        "🎨 Jeux Indie",
        f"{indie_count:,}",
        delta=f"{indie_pct:.1f}% du marché",
        delta_color="off"
    )

with kpi3:
    aa_count = tier_counts.get('AA (Mid-tier)', 0)
    aa_pct = (aa_count / len(df_classified)) * 100
    st.metric(
        "⚙️ Jeux AA",
        f"{aa_count:,}",
        delta=f"{aa_pct:.1f}% du marché",
        delta_color="off"
    )

with kpi4:
    unique_devs = df_classified['Developers'].nunique()
    st.metric(
        "👨‍💻 Developers Uniques",
        f"{unique_devs:,}",
        help="Nombre de studios différents"
    )

st.divider()

# ============================================================
# 5. INSIGHTS AUTOMATIQUES
# ============================================================
st.markdown("### 🎯 Insights Automatiques")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    # Insight 1: Revenus AAA vs Indie
    aaa_revenue = df_classified[df_classified['Game Tier'] == 'AAA']['Estimated revenue'].sum()
    indie_revenue = df_classified[df_classified['Game Tier'] == 'Indie']['Estimated revenue'].sum()
    total_revenue = df_classified['Estimated revenue'].sum()
    
    aaa_revenue_pct = (aaa_revenue / total_revenue) * 100
    indie_revenue_pct = (indie_revenue / total_revenue) * 100
    
    if aaa_revenue_pct > 50:
        st.success(
            f"💰 **Domination AAA Revenus**: Les AAA représentent {aaa_pct:.1f}% des jeux "
            f"mais génèrent **{aaa_revenue_pct:.0f}% des revenus** (${aaa_revenue/1e9:.1f}B)"
        )
    elif indie_revenue_pct > 30:
        st.info(
            f"🎨 **Force Indie**: Les Indie ({indie_pct:.0f}% volume) génèrent "
            f"**{indie_revenue_pct:.0f}% des revenus** — performance remarquable"
        )
    
    # Insight 2: ROI par tier
    aaa_roi = (aaa_revenue / aaa_count) if aaa_count > 0 else 0
    indie_roi = (indie_revenue / indie_count) if indie_count > 0 else 0
    
    if aaa_roi > indie_roi * 10:
        st.warning(
            f"📈 **ROI AAA Supérieur**: Revenu moyen par jeu AAA (${aaa_roi/1e6:.1f}M) "
            f"est **{aaa_roi/indie_roi:.0f}x** celui d'un Indie (${indie_roi/1e6:.2f}M)"
        )

with col_insight2:
    # Insight 3: Qualité par tier
    aaa_meta = df_classified[(df_classified['Game Tier'] == 'AAA') & (df_classified['Metacritic score'] > 0)]['Metacritic score'].mean()
    indie_meta = df_classified[(df_classified['Game Tier'] == 'Indie') & (df_classified['Metacritic score'] > 0)]['Metacritic score'].mean()
    
    if not pd.isna(aaa_meta) and not pd.isna(indie_meta):
        if aaa_meta > indie_meta + 5:
            st.success(
                f"⭐ **Qualité AAA**: Metacritic moyen AAA ({aaa_meta:.1f}) surpasse "
                f"Indie ({indie_meta:.1f}) de {aaa_meta - indie_meta:.1f} points"
            )
        elif indie_meta > aaa_meta:
            st.info(
                f"🎨 **Surprise Qualité Indie**: Les Indie ({indie_meta:.1f}) dépassent "
                f"les AAA ({aaa_meta:.1f}) en score critique moyen"
            )
    
    # Insight 4: Top Developer
    top_dev_volume = df_classified['Developers'].value_counts().head(1)
    if len(top_dev_volume) > 0:
        top_dev_name = top_dev_volume.index[0]
        top_dev_count = top_dev_volume.iloc[0]
        
        st.info(
            f"🏆 **Studio le Plus Prolifique**: **{top_dev_name}** avec "
            f"**{top_dev_count:,} jeux** sur Steam"
        )

st.divider()

# ============================================================
# 6. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 AAA vs Indie",
    "🏢 Top Studios",
    "📊 Concentration Marché",
    "🎯 Spécialisation Genres"
])

# ============================================================
# TAB 1: AAA vs INDIE COMPARISON
# ============================================================
with tab1:
    st.markdown("### 🏆 Comparaison AAA vs Indie vs AA")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("#### Volume vs Revenus")
        
        # Données pour comparaison
        tier_data = df_classified.groupby('Game Tier').agg({
            'AppID': 'count',
            'Estimated revenue': 'sum'
        }).reset_index()
        tier_data.columns = ['Tier', 'Volume', 'Revenus']
        tier_data['Revenus (Milliards)'] = tier_data['Revenus'] / 1e9
        
        # Graphique double axe
        fig_tier = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_tier.add_trace(
            go.Bar(
                x=tier_data['Tier'],
                y=tier_data['Volume'],
                name='Volume de jeux',
                marker_color=COLORS['primary']
            ),
            secondary_y=False
        )
        
        fig_tier.add_trace(
            go.Bar(
                x=tier_data['Tier'],
                y=tier_data['Revenus (Milliards)'],
                name='Revenus (Milliards $)',
                marker_color=COLORS['secondary']
            ),
            secondary_y=True
        )
        
        fig_tier.update_layout(
            **PLOTLY_LAYOUT,
            height=400
        )
        
        fig_tier.update_yaxes(title_text="Volume de jeux", secondary_y=False)
        fig_tier.update_yaxes(title_text="Revenus (Milliards $)", secondary_y=True)
        
        st.plotly_chart(fig_tier, use_container_width=True)
    
    with col_comp2:
        st.markdown("#### Distribution Marché")
        
        # Pie chart répartition
        fig_pie = px.pie(
            tier_data,
            values='Volume',
            names='Tier',
            color_discrete_sequence=COLORS['chart'],
            hole=0.4
        )
        
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # Comparaison métriques détaillées
    st.markdown("#### 📊 Métriques Comparatives")
    
    tier_metrics = df_classified.groupby('Game Tier').agg({
        'AppID': 'count',
        'Estimated revenue': ['sum', 'mean'],
        'Price': 'mean',
        'Metacritic score': lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0,
        'Peak CCU': 'median',
        'Median playtime forever': 'median'
    }).reset_index()
    
    tier_metrics.columns = ['Tier', 'Nb Jeux', 'CA Total (USD)', 'CA Moyen/Jeu', 
                            'Prix Moyen', 'Metacritic Moyen', 'CCU Médian', 'Playtime Médian (min)']
    
    tier_metrics['CA Total (B$)'] = (tier_metrics['CA Total (USD)'] / 1e9).round(2)
    tier_metrics['CA Moyen/Jeu (M$)'] = (tier_metrics['CA Moyen/Jeu'] / 1e6).round(2)
    tier_metrics['Prix Moyen'] = tier_metrics['Prix Moyen'].round(2)
    tier_metrics['Metacritic Moyen'] = tier_metrics['Metacritic Moyen'].round(1)
    tier_metrics['Playtime (h)'] = (tier_metrics['Playtime Médian (min)'] / 60).round(1)
    
    display_metrics = tier_metrics[['Tier', 'Nb Jeux', 'CA Total (B$)', 'CA Moyen/Jeu (M$)', 
                                    'Prix Moyen', 'Metacritic Moyen', 'CCU Médian', 'Playtime (h)']]
    
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)

# ============================================================
# TAB 2: TOP STUDIOS
# ============================================================
with tab2:
    st.markdown("### 🏢 Top Studios par Performance")
    
    col_studio1, col_studio2 = st.columns(2)
    
    with col_studio1:
        st.markdown("#### Top 15 Developers (Volume)")
        
        top_devs = df_classified['Developers'].value_counts().head(15).reset_index()
        top_devs.columns = ['Developer', 'Nb Jeux']
        
        fig_devs = px.bar(
            top_devs,
            x='Nb Jeux',
            y='Developer',
            orientation='h',
            color='Nb Jeux',
            color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['chart'][5]]]
        )
        
        fig_devs.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        
        st.plotly_chart(fig_devs, use_container_width=True)
    
    with col_studio2:
        st.markdown("#### Top 15 Publishers (Volume)")
        
        top_pubs = df_classified['Publishers'].value_counts().head(15).reset_index()
        top_pubs.columns = ['Publisher', 'Nb Jeux']
        
        fig_pubs = px.bar(
            top_pubs,
            x='Nb Jeux',
            y='Publisher',
            orientation='h',
            color='Nb Jeux',
            color_continuous_scale=[[0, COLORS['chart'][1]], [1, COLORS['chart'][4]]]
        )
        
        fig_pubs.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        
        st.plotly_chart(fig_pubs, use_container_width=True)
    
    st.divider()
    
    # Top par revenus
    st.markdown("#### 💰 Top 10 Developers par Revenus Cumulés")
    
    dev_revenue = df_classified.groupby('Developers')['Estimated revenue'].sum().nlargest(10).reset_index()
    dev_revenue.columns = ['Developer', 'Revenus']
    dev_revenue['Revenus (Milliards $)'] = (dev_revenue['Revenus'] / 1e9).round(2)
    
    fig_dev_rev = px.bar(
        dev_revenue,
        x='Revenus (Milliards $)',
        y='Developer',
        orientation='h',
        color='Revenus (Milliards $)',
        color_continuous_scale=[[0, COLORS['warning']], [1, COLORS['danger']]]
    )
    
    fig_dev_rev.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending")
    )
    
    st.plotly_chart(fig_dev_rev, use_container_width=True)

# ============================================================
# TAB 3: CONCENTRATION MARCHÉ
# ============================================================
with tab3:
    st.markdown("### 📊 Analyse de Concentration du Marché")
    
    # Top X% developers représentent Y% revenus
    dev_rev_sorted = df_classified.groupby('Developers')['Estimated revenue'].sum().sort_values(ascending=False)
    cumsum_pct = (dev_rev_sorted.cumsum() / dev_rev_sorted.sum() * 100)
    
    # Trouver combien de devs pour 80% revenus
    devs_for_80pct = (cumsum_pct <= 80).sum()
    pct_devs_for_80 = (devs_for_80pct / len(dev_rev_sorted)) * 100
    
    col_conc1, col_conc2 = st.columns([2, 1])
    
    with col_conc1:
        st.markdown("#### Courbe de Concentration (Pareto)")
        
        # Limite à top 100 pour lisibilité
        pareto_data = pd.DataFrame({
            'Rang': range(1, min(101, len(cumsum_pct) + 1)),
            '% Revenus Cumulés': cumsum_pct.head(100).values
        })
        
        fig_pareto = px.area(
            pareto_data,
            x='Rang',
            y='% Revenus Cumulés',
            color_discrete_sequence=[COLORS['primary']]
        )
        
        fig_pareto.add_hline(y=80, line_dash="dash", line_color=COLORS['danger'],
                            annotation_text="80% des revenus")
        
        fig_pareto.update_layout(
            **PLOTLY_LAYOUT,
            height=400
        )
        
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    with col_conc2:
        st.markdown("#### 📈 Métriques Concentration")
        
        st.metric(
            "Top Devs pour 80% CA",
            f"{devs_for_80pct:,}",
            delta=f"{pct_devs_for_80:.1f}% des studios",
            delta_color="off"
        )
        
        top10_revenue_pct = (dev_rev_sorted.head(10).sum() / dev_rev_sorted.sum() * 100)
        st.metric(
            "Part Top 10 Devs",
            f"{top10_revenue_pct:.1f}%",
            help="% des revenus générés par top 10"
        )
        
        top1_revenue_pct = (dev_rev_sorted.iloc[0] / dev_rev_sorted.sum() * 100)
        st.metric(
            "Part #1 Developer",
            f"{top1_revenue_pct:.1f}%",
            help="% des revenus du leader"
        )

# ============================================================
# TAB 4: SPÉCIALISATION GENRES
# ============================================================
with tab4:
    st.markdown("### 🎯 Spécialisation par Genre des Top Studios")
    
    # Top 10 developers + leurs genres principaux
    top10_devs = df_classified['Developers'].value_counts().head(10).index.tolist()
    
    df_top_devs = df_classified[df_classified['Developers'].isin(top10_devs)].copy()
    
    # Explosion genres
    df_genres_dev = df_top_devs.explode('Genres').dropna(subset=['Genres'])
    
    # Heatmap Developer x Genre
    genre_dev_matrix = pd.crosstab(df_genres_dev['Developers'], df_genres_dev['Genres'])
    
    # Garder top 15 genres
    top_genres = genre_dev_matrix.sum(axis=0).nlargest(15).index
    genre_dev_matrix = genre_dev_matrix[top_genres]
    
    fig_heatmap = px.imshow(
        genre_dev_matrix,
        labels=dict(x="Genre", y="Developer", color="Nb Jeux"),
        x=genre_dev_matrix.columns,
        y=genre_dev_matrix.index,
        color_continuous_scale=[[0, COLORS['background']], [1, COLORS['primary']]],
        text_auto=True
    )
    
    fig_heatmap.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🏢 **GameData360 — Developers & Publishers** | "
    f"Analyse de {unique_devs:,} studios sur {len(df_classified):,} jeux"
)
