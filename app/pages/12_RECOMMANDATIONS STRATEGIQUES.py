# -*- coding: utf-8 -*-
"""
GameData360 - Page Recommandations Stratégiques
===============================================
Synthèse des analyses pour orienter actionnaires, producteurs et joueurs.
Insights actionnables basés sur l'ensemble des données du marché.

Auteur: GameData360 Team
Version: 1.0 (Strategic Recommendations Edition)
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
    get_unique_values,
    apply_list_filter,
    explode_genres,
    format_number
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — Recommandations Stratégiques",
    page_icon="🎯",
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
    
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Info boxes stylisés */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(0,255,136,0.05), rgba(0,255,255,0.05));
        border-left: 4px solid #00ff88;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .stakeholder-section {
        background: rgba(0,255,136,0.03);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
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
st.markdown("# 🎯 RECOMMANDATIONS STRATÉGIQUES")
st.markdown("##### Synthèse des analyses pour actionnaires, producteurs et joueurs")

# ============================================================
# 2. CHARGEMENT DES DONNÉES (CACHE OPTIMISÉ)
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
# 3. CALCULS GLOBAUX ET PRÉPARATIONS
# ============================================================
@st.cache_data(show_spinner=False)
def calculate_global_insights(_df):
    """Calcule les insights globaux pour toutes les sections."""
    
    insights = {}
    
    # Métriques globales
    insights['total_games'] = len(_df)
    insights['total_revenue'] = _df['Estimated revenue'].sum()
    insights['avg_price'] = _df['Price'].mean()
    insights['median_metacritic'] = _df[_df['Metacritic score'] > 0]['Metacritic score'].median()
    insights['f2p_ratio'] = (_df['Price'] == 0).sum() / len(_df) * 100
    
    # Top genres par revenus
    df_genres = explode_genres(_df)
    genre_revenue = df_genres.groupby('Genres')['Estimated revenue'].sum().sort_values(ascending=False)
    insights['top_genres_revenue'] = genre_revenue.head(10)
    
    # Analyse Pareto
    df_pareto = _df[_df['Estimated revenue'] > 0].copy()
    df_pareto = df_pareto.sort_values('Estimated revenue', ascending=False).reset_index(drop=True)
    df_pareto['cumul_pct'] = (df_pareto['Estimated revenue'].cumsum() / df_pareto['Estimated revenue'].sum()) * 100
    df_pareto['game_pct'] = ((df_pareto.index + 1) / len(df_pareto)) * 100
    
    # Trouver le point 80%
    idx_80 = (df_pareto['cumul_pct'] >= 80).idxmax()
    insights['pareto_80_pct'] = df_pareto.loc[idx_80, 'game_pct']
    
    # ROI par segment de prix
    price_segments = {
        'Free-to-Play': (0, 0),
        'Budget (< $10)': (0.01, 9.99),
        'Mid-range ($10-$30)': (10, 29.99),
        'Premium ($30-$60)': (30, 59.99),
        'AAA ($60+)': (60, 1000)
    }
    
    roi_by_segment = {}
    for segment, (min_p, max_p) in price_segments.items():
        if min_p == 0 and max_p == 0:
            mask = _df['Price'] == 0
        else:
            mask = (_df['Price'] >= min_p) & (_df['Price'] <= max_p)
        
        segment_df = _df[mask]
        if len(segment_df) > 0:
            avg_revenue = segment_df['Estimated revenue'].mean()
            avg_price = segment_df['Price'].mean()
            avg_ccu = segment_df['Peak CCU'].mean()
            avg_score = segment_df[segment_df['Positive'] > 0]['Positive'].sum() / (segment_df['Positive'].sum() + segment_df['Negative'].sum()+ 1) * 100 if len(segment_df[segment_df['Positive'] > 0]) > 0 else 0
            
            roi_by_segment[segment] = {
                'count': len(segment_df),
                'avg_revenue': avg_revenue,
                'avg_price': avg_price,
                'avg_ccu': avg_ccu,
                'avg_score': avg_score if not pd.isna(avg_score) else 0
            }
    
    insights['roi_by_segment'] = roi_by_segment
    
    # Tendances temporelles (2020-2024)
    recent_df = _df[_df['Release Year'] >= 2020]
    yearly_growth = recent_df.groupby('Release Year').size()
    if len(yearly_growth) >= 2:
        growth_rate = ((yearly_growth.iloc[-1] - yearly_growth.iloc[0]) / yearly_growth.iloc[0]) * 100
        insights['recent_growth_rate'] = growth_rate
    else:
        insights['recent_growth_rate'] = 0
    
    # Sweet spots (prix vs qualité)
    df_quality = _df[(_df['Metacritic score'] > 0) & (_df['Price'] > 0)]
    insights['sweet_spots'] = df_quality[['Name', 'Price', 'Metacritic score', 'Peak CCU', 'Genres']]
    
    return insights

global_insights = calculate_global_insights(df_analyse)

# ============================================================
# 4. TOP-LEVEL KPIs (SYNTHÈSE GLOBALE)
# ============================================================
st.markdown("### 📊 Vue d'Ensemble du Marché")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "🎮 Total Jeux",
        f"{global_insights['total_games']:,}",
        help="Nombre total de jeux analysés"
    )

with kpi2:
    total_rev_b = global_insights['total_revenue'] / 1e9
    st.metric(
        "💰 Revenus Estimés",
        f"${total_rev_b:.1f}B",
        help="Revenus cumulés estimés (milliards USD)"
    )

with kpi3:
    st.metric(
        "⭐ Qualité Médiane",
        f"{global_insights['median_metacritic']:.0f}",
        help="Score Metacritic médian"
    )

with kpi4:
    st.metric(
        "🆓 Part F2P",
        f"{global_insights['f2p_ratio']:.0f}%",
        help="Pourcentage de jeux gratuits"
    )

st.divider()

# ============================================================
# 5. ONGLETS PAR STAKEHOLDER
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "💼 Actionnaires / Investisseurs",
    "🏭 Producteurs / Développeurs",
    "🎮 Joueurs"
])

# ============================================================
# TAB 1: ACTIONNAIRES / INVESTISSEURS
# ============================================================
with tab1:
    st.markdown("## 💼 Recommandations pour Actionnaires et Investisseurs")
    st.markdown("Analyse des opportunités d'investissement et tendances du marché")
    
    # Section 1: Principe de Pareto
    st.markdown("### 📊 Concentration du Marché (Principe de Pareto)")
    
    col_pareto1, col_pareto2 = st.columns([2, 1])
    
    with col_pareto1:
        # Courbe de Lorenz
        df_pareto = df_analyse[df_analyse['Estimated revenue'] > 0].copy()
        df_pareto = df_pareto.sort_values('Estimated revenue', ascending=False).reset_index(drop=True)
        df_pareto['cumul_pct'] = (df_pareto['Estimated revenue'].cumsum() / df_pareto['Estimated revenue'].sum()) * 100
        df_pareto['game_pct'] = ((df_pareto.index + 1) / len(df_pareto)) * 100
        
        fig_lorenz = go.Figure()
        
        # Courbe de Lorenz
        fig_lorenz.add_trace(go.Scatter(
            x=df_pareto['game_pct'],
            y=df_pareto['cumul_pct'],
            mode='lines',
            name='Courbe de Lorenz',
            line=dict(color=COLORS['primary'], width=3),
            fill='tonexty',
            fillcolor='rgba(0,255,136,0.2)'
        ))
        
        # Ligne d'égalité parfaite
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Égalité parfaite',
            line=dict(color=COLORS['danger'], width=2, dash='dash')
        ))
        
        fig_lorenz.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="% des jeux",
            yaxis_title="% des revenus cumulés",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_lorenz, use_container_width=True)
    
    with col_pareto2:
        st.markdown("#### 💡 Insight Clé")
        
        st.markdown(f"""
        <div class="recommendation-box">
        <h4>🎯 Concentration Élevée</h4>
        <p><strong>{global_insights['pareto_80_pct']:.1f}%</strong> des jeux génèrent <strong>80%</strong> des revenus.</p>
        <p><strong>Recommandation:</strong> Investir dans les franchises établies et les studios AAA pour sécuriser le ROI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top performers
        top_5_revenue = df_analyse.nlargest(5, 'Estimated revenue')[['Name', 'Estimated revenue']]
        top_5_revenue['Revenus (M$)'] = (top_5_revenue['Estimated revenue'] / 1e6).round(1)
        
        st.markdown("##### 🏆 Top 5 Performers")
        st.dataframe(
            top_5_revenue[['Name', 'Revenus (M$)']],
            hide_index=True,
            use_container_width=True,
            height=220
        )
    
    st.divider()
    
    # Section 2: ROI par Segment de Prix
    st.markdown("### 💵 ROI par Segment de Prix")
    
    roi_data = []
    for segment, metrics in global_insights['roi_by_segment'].items():
        roi_data.append({
            'Segment': segment,
            'Nombre': metrics['count'],
            'Rev. Moyen (M$)': metrics['avg_revenue'] / 1e6,
            'Prix Moyen': metrics['avg_price'],
            'CCU Moyen': metrics['avg_ccu'],
            'Score Moyen': metrics['avg_score']
        })
    
    df_roi = pd.DataFrame(roi_data)
    
    col_roi1, col_roi2 = st.columns(2)
    
    with col_roi1:
        # Graphique revenus moyens par segment
        fig_roi = px.bar(
            df_roi,
            x='Segment',
            y='Rev. Moyen (M$)',
            color='Rev. Moyen (M$)',
            color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['chart'][4]]]
        )
        
        fig_roi.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Segment de Prix",
            yaxis_title="Revenus Moyens (M$)",
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col_roi2:
        # Table détaillée
        st.markdown("#### 📋 Métriques Détaillées")
        st.dataframe(
            df_roi.style.format({
                'Nombre': '{:,}',
                'Rev. Moyen (M$)': '{:.2f}',
                'Prix Moyen': '${:.2f}',
                'CCU Moyen': '{:,.0f}',
                'Score Moyen': '{:.1f}'
            }),
            hide_index=True,
            use_container_width=True,
            height=350
        )
    
    # Recommandations investisseurs
    st.markdown("### 🎯 Recommandations d'Investissement")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🏆 AAA Premium</h4>
        <p><strong>ROI:</strong> Très élevé mais risqué</p>
        <p><strong>Profil:</strong> Investisseurs avec fort capital</p>
        <p><strong>Action:</strong> Investir dans les franchises établies (séquelles, licences)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_rec2:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🎨 Indie Mid-Range</h4>
        <p><strong>ROI:</strong> Modéré avec faible risque</p>
        <p><strong>Profil:</strong> Portfolio diversifié</p>
        <p><strong>Action:</strong> Soutenir les studios indés prometteurs avec bon track record</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_rec3:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🆓 Free-to-Play</h4>
        <p><strong>ROI:</strong> Très variable (hit or miss)</p>
        <p><strong>Profil:</strong> Investisseurs aguerris</p>
        <p><strong>Action:</strong> Focus sur les jeux avec forte rétention et monétisation éprouvée</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section 3: Tendances de Croissance
    st.markdown("### 📈 Genres en Croissance (Opportunités)")
    
    # Top 10 genres par revenus
    top_genres_df = global_insights['top_genres_revenue'].reset_index()
    top_genres_df.columns = ['Genre', 'Revenus']
    top_genres_df['Revenus (M$)'] = (top_genres_df['Revenus'] / 1e6).round(1)
    
    fig_genres_inv = px.treemap(
        top_genres_df.head(10),
        path=['Genre'],
        values='Revenus',
        color='Revenus',
        color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['primary']]],
        hover_data={'Revenus (M$)': True}
    )
    
    fig_genres_inv.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_genres_inv, use_container_width=True)
    
    # Insight de croissance récente
    if global_insights['recent_growth_rate'] > 0:
        st.success(
            f"📈 **Croissance Récente Positive**: Le marché a crû de **{global_insights['recent_growth_rate']:.1f}%** "
            f"sur la période 2020-2024. Opportunité d'investissement favorable."
        )
    else:
        st.warning(
            f"⚠️ **Ralentissement Récent**: Le marché a ralenti de **{abs(global_insights['recent_growth_rate']):.1f}%** "
            f"sur la période 2020-2024. Prudence recommandée."
        )

# ============================================================
# TAB 2: PRODUCTEURS / DÉVELOPPEURS
# ============================================================
with tab2:
    st.markdown("## 🏭 Recommandations pour Producteurs et Développeurs")
    st.markdown("Gaps de marché, stratégies de pricing et opportunités de production")
    
    # Section 1: Sweet Spots Prix/Qualité
    st.markdown("### 💎 Sweet Spots Prix vs Qualité")
    
    df_sweet = global_insights['sweet_spots'].copy()
    df_sweet = df_sweet[(df_sweet['Metacritic score'] >= 70) & (df_sweet['Price'] <= 60)]
    
    # Scatter plot
    fig_sweet = px.scatter(
        df_sweet.head(200),  # Limiter pour la lisibilité
        x='Price',
        y='Metacritic score',
        size='Peak CCU',
        color='Metacritic score',
        hover_data=['Name'],
        color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]],
        size_max=30
    )
    
    # Zones optimales
    fig_sweet.add_shape(
        type="rect",
        x0=10, y0=75, x1=30, y1=90,
        line=dict(color=COLORS['primary'], width=2, dash='dash'),
        fillcolor='rgba(0,255,136,0.1)'
    )
    
    fig_sweet.add_annotation(
        x=20, y=87,
        text="Zone Optimale",
        showarrow=False,
        font=dict(size=14, color=COLORS['primary'])
    )
    
    fig_sweet.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Prix ($)",
        yaxis_title="Score Metacritic",
        height=450,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_sweet, use_container_width=True)
    
    col_sweet1, col_sweet2 = st.columns(2)
    
    with col_sweet1:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🎯 Zone Optimale Identifiée</h4>
        <p><strong>Prix:</strong> $10 - $30</p>
        <p><strong>Qualité:</strong> Metacritic 75-90</p>
        <p><strong>Stratégie:</strong> Viser ce corridor pour maximiser l'acceptation marché et les ventes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sweet2:
        st.markdown("""
        <div class="recommendation-box">
        <h4>⚠️ Zones à Éviter</h4>
        <p><strong>Low Quality/High Price:</strong> Rejet du marché</p>
        <p><strong>High Quality/Low Price:</strong> Sous-valorisation, revenus perdus</p>
        <p><strong>Action:</strong> Calibrer le pricing en fonction de la qualité objective</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section 2: Analyse des Gaps de Marché
    st.markdown("### 🔍 Gaps de Marché (Opportunités)")
    
    # Analyse des combinaisons genre/catégorie sous-exploitées
    df_exploded = df_analyse.explode('Genres').explode('Categories')
    df_exploded = df_exploded[df_exploded['Genres'].notna() & df_exploded['Categories'].notna()]
    
    # Top combinaisons actuelles
    combo_counts = df_exploded.groupby(['Genres', 'Categories']).size().reset_index(name='Count')
    combo_counts = combo_counts.sort_values('Count', ascending=False)
    
    # Genres populaires
    top_genres = df_exploded['Genres'].value_counts().head(8).index.tolist()
    top_cats = df_exploded['Categories'].value_counts().head(8).index.tolist()
    
    # Heatmap des combinaisons
    heatmap_data = df_exploded[
        df_exploded['Genres'].isin(top_genres) & 
        df_exploded['Categories'].isin(top_cats)
    ].groupby(['Genres', 'Categories']).size().reset_index(name='Count')
    
    heatmap_pivot = heatmap_data.pivot(index='Genres', columns='Categories', values='Count').fillna(0)
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        color_continuous_scale=[[0, '#1a1a1a'], [0.5, COLORS['warning']], [1, COLORS['primary']]],
        aspect='auto',
        labels=dict(color="Nombre de jeux")
    )
    
    fig_heatmap.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Catégories",
        yaxis_title="Genres",
        height=500
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info(
        "💡 **Interprétation**: Les cases sombres représentent des combinaisons sous-exploitées = opportunités de niche. "
        "Les cases claires indiquent une forte saturation = forte concurrence."
    )
    
    st.divider()
    
    # Section 3: Stratégies de Pricing
    st.markdown("### 💰 Stratégies de Pricing Recommandées")
    
    col_pricing1, col_pricing2, col_pricing3 = st.columns(3)
    
    with col_pricing1:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🎮 Jeux Solo Narratifs</h4>
        <p><strong>Prix Optimal:</strong> $15 - $25</p>
        <p><strong>Durée Cible:</strong> 8-15h</p>
        <p><strong>DLC/Extensions:</strong> Oui, très efficaces</p>
        <p><strong>Exemple:</strong> Adventure, RPG Story-Driven</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_pricing2:
        st.markdown("""
        <div class="recommendation-box">
        <h4>👥 Jeux Multijoueur</h4>
        <p><strong>Modèle:</strong> F2P avec cosmetics</p>
        <p><strong>Alt.:</strong> $20-40 avec Season Pass</p>
        <p><strong>Monétisation:</strong> Battle Pass, skins</p>
        <p><strong>Exemple:</strong> FPS, MOBA, Battle Royale</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_pricing3:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🎨 Jeux Indie Innovants</h4>
        <p><strong>Prix Optimal:</strong> $10 - $20</p>
        <p><strong>Stratégie:</strong> Launch discount 15-20%</p>
        <p><strong>Bundles:</strong> Très recommandés</p>
        <p><strong>Exemple:</strong> Puzzle, Platformer, Roguelite</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section 4: Benchmarks de Succès
    st.markdown("### 📊 Benchmarks de Succès par Genre")
    
    # Calculer les benchmarks par genre
    df_genres_bench = explode_genres(df_analyse)
    top_10_genres = df_genres_bench['Genres'].value_counts().head(10).index
    df_genres_bench = df_genres_bench[df_genres_bench['Genres'].isin(top_10_genres)]
    
    benchmark_stats = df_genres_bench.groupby('Genres').agg({
        'Peak CCU': 'median',
        'Positive': 'sum',
        'Negative': 'sum',
        'Price': 'median',
        'Estimated revenue': 'median'
    }).reset_index()
    
    benchmark_stats.columns = ['Genre', 'CCU Médian', 'Positifs', 'Négatifs', 'Prix Médian', 'Revenus Médians']
    benchmark_stats['Score (%)'] = (benchmark_stats['Positifs'] / (benchmark_stats['Positifs'] + benchmark_stats['Négatifs'] + 1) * 100).round(1)
    benchmark_stats['Revenus (K$)'] = (benchmark_stats['Revenus Médians'] / 1000).round(0)
    
    st.dataframe(
        benchmark_stats[['Genre', 'CCU Médian', 'Score (%)', 'Prix Médian', 'Revenus (K$)']].style.format({
            'CCU Médian': '{:,.0f}',
            'Score (%)': '{:.1f}%',
            'Prix Médian': '${:.2f}',
            'Revenus (K$)': '{:,.0f}K$'
        }),
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    st.success(
        "🎯 **Utilisation**: Ces benchmarks constituent des objectifs réalistes à viser pour votre genre. "
        "Un jeu performant devrait dépasser ces médianes."
    )

# ============================================================
# TAB 3: JOUEURS
# ============================================================
with tab3:
    st.markdown("## 🎮 Recommandations pour Joueurs")
    st.markdown("Trouvez les jeux qui correspondent à votre profil et vos préférences")
    
    # Sélecteur de profil joueur
    st.markdown("### 🎯 Définissez Votre Profil")
    
    col_profile1, col_profile2, col_profile3 = st.columns(3)
    
    with col_profile1:
        play_style = st.selectbox(
            "🎮 Style de jeu",
            ["Casual", "Hardcore", "Équilibré"],
            help="Casual: jeux courts et accessibles | Hardcore: jeux longs et exigeants"
        )
    
    with col_profile2:
        budget = st.selectbox(
            "💰 Budget",
            ["Free-to-Play uniquement", "Budget (< $10)", "Moyen ($10-$30)", "Premium ($30+)", "Illimité"],
            help="Votre budget pour acheter des jeux"
        )
    
    with col_profile3:
        social_pref = st.selectbox(
            "👥 Préférence sociale",
            ["Solo uniquement", "Principalement Solo", "Mixte", "Principalement Multi", "Multi uniquement"],
            help="Solo: aventure narrative | Multi: jeu en ligne avec d'autres"
        )
    
    # Genres préférés - EXTRACTION DIRECTE (comme page 9) pour éviter problème de casse
    all_genres = sorted(set([
        g for genres in df_analyse["Genres"] 
        if isinstance(genres, list) 
        for g in genres
    ]))
    
    selected_genres_player = st.multiselect(
        "🎨 Genres Préférés (optionnel)",
        all_genres,
        default=[],
        help="Sélectionnez vos genres favoris pour des recommandations ciblées"
    )
    
    # Bouton de recommandation
    if st.button("🔍 Trouver mes jeux", type="primary", use_container_width=True):
        
        # Filtrage basé sur le profil - UTILISER df_analyse pour garantir toutes les colonnes
        df_filtered = df_analyse.copy()
        
        # Filtre budget
        if budget == "Free-to-Play uniquement":
            df_filtered = df_filtered[df_filtered['Price'] == 0]
        elif budget == "Budget (< $10)":
            df_filtered = df_filtered[df_filtered['Price'] < 10]
        elif budget == "Moyen ($10-$30)":
            df_filtered = df_filtered[(df_filtered['Price'] >= 10) & (df_filtered['Price'] <= 30)]
        elif budget == "Premium ($30+)":
            df_filtered = df_filtered[df_filtered['Price'] >= 30]
        
        # Filtre style de jeu (basé sur playtime médian)
        if play_style == "Casual":
            # Jeux avec moins de 300 min (5h) de playtime médian
            df_filtered = df_filtered[df_filtered['Median playtime forever'] < 300]
        elif play_style == "Hardcore":
            # Jeux avec plus de 600 min (10h) de playtime médian
            df_filtered = df_filtered[df_filtered['Median playtime forever'] > 600]
        
        # Filtre social (Solo/Multi)
        if social_pref in ["Solo uniquement", "Principalement Solo"]:
            df_filtered = df_filtered[
                df_filtered['Categories'].apply(
                    lambda x: isinstance(x, list) and 'Single-player' in x
                )
            ]
        elif social_pref in ["Multi uniquement", "Principalement Multi"]:
            df_filtered = df_filtered[
                df_filtered['Categories'].apply(
                    lambda x: isinstance(x, list) and any(
                        cat in x for cat in ['Multi-player', 'Online PvP', 'Co-op', 'MMO']
                    )
                )
            ]
        
        # Filtre genres
        if selected_genres_player:
            df_filtered = df_filtered[
                df_filtered['Genres'].apply(
                    lambda x: isinstance(x, list) and any(g in x for g in selected_genres_player)
                )
            ]
        
        # Tri par qualité et popularité avec colonnes garanties
        # Construction robuste du score composite
        score_components = []
        weights = []
        
        # Peak CCU (priorité 1)
        if 'Peak CCU' in df_filtered.columns:
            score_components.append(np.log10(df_filtered['Peak CCU'].fillna(1) + 1) * 10)
            weights.append(0.4)
        
        # Recommendations (priorité 2)
        if 'Recommendations' in df_filtered.columns:
            score_components.append(np.log10(df_filtered['Recommendations'].fillna(1) + 1) * 5)
            weights.append(0.3)
        
        # Metacritic score (priorité 3)
        if 'Metacritic score' in df_filtered.columns:
            score_components.append(df_filtered['Metacritic score'].fillna(50))
            weights.append(0.3)
        
        # Calculer le score composite avec normalisation des poids
        if score_components:
            total_weight = sum(weights)
            df_filtered['score_composite'] = sum(
                comp * (w / total_weight) for comp, w in zip(score_components, weights)
            )
        else:
            # Fallback: score aléatoire
            df_filtered['score_composite'] = np.random.rand(len(df_filtered)) * 100


        
        df_recommendations = df_filtered.nlargest(20, 'score_composite')
        
        if len(df_recommendations) > 0:
            st.success(f"✅ {len(df_recommendations)} jeux trouvés correspondant à votre profil !")
            
            # Affichage des recommandations
            st.markdown("### 🎮 Vos Recommandations Personnalisées")
            
            # Préparer l'affichage avec les colonnes disponibles
            base_columns = ['Name', 'Price', 'Genres', 'Peak CCU', 'Median playtime forever', 'Recommendations']
            optional_columns = []
            
            # Ajouter Metacritic si disponible
            if 'Metacritic score' in df_recommendations.columns:
                optional_columns.append('Metacritic score')
            
            display_df = df_recommendations[base_columns + optional_columns].copy()
            
            display_df['Prix'] = display_df['Price'].apply(lambda x: "Gratuit" if x == 0 else f"${x:.2f}")
            display_df['Genres'] = display_df['Genres'].apply(
                lambda x: ", ".join(x[:3]) if isinstance(x, list) else "N/A"
            )
            
            if 'Metacritic score' in display_df.columns:
                display_df['Metacritic'] = display_df['Metacritic score'].apply(
                    lambda x: f"{x:.0f}" if x > 0 else "N/A"
                )
            
            display_df['Peak CCU'] = display_df['Peak CCU'].apply(lambda x: f"{x:,}")
            display_df['Playtime (h)'] = (display_df['Median playtime forever'] / 60).round(1)
            display_df['Reco'] = display_df['Recommendations'].apply(lambda x: f"{x:,}")

            
            # Construire les colonnes finales dynamiquement
            final_cols = ['Name', 'Prix', 'Genres']
            final_names = ['Jeu', 'Prix', 'Genres']
            
            if 'Metacritic' in display_df.columns:
                final_cols.append('Metacritic')
                final_names.append('Metacritic')
            
            final_cols.extend(['Peak CCU', 'Reco', 'Playtime (h)'])
            final_names.extend(['Peak CCU', 'Recommandations', 'Durée (h)'])
            
            final_display = display_df[final_cols]
            final_display.columns = final_names
            
            st.dataframe(
                final_display,
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # Stats du profil
            st.markdown("### 📊 Statistiques de Votre Profil")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                avg_price = df_recommendations['Price'].mean()
                st.metric("💵 Prix Moyen", f"${avg_price:.2f}")
            
            with col_stat2:
                if 'Metacritic score' in df_recommendations.columns:
                    avg_meta = df_recommendations[df_recommendations['Metacritic score'] > 0]['Metacritic score'].mean()
                    st.metric("⭐ Qualité Moyenne", f"{avg_meta:.0f}" if not pd.isna(avg_meta) else "N/A")
                else:
                    avg_reco = df_recommendations['Recommendations'].mean()
                    st.metric("📊 Reco Moyenne", f"{avg_reco:,.0f}")
            
            with col_stat3:
                avg_playtime = df_recommendations['Median playtime forever'].mean() / 60
                st.metric("⏱️ Durée Moy.", f"{avg_playtime:.1f}h")
            
            with col_stat4:
                genres_list = [g for genres in df_recommendations['Genres'] if isinstance(genres, list) for g in genres]
                top_genre = pd.Series(genres_list).value_counts().index[0] if genres_list else "N/A"
                st.metric("🎨 Genre Principal", top_genre)
            
        else:
            st.warning(
                "⚠️ Aucun jeu trouvé correspondant exactement à vos critères. "
                "Essayez d'élargir vos filtres (budget, genres)."
            )
    
    else:
        st.info("👆 Définissez votre profil ci-dessus et cliquez sur 'Trouver mes jeux' pour obtenir des recommandations personnalisées !")
    
    st.divider()
    
    # Section: Jeux Populaires par Catégorie
    st.markdown("### 🏆 Jeux les Plus Populaires par Catégorie")
    
    category_choice = st.selectbox(
        "Choisir une catégorie",
        ["Top Général", "Free-to-Play", "Solo", "Multijoueur", "Nouveautés 2023-2024"]
    )
    
    if category_choice == "Top Général":
        top_games = df_analyse.nlargest(10, 'Peak CCU')
    elif category_choice == "Free-to-Play":
        top_games = df_analyse[df_analyse['Price'] == 0].nlargest(10, 'Peak CCU')
    elif category_choice == "Solo":
        df_solo = df_analyse[
            df_analyse['Categories'].apply(
                lambda x: isinstance(x, list) and 'Single-player' in x
            )
        ]
        top_games = df_solo.nlargest(10, 'Peak CCU')
    elif category_choice == "Multijoueur":
        df_multi = df_analyse[
            df_analyse['Categories'].apply(
                lambda x: isinstance(x, list) and any(
                    cat in x for cat in ['Multi-player', 'Online PvP', 'MMO']
                )
            )
        ]
        top_games = df_multi.nlargest(10, 'Peak CCU')
    else:  # Nouveautés
        df_recent = df_analyse[df_analyse['Release Year'] >= 2023]
        top_games = df_recent.nlargest(10, 'Peak CCU')
    
    # Affichage avec colonnes disponibles
    base_cols = ['Name', 'Price', 'Genres', 'Peak CCU']
    has_metacritic = 'Metacritic score' in top_games.columns
    
    if has_metacritic:
        base_cols.append('Metacritic score')
    
    display_top = top_games[base_cols].copy()
    display_top['Prix'] = display_top['Price'].apply(lambda x: "Gratuit" if x == 0 else f"${x:.2f}")
    display_top['Genres'] = display_top['Genres'].apply(
        lambda x: ", ".join(x[:2]) if isinstance(x, list) else "N/A"
    )
    display_top['Peak CCU'] = display_top['Peak CCU'].apply(lambda x: f"{x:,}")
    
    if has_metacritic:
        display_top['Score'] = display_top['Metacritic score'].apply(
            lambda x: f"{x:.0f}" if x > 0 else "N/A"
        )
        final_top = display_top[['Name', 'Prix', 'Genres', 'Peak CCU', 'Score']]
        final_top.columns = ['Jeu', 'Prix', 'Genres', 'Peak CCU', 'Score']
    else:
        final_top = display_top[['Name', 'Prix', 'Genres', 'Peak CCU']]
        final_top.columns = ['Jeu', 'Prix', 'Genres', 'Peak CCU']
    
    st.dataframe(
        final_top,
        hide_index=True,
        use_container_width=True,
        height=400
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🎯 **GameData360 — Recommandations Stratégiques** | "
    f"Synthèse sur {len(df_analyse):,} jeux analysés"
)
