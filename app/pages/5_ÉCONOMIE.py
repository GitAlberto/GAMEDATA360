# -*- coding: utf-8 -*-
"""
GameData360 - Page Économie
============================
Analyse économique professionnelle du marché du jeu vidéo.
Insights: Pareto, Value for Money, Pricing Power, Indie vs AAA, Market Share.

Auteur: GameData360 Team
Version: 3.0 (Chief Gaming Economist Edition)
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
    PLOTLY_AXIS,
    FILE_PATH,
    NON_GAME_GENRES
)
from utils.data_helpers import (
    load_game_data,
    get_unique_values,
    apply_all_filters,
    explode_genres,
    format_number
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — Économie",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le thème gaming (cohérent avec page 2)
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
st.markdown("# 💰 ÉCONOMIE DU GAMING")
st.markdown("##### Analyse de marché professionnelle : Pricing, Monétisation & Market Share")

# ============================================================
# 2. CHARGEMENT DES DONNÉES (CACHE OPTIMISÉ)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    """Charge et prépare les données avec cache."""
    return load_game_data(str(FILE_PATH))

# Chargement avec indicateur
try:
    with st.spinner('⚡ Chargement des données économiques...'):
        df_analyse = load_data()
        
        # Filtrage des genres non-jeux (logiciels)
        def is_game(genres_list):
            """Retourne False si le jeu contient un genre de logiciel."""
            if not isinstance(genres_list, list):
                return True
            # Comparaison case-insensitive (NON_GAME_GENRES est en lowercase)
            genres_lower = [g.lower() for g in genres_list]
            return not any(genre in genres_lower for genre in NON_GAME_GENRES)
        
        initial_count = len(df_analyse)
        df_analyse = df_analyse[df_analyse["Genres"].apply(is_game)].copy()
        excluded_count = initial_count - len(df_analyse)
        
        if excluded_count > 0:
            st.sidebar.success(f"🎮 {excluded_count:,} logiciels exclus (seuls les jeux sont analysés)")
        
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable : {FILE_PATH}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. SIDEBAR - FILTRES GLOBAUX
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Filtres Économiques")
    
    # Extraction des valeurs uniques
    unique_genres = get_unique_values(df_analyse, "Genres")
    unique_cats = get_unique_values(df_analyse, "Categories")
    
    # Filtre Année de sortie
    st.markdown("### 📅 Période")
    if "Release Year" in df_analyse.columns:
        min_year = int(df_analyse["Release Year"].min())
        max_year = int(df_analyse["Release Year"].max())
        year_range = st.slider(
            "Années de sortie",
            min_year, max_year, (min_year, max_year),
            help="Filtrer par période de sortie"
        )
    else:
        year_range = None
    
    # Filtre Prix
    st.markdown("### 💵 Fourchette de Prix")
    min_price = float(df_analyse["Price"].min())
    max_price = float(df_analyse["Price"].max())
    price_range = st.slider(
        "Prix ($)",
        min_price, max_price, (min_price, max_price),
        help="Filtrer par gamme de prix"
    )
    
    # Filtres standards
    st.markdown("### 🎮 Contenu")
    selected_genres = st.multiselect(
        "Genres",
        unique_genres,
        help="Sélectionnez un ou plusieurs genres"
    )
    
    selected_categories = st.multiselect(
        "Catégories",
        unique_cats,
        help="Solo, Multi, Co-op, etc."
    )
    
    # Bouton Reset
    if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
        st.rerun()
    
    st.divider()
    
    # Info dataset
    st.markdown("### 📊 Dataset")
    st.caption(f"**{len(df_analyse):,}** jeux dans la base")

# ============================================================
# 4. APPLICATION DES FILTRES
# ============================================================
df_filtered = apply_all_filters(
    df_analyse,
    genres=selected_genres,
    categories=selected_categories
)

# Filtres additionnels (année, prix)
if year_range:
    df_filtered = df_filtered[
        (df_filtered["Release Year"] >= year_range[0]) &
        (df_filtered["Release Year"] <= year_range[1])
    ]

df_filtered = df_filtered[
    (df_filtered["Price"] >= price_range[0]) &
    (df_filtered["Price"] <= price_range[1])
]

# Indicateur de filtrage
if len(df_filtered) < len(df_analyse):
    st.success(
        f"🎯 **{len(df_filtered):,}** jeux correspondent aux filtres "
        f"(sur {len(df_analyse):,} total)"
    )
else:
    st.info(f"📊 **{len(df_analyse):,}** jeux affichés (aucun filtre)")

# ============================================================
# 5. TOP-LEVEL KPIs (Métriques Économiques Clés)
# ============================================================
st.markdown("### 📈 Indicateurs Économiques Clés")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    # Valeur totale du marché
    total_revenue = df_filtered["Estimated revenue"].sum()
    #total_revenue_all = df_analyse["Estimated revenue"].sum()
    #delta_revenue = total_revenue - total_revenue_all
    
    st.metric(
        "💰 Valeur Marché",
        format_number(total_revenue),
        #delta=format_number(delta_revenue) if selected_genres or selected_categories else None,
        help="Revenu total estimé du segment de marché"
    )

with kpi2:
    # Prix médian
    median_price = df_filtered["Price"].median()
    mean_price = df_filtered["Price"].mean()
    
    st.metric(
        "💵 Prix Médian",
        f"${median_price:.2f}",
        delta=f"Moyen: ${mean_price:.2f}",
        delta_color="off",
        help="Prix médian (50% des jeux en dessous)"
    )

with kpi3:
    # Ratio F2P/Payant
    f2p_ratio = (df_filtered["Price"] == 0).sum() / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
    
    st.metric(
        "🎮 % Free-to-Play",
        f"{f2p_ratio:.1f}%",
        help="Pourcentage de jeux gratuits dans le segment"
    )

with kpi4:
    # Value for Money médian ($/heure)
    df_vfm = df_filtered[
        (df_filtered["Price"] > 0) &
        (df_filtered["Median playtime forever"] > 0)
    ].copy()
    
    if len(df_vfm) > 0:
        df_vfm["Cost_per_Hour"] = df_vfm["Price"] / (df_vfm["Median playtime forever"] / 60)
        vfm_median = df_vfm["Cost_per_Hour"].median()
        st.metric(
            "⏱️ Coût/Heure",
            f"${vfm_median:.2f}",
            help="Coût moyen par heure de jeu"
        )
    else:
        st.metric("⏱️ Coût/Heure", "N/A")

st.divider()

# ============================================================
# 6. ONGLETS PRINCIPAUX
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Long Tail & Pareto",
    "💎 Value for Money",
    "🎯 Pricing Power",
    "🏢 Market Share"
])

# ============================================================
# TAB 1: LONG TAIL & PRINCIPE DE PARETO
# ============================================================
with tab1:
    st.markdown("### 📊 Concentration du Marché : Principe de Pareto (80/20)")
    st.caption("Analyse de l'inégalité des revenus dans l'industrie du jeu vidéo")
    
    # Calcul Pareto/Lorenz
    df_pareto = df_filtered[df_filtered["Estimated revenue"] > 0].copy()
    
    if len(df_pareto) > 0:
        # Tri par revenu décroissant
        df_pareto = df_pareto.sort_values("Estimated revenue", ascending=False).reset_index(drop=True)
        
        # Calculs cumulatifs
        total_rev = df_pareto["Estimated revenue"].sum()
        df_pareto["Cumul_Revenue"] = df_pareto["Estimated revenue"].cumsum()
        df_pareto["Cumul_Pct_Revenue"] = (df_pareto["Cumul_Revenue"] / total_rev) * 100
        df_pareto["Game_Pct"] = (np.arange(1, len(df_pareto) + 1) / len(df_pareto)) * 100
        
        # Graphique Courbe de Lorenz
        fig_lorenz = go.Figure()
        
        # Ligne d'égalité parfaite (diagonale)
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Égalité parfaite",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
            hovertemplate="Égalité: %{y:.0f}%<extra></extra>"
        ))
        
        # Courbe de Lorenz réelle
        fig_lorenz.add_trace(go.Scatter(
            x=np.concatenate([[0], df_pareto["Game_Pct"].values]),
            y=np.concatenate([[0], df_pareto["Cumul_Pct_Revenue"].values]),
            mode="lines",
            name="Distribution réelle",
            line=dict(color=COLORS["primary"], width=3),
            fill="tonexty",
            fillcolor="rgba(0,255,136,0.2)",
            hovertemplate="Top %{x:.0f}% des jeux = %{y:.0f}% des revenus<extra></extra>"
        ))
        
        fig_lorenz.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="% de Jeux (triés par revenu décroissant)",
            yaxis_title="% Cumulé des Revenus",
            height=450,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_lorenz, use_container_width=True)
        
        # Calcul des insights Pareto
        top_20_idx = int(len(df_pareto) * 0.2)
        top_20_revenue_pct = df_pareto.iloc[top_20_idx]["Cumul_Pct_Revenue"]
        
        top_10_idx = int(len(df_pareto) * 0.1)
        top_10_revenue_pct = df_pareto.iloc[top_10_idx]["Cumul_Pct_Revenue"]
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.info(
                f"📊 **Principe de Pareto**: Les top **20%** des jeux génèrent "
                f"**{top_20_revenue_pct:.1f}%** des revenus totaux."
            )
        
        with col_insight2:
            st.warning(
                f"🏆 **Ultra-concentration**: Les top **10%** captent "
                f"**{top_10_revenue_pct:.1f}%** du marché."
            )
    else:
        st.warning("⚠️ Pas de données de revenus disponibles pour cette sélection")

# ============================================================
# TAB 2: VALUE FOR MONEY (COST PER HOUR)
# ============================================================
with tab2:
    st.markdown("### 💎 Value for Money : Coût par Heure de Jeu")
    st.caption("Quels genres offrent le meilleur rapport qualité/prix ?")
    
    # Calcul Cost per Hour
    df_vfm_analysis = df_filtered[
        (df_filtered["Median playtime forever"] > 0) &
        (df_filtered["Price"] > 0)
    ].copy()
    
    if len(df_vfm_analysis) > 0:
        df_vfm_analysis["Cost_per_Hour"] = df_vfm_analysis["Price"] / (df_vfm_analysis["Median playtime forever"] / 60)
        
        # Explosion des genres
        df_vfm_exploded = df_vfm_analysis.explode("Genres").dropna(subset=["Genres"])
        
        # Top 15 genres par volume
        top_genres = df_vfm_exploded["Genres"].value_counts().head(15).index.tolist()
        df_vfm_top = df_vfm_exploded[df_vfm_exploded["Genres"].isin(top_genres)]
        
        col_vfm1, col_vfm2 = st.columns([2, 1])
        
        with col_vfm1:
            st.markdown("#### 📊 Distribution du Coût/Heure par Genre")
            
            # Boxplot par genre
            fig_vfm = px.box(
                df_vfm_top,
                x="Cost_per_Hour",
                y="Genres",
                color="Genres",
                orientation="h",
                log_x=True,
                color_discrete_sequence=COLORS["chart"]
            )
            
            fig_vfm.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Coût par Heure ($) - Échelle Log",
                yaxis_title="",
                height=500,
                showlegend=False
            )
            
            fig_vfm.update_yaxes(categoryorder="median ascending")
            
            st.plotly_chart(fig_vfm, use_container_width=True)
        
        with col_vfm2:
            st.markdown("#### 🏆 Meilleur Rapport $/h")
            
            # Top 10 meilleurs rapports (médiane par genre)
            genre_vfm = df_vfm_top.groupby("Genres")["Cost_per_Hour"].median().sort_values().head(10).reset_index()
            genre_vfm.columns = ["Genre", "$/Heure"]
            genre_vfm["$/Heure"] = genre_vfm["$/Heure"].round(2)
            
            st.dataframe(
                genre_vfm,
                hide_index=True,
                use_container_width=True,
                height=460
            )
        
        # Insight économique
        best_genre = genre_vfm.iloc[0]["Genre"]
        best_value = genre_vfm.iloc[0]["$/Heure"]
        
        st.success(
            f"💎 **Insight Économique**: Le genre **{best_genre}** offre le meilleur "
            f"rapport qualité/prix avec seulement **${best_value:.2f}/heure** de jeu. "
            f"Les développeurs de ces genres doivent compenser par le volume de joueurs."
        )
    else:
        st.warning("⚠️ Pas assez de données pour l'analyse Value for Money")

# ============================================================
# TAB 3: PRICING POWER & ÉLASTICITÉ
# ============================================================
with tab3:
    st.markdown("### 🎯 Pricing Power : Prix vs Qualité")
    st.caption("Les jeux chers sont-ils mieux notés ? Analyse de la pression qualité")
    
    col_pricing1, col_pricing2 = st.columns(2)
    
    with col_pricing1:
        st.markdown("#### 📈 Corrélation Prix vs Score Metacritic")
        
        # Filtrage données valides
        df_pricing = df_filtered[
            (df_filtered["Metacritic score"] > 0) &
            (df_filtered["Price"] > 0)
        ].copy()
        
        if len(df_pricing) > 0:
            # Échantillonnage si trop de données
            if len(df_pricing) > 500:
                df_pricing_sample = df_pricing.sample(500, random_state=42)
            else:
                df_pricing_sample = df_pricing
            
            # Scatter avec trendline
            fig_price_quality = px.scatter(
                df_pricing_sample,
                x="Price",
                y="Metacritic score",
                size="Recommendations",
                color="Price",
                hover_name="Name",
                trendline="ols",
                color_continuous_scale=[[0, COLORS["tertiary"]], [0.5, COLORS["primary"]], [1, COLORS["secondary"]]],
                opacity=0.7
            )
            
            fig_price_quality.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Prix ($)",
                yaxis_title="Score Metacritic",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_price_quality, use_container_width=True)
            
            # Calcul de la corrélation
            correlation = df_pricing["Price"].corr(df_pricing["Metacritic score"])
            
            if correlation > 0.3:
                st.info(
                    f"📊 **Corrélation positive** de **{correlation:.2%}** : "
                    f"Les jeux chers ont tendance à être mieux notés (pression qualité)."
                )
            elif correlation < -0.3:
                st.warning(
                    f"⚠️ **Corrélation négative** de **{correlation:.2%}** : "
                    f"Les jeux chers sont jugés plus sévèrement !"
                )
            else:
                st.info(
                    f"📊 **Faible corrélation** de **{correlation:.2%}** : "
                    f"Le prix n'est pas un indicateur de qualité."
                )
        else:
            st.warning("⚠️ Pas assez de données avec scores Metacritic")
    
    with col_pricing2:
        st.markdown("#### 💰 Segmentation Prix par Genre")
        
        # Violin plot des prix par genre
        df_genre_pricing = df_filtered[df_filtered["Price"] > 0].copy()
        df_genre_exploded = df_genre_pricing.explode("Genres").dropna(subset=["Genres"])
        
        # Top 12 genres
        top_genres_pricing = df_genre_exploded["Genres"].value_counts().head(12).index.tolist()
        df_genre_top = df_genre_exploded[df_genre_exploded["Genres"].isin(top_genres_pricing)]
        
        if len(df_genre_top) > 0:
            fig_pricing_genre = px.violin(
                df_genre_top,
                y="Genres",
                x="Price",
                box=True,
                color="Genres",
                color_discrete_sequence=COLORS["chart"]
            )
            
            fig_pricing_genre.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Prix ($)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            
            fig_pricing_genre.update_yaxes(categoryorder="median descending")
            
            st.plotly_chart(fig_pricing_genre, use_container_width=True)
            
            # Insight Premium vs Commodity
            genre_prices = df_genre_top.groupby("Genres")["Price"].median().sort_values(ascending=False)
            premium_genre = genre_prices.index[0]
            premium_price = genre_prices.iloc[0]
            
            st.success(
                f"💎 **Genre Premium**: **{premium_genre}** avec un prix médian de "
                f"**${premium_price:.2f}** — un marché de niche à forte valorisation."
            )

    st.divider()
    
    # Évolution du pricing dans le temps
    st.markdown("#### 📈 Évolution du Prix Moyen au Fil des Années")
    
    if "Release Year" in df_filtered.columns:
        df_pricing_evolution = df_filtered[
            (df_filtered["Release Year"] >= 1997) &
            (df_filtered["Price"] > 0)
        ].copy()
        
        yearly_pricing = df_pricing_evolution.groupby("Release Year")["Price"].agg(["mean", "median"]).reset_index()
        
        fig_evolution = go.Figure()
        
        fig_evolution.add_trace(go.Scatter(
            x=yearly_pricing["Release Year"],
            y=yearly_pricing["mean"],
            mode="lines+markers",
            name="Prix Moyen",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8)
        ))
        
        fig_evolution.add_trace(go.Scatter(
            x=yearly_pricing["Release Year"],
            y=yearly_pricing["median"],
            mode="lines+markers",
            name="Prix Médian",
            line=dict(color=COLORS["tertiary"], width=3),
            marker=dict(size=8)
        ))
        
        fig_evolution.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Année de Sortie",
            yaxis_title="Prix ($)",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Calcul inflation gaming
        if len(yearly_pricing) >= 10:
            price_10y_ago = yearly_pricing[yearly_pricing["Release Year"] == yearly_pricing["Release Year"].iloc[0]]["mean"].iloc[0]
            price_recent = yearly_pricing[yearly_pricing["Release Year"] == yearly_pricing["Release Year"].iloc[-1]]["mean"].iloc[0]
            inflation = ((price_recent - price_10y_ago) / price_10y_ago) * 100
            
            st.info(
                f"📊 **Inflation Gaming**: Le prix moyen a évolué de **{inflation:+.1f}%** "
                f"sur la période analysée."
            )

# ============================================================
# TAB 4: MARKET SHARE & SEGMENTATION
# ============================================================
with tab4:
    st.markdown("### 🏢 Segmentation de Marché & Parts de Marché")
    st.caption("Visualisation de la domination par genre et analyse Indie vs AAA")
    
    col_market1, col_market2 = st.columns([2, 1])
    
    with col_market1:
        st.markdown("#### 🗺️ Treemap : Parts de Marché par Genre")
        
        # Préparation données treemap
        df_treemap = df_filtered[df_filtered["Estimated revenue"] > 0].copy()
        df_treemap_exploded = df_treemap.explode("Genres").dropna(subset=["Genres"])
        
        genre_revenue = df_treemap_exploded.groupby("Genres").agg({
            "Estimated revenue": "sum",
            "AppID": "count"
        }).reset_index()
        genre_revenue.columns = ["Genre", "Revenue", "Count"]
        genre_revenue = genre_revenue.sort_values("Revenue", ascending=False).head(20)
        
        if len(genre_revenue) > 0:
            fig_treemap = px.treemap(
                genre_revenue,
                path=["Genre"],
                values="Revenue",
                color="Revenue",
                hover_data={"Count": True},
                color_continuous_scale=[[0, COLORS["chart"][0]], [0.5, COLORS["chart"][2]], [1, COLORS["chart"][4]]],
            )
            
            fig_treemap.update_layout(
                **PLOTLY_LAYOUT,
                height=500
            )
            
            fig_treemap.update_traces(
                textposition="middle center",
                marker=dict(line=dict(width=2, color="rgba(0,0,0,0.5)"))
            )
            
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Top genre
            top_genre_market = genre_revenue.iloc[0]["Genre"]
            top_revenue_market = genre_revenue.iloc[0]["Revenue"]
            total_revenue_market = genre_revenue["Revenue"].sum()
            top_pct = (top_revenue_market / total_revenue_market) * 100
            
            st.success(
                f"🏆 **Genre Dominant**: **{top_genre_market}** capte "
                f"**{top_pct:.1f}%** du marché (top 20 genres) avec {format_number(top_revenue_market)}."
            )
        else:
            st.warning("⚠️ Pas de données de revenus disponibles")
    
    with col_market2:
        st.markdown("#### 📊 Top 10 Genres")
        
        if len(genre_revenue) > 0:
            top10_display = genre_revenue.head(10)[["Genre", "Revenue", "Count"]].copy()
            top10_display["Revenue"] = top10_display["Revenue"].apply(lambda x: format_number(x))
            top10_display.columns = ["Genre", "Revenu", "Jeux"]
            
            st.dataframe(
                top10_display,
                hide_index=True,
                use_container_width=True,
                height=460
            )
    
    st.divider()
    
    # Analyse Indie vs AAA
    st.markdown("### 🎮 Analyse Indie vs AAA : Santé du Marché")
    
    if "Estimated revenue" in df_filtered.columns and len(df_filtered) > 0:
        df_segment = df_filtered[df_filtered["Estimated revenue"] > 0].copy()
        
        # Définition AAA = top 10% par revenu
        threshold_aaa = df_segment["Estimated revenue"].quantile(0.90)
        df_segment["Segment"] = df_segment["Estimated revenue"].apply(
            lambda x: "AAA" if x >= threshold_aaa else "Indie"
        )
        
        col_indie1, col_indie2, col_indie3 = st.columns(3)
        
        with col_indie1:
            st.markdown("#### 💰 Revenus Médians")
            
            segment_medians = df_segment.groupby("Segment")["Estimated revenue"].median().reset_index()
            segment_medians.columns = ["Segment", "Median_Revenue"]
            
            fig_segment_median = px.bar(
                segment_medians,
                x="Segment",
                y="Median_Revenue",
                color="Segment",
                color_discrete_map={"AAA": COLORS["secondary"], "Indie": COLORS["tertiary"]},
                text="Median_Revenue"
            )
            
            fig_segment_median.update_traces(
                texttemplate='%{text:.2s}',
                textposition='outside'
            )
            
            fig_segment_median.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=False,
                yaxis_title="Revenu Médian ($)",
                xaxis_title="",
                height=300
            )
            
            st.plotly_chart(fig_segment_median, use_container_width=True)
        
        with col_indie2:
            st.markdown("#### 🎮 Volume de Jeux")
            
            segment_counts = df_segment["Segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Count"]
            
            fig_segment_count = px.pie(
                segment_counts,
                values="Count",
                names="Segment",
                color="Segment",
                color_discrete_map={"AAA": COLORS["secondary"], "Indie": COLORS["tertiary"]},
                hole=0.4
            )
            
            fig_segment_count.update_layout(
                **PLOTLY_LAYOUT,
                height=300
            )
            
            fig_segment_count.update_traces(
                textposition="inside",
                textinfo="percent+value"
            )
            
            st.plotly_chart(fig_segment_count, use_container_width=True)
        
        with col_indie3:
            st.markdown("#### ⭐ Qualité Moyenne")
            
            df_segment_quality = df_segment[df_segment["Metacritic score"] > 0]
            
            if len(df_segment_quality) > 0:
                segment_quality = df_segment_quality.groupby("Segment")["Metacritic score"].mean().reset_index()
                segment_quality.columns = ["Segment", "Score"]
                
                fig_segment_quality = px.bar(
                    segment_quality,
                    x="Segment",
                    y="Score",
                    color="Segment",
                    color_discrete_map={"AAA": COLORS["secondary"], "Indie": COLORS["tertiary"]},
                    text="Score"
                )
                
                fig_segment_quality.update_traces(
                    texttemplate='%{text:.1f}',
                    textposition='outside'
                )
                
                fig_segment_quality.update_layout(
                    **PLOTLY_LAYOUT,
                    showlegend=False,
                    yaxis_title="Score Metacritic Moyen",
                    xaxis_title="",
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_segment_quality, use_container_width=True)
            else:
                st.metric("Score Moyen", "N/A")
        
        # Insight marché
        indie_count = segment_counts[segment_counts["Segment"] == "Indie"]["Count"].iloc[0]
        total_count = segment_counts["Count"].sum()
        indie_pct = (indie_count / total_count) * 100
        
        st.info(
            f"🎮 **Marché Indie**: **{indie_pct:.0f}%** des jeux sont classés Indie (hors top 10% revenus). "
            f"Le marché reste très accessible aux petits studios, mais la monétisation est concentrée sur les AAA."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "💰 **GameData360 — Économie** | "
    f"Analyse économique professionnelle sur {len(df_analyse):,} jeux"
)