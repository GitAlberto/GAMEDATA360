# -*- coding: utf-8 -*-
"""
GameData360 - Page Marché Global
=================================
Vue d'ensemble complète du marché du jeu vidéo avec graphiques essentiels.
KPIs principaux, évolution temporelle, distribution prix/genres/plateformes.

Auteur: GameData360 Team
Version: 3.0 (Market Overview Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from utils.data_helpers import explode_genres

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
    page_title="GameData360 — Marché Global",
    page_icon="🌍",
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
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("# 🌍 MARCHÉ GLOBAL")
st.markdown("##### Vue d'ensemble du marché du jeu vidéo : Graphiques essentiels")

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
# 3. SIDEBAR - FILTRES ESSENTIELS
# ============================================================
st.sidebar.markdown("# 🎛️ Filtres Globaux")
st.sidebar.markdown("Filtrez les données affichées dans tous les graphiques !")

# Filtre Genres
st.sidebar.markdown("### 🎮 Genres")
all_genres = sorted(set([g for genres in df_analyse["Genres"] if isinstance(genres, list) for g in genres])) # extraction des genres
selected_genres = st.sidebar.multiselect(
    "Sélectionner genres",
    options=all_genres,# Injection des genres
    default=[],
    key="market_genres"
)

# Filtre Catégories
st.sidebar.markdown("### 📂 Catégories")
all_categories = sorted(set([c for cats in df_analyse["Categories"] if isinstance(cats, list) for c in cats])) # extraction des catégories
selected_categories = st.sidebar.multiselect(
    "Sélectionner catégories",
    options=all_categories,# Injection des catégories
    default=[],
    key="market_categories"
)

# Filtre Tags
st.sidebar.markdown("### 🏷️ Tags")
all_tags = sorted(set([t for tags in df_analyse["Tags"] if isinstance(tags, list) for t in tags])) # extraction des tags
selected_tags = st.sidebar.multiselect(
    "Sélectionner tags",
    options=all_tags,# Injection des tags
    default=[],
    key="market_tags"
)

# Filtre Prix
st.sidebar.markdown("### 💰 Prix")
price_min = float(df_analyse["Price"].min()) # extraction du prix minimum
price_max = float(df_analyse["Price"].max()) # extraction du prix maximum

price_range = st.sidebar.slider(
    "Range de prix ($)", # Intitulé du slider
    min_value=price_min, # Borne minimum
    max_value=price_max, # Borne maximum
    value=(price_min, price_max), # Plage par défaut
    step=1.0, # Pas
    key="market_price"
)

# Filtre Année
st.sidebar.markdown("### 📅 Année de Sortie")
year_min = int(df_analyse["Release Year"].min()) # extraction de l'année minimum
year_max = int(df_analyse["Release Year"].max()) # extraction de l'année maximum

year_range = st.sidebar.slider(
    "Période", # Intitulé du slider
    min_value=year_min, # Borne minimum
    max_value=year_max, # Borne maximum
    value=(year_min, year_max), # Plage par défaut
    step=1, # Pas
    key="market_year"
)

# Bouton Reset
if st.sidebar.button("🔄 Réinitialiser les filtres"):
    st.rerun()

# ============================================================
# 4. APPLICATION DES FILTRES
# ============================================================
@st.cache_data(show_spinner=False)
def apply_market_filters(_df, genres, categories, tags, price_rng, year_rng):
    """Applique les filtres de marché."""
    filtered = _df.copy()
    
    # Filtre Prix
    filtered = filtered[
        (filtered["Price"] >= price_rng[0]) &
        (filtered["Price"] <= price_rng[1])
    ]
    
    # Filtre Année
    filtered = filtered[
        (filtered["Release Year"] >= year_rng[0]) &
        (filtered["Release Year"] <= year_rng[1])
    ]
    
    # Filtre Genres
    if genres:
        filtered = filtered[
            filtered["Genres"].apply(
                lambda x: isinstance(x, list) and any(g in x for g in genres)
            )
        ]
    
    # Filtre Catégories
    if categories:
        filtered = filtered[
            filtered["Categories"].apply(
                lambda x: isinstance(x, list) and any(c in x for c in categories)
            )
        ]
    
    # Filtre Tags
    if tags:
        filtered = filtered[
            filtered["Tags"].apply(
                lambda x: isinstance(x, list) and any(t in x for t in tags)
            )
        ]
    
    return filtered

# Application des filtres
df_filtered = apply_market_filters(
    df_analyse,
    selected_genres,
    selected_categories,
    selected_tags,
    price_range,
    year_range
)

# Info filtrage
filters_active = bool(selected_genres or selected_categories or selected_tags or 
                     price_range != (price_min, price_max) or 
                     year_range != (year_min, year_max))

if filters_active:
    st.info(f"📊 **{len(df_filtered):,}** jeux après filtrage (sur {len(df_analyse):,} total)")
else:
    st.success(f"📊 Affichage de tous les **{len(df_analyse):,}** jeux")

# ============================================================
# 5. TOP-LEVEL KPIs ESSENTIELS
# ============================================================
st.markdown("### 📊 Indicateurs Clés du Marché")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# KPI 1: Total Jeux
with kpi1:
    st.metric(
        "🎮 Total Jeux",
        f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df_analyse):,}" if filters_active else None,
        help="Nombre total de jeux"
    )

# KPI 2: Revenus Totaux
total_revenue = df_filtered["Estimated revenue"].sum() / 1e9
with kpi2:
    st.metric(
        "💰 Revenus Totaux",
        f"${total_revenue:.1f}B",
        help="Revenus estimés cumulés (milliards USD)"
    )

# KPI 3: Prix Moyen
avg_price = df_filtered["Price"].mean()
with kpi3:
    st.metric(
        "💵 Prix Moyen",
        f"${avg_price:.2f}",
        help="Prix moyen d'un jeu"
    )

# KPI 4: % Free-to-Play
f2p_count = len(df_filtered[df_filtered["Price"] == 0])
f2p_pct = (f2p_count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
with kpi4:
    st.metric(
        "🆓 % Free-to-Play",
        f"{f2p_pct:.0f}%",
        help="Pourcentage de jeux gratuits"
    )

# KPI 5: Metacritic Médian
median_meta = df_filtered[df_filtered["Metacritic score"] > 0]["Metacritic score"].median()
with kpi5:
    st.metric(
        "⭐ Metacritic Médian",
        f"{median_meta:.0f}" if not pd.isna(median_meta) else "N/A",
        help="Score critique médian"
    )

st.divider()

# ============================================================
# 4. GRAPHIQUES ESSENTIELS
# ============================================================

# ROW 1: Évolution Temporelle
st.markdown("### 📈 Évolution du Marché")

col_evo1, col_evo2 = st.columns(2)

with col_evo1:
    st.markdown("#### Volume de Jeux par Année à partir de l'an 2000")
    
    # Préparation données
    yearly_volume = df_filtered.groupby("Release Year").size().reset_index(name="Nombre de jeux")
    yearly_volume = yearly_volume[yearly_volume["Release Year"] >= 2000]  # Focus post-2000
    
    fig_volume = px.area(
        yearly_volume,
        x="Release Year",
        y="Nombre de jeux",
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig_volume.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

with col_evo2:
    st.markdown("#### Revenus Cumulés par Année à partir de l'an 2000")
    
    # Revenus par année
    yearly_revenue = df_filtered.groupby("Release Year")["Estimated revenue"].sum().reset_index()
    yearly_revenue = yearly_revenue[yearly_revenue["Release Year"] >= 2000]
    yearly_revenue["Revenue (Milliards)"] = yearly_revenue["Estimated revenue"] / 1e9
    
    fig_revenue = px.bar(
        yearly_revenue,
        x="Release Year",
        y="Revenue (Milliards)",
        color_discrete_sequence=[COLORS['solo']]
    )
    
    fig_revenue.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        showlegend=False,
        yaxis_title="Revenus (Milliards USD)"
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)

st.divider()

# ROW 2: Distributions
st.markdown("### 📊 Distributions Marché")

col_dist1, col_dist2, col_dist3 = st.columns(3)

with col_dist1:
    st.markdown("#### Distribution Prix")
    
    # Histogramme prix (excluant F2P pour lisibilité)
    df_paid = df_filtered[df_filtered["Price"] > 0]
    
    fig_price = px.histogram(
        df_paid,
        x="Price",
        nbins=50,
        color_discrete_sequence=[COLORS['chart'][2]]
    )
    
    fig_price.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=False,
        xaxis_title="Prix ($)",
        yaxis_title="Nombre de jeux"
    )
    
    # Ligne verticale prix médian
    median_price = df_paid["Price"].median()
    fig_price.add_vline(x=median_price, line_dash="dash", line_color=COLORS['danger'], 
                       annotation_text=f"Médiane: ${median_price:.2f}")
    
    st.plotly_chart(fig_price, use_container_width=True)

with col_dist2:
    st.markdown("#### Distribution Metacritic")
    
    df_with_meta = df_filtered[df_filtered["Metacritic score"] > 0]
    
    fig_meta = px.histogram(
        df_with_meta,
        x="Metacritic score",
        nbins=40,
        color_discrete_sequence=[COLORS['chart'][4]]
    )
    
    fig_meta.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=False,
        xaxis_title="Metacritic Score",
        yaxis_title="Nombre de jeux"
    )
    
    # Ligne verticale médiane
    fig_meta.add_vline(x=median_meta, line_dash="dash", line_color=COLORS['primary'], 
                      annotation_text=f"Médiane: {median_meta:.0f}")
    
    st.plotly_chart(fig_meta, use_container_width=True)

with col_dist3:
    st.markdown("#### F2P vs Payant")
    
    # Pie chart
    paid_count = len(df_filtered) - f2p_count  # Correction: utiliser df_filtered
    f2p_data = pd.DataFrame({
        'Type': ['Free-to-Play', 'Payant'],
        'Count': [f2p_count, paid_count]
    })
    
    fig_f2p = px.pie(
        f2p_data,
        values='Count',
        names='Type',
        color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
        hole=0.4
    )
    
    fig_f2p.update_layout(
        **PLOTLY_LAYOUT,
        height=300
    )
    
    st.plotly_chart(fig_f2p, use_container_width=True)

st.divider()

# ROW 3: Top Genres & Plateformes
st.markdown("### 🎯 Répartition par Genre & Plateforme")

col_genre, col_platform = st.columns(2)

with col_genre:
    st.markdown("#### Top 10 Genres")
    
    # Explosion genres
    df_genres = explode_genres(df_filtered)
    genre_counts = df_genres["Genres"].value_counts().head(10).reset_index()
    genre_counts.columns = ["Genre", "Count"]
    
    fig_genres = px.bar(
        genre_counts,
        x="Count",
        y="Genre",
        orientation="h",
        color="Count",
        color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['chart'][5]]]
    )
    
    fig_genres.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Nombre de jeux"
    )
    
    st.plotly_chart(fig_genres, use_container_width=True)

with col_platform:
    st.markdown("#### Support Plateformes")
    
    # Comptage plateformes
    platform_data = pd.DataFrame({
        'Plateforme': ['Windows', 'Mac', 'Linux'],
        'Jeux': [
            df_filtered['Windows'].sum(),
            df_filtered['Mac'].sum(),
            df_filtered['Linux'].sum()
        ]
    })
    
    fig_platform = px.bar(
        platform_data,
        x='Plateforme',
        y='Jeux',
        color='Plateforme',
        color_discrete_map={
            'Windows': COLORS['solo'],
            'Mac': COLORS['multi'],
            'Linux': COLORS['tertiary']
        }
    )
    
    fig_platform.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        showlegend=False,
        yaxis_title="Nombre de jeux"
    )
    
    st.plotly_chart(fig_platform, use_container_width=True)

st.divider()

# ROW 4: Évolution F2P vs Payant
st.markdown("### 🆓 Évolution Free-to-Play vs Payant à partir de l'année 2000")

# Évolution temporelle F2P
df_filtered["Type"] = df_filtered["Price"].apply(lambda x: "Free-to-Play" if x == 0 else "Payant")
f2p_evolution = df_filtered.groupby(["Release Year", "Type"]).size().reset_index(name="Count")
f2p_evolution = f2p_evolution[f2p_evolution["Release Year"] >= 2000]

fig_f2p_evo = px.area(
    f2p_evolution,
    x="Release Year",
    y="Count",
    color="Type",
    color_discrete_map={
        "Free-to-Play": COLORS['primary'],      # Vert (en bas, petite aire)
        "Payant": COLORS['secondary']            # Violet/Magenta (au-dessus, grande aire dominante)
    },
    category_orders={"Type": ["Free-to-Play", "Payant"]}  # F2P en 1er = bas, Payant en 2ème = dessus
)

fig_f2p_evo.update_layout(
    **PLOTLY_LAYOUT,
    height=400,
    yaxis_title="Nombre de jeux"
)

st.plotly_chart(fig_f2p_evo, use_container_width=True)

st.divider()

# ROW 5: Top Jeux (Peak CCU)
st.markdown("### 🏆 Top 20 Jeux les Plus Populaires (Peak CCU)")

top_20 = df_filtered.nlargest(20, "Peak CCU")[["Name", "Peak CCU", "Price", "Metacritic score", "Genres", "Release Year"]]
top_20["Peak CCU"] = top_20["Peak CCU"].apply(lambda x: f"{x:,}")
top_20["Price"] = top_20["Price"].apply(lambda x: f"${x:.2f}")
top_20["Genres"] = top_20["Genres"].apply(lambda x: ", ".join(x[:3]) if isinstance(x, list) else "N/A")
top_20.columns = ["Nom", "Peak CCU", "Prix", "Metacritic", "Genres", "Année"]

st.dataframe(top_20, hide_index=True, use_container_width=True, height=400)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🌍 **GameData360 — Marché Global** | "
    f"Vue d'ensemble sur {len(df_filtered):,} jeux"
)