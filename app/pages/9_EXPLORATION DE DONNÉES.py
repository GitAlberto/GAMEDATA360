# -*- coding: utf-8 -*-
"""
GameData360 - Page Exploration de Données
==========================================
Le "Google du Jeu Vidéo" - Outil d'exploration ultra-complet et granulaire.
Features: Fuzzy Search, Granular Filters, Interactive Grid, Game Comparator.

Auteur: GameData360 Team
Version: 4.0 (Data Browser Masterclass Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="GameData360 — Data Browser",
    page_icon="🔍",
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
    
    /* Style pour la search bar */
    .stTextInput input {
        font-size: 18px !important;
        font-family: 'Rajdhani', sans-serif !important;
        border: 2px solid rgba(0,255,136,0.5) !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    
    .stTextInput input:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 10px rgba(0,255,136,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("# 🔍 DATA BROWSER — Le Google du Jeu Vidéo")
st.markdown("##### Explorez, comparez, découvrez — Recherche intelligente & filtrage granulaire")

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
        
        # Filtrage des genres non-jeux (logiciels) - EXACTEMENT comme page 2
        def is_game(genres_list):
            """Retourne False si le jeu contient un genre de logiciel."""
            if not isinstance(genres_list, list):
                return True
            # Comparaison case-insensitive (NON_GAME_GENRES est en lowercase)
            genres_lower = [g.lower() for g in genres_list]
            return not any(genre in genres_lower for genre in NON_GAME_GENRES)
        
        initial_count = len(df_analyse)
        df_full = df_analyse[df_analyse["Genres"].apply(is_game)].copy()
        excluded_count = initial_count - len(df_full)
        
        if excluded_count > 0:
            st.sidebar.success(f"🎮 {excluded_count:,} logiciels exclus (seuls les jeux sont analysés)")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. SEARCH BAR INTELLIGENTE (TOP PRIORITY)
# ============================================================
st.markdown("### 🔎 Recherche Intelligente")

search_query = st.text_input(
    "Recherchez un jeu par nom...",
    placeholder="Ex: Witcher, Portal, Half-Life...",
    help="Recherche insensible à la casse, trouve les jeux dont le nom contient votre recherche",
    label_visibility="collapsed"
)

# ============================================================
# 4. SIDEBAR - FILTRES GRANULAIRES
# ============================================================
st.sidebar.markdown("# 🎛️ Filtres Avancés")

# Filtre Prix
st.sidebar.markdown("### 💰 Prix")
price_min = float(df_full["Price"].min())
price_max = float(df_full["Price"].max())

price_range = st.sidebar.slider(
    "Range de prix ($)",
    min_value=price_min,
    max_value=price_max,  # Utilise le VRAI max (pas de cap)
    value=(price_min, price_max),
    step=1.0
)

# Filtre Metacritic
st.sidebar.markdown("### ⭐ Metacritic Score")
df_with_meta = df_full[df_full["Metacritic score"] > 0]

if len(df_with_meta) > 0:
    meta_min = float(df_with_meta["Metacritic score"].min())
    meta_max = 100.0  # Metacritic max = 100 (pas 97)
    
    meta_range = st.sidebar.slider(
        "Score critique",
        min_value=meta_min,
        max_value=meta_max,
        value=(meta_min, meta_max),
        step=1.0
    )
else:
    meta_range = (0.0, 100.0)

# Filtre Année
st.sidebar.markdown("### 📅 Année de Sortie")
year_min = int(df_full["Release Year"].min())
year_max = int(df_full["Release Year"].max())

year_range = st.sidebar.slider(
    "Période",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)

# Filtre Genres
st.sidebar.markdown("### 🎮 Genres")
all_genres = sorted(set([g for genres in df_full["Genres"] if isinstance(genres, list) for g in genres]))
selected_genres = st.sidebar.multiselect(
    "Sélectionner genres",
    options=all_genres,
    default=[]
)

# Filtre Support (Windows, Mac, Linux)
st.sidebar.markdown("### 💻 Support Plateforme")

filter_windows = st.sidebar.checkbox("Windows", value=True)
filter_mac = st.sidebar.checkbox("Mac", value=True)
filter_linux = st.sidebar.checkbox("Linux", value=True)

# Filtre Peak CCU (Popularité)
st.sidebar.markdown("### 🔥 Popularité (Peak CCU)")

df_with_ccu = df_full[df_full["Peak CCU"] > 0]
if len(df_with_ccu) > 0:
    ccu_min = 0.0  # Float pour cohérence
    ccu_max = float(df_with_ccu["Peak CCU"].max())  # Utilise le VRAI max (pas percentile)
    
    ccu_range = st.sidebar.slider(
        "Concurrent Users",
        min_value=ccu_min,
        max_value=ccu_max,
        value=(ccu_min, ccu_max),
        step=100.0
    )
else:
    ccu_range = (0.0, 1000000.0)

# Bouton Reset
if st.sidebar.button("🔄 Réinitialiser tous les filtres"):
    st.rerun()

# ============================================================
# 5. APPLICATION DES FILTRES (CACHED)
# ============================================================
@st.cache_data(show_spinner=False)
def apply_filters(_df, search, price_rng, meta_rng, year_rng, genres, win, mac, linux, ccu_rng):
    """Applique tous les filtres de manière optimisée."""
    filtered = _df.copy()
    
    # Recherche textuelle (fuzzy)
    if search:
        filtered = filtered[
            filtered["Name"].str.contains(search, case=False, na=False)
        ]
    
    # Filtres numériques
    filtered = filtered[
        (filtered["Price"] >= price_rng[0]) &
        (filtered["Price"] <= price_rng[1]) &
        (filtered["Release Year"] >= year_rng[0]) &
        (filtered["Release Year"] <= year_rng[1])
    ]
    
    # Metacritic (seulement si score > 0)
    filtered = filtered[
        (filtered["Metacritic score"] == 0) |
        ((filtered["Metacritic score"] >= meta_rng[0]) & (filtered["Metacritic score"] <= meta_rng[1]))
    ]
    
    # CCU
    filtered = filtered[
        (filtered["Peak CCU"] == 0) |
        ((filtered["Peak CCU"] >= ccu_rng[0]) & (filtered["Peak CCU"] <= ccu_rng[1]))
    ]
    
    # Genres
    if genres:
        filtered = filtered[
            filtered["Genres"].apply(
                lambda x: isinstance(x, list) and any(g in x for g in genres)
            )
        ]
    
    # Support plateformes
    platform_filters = []
    if win:
        platform_filters.append(filtered["Windows"] == True)
    if mac:
        platform_filters.append(filtered["Mac"] == True)
    if linux:
        platform_filters.append(filtered["Linux"] == True)
    
    if platform_filters:
        from functools import reduce
        import operator
        filtered = filtered[reduce(operator.or_, platform_filters)]
    
    return filtered

# Application des filtres
df_filtered = apply_filters(
    df_full,
    search_query,
    price_range,
    meta_range,
    year_range,
    selected_genres,
    filter_windows,
    filter_mac,
    filter_linux,
    ccu_range
)

# Feedback temps réel
st.markdown(f"### 📊 **{len(df_filtered):,}** jeux correspondent à vos critères")

if len(df_filtered) == 0:
    st.warning("Aucun jeu ne correspond à vos critères. Essayez d'ajuster les filtres.")
    st.stop()

st.divider()

# ============================================================
# 6. ONGLETS PRINCIPAUX
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Super Grid (Explorer)",
    "⚔️ Duel Mode (Comparer 2 Jeux)",
    "🎮 Fiche d'Identité (Deep Dive)"
])

# ============================================================
# TAB 1: SUPER GRID
# ============================================================
with tab1:
    st.markdown("### 🗂️ Tableau Interactif Avancé")
    
    # Préparation données pour affichage
    df_display = df_filtered[[
        "Name",
        "Release Year",
        "Price",
        "Metacritic score",
        "Positive",
        "Negative",
        "Peak CCU",
        "Median playtime forever",
        "Genres",
        "Windows",
        "Mac",
        "Linux"
    ]].copy()
    
    # Calculs dérivés
    df_display["Total Reviews"] = df_display["Positive"] + df_display["Negative"]
    df_display["% Positif"] = (
        (df_display["Positive"] / df_display["Total Reviews"] * 100)
        .fillna(0)
        .round(1)
    )
    df_display["Playtime (h)"] = (df_display["Median playtime forever"] / 60).round(1)
    
    # Sélection colonnes finales
    grid_cols = [
        "Name",
        "Release Year",
        "Price",
        "Metacritic score",
        "% Positif",
        "Total Reviews",
        "Peak CCU",
        "Playtime (h)",
        "Genres"
    ]
    
    df_grid = df_display[grid_cols].copy()
    
    # Configuration des colonnes avec st.column_config
    column_config = {
        "Name": st.column_config.TextColumn(
            "🎮 Nom du Jeu",
            width="large",
            help="Nom du jeu"
        ),
        "Release Year": st.column_config.NumberColumn(
            "📅 Année",
            format="%d",
            width="small"
        ),
        "Price": st.column_config.NumberColumn(
            "💰 Prix",
            format="$%.2f",
            width="small"
        ),
        "Metacritic score": st.column_config.ProgressColumn(
            "⭐ Metacritic",
            format="%d",
            min_value=0,
            max_value=100,
            width="medium"
        ),
        "% Positif": st.column_config.ProgressColumn(
            "👍 % Positif",
            format="%.1f%%",
            min_value=0,
            max_value=100,
            width="medium"
        ),
        "Total Reviews": st.column_config.NumberColumn(
            "💬 Reviews",
            format="%d",
            width="small"
        ),
        "Peak CCU": st.column_config.NumberColumn(
            "🔥 Peak CCU",
            format="%d",
            width="medium"
        ),
        "Playtime (h)": st.column_config.NumberColumn(
            "⏱️ Playtime",
            format="%.1f h",
            width="small"
        ),
        "Genres": st.column_config.ListColumn(
            "🎯 Genres",
            width="large"
        )
    }
    
    # Affichage du data editor
    st.dataframe(
        df_grid,
        column_config=column_config,
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Statistiques rapides
    st.markdown("#### 📈 Statistiques sur la sélection")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        avg_price = df_filtered["Price"].mean()
        st.metric("💰 Prix Moyen", f"${avg_price:.2f}")
    
    with col_stat2:
        avg_meta = df_filtered[df_filtered["Metacritic score"] > 0]["Metacritic score"].mean()
        st.metric("⭐ Metacritic Moyen", f"{avg_meta:.1f}")
    
    with col_stat3:
        total_reviews = df_display["Total Reviews"].sum()
        st.metric("💬 Total Reviews", f"{total_reviews:,}")
    
    with col_stat4:
        avg_playtime = df_display["Playtime (h)"].mean()
        st.metric("⏱️ Playtime Moyen", f"{avg_playtime:.1f}h")

# ============================================================
# TAB 2: DUEL MODE (COMPARATEUR)
# ============================================================
with tab2:
    st.markdown("### ⚔️ Comparateur de Jeux — Duel Mode")
    st.caption("Sélectionnez 2 jeux pour les comparer côte à côte")
    
    col_duel1, col_duel2 = st.columns(2)
    
    # Liste des jeux disponibles
    game_names = df_filtered["Name"].tolist()
    
    with col_duel1:
        game1 = st.selectbox(
            "🎮 Jeu 1",
            options=game_names,
            index=0 if len(game_names) > 0 else None
        )
    
    with col_duel2:
        game2 = st.selectbox(
            "🎮 Jeu 2",
            options=game_names,
            index=1 if len(game_names) > 1 else 0
        )
    
    if game1 and game2 and game1 != game2:
        # Récupération des données
        data1 = df_filtered[df_filtered["Name"] == game1].iloc[0]
        data2 = df_filtered[df_filtered["Name"] == game2].iloc[0]
        
        # Radar Chart Comparatif
        st.markdown("#### 🕸️ Radar Chart Comparatif")
        
        # Normalisation des valeurs pour le radar (0-100)
        def normalize(val, min_val, max_val):
            if max_val == min_val:
                return 50
            return ((val - min_val) / (max_val - min_val)) * 100
        
        # Axes du radar
        categories = ["Prix\n(inverse)", "Metacritic", "% Positif", "Popularité\n(CCU)", "Playtime"]
        
        # Calcul valeurs normalisées
        price1_norm = 100 - normalize(data1["Price"], df_filtered["Price"].min(), df_filtered["Price"].max())
        price2_norm = 100 - normalize(data2["Price"], df_filtered["Price"].min(), df_filtered["Price"].max())
        
        meta1_norm = data1["Metacritic score"] if data1["Metacritic score"] > 0 else 50
        meta2_norm = data2["Metacritic score"] if data2["Metacritic score"] > 0 else 50
        
        pos1 = (data1["Positive"] / (data1["Positive"] + data1["Negative"]) * 100) if (data1["Positive"] + data1["Negative"]) > 0 else 50
        pos2 = (data2["Positive"] / (data2["Positive"] + data2["Negative"]) * 100) if (data2["Positive"] + data2["Negative"]) > 0 else 50
        
        ccu1_norm = normalize(data1["Peak CCU"], df_filtered["Peak CCU"].min(), df_filtered["Peak CCU"].max())
        ccu2_norm = normalize(data2["Peak CCU"], df_filtered["Peak CCU"].min(), df_filtered["Peak CCU"].max())
        
        play1_norm = normalize(data1["Median playtime forever"], df_filtered["Median playtime forever"].min(), df_filtered["Median playtime forever"].max())
        play2_norm = normalize(data2["Median playtime forever"], df_filtered["Median playtime forever"].min(), df_filtered["Median playtime forever"].max())
        
        values1 = [price1_norm, meta1_norm, pos1, ccu1_norm, play1_norm]
        values2 = [price2_norm, meta2_norm, pos2, ccu2_norm, play2_norm]
        
        # Création radar
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values1 + [values1[0]],  # Fermer le polygone
            theta=categories + [categories[0]],
            fill='toself',
            name=game1,
            line_color=COLORS['primary']
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values2 + [values2[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=game2,
            line_color=COLORS['danger']
        ))
        
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.divider()
        
        # Tableau Side-by-Side
        st.markdown("#### 📊 Comparaison Détaillée")
        
        comparison_data = {
            "Métrique": [
                "Prix",
                "Metacritic",
                "Reviews Positives",
                "Reviews Négatives",
                "% Positif",
                "Peak CCU",
                "Playtime Médian (h)",
                "Année de Sortie",
                "Genres"
            ],
            game1: [
                f"${data1['Price']:.2f}",
                f"{data1['Metacritic score']:.0f}" if data1['Metacritic score'] > 0 else "N/A",
                f"{data1['Positive']:,}",
                f"{data1['Negative']:,}",
                f"{pos1:.1f}%",
                f"{data1['Peak CCU']:,}",
                f"{(data1['Median playtime forever'] / 60):.1f}",
                f"{data1['Release Year']:.0f}",
                ", ".join(data1['Genres'][:3]) if isinstance(data1['Genres'], list) else "N/A"
            ],
            game2: [
                f"${data2['Price']:.2f}",
                f"{data2['Metacritic score']:.0f}" if data2['Metacritic score'] > 0 else "N/A",
                f"{data2['Positive']:,}",
                f"{data2['Negative']:,}",
                f"{pos2:.1f}%",
                f"{data2['Peak CCU']:,}",
                f"{(data2['Median playtime forever'] / 60):.1f}",
                f"{data2['Release Year']:.0f}",
                ", ".join(data2['Genres'][:3]) if isinstance(data2['Genres'], list) else "N/A"
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)
    
    else:
        st.info("Veuillez sélectionner 2 jeux différents pour activer le comparateur")

# ============================================================
# TAB 3: FICHE D'IDENTITÉ (DEEP DIVE)
# ============================================================
with tab3:
    st.markdown("### 🎮 Fiche d'Identité — Deep Dive")
    st.caption("Sélectionnez un jeu pour voir ses détails complets")
    
    # Sélection du jeu
    selected_game = st.selectbox(
        "Choisir un jeu",
        options=game_names,
        index=0 if len(game_names) > 0 else None,
        key="deep_dive_selector"
    )
    
    if selected_game:
        game_data = df_filtered[df_filtered["Name"] == selected_game].iloc[0]
        
        # En-tête
        st.markdown(f"## 🎮 {game_data['Name']}")
        
        # KPIs principaux
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        
        with col_kpi1:
            st.metric("💰 Prix", f"${game_data['Price']:.2f}")
        
        with col_kpi2:
            meta_score = game_data['Metacritic score']
            st.metric("⭐ Metacritic", f"{meta_score:.0f}" if meta_score > 0 else "N/A")
        
        with col_kpi3:
            total_rev = game_data['Positive'] + game_data['Negative']
            pos_pct = (game_data['Positive'] / total_rev * 100) if total_rev > 0 else 0
            st.metric("👍 % Positif", f"{pos_pct:.0f}%")
        
        with col_kpi4:
            st.metric("🔥 Peak CCU", f"{game_data['Peak CCU']:,}")
        
        with col_kpi5:
            playtime_h = game_data['Median playtime forever'] / 60
            st.metric("⏱️ Playtime", f"{playtime_h:.1f}h")
        
        st.divider()
        
        # Informations détaillées
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("#### 📋 Informations Générales")
            
            info_data = {
                "Année de Sortie": int(game_data['Release Year']),
                "Prix": f"${game_data['Price']:.2f}",
                "Développeur": game_data.get('Developers', 'N/A'),
                "Éditeur": game_data.get('Publishers', 'N/A'),
                "Genres": ", ".join(game_data['Genres']) if isinstance(game_data['Genres'], list) else "N/A"
            }
            
            for key, val in info_data.items():
                st.markdown(f"**{key}:** {val}")
        
        with col_info2:
            st.markdown("#### 💻 Support Plateforme")
            
            platforms = []
            if game_data.get('Windows', False):
                platforms.append("🪟 Windows")
            if game_data.get('Mac', False):
                platforms.append("🍎 Mac")
            if game_data.get('Linux', False):
                platforms.append("🐧 Linux")
            
            if platforms:
                for plat in platforms:
                    st.markdown(f"✅ {plat}")
            else:
                st.markdown("N/A")
        
        st.divider()
        
        # Tags populaires
        if isinstance(game_data.get('Tags'), list) and len(game_data['Tags']) > 0:
            st.markdown("#### 🏷️ Tags Populaires")
            
            tags_display = game_data['Tags'][:10]  # Top 10 tags
            st.markdown(" • ".join([f"`{tag}`" for tag in tags_display]))
        
        st.divider()
        
        # Positionnement vs Genre
        st.markdown("#### 📊 Positionnement vs Moyenne du Genre")
        
        if isinstance(game_data['Genres'], list) and len(game_data['Genres']) > 0:
            main_genre = game_data['Genres'][0]
            
            # Calculer moyennes du genre
            genre_games = df_full[
                df_full['Genres'].apply(
                    lambda x: isinstance(x, list) and main_genre in x
                )
            ]
            
            if len(genre_games) > 1:
                col_pos1, col_pos2, col_pos3 = st.columns(3)
                
                with col_pos1:
                    genre_avg_price = genre_games['Price'].mean()
                    delta_price = game_data['Price'] - genre_avg_price
                    
                    st.metric(
                        f"Prix vs {main_genre}",
                        f"${game_data['Price']:.2f}",
                        delta=f"${delta_price:+.2f}",
                        delta_color="inverse"
                    )
                
                with col_pos2:
                    genre_avg_meta = genre_games[genre_games['Metacritic score'] > 0]['Metacritic score'].mean()
                    delta_meta = game_data['Metacritic score'] - genre_avg_meta if game_data['Metacritic score'] > 0 else 0
                    
                    st.metric(
                        f"Metacritic vs {main_genre}",
                        f"{game_data['Metacritic score']:.0f}" if game_data['Metacritic score'] > 0 else "N/A",
                        delta=f"{delta_meta:+.1f}" if game_data['Metacritic score'] > 0 else None
                    )
                
                with col_pos3:
                    genre_avg_playtime = genre_games['Median playtime forever'].mean() / 60
                    delta_playtime = playtime_h - genre_avg_playtime
                    
                    st.metric(
                        f"Playtime vs {main_genre}",
                        f"{playtime_h:.1f}h",
                        delta=f"{delta_playtime:+.1f}h"
                    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🔍 **GameData360 — Data Browser** | "
    f"Exploration sur {len(df_full):,} jeux | {excluded_count:,} logiciels exclus"
)
