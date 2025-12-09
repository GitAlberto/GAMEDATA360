# -*- coding: utf-8 -*-
"""
GameData360 - Page Genres & Tags Insights
==========================================
Analyse des genres, tags et leurs combinaisons gagnantes.
Insights: Combinations High-Quality, Tags Émergents, Genre Mix Optimal.

Auteur: GameData360 Team
Version: 3.0 (Combinatorial Intelligence Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
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
    page_title="GameData360 — Genres & Tags",
    page_icon="🎮",
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
st.markdown("# 🎮 GENRES & TAGS INSIGHTS")
st.markdown("##### Analyse des combinaisons, co-occurrences & formules gagnantes")

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
# 3. TOP-LEVEL KPIs
# ============================================================
st.markdown("### 📊 Statistiques Globales")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1: Nombre de Genres Uniques
df_genres_exploded = df_analyse.explode("Genres").dropna(subset=["Genres"])
unique_genres_count = df_genres_exploded["Genres"].nunique()

with kpi1:
    st.metric(
        "🎯 Genres Uniques",
        unique_genres_count,
        help="Nombre de genres différents"
    )

# KPI 2: Nombre de Tags Uniques
df_tags_exploded = df_analyse.explode("Tags").dropna(subset=["Tags"])
unique_tags_count = df_tags_exploded["Tags"].nunique()

with kpi2:
    st.metric(
        "🏷️ Tags Uniques",
        unique_tags_count,
        help="Nombre de tags différents"
    )

# KPI 3: Genre Dominant
dominant_genre = df_genres_exploded["Genres"].value_counts().index[0] if len(df_genres_exploded) > 0 else "N/A"
dominant_genre_count = df_genres_exploded["Genres"].value_counts().iloc[0] if len(df_genres_exploded) > 0 else 0

with kpi3:
    st.metric(
        "👑 Genre Dominant",
        dominant_genre,
        delta=f"{dominant_genre_count:,} jeux",
        delta_color="off",
        help="Genre le plus répandu"
    )

# KPI 4: Tag Dominant
dominant_tag = df_tags_exploded["Tags"].value_counts().index[0] if len(df_tags_exploded) > 0 else "N/A"

with kpi4:
    st.metric(
        "🔥 Tag Dominant",
        dominant_tag,
        help="Tag le plus utilisé"
    )

st.divider()

# ============================================================
# 4. INSIGHTS AUTOMATIQUES
# ============================================================
st.markdown("### 🎯 Insights Automatiques")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    # Insight 1: Genre Mix Optimal (meilleur Metacritic moyen)
    df_multi_genre = df_analyse[df_analyse["Genres"].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
    
    if len(df_multi_genre) > 0 and len(df_analyse[df_analyse["Metacritic score"] > 0]) > 0:
        avg_meta_multi = df_multi_genre[df_multi_genre["Metacritic score"] > 0]["Metacritic score"].mean()
        avg_meta_single = df_analyse[
            (df_analyse["Genres"].apply(lambda x: len(x) == 1 if isinstance(x, list) else False)) &
            (df_analyse["Metacritic score"] > 0)
        ]["Metacritic score"].mean()
        
        if avg_meta_multi > avg_meta_single + 3:
            st.success(
                f"🎮 **Genre Mix Gagnant**: Les jeux multi-genres ont un Metacritic moyen de "
                f"{avg_meta_multi:.1f} vs {avg_meta_single:.1f} pour mono-genre (+{avg_meta_multi - avg_meta_single:.1f} pts)"
            )
    
    # Insight 2: Combinaison Genre Gagnante
    @st.cache_data
    def find_winning_genre_combo(_df):
        genre_pairs = []
        
        for genres_list in _df["Genres"]:
            if isinstance(genres_list, list) and len(genres_list) >= 2:
                for g1, g2 in combinations(sorted(genres_list), 2):
                    genre_pairs.append((g1, g2))
        
        pair_counts = pd.Series(genre_pairs).value_counts()
        return pair_counts.head(5)
    
    top_combos = find_winning_genre_combo(df_analyse)
    
    if len(top_combos) > 0:
        best_combo = top_combos.index[0]
        combo_count = top_combos.iloc[0]
        
        st.info(
            f"💎 **Combinaison Populaire**: **{best_combo[0]} + {best_combo[1]}** "
            f"apparaît dans {combo_count:,} jeux"
        )

with col_insight2:
    # Insight 3: Tags Haute Qualité
    df_tags_quality = df_tags_exploded[df_tags_exploded["Metacritic score"] > 0].copy()
    
    if len(df_tags_quality) > 0:
        tag_meta_avg = df_tags_quality.groupby("Tags")["Metacritic score"].agg(['mean', 'count']).reset_index()
        tag_meta_avg = tag_meta_avg[tag_meta_avg['count'] >= 10]  # Au moins 10 jeux
        
        if len(tag_meta_avg) > 0:
            best_quality_tag = tag_meta_avg.loc[tag_meta_avg['mean'].idxmax()]
            
            st.success(
                f"⭐ **Tag Qualité**: Le tag **{best_quality_tag['Tags']}** a le plus haut "
                f"Metacritic moyen ({best_quality_tag['mean']:.1f}) sur {int(best_quality_tag['count'])} jeux"
            )
    
    # Insight 4: Tags Émergents (forte croissance)
    df_recent = df_analyse[df_analyse["Release Year"] >= 2020]
    df_old = df_analyse[df_analyse["Release Year"] < 2020]
    
    if len(df_recent) > 0 and len(df_old) > 0:
        recent_tags = df_recent.explode("Tags")["Tags"].value_counts()
        old_tags = df_old.explode("Tags")["Tags"].value_counts()
        
        # Normaliser par volume total
        recent_pct = (recent_tags / len(df_recent)) * 100
        old_pct = (old_tags / len(df_old)) * 100
        
        # Calculer croissance
        growth = recent_pct - old_pct
        growth = growth[growth > 2]  # Croissance >2 points de %
        
        if len(growth) > 0:
            emerging_tag = growth.index[0]
            growth_val = growth.iloc[0]
            
            st.warning(
                f"🚀 **Tag Émergent**: **{emerging_tag}** a crû de "
                f"{growth_val:.1f} points de % depuis 2020 — tendance en hausse"
            )

st.divider()

# ============================================================
# 5. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Top Genres & Tags",
    "💎 Combinaisons Gagnantes",
    "🔗 Co-occurrence Network",
    "📊 Qualité par Genre/Tag"
])

# ============================================================
# TAB 1: TOP GENRES & TAGS
# ============================================================
with tab1:
    col_top1, col_top2 = st.columns(2)
    
    with col_top1:
        st.markdown("### 🎯 Top 15 Genres (Volume)")
        
        genre_counts = df_genres_exploded["Genres"].value_counts().head(15).reset_index()
        genre_counts.columns = ["Genre", "Count"]
        
        fig_top_genres = px.bar(
            genre_counts,
            x="Count",
            y="Genre",
            orientation="h",
            color="Count",
            color_continuous_scale=[[0, COLORS['chart'][0]], [1, COLORS['chart'][4]]]
        )
        
        fig_top_genres.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        
        st.plotly_chart(fig_top_genres, use_container_width=True)
    
    with col_top2:
        st.markdown("### 🏷️ Top 15 Tags (Volume)")
        
        tag_counts = df_tags_exploded["Tags"].value_counts().head(15).reset_index()
        tag_counts.columns = ["Tag", "Count"]
        
        fig_top_tags = px.bar(
            tag_counts,
            x="Count",
            y="Tag",
            orientation="h",
            color="Count",
            color_continuous_scale=[[0, COLORS['chart'][1]], [1, COLORS['chart'][5]]]
        )
        
        fig_top_tags.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        
        st.plotly_chart(fig_top_tags, use_container_width=True)
    
    st.divider()
    
    # Tags par Engagement
    st.markdown("### ⏱️ Top Tags par Engagement (Playtime Médian)")
    
    df_playtime = df_tags_exploded[df_tags_exploded["Median playtime forever"] > 0].copy()
    tag_playtime = df_playtime.groupby("Tags")["Median playtime forever"].agg(['median', 'count']).reset_index()
    tag_playtime = tag_playtime[tag_playtime['count'] >= 10]  # Au moins 10 jeux
    tag_playtime = tag_playtime.nlargest(15, "median")
    tag_playtime["Heures"] = (tag_playtime["median"] / 60).round(1)
    tag_playtime.columns = ["Tag", "Median", "Count", "Heures"]
    
    fig_playtime = px.bar(
        tag_playtime,
        x="Heures",
        y="Tag",
        orientation="h",
        color="Heures",
        color_continuous_scale=[[0, COLORS['warning']], [1, COLORS['danger']]]
    )
    
    fig_playtime.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending")
    )
    
    st.plotly_chart(fig_playtime, use_container_width=True)

# ============================================================
# TAB 2: COMBINAISONS GAGNANTES
# ============================================================
with tab2:
    st.markdown("### 💎 Top Combinaisons de Genres")
    
    # Table top combos
    top_10_combos = top_combos.head(10).reset_index()
    top_10_combos.columns = ["Combinaison", "Nombre de jeux"]
    top_10_combos["Genre 1"] = top_10_combos["Combinaison"].apply(lambda x: x[0])
    top_10_combos["Genre 2"] = top_10_combos["Combinaison"].apply(lambda x: x[1])
    
    display_combos = top_10_combos[["Genre 1", "Genre 2", "Nombre de jeux"]]
    
    st.dataframe(display_combos, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Qualité par combinaison
    st.markdown("### ⭐ Meilleures Combinaisons (Metacritic Moyen)")
    
    combo_quality = []
    
    for combo, count in top_combos.head(15).items():
        if count >= 5:  # Au moins 5 jeux
            combo_games = df_analyse[
                df_analyse["Genres"].apply(
                    lambda x: isinstance(x, list) and combo[0] in x and combo[1] in x
                )
            ]
            
            combo_meta = combo_games[combo_games["Metacritic score"] > 0]["Metacritic score"].mean()
            
            if not np.isnan(combo_meta):
                combo_quality.append({
                    "Combinaison": f"{combo[0]} + {combo[1]}",
                    "Metacritic Moyen": round(combo_meta, 1),
                    "Nb Jeux": count
                })
    
    if combo_quality:
        df_combo_quality = pd.DataFrame(combo_quality).sort_values("Metacritic Moyen", ascending=False)
        
        fig_combo_quality = px.bar(
            df_combo_quality.head(10),
            x="Metacritic Moyen",
            y="Combinaison",
            orientation="h",
            color="Metacritic Moyen",
            color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]]
        )
        
        fig_combo_quality.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        
        st.plotly_chart(fig_combo_quality, use_container_width=True)

# ============================================================
# TAB 3: CO-OCCURRENCE NETWORK
# ============================================================
with tab3:
    st.markdown("### 🔗 Co-occurrence des Tags (Top 30)")
    
    # Top 30 tags
    all_tags = [t for sublist in df_analyse["Tags"].dropna() for t in sublist if isinstance(sublist, list)]
    top_tags = pd.Series(all_tags).value_counts().head(30).index.tolist()
    
    # Calculer co-occurrences
    @st.cache_data
    def calculate_cooccurrence(_df, allowed_tags):
        pair_counts = {}
        
        for tags in _df["Tags"].dropna():
            if isinstance(tags, list):
                relevant = sorted([t for t in tags if t in allowed_tags])
                for t1, t2 in combinations(relevant, 2):
                    pair = (t1, t2)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        return pair_counts
    
    pairs = calculate_cooccurrence(df_analyse, top_tags)
    df_pairs = pd.DataFrame([(k[0], k[1], v) for k, v in pairs.items()], columns=['Tag1', 'Tag2', 'Count'])
    df_pairs = df_pairs.nlargest(100, 'Count')  # Top 100 paires
    
    if len(df_pairs) > 0:
        # Heatmap interactive
        fig_cooccur = px.density_heatmap(
            df_pairs,
            x="Tag1",
            y="Tag2",
            z="Count",
            color_continuous_scale=[[0, COLORS['background']], [1, COLORS['primary']]]
        )
        
        fig_cooccur.update_layout(
            **PLOTLY_LAYOUT,
            height=600,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_cooccur, use_container_width=True)
        
        # Table top paires
        st.markdown("#### Top 10 Paires de Tags")
        
        top_pairs_display = df_pairs.head(10).copy()
        st.dataframe(top_pairs_display, hide_index=True, use_container_width=True)

# ============================================================
# TAB 4: QUALITÉ PAR GENRE/TAG
# ============================================================
with tab4:
    st.markdown("### ⭐ Qualité Moyenne par Genre (Metacritic)")
    
    df_genre_quality = df_genres_exploded[df_genres_exploded["Metacritic score"] > 0].copy()
    
    genre_meta = df_genre_quality.groupby("Genres")["Metacritic score"].agg(['mean', 'median', 'count']).reset_index()
    genre_meta = genre_meta[genre_meta['count'] >= 10]  # Au moins 10 jeux
    genre_meta = genre_meta.sort_values("mean", ascending=False).head(15)
    
    genre_meta["mean"] = genre_meta["mean"].round(1)
    genre_meta["median"] = genre_meta["median"].round(1)
    genre_meta.columns = ["Genre", "Moyenne", "Médiane", "Nb Jeux"]
    
    fig_genre_quality = px.bar(
        genre_meta,
        x="Moyenne",
        y="Genre",
        orientation="h",
        color="Moyenne",
        color_continuous_scale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['primary']]]
    )
    
    fig_genre_quality.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending")
    )
    
    st.plotly_chart(fig_genre_quality, use_container_width=True)
    
    st.divider()
    
    # Table détaillée
    st.markdown("#### 📊 Statistiques Détaillées")
    
    st.dataframe(genre_meta, hide_index=True, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🎮 **GameData360 — Genres & Tags Insights** | "
    f"{unique_genres_count} genres, {unique_tags_count} tags analysés"
)
