# -*- coding: utf-8 -*-
"""
GameData360 - Page Comportement Joueurs
========================================
Analyse du comportement des joueurs : rétention, engagement, dépenses.

Auteur: GameData360 Team
Version: 2.0 (Refactorisée)
"""

import streamlit as st
import pandas as pd
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
    TAB_CONFIG,
    FILE_PATH
)
from utils.data_helpers import (
    load_game_data,
    get_unique_values,
    apply_all_filters,
    explode_genres,
    categorize_game_mode,
    calculate_genre_stats,
    format_number
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — Comportement Joueurs",
    page_icon="🎮",
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
    
    /* Success message stylisé */
    .stSuccess {
        background: linear-gradient(90deg, rgba(0,255,136,0.2), transparent);
        border-left: 4px solid #00ff88;
    }
    
    /* Dataframes stylisés */
    .stDataFrame {
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("# 🎮 COMPORTEMENT JOUEURS")
st.markdown("##### Analyse détaillée de l'engagement, des recommandations et du temps de jeu")

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
    st.markdown("## 🎯 Filtres")
    
    # Extraction des valeurs uniques (cachée)
    unique_genres = get_unique_values(df_analyse, "Genres")
    unique_cats = get_unique_values(df_analyse, "Categories")
    unique_tags = get_unique_values(df_analyse, "Tags")
    
    # Filtres avec style
    selected_genres = st.multiselect(
        "🎮 Genres",
        unique_genres,
        help="Sélectionnez un ou plusieurs genres"
    )
    
    selected_categories = st.multiselect(
        "📂 Catégories",
        unique_cats,
        help="Solo, Multi, Co-op, etc."
    )
    
    selected_tags = st.multiselect(
        "🏷️ Tags",
        unique_tags,
        help="Tags descriptifs des jeux"
    )
    
    # Bouton Reset stylisé
    if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
        st.rerun()
    
    st.divider()
    
    # Info dataset
    st.markdown("### 📊 Dataset")
    st.caption(f"**{len(df_analyse):,}** jeux dans la base")

# ============================================================
# 4. APPLICATION DES FILTRES (VECTORISÉ)
# ============================================================
df_filtered = apply_all_filters(
    df_analyse,
    genres=selected_genres,
    categories=selected_categories,
    tags=selected_tags
)

# Indicateur de filtrage
col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    if len(df_filtered) < len(df_analyse):
        st.success(
            f"🎯 **{len(df_filtered):,}** jeux correspondent aux filtres "
            f"(sur {len(df_analyse):,} total)"
        )
    else:
        st.info(f"📊 **{len(df_analyse):,}** jeux affichés (aucun filtre)")

# ============================================================
# 5. ONGLETS PRINCIPAUX
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    TAB_CONFIG["vue_ensemble"],
    TAB_CONFIG["recommandations"],
    TAB_CONFIG["temps_jeu"],
    TAB_CONFIG["solo_multi"]
])

# ============================================================
# TAB 1: VUE D'ENSEMBLE
# ============================================================
with tab1:
    st.markdown("### 📈 Indicateurs Clés de Performance")
    
    # KPIs en 4 colonnes
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            "🎮 Nombre de Jeux",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df_analyse):,}" if selected_genres or selected_categories or selected_tags else None,
            help="Nombre total de jeux après filtrage"
        )
    
    with kpi2:
        if "Estimated revenue" in df_filtered.columns:
            total_revenue = df_filtered["Estimated revenue"].sum()
            st.metric(
                "💰 Revenu Total Estimé",
                format_number(total_revenue),
                help="Somme des revenus estimés"
            )
        else:
            st.metric("💰 Revenu", "N/A")
    
    with kpi3:
        if "Price" in df_filtered.columns:
            avg_price = df_filtered["Price"].mean()
            st.metric(
                "💵 Prix Moyen",
                f"${avg_price:.2f}",
                help="Prix moyen des jeux"
            )
    
    with kpi4:
        if "Price" in df_filtered.columns:
            median_price = df_filtered["Price"].median()
            st.metric(
                "📊 Prix Médian",
                f"${median_price:.2f}",
                help="Prix médian (50% des jeux sont en dessous)"
            )
    
    st.divider()
    
    # Graphiques Vue d'ensemble
    col_overview1, col_overview2 = st.columns(2)
    
    with col_overview1:
        st.markdown("#### 🏆 Top 10 Genres par Nombre de Jeux")
        
        # Explosion une seule fois, mise en cache
        df_genres = explode_genres(df_filtered)
        genre_counts = df_genres["Genres"].value_counts().head(10).reset_index()
        genre_counts.columns = ["Genre", "Nombre"]
        
        fig_genres = px.bar(
            genre_counts,
            x="Nombre",
            y="Genre",
            orientation="h",
            color="Nombre",
            color_continuous_scale=[[0, COLORS["chart"][0]], [1, COLORS["chart"][1]]],
        )
        fig_genres.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            coloraxis_showscale=False,
            height=400
        )
        fig_genres.update_yaxes(categoryorder="total ascending")
        fig_genres.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x:,} jeux<extra></extra>"
        )
        st.plotly_chart(fig_genres, use_container_width=True)
    
    with col_overview2:
        st.markdown("#### 🔥 Top 10 Genres par Peak CCU")
        
        if "Peak CCU" in df_filtered.columns:
            genre_ccu = calculate_genre_stats(df_filtered, "Peak CCU", "sum", 10)
            
            fig_ccu = px.bar(
                genre_ccu,
                x="Peak CCU",
                y="Genre",
                orientation="h",
                color="Peak CCU",
                color_continuous_scale=[[0, COLORS["chart"][2]], [1, COLORS["chart"][3]]],
            )
            fig_ccu.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=False,
                coloraxis_showscale=False,
                height=400
            )
            fig_ccu.update_yaxes(categoryorder="total ascending")
            fig_ccu.update_traces(
                hovertemplate="<b>%{y}</b><br>Peak CCU: %{x:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig_ccu, use_container_width=True)
        else:
            st.warning("⚠️ Colonne 'Peak CCU' non disponible")

# ============================================================
# TAB 2: RECOMMANDATIONS & POPULARITÉ
# ============================================================
with tab2:
    st.markdown("### 🏆 Analyse des Recommandations")
    st.caption("Les jeux les plus recommandés par la communauté Steam")
    
    col_rec1, col_rec2 = st.columns([2, 1])
    
    with col_rec1:
        st.markdown("#### Top 15 Jeux les Plus Recommandés")
        
        top_recommended = df_filtered.nlargest(15, "Recommendations")[
            ["Name", "Recommendations", "Estimated revenue", "Peak CCU"]
        ].copy()
        
        # Formatage pour le hover
        top_recommended["Revenue_fmt"] = top_recommended["Estimated revenue"].apply(
            lambda x: format_number(x) if pd.notna(x) else "N/A"
        )
        
        fig_rec = px.bar(
            top_recommended,
            x="Recommendations",
            y="Name",
            orientation="h",
            color="Recommendations",
            color_continuous_scale=[
                [0, COLORS["primary"]],
                [0.5, COLORS["tertiary"]],
                [1, COLORS["secondary"]]
            ],
            custom_data=["Revenue_fmt", "Peak CCU"]
        )
        fig_rec.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            coloraxis_showscale=False,
            height=500
        )
        fig_rec.update_yaxes(categoryorder="total ascending")
        fig_rec.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "📊 Recommandations: %{x:,.0f}<br>"
                "💰 Revenu: %{customdata[0]}<br>"
                "👥 Peak CCU: %{customdata[1]:,.0f}"
                "<extra></extra>"
            )
        )
        st.plotly_chart(fig_rec, use_container_width=True)
    
    with col_rec2:
        st.markdown("#### 📋 Détails")
        
        # Tableau interactif
        display_df = top_recommended[["Name", "Recommendations"]].copy()
        display_df["Recommendations"] = display_df["Recommendations"].apply(lambda x: f"{x:,}")
        display_df.columns = ["Jeu", "Recommandations"]
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=460
        )
    
    st.divider()
    
    # Corrélation Recommandations vs Revenu
    st.markdown("#### 📈 Corrélation Recommandations vs Revenu")
    
    if "Estimated revenue" in df_filtered.columns:
        # Filtrer les valeurs valides
        df_scatter = df_filtered[
            (df_filtered["Recommendations"] > 0) & 
            (df_filtered["Estimated revenue"] > 0)
        ].copy()
        
        fig_scatter = px.scatter(
            df_scatter.nlargest(200, "Recommendations"),
            x="Recommendations",
            y="Estimated revenue",
            size="Peak CCU" if "Peak CCU" in df_scatter.columns else None,
            color="Price",
            hover_name="Name",
            log_x=True,
            log_y=True,
            color_continuous_scale="viridis",
            opacity=0.7
        )
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis_title="Recommandations (log)",
            yaxis_title="Revenu estimé (log)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Insight textuel
        correlation = df_scatter["Recommendations"].corr(df_scatter["Estimated revenue"])
        st.info(
            f"📊 **Insight**: Corrélation de **{correlation:.2%}** entre les recommandations "
            f"et le revenu. Les jeux populaires génèrent généralement plus de revenus."
        )

# ============================================================
# TAB 3: TEMPS DE JEU & ENGAGEMENT
# ============================================================
with tab3:
    st.markdown("### ⏱️ Analyse du Temps de Jeu")
    st.caption("Mesure de l'engagement des joueurs par genre")
    
    # Filtrage des jeux avec temps de jeu > 0
    col_ref = "Median playtime forever"
    df_active = df_filtered[df_filtered[col_ref] > 0].copy()
    
    if len(df_active) == 0:
        st.warning("⚠️ Aucun jeu avec temps de jeu enregistré dans la sélection")
    else:
        # KPIs Engagement
        kpi_eng1, kpi_eng2, kpi_eng3 = st.columns(3)
        
        global_median = df_active[col_ref].median()
        global_mean = df_active[col_ref].mean()
        
        with kpi_eng1:
            st.metric(
                "⏱️ Temps Médian Global",
                f"{global_median:.0f} min",
                help="Temps de jeu médian de tous les jeux actifs"
            )
        
        with kpi_eng2:
            st.metric(
                "📊 Temps Moyen Global",
                f"{global_mean:.0f} min",
                delta=f"+{global_mean - global_median:.0f} vs médiane"
            )
        
        with kpi_eng3:
            st.metric(
                "🎮 Jeux Actifs",
                f"{len(df_active):,}",
                help="Jeux avec au moins 1 minute jouée"
            )
        
        st.divider()
        
        col_pt1, col_pt2 = st.columns(2)
        
        with col_pt1:
            st.markdown("#### 🎯 Top 15 Genres par Temps de Jeu Médian")
            
            df_genre_time = calculate_genre_stats(df_active, col_ref, "median", 15)
            df_genre_time.columns = ["Genre", "Minutes"]
            
            fig_time = px.bar(
                df_genre_time,
                x="Minutes",
                y="Genre",
                orientation="h",
                color="Minutes",
                color_continuous_scale=[[0, COLORS["warning"]], [1, COLORS["danger"]]],
            )
            fig_time.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=False,
                coloraxis_showscale=False,
                height=450
            )
            fig_time.update_yaxes(categoryorder="total ascending")
            fig_time.update_traces(
                hovertemplate="<b>%{y}</b><br>Temps médian: %{x:.0f} min<extra></extra>"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_pt2:
            st.markdown("#### 📊 Performance Genre vs Marché")
            
            # Benchmark comparatif
            df_benchmark = calculate_genre_stats(df_active, col_ref, "median", 15)
            df_benchmark.columns = ["Genre", "Genre_Median"]
            df_benchmark["Global_Median"] = global_median
            df_benchmark["Delta"] = df_benchmark["Genre_Median"] - global_median
            df_benchmark["Au_dessus"] = df_benchmark["Delta"] > 0
            
            fig_benchmark = go.Figure()
            
            # Barres pour chaque genre
            fig_benchmark.add_trace(go.Bar(
                x=df_benchmark["Genre_Median"],
                y=df_benchmark["Genre"],
                orientation="h",
                marker_color=[
                    COLORS["primary"] if x else COLORS["danger"] 
                    for x in df_benchmark["Au_dessus"]
                ],
                opacity=0.8,
                name="Médiane Genre",
                hovertemplate="<b>%{y}</b><br>Médiane: %{x:.0f} min<extra></extra>"
            ))
            
            # Ligne de référence globale
            fig_benchmark.add_vline(
                x=global_median,
                line_dash="dash",
                line_color=COLORS["benchmark"],
                line_width=3,
                annotation_text=f"Marché: {global_median:.0f} min",
                annotation_position="top"
            )
            
            fig_benchmark.update_layout(
                **PLOTLY_LAYOUT,
                height=450,
                showlegend=False
            )
            fig_benchmark.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_benchmark, use_container_width=True)
        
        # Insight textuel
        top_genre = df_genre_time.iloc[0]["Genre"]
        top_time = df_genre_time.iloc[0]["Minutes"]
        st.success(
            f"🏆 **Top Engagement**: Le genre **{top_genre}** domine avec un temps "
            f"de jeu médian de **{top_time:.0f} minutes** — "
            f"soit **{(top_time/global_median - 1)*100:.0f}%** au-dessus de la médiane globale."
        )

# ============================================================
# TAB 4: SOLO VS MULTIJOUEUR
# ============================================================
with tab4:
    st.markdown("### 🎯 Engagement : Solo vs Multijoueur")
    st.caption("Comparaison des patterns d'engagement selon le mode de jeu")
    
    # Classification des jeux
    df_mode = df_filtered.copy()
    df_mode["Mode"] = df_mode["Categories"].apply(categorize_game_mode)
    
    # Filtrage Solo / Multi avec temps de jeu > 0
    df_mode = df_mode[
        (df_mode["Mode"].isin(["Solo", "Multijoueur / Co-op"])) &
        (df_mode["Median playtime forever"] > 0)
    ].copy()
    
    # Conversion en heures
    df_mode["Playtime_Hours"] = df_mode["Median playtime forever"] / 60
    
    if len(df_mode) == 0:
        st.warning("⚠️ Pas assez de données pour la comparaison Solo/Multi")
    else:
        # KPIs comparatifs
        kpi_solo, kpi_multi, kpi_diff = st.columns(3)
        
        median_solo = df_mode[df_mode["Mode"] == "Solo"]["Playtime_Hours"].median()
        median_multi = df_mode[df_mode["Mode"] == "Multijoueur / Co-op"]["Playtime_Hours"].median()
        delta = median_multi - median_solo
        
        with kpi_solo:
            count_solo = len(df_mode[df_mode["Mode"] == "Solo"])
            st.metric(
                "🎮 Solo",
                f"{median_solo:.1f}h",
                delta=f"{count_solo:,} jeux",
                delta_color="off"
            )
        
        with kpi_multi:
            count_multi = len(df_mode[df_mode["Mode"] == "Multijoueur / Co-op"])
            st.metric(
                "👥 Multijoueur",
                f"{median_multi:.1f}h",
                delta=f"{count_multi:,} jeux",
                delta_color="off"
            )
        
        with kpi_diff:
            st.metric(
                "📊 Différence",
                f"{abs(delta):.1f}h",
                delta=f"{'Multi' if delta > 0 else 'Solo'} gagne",
                delta_color="normal" if delta > 0 else "inverse"
            )
        
        st.divider()
        
        col_mode1, col_mode2 = st.columns(2)
        
        with col_mode1:
            st.markdown("#### 📊 Distribution du Temps de Jeu")
            
            fig_box = px.box(
                df_mode,
                x="Mode",
                y="Playtime_Hours",
                color="Mode",
                color_discrete_map={
                    "Solo": COLORS["solo"],
                    "Multijoueur / Co-op": COLORS["multi"]
                },
                log_y=True,
                points="outliers"
            )
            fig_box.update_layout(
                **PLOTLY_LAYOUT,
                height=400,
                showlegend=False,
                yaxis_title="Heures (échelle log)",
                xaxis_title=""
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col_mode2:
            st.markdown("#### 🎯 Répartition des Jeux")
            
            mode_counts = df_mode["Mode"].value_counts().reset_index()
            mode_counts.columns = ["Mode", "Count"]
            
            fig_pie = px.pie(
                mode_counts,
                values="Count",
                names="Mode",
                color="Mode",
                color_discrete_map={
                    "Solo": COLORS["solo"],
                    "Multijoueur / Co-op": COLORS["multi"]
                },
                hole=0.4
            )
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+value",
                hovertemplate="<b>%{label}</b><br>%{value:,} jeux<br>%{percent}<extra></extra>"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Strip plot avec jitter
        st.markdown("#### 🔬 Vue Détaillée : Densité des Temps de Jeu")
        
        # Échantillonnage pour performance
        sample_size = min(500, len(df_mode))
        df_sample = df_mode.sample(n=sample_size, random_state=42)
        
        fig_strip = px.strip(
            df_sample,
            x="Playtime_Hours",
            y="Mode",
            color="Mode",
            color_discrete_map={
                "Solo": COLORS["solo"],
                "Multijoueur / Co-op": COLORS["multi"]
            },
            hover_name="Name",
            log_x=True,
            stripmode="overlay"
        )
        fig_strip.update_traces(
            marker=dict(size=8, opacity=0.6),
            jitter=0.4
        )
        fig_strip.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            showlegend=False,
            xaxis_title="Temps de Jeu (heures, échelle log)",
            yaxis_title=""
        )
        st.plotly_chart(fig_strip, use_container_width=True)
        
        # Insight final
        winner = "Multijoueur" if delta > 0 else "Solo"
        st.info(
            f"🎮 **Insight**: Les jeux **{winner}** génèrent en moyenne "
            f"**{abs(delta):.1f} heures** de jeu supplémentaires. "
            f"L'engagement social stimule la rétention des joueurs."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🎮 **GameData360** — Analyse des comportements joueurs | "
    f"Dataset: {len(df_analyse):,} jeux"
)