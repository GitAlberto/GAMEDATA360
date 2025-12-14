# -*- coding: utf-8 -*-
"""
GameData360 - Page Published vs Beta
=====================================
Analyse comparative entre jeux publiés et jeux en Early Access/Beta.
Insights: Pricing, Qualité, Engagement, Success Rate.

Auteur: GameData360 Team
Version: 3.0 (Comparative Analysis Edition)
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
    page_title="GameData360 — Published vs Beta",
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
st.markdown("# 🎯 PUBLISHED vs EARLY ACCESS")
st.markdown("##### Analyse comparative : Stratégies, Performance & Market Dynamics")

# ============================================================
# 2. CHARGEMENT DES DONNÉES (CACHE OPTIMISÉ)
# ============================================================
@st.cache_data(show_spinner=False)
def load_published_data():
    """Charge les jeux publiés et filtre les logiciels."""
    df = load_game_data(str(FILE_PATH))
    
    # Filtrage des genres non-jeux
    def is_game(genres_list):
        if not isinstance(genres_list, list):
            return True
        genres_lower = [g.lower() for g in genres_list]
        return not any(genre in genres_lower for genre in NON_GAME_GENRES)
    
    initial_count = len(df)
    df = df[df["Genres"].apply(is_game)].copy()
    excluded_count = initial_count - len(df)
    
    df['Status'] = 'Published'
    return df, excluded_count

@st.cache_data(show_spinner=False)
def load_beta_data():
    """Charge les jeux en Early Access/Beta et filtre les logiciels."""
    beta_path = FILE_PATH.parent / "jeux_beta.csv"
    
    try:
        df = pd.read_csv(beta_path)
        
        # Parsing des colonnes listes (format CSV: "Action,Indie")
        for col in ["Genres", "Categories", "Tags"]:
            if col in df.columns:
                df[col] = df[col].fillna("").apply(
                    lambda x: [g.strip() for g in x.split(',')] if x else []
                )
        
        # Filtrage des genres non-jeux
        def is_game(genres_list):
            if not isinstance(genres_list, list):
                return True
            genres_lower = [g.lower() for g in genres_list]
            return not any(genre in genres_lower for genre in NON_GAME_GENRES)
        
        initial_count = len(df)
        df = df[df["Genres"].apply(is_game)].copy()
        excluded_count = initial_count - len(df)
        
        df['Status'] = 'Early Access'
        return df, excluded_count
    
    except FileNotFoundError:
        st.error(f"❌ Fichier beta introuvable : {beta_path}")
        st.stop()

# Chargement avec indicateurs
try:
    with st.spinner('⚡ Chargement des données Published...'):
        df_published, excluded_pub = load_published_data()
    
    with st.spinner('⚡ Chargement des données Early Access...'):
        df_beta, excluded_beta = load_beta_data()
    
    # Message d'info
    total_excluded = excluded_pub + excluded_beta
    if total_excluded > 0:
        st.sidebar.success(f"🎮 {total_excluded:,} logiciels exclus au total")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. SIDEBAR - FILTRES & TOGGLE
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Filtres & Options")
    
    # Toggle Published / Beta / Both
    st.markdown("### 🔀 Affichage")
    display_mode = st.radio(
        "Données à afficher",
        ["Les deux", "Published seulement", "Early Access seulement"],
        help="Filtrer par statut de publication"
    )
    
    st.divider()
    
    # Info dataset
    st.markdown("### 📊 Statistiques")
    st.caption(f"**Published:** {len(df_published):,} jeux")
    st.caption(f"**Early Access:** {len(df_beta):,} jeux")
    st.caption(f"**Total:** {len(df_published) + len(df_beta):,} jeux")

# ============================================================
# 4. APPLICATION DU FILTRE DISPLAY MODE
# ============================================================
if display_mode == "Published seulement":
    df_filtered_pub = df_published
    df_filtered_beta = pd.DataFrame()  # Vide
elif display_mode == "Early Access seulement":
    df_filtered_pub = pd.DataFrame()  # Vide
    df_filtered_beta = df_beta
else:
    df_filtered_pub = df_published
    df_filtered_beta = df_beta

# Fusion pour analyses combinées
if not df_filtered_pub.empty and not df_filtered_beta.empty:
    # Colonnes communes
    common_cols = list(set(df_filtered_pub.columns) & set(df_filtered_beta.columns))
    df_combined = pd.concat([
        df_filtered_pub[common_cols],
        df_filtered_beta[common_cols]
    ], ignore_index=True)
elif not df_filtered_pub.empty:
    df_combined = df_filtered_pub
elif not df_filtered_beta.empty:
    df_combined = df_filtered_beta
else:
    df_combined = pd.DataFrame()

# ============================================================
# 5. TOP-LEVEL KPIs COMPARATIFS
# ============================================================
st.markdown("### 📈 Indicateurs Clés Comparatifs")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    # Volume total
    vol_pub = len(df_filtered_pub)
    vol_beta = len(df_filtered_beta)
    total_vol = vol_pub + vol_beta
    
    if total_vol > 0:
        beta_pct = (vol_beta / total_vol) * 100
        st.metric(
            "📊 Volume Total",
            f"{total_vol:,}",
            delta=f"Beta: {beta_pct:.1f}%",
            delta_color="off",
            help="Répartition Published vs Early Access"
        )
    else:
        st.metric("📊 Volume", "N/A")

with kpi2:
    # Prix médian comparatif
    if not df_filtered_pub.empty and not df_filtered_beta.empty:
        median_pub = df_filtered_pub["Price"].median()
        median_beta = df_filtered_beta["Price"].median()
        delta_pct = ((median_beta - median_pub) / median_pub * 100) if median_pub > 0 else 0
        
        st.metric(
            "💵 Prix Médian (Published)",
            f"${median_pub:.2f}",
            delta=f"Beta: {delta_pct:+.0f}%",
            delta_color="inverse",
            help="Comparaison des prix médians"
        )
    elif not df_filtered_pub.empty:
        st.metric("💵 Prix (Published)", f"${df_filtered_pub['Price'].median():.2f}")
    elif not df_filtered_beta.empty:
        st.metric("💵 Prix (Beta)", f"${df_filtered_beta['Price'].median():.2f}")
    else:
        st.metric("💵 Prix", "N/A")

with kpi3:
    # Score moyen
    if "User score" in df_combined.columns and not df_combined.empty:
        df_score = df_combined[df_combined["User score"] > 0]
        
        if not df_score.empty:
            score_pub = df_score[df_score["Status"] == "Published"]["User score"].mean() if "Published" in df_score["Status"].values else 0
            score_beta = df_score[df_score["Status"] == "Early Access"]["User score"].mean() if "Early Access" in df_score["Status"].values else 0
            
            if score_pub > 0 and score_beta > 0:
                delta_score = score_beta - score_pub
                st.metric(
                    "⭐ Score Moyen (Published)",
                    f"{score_pub:.1f}",
                    delta=f"Beta: {delta_score:+.1f}",
                    help="User score moyen comparatif"
                )
            else:
                st.metric("⭐ Score", "N/A")
        else:
            st.metric("⭐ Score", "N/A")
    else:
        st.metric("⭐ Score", "N/A")

with kpi4:
    # CCU médian
    if "Peak CCU" in df_combined.columns and not df_combined.empty:
        df_ccu = df_combined[df_combined["Peak CCU"] > 0]
        
        if not df_ccu.empty:
            ccu_pub = df_ccu[df_ccu["Status"] == "Published"]["Peak CCU"].median() if "Published" in df_ccu["Status"].values else 0
            ccu_beta = df_ccu[df_ccu["Status"] == "Early Access"]["Peak CCU"].median() if "Early Access" in df_ccu["Status"].values else 0
            
            if ccu_pub > 0:
                st.metric(
                    "👥 CCU Médian (Pub)",
                    f"{ccu_pub:,.0f}",
                    help="Peak CCU médian"
                )
            else:
                st.metric("👥 CCU", "N/A")
        else:
            st.metric("👥 CCU", "N/A")
    else:
        st.metric("👥 CCU", "N/A")

st.divider()

# ============================================================
# 6. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'Ensemble",
    "💰 Pricing & Monétisation",
    "🎯 Engagement",
    "🎮 Genres & Market Fit"
])

# ============================================================
# TAB 1: VUE D'ENSEMBLE
# ============================================================
with tab1:
    st.markdown("### 📊 Analyse Comparative Globale")
    
    col_overview1, col_overview2 = st.columns(2)
    
    with col_overview1:
        st.markdown("#### 🎯 Répartition Volumétrique")
        
        # Donut chart
        if not df_combined.empty:
            status_counts = df_combined["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            
            fig_donut = px.pie(
                status_counts,
                values="Count",
                names="Status",
                hole=0.4,
                color="Status",
                color_discrete_map={
                    "Published": COLORS["solo"],
                    "Early Access": COLORS["multi"]
                }
            )
            
            fig_donut.update_layout(
                **PLOTLY_LAYOUT,
                height=350
            )
            
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+value",
                hovertemplate="<b>%{label}</b><br>%{value:,} jeux<br>%{percent}<extra></extra>"
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.warning("⚠️ Aucune donnée disponible")
    
    with col_overview2:
        st.markdown("#### ⭐ Distribution des Scores")
        
        # Violin plot
        if "User score" in df_combined.columns and not df_combined.empty:
            df_score_viz = df_combined[df_combined["User score"] > 0].copy()
            
            if len(df_score_viz) > 0:
                fig_violin = px.violin(
                    df_score_viz,
                    x="Status",
                    y="User score",
                    box=True,
                    color="Status",
                    color_discrete_map={
                        "Published": COLORS["solo"],
                        "Early Access": COLORS["multi"]
                    }
                )
                
                fig_violin.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="",
                    yaxis_title="Score Utilisateur",
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                st.warning("⚠️ Pas de scores disponibles")
        else:
            st.warning("⚠️ Colonne User score non disponible")
    
    st.divider()
    
    # Box plot CCU
    st.markdown("#### 👥 Comparaison Peak CCU")
    
    if "Peak CCU" in df_combined.columns and not df_combined.empty:
        df_ccu_viz = df_combined[df_combined["Peak CCU"] > 0].copy()
        
        if len(df_ccu_viz) > 0:
            fig_ccu = px.box(
                df_ccu_viz,
                x="Status",
                y="Peak CCU",
                color="Status",
                log_y=True,
                color_discrete_map={
                    "Published": COLORS["solo"],
                    "Early Access": COLORS["multi"]
                }
            )
            
            fig_ccu.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="",
                yaxis_title="Peak CCU (échelle log)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_ccu, use_container_width=True)
            
            # Insight
            ccu_pub_median = df_ccu_viz[df_ccu_viz["Status"] == "Published"]["Peak CCU"].median() if "Published" in df_ccu_viz["Status"].values else 0
            ccu_beta_median = df_ccu_viz[df_ccu_viz["Status"] == "Early Access"]["Peak CCU"].median() if "Early Access" in df_ccu_viz["Status"].values else 0
            
            if ccu_pub_median > 0 and ccu_beta_median > 0:
                if ccu_beta_median < ccu_pub_median:
                    diff_pct = ((ccu_pub_median - ccu_beta_median) / ccu_pub_median) * 100
                    st.info(
                        f"📊 **Insight**: Les jeux Early Access ont un CCU médian "
                        f"**{diff_pct:.0f}% inférieur** aux jeux Published — "
                        f"audience plus restreinte en phase beta."
                    )
                else:
                    st.success(
                        f"🚀 Les jeux Early Access performent aussi bien que Published en termes d'audience !"
                    )
        else:
            st.warning("⚠️ Pas de données CCU disponibles")

# ============================================================
# TAB 2: PRICING & MONÉTISATION
# ============================================================
with tab2:
    st.markdown("### 💰 Stratégies Tarifaires Comparatives")
    
    col_price1, col_price2 = st.columns(2)
    
    with col_price1:
        st.markdown("#### 📊 Distribution des Prix")
        
        # Histogramme overlaid
        if not df_combined.empty:
            df_price_viz = df_combined[df_combined["Price"] > 0].copy()
            
            if len(df_price_viz) > 0:
                fig_hist_price = px.histogram(
                    df_price_viz,
                    x="Price",
                    color="Status",
                    nbins=40,
                    barmode="overlay",
                    opacity=0.7,
                    color_discrete_map={
                        "Published": COLORS["solo"],
                        "Early Access": COLORS["multi"]
                    }
                )
                
                fig_hist_price.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Prix ($)",
                    yaxis_title="Nombre de jeux",
                    height=400,
                    xaxis=dict(range=[0, 60])  # Focus sur 0-60$ pour meilleure lisibilité
                )
                
                st.plotly_chart(fig_hist_price, use_container_width=True)
            else:
                st.warning("⚠️ Pas de données de prix")
        else:
            st.warning("⚠️ Aucune donnée disponible")
    
    with col_price2:
        st.markdown("#### 💵 Prix Médian par Statut")
        
        # Bar chart comparatif
        if not df_combined.empty and "Price" in df_combined.columns:
            df_price_agg = df_combined[df_combined["Price"] > 0].copy()
            
            if len(df_price_agg) > 0:
                price_by_status = df_price_agg.groupby("Status")["Price"].median().reset_index()
                price_by_status.columns = ["Status", "Prix Médian"]
                
                fig_price_bar = px.bar(
                    price_by_status,
                    x="Status",
                    y="Prix Médian",
                    color="Status",
                    text="Prix Médian",
                    color_discrete_map={
                        "Published": COLORS["solo"],
                        "Early Access": COLORS["multi"]
                    }
                )
                
                fig_price_bar.update_traces(
                    texttemplate='$%{text:.2f}',
                    textposition='outside'
                )
                
                fig_price_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="",
                    yaxis_title="Prix Médian ($)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_price_bar, use_container_width=True)
                
                # Insight pricing discount
                if len(price_by_status) == 2:
                    pub_price = price_by_status[price_by_status["Status"] == "Published"]["Prix Médian"].iloc[0]
                    beta_price = price_by_status[price_by_status["Status"] == "Early Access"]["Prix Médian"].iloc[0]
                    
                    discount = ((beta_price - pub_price) / pub_price) * 100
                    
                    if discount < 0:
                        st.success(
                            f"💰 **Stratégie Pricing**: Les jeux Early Access sont en moyenne "
                            f"**{abs(discount):.1f}% moins chers** — pricing d'adoption précoce classique."
                        )
                    else:
                        st.info(
                            f"📊 Les jeux Early Access ne sont pas significativement moins chers "
                            f"({discount:+.1f}%)."
                        )
            else:
                st.warning("⚠️ Pas de données de prix")

    st.divider()
    
    # Prix par genre (top 10)
    st.markdown("#### 🎮 Prix Médian par Genre (Top 10)")
    
    if not df_combined.empty and "Genres" in df_combined.columns:
        df_genre_price = df_combined[df_combined["Price"] > 0].copy()
        df_genre_price = df_genre_price.explode("Genres").dropna(subset=["Genres"])
        
        # Top 10 genres par volume total
        top_genres = df_genre_price["Genres"].value_counts().head(10).index.tolist()
        df_genre_price = df_genre_price[df_genre_price["Genres"].isin(top_genres)]
        
        if len(df_genre_price) > 0:
            genre_price_pivot = df_genre_price.groupby(["Genres", "Status"])["Price"].median().reset_index()
            
            fig_genre_price = px.bar(
                genre_price_pivot,
                x="Genres",
                y="Price",
                color="Status",
                barmode="group",
                color_discrete_map={
                    "Published": COLORS["solo"],
                    "Early Access": COLORS["multi"]
                }
            )
            
            fig_genre_price.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="",
                yaxis_title="Prix Médian ($)",
                height=400
            )
            
            st.plotly_chart(fig_genre_price, use_container_width=True)
        else:
            st.warning("⚠️ Pas assez de données par genre")

# ============================================================
# TAB 3: ENGAGEMENT
# ============================================================
with tab3:
    st.markdown("### 🎯 Engagement & Rétention Comparative")
    
    # Recommendations analysis
    if "Recommendations" in df_combined.columns and not df_combined.empty:
        st.markdown("#### 👍 Analyse des Recommandations")
        
        df_reco = df_combined[df_combined["Recommendations"] > 0].copy()
        
        if len(df_reco) > 0:
            col_reco1, col_reco2 = st.columns(2)
            
            with col_reco1:
                # Box plot recommendations
                fig_reco_box = px.box(
                    df_reco,
                    x="Status",
                    y="Recommendations",
                    color="Status",
                    log_y=True,
                    color_discrete_map={
                        "Published": COLORS["solo"],
                        "Early Access": COLORS["multi"]
                    }
                )
                
                fig_reco_box.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="",
                    yaxis_title="Recommandations (log)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_reco_box, use_container_width=True)
            
            with col_reco2:
                # Médiane par status
                reco_median = df_reco.groupby("Status")["Recommendations"].median().reset_index()
                reco_median.columns = ["Status", "Recommandations Médianes"]
                
                fig_reco_bar = px.bar(
                    reco_median,
                    x="Status",
                    y="Recommandations Médianes",
                    color="Status",
                    text="Recommandations Médianes",
                    color_discrete_map={
                        "Published": COLORS["solo"],
                        "Early Access": COLORS["multi"]
                    }
                )
                
                fig_reco_bar.update_traces(
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                )
                
                fig_reco_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="",
                    yaxis_title="Recommandations Médianes",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_reco_bar, use_container_width=True)
        else:
            st.warning("⚠️ Pas de données de recommandations")
    
    st.divider()
    
    # Scatter Recommendations vs CCU
    if "Recommendations" in df_combined.columns and "Peak CCU" in df_combined.columns and not df_combined.empty:
        st.markdown("#### 🔥 Recommandations vs Popularité (CCU)")
        
        df_scatter = df_combined[
            (df_combined["Recommendations"] > 0) &
            (df_combined["Peak CCU"] > 0)
        ].copy()
        
        if len(df_scatter) > 0:
            # Échantillonnage si trop de points
            if len(df_scatter) > 1000:
                df_scatter = df_scatter.sample(1000, random_state=42)
            
            fig_scatter_engagement = px.scatter(
                df_scatter,
                x="Peak CCU",
                y="Recommendations",
                color="Status",
                log_x=True,
                log_y=True,
                opacity=0.6,
                hover_name="Name",
                color_discrete_map={
                    "Published": COLORS["solo"],
                    "Early Access": COLORS["multi"]
                }
            )
            
            fig_scatter_engagement.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Peak CCU (log)",
                yaxis_title="Recommandations (log)",
                height=450
            )
            
            st.plotly_chart(fig_scatter_engagement, use_container_width=True)
            
            # Insight taux de recommandation
            df_scatter["Reco_Rate"] = df_scatter["Recommendations"] / df_scatter["Peak CCU"]
            
            rate_pub = df_scatter[df_scatter["Status"] == "Published"]["Reco_Rate"].median() if "Published" in df_scatter["Status"].values else 0
            rate_beta = df_scatter[df_scatter["Status"] == "Early Access"]["Reco_Rate"].median() if "Early Access" in df_scatter["Status"].values else 0
            
            if rate_pub > 0 and rate_beta > 0:
                if rate_beta > rate_pub:
                    diff = ((rate_beta - rate_pub) / rate_pub) * 100
                    st.success(
                        f"🚀 **Effet Communauté**: Les jeux Early Access ont un taux de "
                        f"recommandation **{diff:.0f}% supérieur** — les early adopters "
                        f"sont plus engagés et évangélistes !"
                    )
                else:
                    st.info("📊 Pas de différence significative dans le taux de recommandation.")
        else:
            st.warning("⚠️ Pas assez de données pour la comparaison")

# ============================================================
# TAB 4: GENRES & MARKET FIT
# ============================================================
with tab4:
    st.markdown("### 🎮 Distribution des Genres")
    
    col_genre1, col_genre2 = st.columns(2)
    
    with col_genre1:
        st.markdown("#### 📊 Top 15 Genres - Published")
        
        if not df_filtered_pub.empty and "Genres" in df_filtered_pub.columns:
            df_pub_genres = df_filtered_pub.explode("Genres").dropna(subset=["Genres"])
            
            if len(df_pub_genres) > 0:
                top_pub_genres = df_pub_genres["Genres"].value_counts().head(15).reset_index()
                top_pub_genres.columns = ["Genre", "Count"]
                
                fig_pub_genres = px.bar(
                    top_pub_genres,
                    x="Count",
                    y="Genre",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, COLORS["solo"]], [1, COLORS["primary"]]]
                )
                
                fig_pub_genres.update_layout(
                    **PLOTLY_LAYOUT,
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=500,
                    yaxis=dict(categoryorder="total ascending")
                )
                
                st.plotly_chart(fig_pub_genres, use_container_width=True)
            else:
                st.warning("⚠️ Pas de données de genres")
        else:
            st.info("ℹ️ Activez l'affichage Published")
    
    with col_genre2:
        st.markdown("#### 🚀 Top 15 Genres - Early Access")
        
        if not df_filtered_beta.empty and "Genres" in df_filtered_beta.columns:
            df_beta_genres = df_filtered_beta.explode("Genres").dropna(subset=["Genres"])
            
            if len(df_beta_genres) > 0:
                top_beta_genres = df_beta_genres["Genres"].value_counts().head(15).reset_index()
                top_beta_genres.columns = ["Genre", "Count"]
                
                fig_beta_genres = px.bar(
                    top_beta_genres,
                    x="Count",
                    y="Genre",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, COLORS["multi"]], [1, COLORS["secondary"]]]
                )
                
                fig_beta_genres.update_layout(
                    **PLOTLY_LAYOUT,
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=500,
                    yaxis=dict(categoryorder="total ascending")
                )
                
                st.plotly_chart(fig_beta_genres, use_container_width=True)
            else:
                st.warning("⚠️ Pas de données de genres")
        else:
            st.info("ℹ️ Activez l'affichage Early Access")
    
    st.divider()
    
    # Table comparative si les deux sont actifs
    if not df_filtered_pub.empty and not df_filtered_beta.empty and "Genres" in df_filtered_pub.columns and "Genres" in df_filtered_beta.columns:
        st.markdown("#### 📋 Comparaison des Genres (Top 10)")
        
        # Calcul pour Published
        df_pub_g = df_filtered_pub.explode("Genres").dropna(subset=["Genres"])
        pub_genre_counts = df_pub_g["Genres"].value_counts()
        pub_total = len(df_filtered_pub)
        
        # Calcul pour Beta
        df_beta_g = df_filtered_beta.explode("Genres").dropna(subset=["Genres"])
        beta_genre_counts = df_beta_g["Genres"].value_counts()
        beta_total = len(df_filtered_beta)
        
        # Union des top genres
        all_genres = set(pub_genre_counts.head(10).index) | set(beta_genre_counts.head(10).index)
        
        comparison_data = []
        for genre in all_genres:
            pub_count = pub_genre_counts.get(genre, 0)
            beta_count = beta_genre_counts.get(genre, 0)
            
            pub_pct = (pub_count / pub_total * 100) if pub_total > 0 else 0
            beta_pct = (beta_count / beta_total * 100) if beta_total > 0 else 0
            
            delta = beta_pct - pub_pct
            
            comparison_data.append({
                "Genre": genre,
                "% Published": pub_pct,
                "% Early Access": beta_pct,
                "Delta": delta,
                "Insight": "Over-index Beta" if delta > 5 else ("Under-index Beta" if delta < -5 else "Équilibré")
            })
        
        df_comparison = pd.DataFrame(comparison_data).sort_values("Delta", ascending=False).head(10)
        
        # Formattage
        df_comparison["% Published"] = df_comparison["% Published"].apply(lambda x: f"{x:.1f}%")
        df_comparison["% Early Access"] = df_comparison["% Early Access"].apply(lambda x: f"{x:.1f}%")
        df_comparison["Delta"] = df_comparison["Delta"].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)
        
        # Insight sur le genre qui over-index le plus
        top_overindex = comparison_data[0] if comparison_data else None
        if top_overindex and top_overindex["Delta"] > 5:
            st.warning(
                f"🎮 **Market Shift**: Le genre **{top_overindex['Genre']}** est "
                f"sur-représenté en Early Access ({top_overindex['Delta']:+.1f}% vs Published) — "
                f"signe d'un marché indie très actif dans cette catégorie."
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🎯 **GameData360 — Published vs Early Access** | "
    f"Analyse comparative sur {len(df_published) + len(df_beta):,} jeux"
)