# -*- coding: utf-8 -*-
"""
GameData360 - Page ML Clustering & Recommandations
===================================================
Machine Learning pour découvrir des patterns cachés et recommander des jeux.
Algorithmes: K-Means, DBSCAN, Hierarchical + UMAP/T-SNE visualization.

Auteur: GameData360 Team
Version: 1.0 (ML Intelligence Edition)
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
from utils.ml_helpers import (
    prepare_features_for_clustering,
    perform_dbscan_clustering,
    calculate_k_distance,
    reduce_dimensions_umap,
    calculate_single_game_similarity,
    get_recommendations,
    get_cluster_statistics
)

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — ML Clustering",
    page_icon="🤖",
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
    
    /* Selectbox custom */
    .stSelectbox label {
        font-family: 'Rajdhani', sans-serif !important;
        color: #00ff88 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("# 🤖 ML CLUSTERING & RECOMMANDATIONS")
st.markdown("##### Intelligence Artificielle : Découverte de Patterns Cachés et Système de Recommandation")

# ============================================================
# 2. CHARGEMENT ET PREPROCESSING
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
    
    df = df[df["Genres"].apply(is_game)].copy()
    
    # Filtrage des jeux Early Access / Beta
    def is_not_beta(categories):
        if not isinstance(categories, list):
            return True
        cats_lower = [c.lower() for c in categories]
        beta_keywords = ['early access', 'beta']
        return not any(keyword in cat for cat in cats_lower for keyword in beta_keywords)
    
    df = df[df["Categories"].apply(is_not_beta)].copy()
    
    # Filtrer les jeux avec données suffisantes pour le ML
    df = df[
        (df['Name'].notna()) &
        (df['Price'].notna()) &
        (df['Peak CCU'].notna())
    ].copy()
    
    return df

# Chargement avec indicateur
try:
    with st.spinner('🤖 Chargement des données et préparation ML...'):
        df_ml = load_and_filter_data()
        
        # Preprocessing pour le clustering
        features, df_clean, scaler = prepare_features_for_clustering(df_ml)
        
        st.sidebar.success(f"✅ {len(df_ml):,} jeux chargés pour l'analyse ML")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 3. CALCUL DES CLUSTERS (CACHE) - VERSION OPTIMISÉE
# ============================================================
with st.spinner('🔬 Calcul DBSCAN optimisé (PCA + k-distance)...'):
    # Calculer k-distance pour trouver eps optimal
    k_distances, suggested_eps = calculate_k_distance(features, k=100)
    
    # DBSCAN avec paramètres optimisés
    # min_samples=100 évite les micro-clusters
    # eps suggéré par k-distance
    dbscan_labels, dbscan_n_clusters, dbscan_n_outliers = perform_dbscan_clustering(
        features, 
        eps=suggested_eps, 
        min_samples=100
    )

# Ajout des labels au DataFrame principal
df_ml['cluster_dbscan'] = dbscan_labels

# Variables pour compatibilité
selected_labels = dbscan_labels
cluster_col = 'cluster_dbscan'

# ============================================================
# 4. SIDEBAR - INFO
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 Info ML")
    st.caption(f"**Total jeux analysés:** {len(df_ml):,}")
    st.caption(f"**Features après PCA:** {features.shape[1]} (optimisé)")
    st.caption(f"**Algorithme:** DBSCAN optimisé")
    st.caption(f"**Min samples:** 100 (anti-micro-clusters)")
    st.caption(f"**Eps optimal:** {suggested_eps:.3f}")
    st.caption(f"**Clusters trouvés:** {dbscan_n_clusters}")
    st.caption(f"**Outliers détectés:** {dbscan_n_outliers}")

# ============================================================
# 5. TOP-LEVEL KPIs
# ============================================================
st.markdown("### 📊 Métriques ML DBSCAN")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "🔍 Clusters Détectés",
        dbscan_n_clusters,
        delta="Groupes naturels",
        help="DBSCAN a détecté des clusters de forme arbitraire"
    )

with kpi2:
    st.metric(
        "🎲 Outliers",
        dbscan_n_outliers,
        delta="Jeux atypiques",
        help="Jeux qui ne correspondent à aucun cluster (nichés/uniques)"
    )

with kpi3:
    pct_outliers = (dbscan_n_outliers / len(df_ml)) * 100
    st.metric(
        "📊 % Outliers",
        f"{pct_outliers:.1f}%",
        delta=f"{dbscan_n_outliers:,} jeux",
        help="Pourcentage de jeux atypiques"
    )

with kpi4:
    avg_cluster_size = (len(df_ml) - dbscan_n_outliers) / dbscan_n_clusters if dbscan_n_clusters > 0 else 0
    st.metric(
        "👥 Taille Moy. Cluster",
        f"{avg_cluster_size:,.0f}",
        delta="jeux/cluster",
        delta_color="off",
        help="Nombre moyen de jeux par cluster"
    )

st.divider()

# ============================================================
# 6. ONGLETS D'ANALYSE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'Ensemble",
    "🗺️ Visualisation Interactive",
    "🔍 Analyse des Clusters",
    "🎮 Recommandations"
])

# ============================================================
# TAB 1: VUE D'ENSEMBLE
# ============================================================
with tab1:
    st.markdown("### 📈 Distribution DBSCAN des Jeux par Cluster")
    
    # Distribution DBSCAN
    dbscan_dist = pd.Series(dbscan_labels).value_counts().sort_index()
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        fig_dbscan_dist = px.bar(
            x=dbscan_dist.index,
            y=dbscan_dist.values,
            labels={'x': 'Cluster ID (-1 = Outliers)', 'y': 'Nombre de jeux'},
            title=f"Distribution DBSCAN ({dbscan_n_clusters} clusters + {dbscan_n_outliers} outliers)",
            color=dbscan_dist.values,
            color_continuous_scale=[[0, COLORS['danger']], [1, COLORS['primary']]]
        )
        
        fig_dbscan_dist.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_dbscan_dist, use_container_width=True)
    
    with col_stats:
        st.markdown("#### 📊 Statistiques Clusters")
        
        stats_clusters = pd.DataFrame({
            'Métrique': [
                'Total Clusters',
                'Plus Grand',
                'Plus Petit',
                'Outliers',
                '% Outliers'
            ],
            'Valeur': [
                f"{dbscan_n_clusters}",
                f"{dbscan_dist[dbscan_dist.index != -1].max():,}",
                f"{dbscan_dist[dbscan_dist.index != -1].min():,}",
                f"{dbscan_n_outliers:,}",
                f"{pct_outliers:.1f}%"
            ]
        })
        
        st.dataframe(stats_clusters, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Insights DBSCAN
    st.markdown("### 💡 Insights Automatiques")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        if dbscan_n_clusters > 0:
            st.success(
                f"✅ **{dbscan_n_clusters} groupes naturels** détectés par DBSCAN. "
                f"Chaque cluster représente des jeux avec des caractéristiques similaires (prix, CCU, genres, tags)."
            )
    
    with col_insight2:
        if pct_outliers > 5:
            st.warning(
                f"🎲 **{pct_outliers:.1f}% de jeux atypiques** identifiés ! "
                f"Ces {dbscan_n_outliers:,} jeux sont uniques/nichés et ne s'intègrent dans aucun groupe majeur."
            )
        else:
            st.info(
                f"📊 Seulement **{pct_outliers:.1f}% d'outliers** — La majorité des jeux s'inscrit dans des tendances claires."
            )
    
    st.divider()
    
    # K-Distance Plot (optimisé)
    st.markdown("### 📏 K-Distance Plot (Optimisation Eps)")
    st.caption("Graphique utilisé pour trouver le eps optimal via la méthode du coude")
    
    # Réduire le nombre de points à afficher (échantillonnage)
    step = max(1, len(k_distances) // 500)  # Max 500 points
    k_dist_sample = k_distances[::step]
    x_sample = list(range(0, len(k_distances), step))
    
    fig_kdist = go.Figure()
    
    fig_kdist.add_trace(go.Scatter(
        x=x_sample,
        y=k_dist_sample,
        mode='lines',
        name='K-Distance',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    # Ligne eps suggéré
    fig_kdist.add_hline(
        y=suggested_eps,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Eps optimal = {suggested_eps:.3f}",
        annotation_position="right"
    )
    
    fig_kdist.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Points (triés par distance)",
        yaxis_title="Distance au 100e voisin",
        height=350
    )
    
    st.plotly_chart(fig_kdist, use_container_width=True)
    
    st.info(
        f"💡 **Eps Optimal = {suggested_eps:.3f}** : Ce seuil sépare les points denses (clusters) "
        f"des points épars (outliers). Le 'coude' du graphique indique la distance maximale optimale."
    )

# ============================================================
# TAB 2: VISUALISATION INTERACTIVE (UMAP)
# ============================================================

# Cache des coordonnées UMAP pour éviter les recalculs
@st.cache_data(show_spinner=False)
def get_umap_coords(features_array, n_comp):
    """Cache les coordonnées UMAP pour ne pas recalculer."""
    return reduce_dimensions_umap(features_array, n_components=n_comp)

with tab2:
    st.markdown("### 🗺️ Carte Interactive UMAP des Jeux")
    st.caption("UMAP (Uniform Manifold Approximation and Projection) pour visualisation rapide et précise")
    
    # Checkbox 3D
    dim_3d = st.checkbox("Visualisation 3D", value=False, help="Activer la visualisation 3D (plus immersive)")
    
    # Calcul des coordonnées UMAP avec cache
    n_components = 3 if dim_3d else 2
    
    with st.spinner('🔄 Calcul UMAP (réduction dimensionnelle)...'):
        coords = get_umap_coords(features, n_components)
    
    # Créer DataFrame de visualisation
    df_viz = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2] if dim_3d else 0,
        'Name': df_ml['Name'].values,
        'Price': df_ml['Price'].values,
        'CCU': df_ml['Peak CCU'].values,
        'Cluster': selected_labels
    })
    
    # Graphique 2D ou 3D
    if dim_3d:
        fig_viz = px.scatter_3d(
            df_viz,
            x='x',
            y='y',
            z='z',
            color='Cluster',
            hover_data=['Name', 'Price', 'CCU'],
            color_continuous_scale=COLORS['chart'],
            title=f"UMAP 3D - Clustering DBSCAN ({dbscan_n_clusters} clusters)"
        )
    else:
        fig_viz = px.scatter(
            df_viz,
            x='x',
            y='y',
            color='Cluster',
            hover_data=['Name', 'Price', 'CCU'],
            color_continuous_scale=COLORS['chart'],
            title=f"UMAP 2D - Clustering DBSCAN ({dbscan_n_clusters} clusters)"
        )
    
    fig_viz.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        showlegend=True
    )
    
    fig_viz.update_traces(marker=dict(size=3, opacity=0.6))
    
    st.plotly_chart(fig_viz, use_container_width=True)
    
    st.success(
        f"🎯 **Visualisation Active**: Chaque point = un jeu. Les couleurs représentent les {dbscan_n_clusters} clusters DBSCAN. "
        f"Les outliers (cluster -1) sont également visibles. Survolez les points pour voir les détails !"
    )

# ============================================================
# TAB 3: ANALYSE DES CLUSTERS
# ============================================================
with tab3:
    st.markdown("### 🔍 Analyse Détaillée des Clusters")
    
    # Sélecteur de cluster
    unique_clusters = sorted([c for c in np.unique(selected_labels) if c != -1])
    
    selected_cluster = st.selectbox(
        "Sélectionner un cluster à analyser",
        unique_clusters,
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Récupérer les statistiques
    stats = get_cluster_statistics(selected_cluster, df_ml, selected_labels)
    
    # KPIs du cluster
    st.markdown(f"#### 📊 Métriques du Cluster {selected_cluster}")
    
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    
    with kpi_c1:
        st.metric("🎮 Jeux", f"{stats['size']:,}")
    
    with kpi_c2:
        st.metric("💰 Prix Moyen", f"${stats['avg_price']:.2f}")
    
    with kpi_c3:
        st.metric("👥 CCU Moyen", f"{stats['avg_ccu']:,.0f}")
    
    with kpi_c4:
        st.metric("⭐ Metacritic", f"{stats['avg_metacritic']:.1f}" if stats['avg_metacritic'] > 0 else "N/A")
    
    with kpi_c5:
        st.metric("🆓 % F2P", f"{stats['pct_f2p']:.0f}%")
    
    st.divider()
    
    # Genres et tags en format simplifié (sans graphiques)
    col_genre, col_tags = st.columns(2)
    
    with col_genre:
        st.markdown("#### 🎮 Top 5 Genres")
        
        if stats.get('top_genres'):
            genres_list = list(stats['top_genres'].items())[:5]
            for genre, count in genres_list:
                st.caption(f"**{genre}**: {count:,} jeux")
    
    with col_tags:
        st.markdown("#### 🏷️ Top 5 Tags")
        
        if stats.get('top_tags'):
            tags_list = list(stats['top_tags'].items())[:5]
            for tag, count in tags_list:
                st.caption(f"**{tag}**: {count:,} jeux")
    
    st.divider()
    
    # Top jeux du cluster (limité à 5 pour la performance)
    st.markdown(f"#### 🏆 Top 5 Jeux Représentatifs du Cluster {selected_cluster}")
    
    cluster_games = df_ml[selected_labels == selected_cluster].copy()
    cluster_games = cluster_games.nlargest(5, 'Peak CCU')[['Name', 'Price', 'Peak CCU']]
    
    st.dataframe(cluster_games, use_container_width=True, hide_index=True)

# ============================================================
# TAB 4: SYSTÈME DE RECOMMANDATION
# ============================================================
with tab4:
    st.markdown("### 🎮 Système de Recommandation Intelligent")
    st.caption("Sélectionnez un jeu pour obtenir des recommandations basées sur le machine learning")
    
    # Recherche de jeu
    game_search = st.selectbox(
        "🔍 Rechercher un jeu",
        options=df_ml['Name'].tolist(),
        index=0,
        help="Tapez pour rechercher un jeu"
    )
    
    if game_search:
        # Trouver l'index du jeu
        game_idx = df_ml[df_ml['Name'] == game_search].index[0]
        game_info = df_ml.loc[game_idx]
        
        # Calculer la similarité pour CE jeu uniquement (économie mémoire)
        with st.spinner('🔄 Calcul des similarités pour ce jeu...'):
            similarities = calculate_single_game_similarity(game_idx, features)
        
        # Afficher les infos du jeu sélectionné
        st.markdown(f"#### 🎯 Jeu Sélectionné: **{game_search}**")
        
        col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
        
        with col_info1:
            st.metric("💰 Prix", f"${game_info['Price']:.2f}" if game_info['Price'] > 0 else "F2P")
        
        with col_info2:
            st.metric("👥 Peak CCU", f"{game_info['Peak CCU']:,.0f}")
        
        with col_info3:
            metacritic = game_info['Metacritic score']
            st.metric("⭐ Metacritic", f"{metacritic:.0f}" if metacritic > 0 else "N/A")
        
        with col_info4:
            user_score = game_info['User score']
            st.metric("👤 User Score", f"{user_score:.0f}" if user_score > 0 else "N/A")
        
        with col_info5:
            cluster_id = selected_labels[game_idx]
            st.metric("🎯 Cluster", f"{cluster_id}")
        
        st.divider()
        
        # Recommandations
        col_reco1, col_reco2 = st.columns(2)
        
        with col_reco1:
            st.markdown("#### 🎮 Jeux Similaires (Même Cluster)")
            
            reco_same = get_recommendations(
                game_idx,
                similarities,
                df_ml,
                cluster_labels=selected_labels,
                n=5,
                same_cluster_only=True
            )
            
            if len(reco_same) > 0:
                reco_same_display = reco_same[['Name', 'Price', 'Peak CCU', 'Similarity']].copy()
                reco_same_display['Similarity'] = reco_same_display['Similarity'].apply(lambda x: f"{x:.3f}")
                reco_same_display['Price'] = reco_same_display['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "F2P")
                
                st.dataframe(reco_same_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Pas assez de jeux dans ce cluster pour des recommandations.")
        
        with col_reco2:
            st.markdown("#### 🌟 Découvertes (Autres Clusters)")
            
            reco_other = get_recommendations(
                game_idx,
                similarities,
                df_ml,
                cluster_labels=selected_labels,
                n=10,
                same_cluster_only=False
            )
            
            # Filtrer pour avoir uniquement les autres clusters
            reco_other = reco_other[reco_other['Cluster'] != cluster_id].head(5)
            
            if len(reco_other) > 0:
                reco_other_display = reco_other[['Name', 'Price', 'Peak CCU', 'Cluster', 'Similarity']].copy()
                reco_other_display['Similarity'] = reco_other_display['Similarity'].apply(lambda x: f"{x:.3f}")
                reco_other_display['Price'] = reco_other_display['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "F2P")
                
                st.dataframe(reco_other_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucune découverte disponible.")
        
        st.divider()
        
        # Visualisation optionnelle (désactivée par défaut pour performance)
        if st.checkbox("🗺️ Afficher la position dans l'espace ML", value=False, key="show_umap_reco"):
            st.markdown("#### 🗺️ Position du Jeu dans l'Espace ML")
            
            with st.spinner('Calcul UMAP 2D...'):
                coords_2d = get_umap_coords(features, 2)
            
            df_viz_reco = pd.DataFrame({
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1],
                'Name': df_ml['Name'].values,
                'Cluster': selected_labels,
                'Selected': ['Sélectionné' if i == game_idx else 'Autre' for i in range(len(df_ml))]
            })
            
            fig_reco = px.scatter(
                df_viz_reco,
                x='x',
                y='y',
                color='Selected',
                hover_data=['Name', 'Cluster'],
                color_discrete_map={'Sélectionné': COLORS['danger'], 'Autre': COLORS['primary']},
                title=f"Position de '{game_search}' dans l'espace UMAP"
            )
            
            fig_reco.update_layout(
                **PLOTLY_LAYOUT,
                height=400
            )
            
            fig_reco.update_traces(marker=dict(size=4, opacity=0.5))
            
            # Agrandir le point sélectionné
            fig_reco.add_trace(go.Scatter(
                x=[coords_2d[game_idx, 0]],
                y=[coords_2d[game_idx, 1]],
                mode='markers',
                marker=dict(size=15, color=COLORS['warning'], symbol='star', line=dict(width=2, color='white')),
                name='Jeu Sélectionné',
                showlegend=True
            ))
            
            st.plotly_chart(fig_reco, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🤖 **GameData360 — ML Clustering & Recommandations** | "
    f"Analyse DBSCAN sur {len(df_ml):,} jeux | "
    f"Powered by UMAP & scikit-learn"
)
