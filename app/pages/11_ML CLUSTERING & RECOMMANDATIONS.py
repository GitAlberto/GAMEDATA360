# -*- coding: utf-8 -*-
"""
GameData360 - Page ML Clustering (Version Lite)
================================================
Version allégée pour machines avec 8GB RAM.
Utilise PCA + K-Means simple sans UMAP ni algorithmes lourds.

Auteur: GameData360 Team
Version: 2.0 (Lite Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
from utils.data_helpers import load_game_data

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — ML Clustering Lite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS minimal
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(255,0,255,0.1) 100%);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 ML CLUSTERING LITE")
st.caption("Version allégée pour 8GB RAM — K-Means + PCA simple")

# ============================================================
# 2. FONCTIONS SIMPLIFIÉES (INLINE)
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_prepare_lite():
    """Charge et prépare les données avec échantillonnage."""
    df = load_game_data(str(FILE_PATH))
    
    # Filtrage basique
    df = df[
        (df['Name'].notna()) &
        (df['Price'].notna()) &
        (df['Peak CCU'].notna())
    ].copy()
    
    # ÉCHANTILLONNAGE pour économiser la RAM (max 5000 jeux)
    max_games = 10000
    if len(df) > max_games:
        # Garder les jeux les plus populaires
        df = df.nlargest(max_games, 'Peak CCU')
    
    return df

@st.cache_data(show_spinner=False)
def prepare_simple_features(df):
    """Prépare des features minimales (seulement 4 colonnes)."""
    from sklearn.preprocessing import StandardScaler
    
    # Features minimales
    features_df = pd.DataFrame({
        'price_log': np.log1p(df['Price'].fillna(0)),
        'ccu_log': np.log1p(df['Peak CCU'].fillna(0)),
        'metacritic': df['Metacritic score'].fillna(0) / 100,
        'is_f2p': (df['Price'] == 0).astype(int)
    })
    
    # Normalisation simple
    scaler = StandardScaler()
    features = scaler.fit_transform(features_df)
    
    return features

@st.cache_data(show_spinner=False)
def simple_kmeans(features, n_clusters=5):
    """K-Means simple sans recherche d'optimal."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    labels = kmeans.fit_predict(features)
    
    return labels

@st.cache_data(show_spinner=False)
def simple_pca_2d(features):
    """PCA ultra-simple pour visualisation 2D."""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)
    
    return coords

# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================
try:
    with st.spinner('⚡ Chargement rapide...'):
        df_ml = load_and_prepare_lite()
        st.sidebar.success(f"✅ {len(df_ml):,} jeux (échantillon)")

except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# ============================================================
# 4. SIDEBAR - PARAMÈTRES SIMPLES
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    n_clusters = st.slider("Nombre de clusters", 3, 8, 5)
    
    st.divider()
    st.caption(f"**Jeux analysés:** {len(df_ml):,}")
    st.caption(f"**RAM Mode:** Lite")

# ============================================================
# 5. CALCULS ML SIMPLES
# ============================================================
with st.spinner('🔬 Calcul K-Means...'):
    features = prepare_simple_features(df_ml)
    labels = simple_kmeans(features, n_clusters)
    df_ml['cluster'] = labels

# ============================================================
# 6. KPIs SIMPLES
# ============================================================
st.markdown("### 📊 Résumé Clustering")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔍 Clusters", n_clusters)

with col2:
    avg_size = len(df_ml) // n_clusters
    st.metric("👥 Taille Moy.", f"{avg_size:,}")

with col3:
    st.metric("🎮 Total Jeux", f"{len(df_ml):,}")

with col4:
    biggest = pd.Series(labels).value_counts().max()
    st.metric("📈 Plus Grand", f"{biggest:,}")

st.divider()

# ============================================================
# 7. VISUALISATION SIMPLE (PCA 2D)
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🗺️ Vue 2D", "🎮 Recommandations"])

with tab1:
    st.markdown("### Distribution des Clusters")
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    fig_dist = px.bar(
        x=[f"Cluster {i}" for i in cluster_counts.index],
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Nombre de jeux'},
        color=cluster_counts.values,
        color_continuous_scale=[[0, COLORS['primary']], [1, COLORS['secondary']]]
    )
    
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        showlegend=False,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Stats par cluster
    st.markdown("### 📋 Statistiques par Cluster")
    
    stats_data = []
    for c in range(n_clusters):
        cluster_df = df_ml[df_ml['cluster'] == c]
        stats_data.append({
            'Cluster': f"Cluster {c}",
            'Jeux': len(cluster_df),
            'Prix Moy.': f"${cluster_df['Price'].mean():.2f}",
            'CCU Moy.': f"{cluster_df['Peak CCU'].mean():,.0f}",
            '% F2P': f"{(cluster_df['Price'] == 0).mean() * 100:.0f}%"
        })
    
    st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### 🗺️ Visualisation PCA 2D")
    st.caption("Projection simple sans UMAP (économie de RAM)")
    
    # Calculer PCA 2D seulement si demandé
    if st.button("📊 Générer la visualisation", key="gen_pca"):
        with st.spinner('Calcul PCA...'):
            coords = simple_pca_2d(features)
        
        # Limiter les points affichés (max 2000)
        n_display = min(2000, len(df_ml))
        sample_idx = np.random.choice(len(df_ml), n_display, replace=False)
        
        df_viz = pd.DataFrame({
            'x': coords[sample_idx, 0],
            'y': coords[sample_idx, 1],
            'Name': df_ml.iloc[sample_idx]['Name'].values,
            'Price': df_ml.iloc[sample_idx]['Price'].values,
            'Cluster': labels[sample_idx]
        })
        
        fig_pca = px.scatter(
            df_viz,
            x='x', y='y',
            color='Cluster',
            hover_data=['Name', 'Price'],
            title=f"PCA 2D ({n_display:,} jeux affichés)",
            color_continuous_scale=COLORS['chart']
        )
        
        fig_pca.update_traces(marker=dict(size=4, opacity=0.6))
        fig_pca.update_layout(**PLOTLY_LAYOUT, height=450)
        
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("💡 Cliquez sur le bouton pour générer la visualisation (peut prendre quelques secondes)")

with tab3:
    st.markdown("### 🎮 Recommandations Simples")
    
    # Sélection de jeu (liste limitée pour performance)
    popular_games = df_ml.nlargest(500, 'Peak CCU')['Name'].tolist()
    
    game_search = st.selectbox(
        "🔍 Sélectionner un jeu populaire",
        options=popular_games,
        index=0
    )
    
    if game_search:
        game_info = df_ml[df_ml['Name'] == game_search].iloc[0]
        game_cluster = game_info['cluster']
        
        # Infos du jeu
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("💰 Prix", f"${game_info['Price']:.2f}" if game_info['Price'] > 0 else "F2P")
        with col_b:
            st.metric("👥 Peak CCU", f"{game_info['Peak CCU']:,.0f}")
        with col_c:
            st.metric("🎯 Cluster", f"{game_cluster}")
        
        st.divider()
        
        # Jeux du même cluster
        st.markdown(f"#### 🎮 Jeux Similaires (Cluster {game_cluster})")
        
        same_cluster = df_ml[
            (df_ml['cluster'] == game_cluster) & 
            (df_ml['Name'] != game_search)
        ].nlargest(10, 'Peak CCU')[['Name', 'Price', 'Peak CCU']]
        
        same_cluster['Price'] = same_cluster['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "F2P")
        
        st.dataframe(same_cluster, hide_index=True, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    f"🤖 **GameData360 — ML Clustering Lite** | "
    f"{len(df_ml):,} jeux | K-Means + PCA simple"
)
