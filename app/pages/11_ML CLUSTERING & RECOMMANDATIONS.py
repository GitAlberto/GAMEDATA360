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

# CSS personnalisé pour le thème gaming (cohérent avec les autres pages)
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

st.markdown("# 🤖 ML CLUSTERING LITE")
st.caption("K-Means et PCA")

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
    max_games = 5000
    if len(df) > max_games:
        df = df.nlargest(max_games, 'Peak CCU')
    
    return df

@st.cache_data(show_spinner=False)
def prepare_simple_features(df):
    """Prépare des features minimales (seulement 4 colonnes)."""
    from sklearn.preprocessing import StandardScaler
    
    # Transformations à effectuer
    features_df = pd.DataFrame({
        'price_log': np.log1p(df['Price'].fillna(0)),
        'ccu_log': np.log1p(df['Peak CCU'].fillna(0)),
        'metacritic': df['Metacritic score'].fillna(0) / 100,
        'is_f2p': (df['Price'] == 0).astype(int)
    })
    
    # Standardisation des features
    scaler = StandardScaler()
    features = scaler.fit_transform(features_df) # Application des transformations
    
    return features

@st.cache_data(show_spinner=False)
def simple_kmeans(_features, n_clusters=5):
    """K-Means simple sans recherche d'optimal."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    labels = kmeans.fit_predict(_features)
    
    return labels

@st.cache_data(show_spinner=False)
def simple_pca_2d(_features):
    """PCA ultra-simple pour visualisation 2D."""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_features)
    
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
    n_clusters = st.slider("Nombre de clusters", 3, 8, 5) # min max default
    
    st.divider()
    st.caption(f"**Jeux analysés:** {len(df_ml):,}")
    st.caption(f"**RAM Mode:** Lite")

# ============================================================
# 5. CALCULS ML SIMPLES
# ============================================================
with st.spinner('🔬 Calcul K-Means...'):
    features = prepare_simple_features(df_ml) # Préparation des features
    labels = simple_kmeans(features, n_clusters) # Calcul des clusters
    df_ml['cluster'] = labels # Ajout des clusters au dataframe

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🗺️ Vue 2D", "🎮 Recommandations", "📄 Rapport ML"])

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
    
    if st.button("📊 Générer la visualisation", key="gen_pca"):
        with st.spinner('Calcul PCA...'):
            coords = simple_pca_2d(features)
        
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
        st.info("💡 Cliquez sur le bouton pour générer la visualisation")

with tab3:
    st.markdown("### 🎮 Recommandations")
    
    popular_games = df_ml.nlargest(500, 'Peak CCU')['Name'].tolist()
    
    game_search = st.selectbox(
        "🔍 Sélectionner un jeu populaire",
        options=popular_games,
        index=0
    )
    
    if game_search:
        game_info = df_ml[df_ml['Name'] == game_search].iloc[0]
        game_cluster = game_info['cluster']
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("💰 Prix", f"${game_info['Price']:.2f}" if game_info['Price'] > 0 else "F2P")
        with col_b:
            st.metric("👥 Peak CCU", f"{game_info['Peak CCU']:,.0f}")
        with col_c:
            st.metric("🎯 Cluster", f"{game_cluster}")
        
        st.divider()
        
        st.markdown(f"#### 🎮 Jeux Similaires (Cluster {game_cluster})")
        
        same_cluster = df_ml[
            (df_ml['cluster'] == game_cluster) & 
            (df_ml['Name'] != game_search)
        ].nlargest(10, 'Peak CCU')[['Name', 'Price', 'Peak CCU']]
        
        same_cluster['Price'] = same_cluster['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "F2P")
        
        st.dataframe(same_cluster, hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### 📄 Rapport Explicatif - Analyse ML")
    
    st.markdown("""
    ---
    ## 🎯 Objectif de l'Analyse
    
    Cette page utilise des techniques de **Machine Learning non supervisé** pour :
    1. **Regrouper** les jeux Steam en clusters de jeux similaires
    2. **Identifier** des patterns cachés dans les données  
    3. **Recommander** des jeux similaires basés sur leurs caractéristiques
    
    ---
    ## 🔬 Méthodologie
    
    ### 1. Préparation des Données (Feature Engineering)
    
    Nous utilisons **4 features** pour caractériser chaque jeu :
    
    | Feature | Description | Transformation |
    |---------|-------------|----------------|
    | **Prix** | Prix du jeu en dollars | Log-transform `log(1 + prix)` |
    | **Peak CCU** | Pic de joueurs simultanés | Log-transform `log(1 + ccu)` |
    | **Metacritic** | Score critique (0-100) | Normalisé /100 |
    | **F2P** | Jeu gratuit ou payant | Binaire (0/1) |
    
    **Pourquoi ces transformations ?**
    - Le **log-transform** réduit l'impact des valeurs extrêmes (ex: DOTA 2 avec 1M+ CCU)
    - La **normalisation** met toutes les features sur une échelle comparable
    
    ### 2. Algorithme de Clustering : K-Means
    
    **K-Means** est un algorithme qui partitionne les données en K groupes (clusters) :
    
    ```
    Étape 1: Choisir K centroïdes aléatoires
    Étape 2: Assigner chaque jeu au centroïde le plus proche  
    Étape 3: Recalculer les centroïdes (moyenne des jeux du cluster)
    Étape 4: Répéter jusqu'à convergence
    ```
    
    **Paramètres utilisés :**
    - `n_clusters` = 5 (par défaut, modifiable via sidebar)
    - `n_init` = 10 (nombre d'initialisations différentes)
    - `max_iter` = 100 (itérations max par initialisation)
    
    ### 3. Visualisation : PCA (Analyse en Composantes Principales)
    
    **PCA** permet de réduire les 4 dimensions à 2 dimensions visualisables :
    
    - **PC1** (axe X) : Capture la direction de variance maximale
    - **PC2** (axe Y) : Capture la 2ème direction orthogonale
    
    🔹 Les jeux proches sur le graphique ont des caractéristiques similaires.
    
    ---
    ## 📊 Interprétation des Clusters
    
    Voici comment interpréter typiquement les clusters générés :
    """)
    
    # Génération dynamique de l'interprétation
    for c in range(n_clusters):
        cluster_df = df_ml[df_ml['cluster'] == c]
        avg_price = cluster_df['Price'].mean()
        avg_ccu = cluster_df['Peak CCU'].mean()
        pct_f2p = (cluster_df['Price'] == 0).mean() * 100
        
        # Déterminer le profil du cluster
        if pct_f2p > 50:
            profil = "🆓 **Jeux F2P** - Majoritairement gratuits"
        elif avg_price > 30:
            profil = "💎 **Jeux Premium** - Prix élevé"
        elif avg_ccu > 1000:
            profil = "🔥 **Jeux Populaires** - Fort CCU"
        else:
            profil = "📦 **Jeux Standards** - Profil moyen"
        
        st.markdown(f"""
        **Cluster {c}** ({len(cluster_df):,} jeux) : {profil}
        - Prix moyen : ${avg_price:.2f}
        - CCU moyen : {avg_ccu:,.0f}
        - % F2P : {pct_f2p:.0f}%
        """)
    
    st.markdown("""
    ---
    ## 💡 Limites et Améliorations Possibles
    
    ### Limites actuelles :
    - **Échantillonnage** : Seuls les 5000 jeux les plus populaires sont analysés (économie de RAM)
    - **Features limitées** : 4 features numériques seulement (pas de genres/tags)
    - **K fixe** : Le nombre de clusters est choisi manuellement
    
    ### Améliorations possibles :
    1. **DBSCAN** : Détection automatique du nombre de clusters + outliers
    2. **UMAP** : Meilleure visualisation que PCA (mais plus lourd en RAM)
    3. **Features textuelles** : Encodage des genres/tags avec TF-IDF
    4. **Silhouette Score** : Métrique pour évaluer la qualité du clustering
    
    ---
    ## 📚 Références Techniques
    
    - **K-Means** : MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations"
    - **PCA** : Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space"
    - **Scikit-learn** : Documentation officielle [scikit-learn.org](https://scikit-learn.org)
    """)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    f"🤖 **GameData360 — ML Clustering Lite** | "
    f"{len(df_ml):,} jeux | K-Means + PCA simple"
)
