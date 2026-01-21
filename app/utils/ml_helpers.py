# -*- coding: utf-8 -*-
"""
ML Helpers pour GameData360.
Fonctions de clustering, réduction dimensionnelle et recommandations.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap


# ============================================================
# 1. PREPROCESSING & FEATURE ENGINEERING
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_features_for_clustering(
    df: pd.DataFrame, 
    top_genres: int = 20, 
    top_tags: int = 30,
    n_pca_components: int = 3
) -> Tuple[np.ndarray, pd.DataFrame, object]:
    """
    Prépare la matrice de features pour le clustering.
    
    Améliorations:
    - Log-transform des features numériques pour réduire l'impact des outliers
    - RobustScaler au lieu de StandardScaler (meilleur avec outliers)
    - PCA pour réduction dimensionnelle et élimination du bruit
    
    Args:
        df: DataFrame source
        top_genres: Nombre de genres à encoder
        top_tags: Nombre de tags à encoder
        n_pca_components: Nombre de composantes PCA (défaut: 3)
        
    Returns:
        features: Matrice de features normalisées et réduites (numpy array)
        df_clean: DataFrame nettoyé avec index préservé
        scaler: Scaler utilisé (pour inverse transform si besoin)
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    
    df_work = df.copy()
    
    # ---- FEATURES NUMÉRIQUES ----
    numeric_features = [
        'Price',
        'Peak CCU',
        'Metacritic score',
        'User score',
        'Recommendations',
        'Average playtime forever'
    ]
    
    # Filtrer les colonnes existantes
    numeric_features = [col for col in numeric_features if col in df_work.columns]
    
    # Créer DataFrame numérique
    df_numeric = df_work[numeric_features].copy()
    
    # Feature booléenne F2P
    df_numeric['is_f2p'] = (df_work['Price'] == 0).astype(int)
    
    # LOG-TRANSFORM pour toutes les features > 0 (réduire effet outliers)
    for col in ['Price', 'Peak CCU', 'Recommendations', 'Average playtime forever']:
        if col in df_numeric.columns:
            df_numeric[col] = np.log1p(df_numeric[col])  # log(1 + x) pour gérer les 0
    
    # Imputation des NaN par la médiane
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # ---- FEATURES CATÉGORIELLES ----
    
    # Genres (One-Hot Encoding - Top N)
    if 'Genres' in df_work.columns:
        genres_exploded = df_work.explode('Genres')['Genres'].dropna()
        top_genres_list = genres_exploded.value_counts().head(top_genres).index.tolist()
        
        for genre in top_genres_list:
            df_numeric[f'genre_{genre.lower().replace(" ", "_")}'] = df_work['Genres'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
    
    # Tags (One-Hot Encoding - Top N)
    if 'Tags' in df_work.columns:
        tags_exploded = df_work.explode('Tags')['Tags'].dropna()
        top_tags_list = tags_exploded.value_counts().head(top_tags).index.tolist()
        
        for tag in top_tags_list:
            df_numeric[f'tag_{tag.lower().replace(" ", "_").replace("-", "_")}'] = df_work['Tags'].apply(
                lambda x: 1 if isinstance(x, list) and tag in x else 0
            )
    
    # NORMALISATION avec RobustScaler (résistant aux outliers) ----
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(df_numeric)
    
    # ---- PCA pour réduction dimensionnelle ----
    pca = PCA(n_components=n_pca_components, random_state=42)
    features = pca.fit_transform(features_scaled)
    
    # Garder un DataFrame avec les IDs pour mapping
    df_clean = df_work[['AppID', 'Name']].copy()
    df_clean['feature_index'] = range(len(df_clean))
    
    return features, df_clean, (scaler, pca)


# ============================================================
# 2. ALGORITHMES DE CLUSTERING
# ============================================================

@st.cache_data(show_spinner=False)
def perform_kmeans_clustering(_features: np.ndarray, k_range: Tuple[int, int] = (2, 15)) -> Tuple[np.ndarray, int, float, List[float]]:
    """
    K-Means avec recherche du K optimal (Elbow Method + Silhouette).
    
    Args:
        _features: Matrice de features
        k_range: Range de K à tester (min, max)
        
    Returns:
        labels: Labels de cluster
        best_k: Nombre optimal de clusters
        silhouette: Score silhouette moyen
        inertias: Liste des inertias pour Elbow plot
    """
    inertias = []
    silhouettes = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(_features)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette nécessite au moins 2 clusters
            silhouettes.append(silhouette_score(_features, labels))
        else:
            silhouettes.append(0)
    
    # Trouver le meilleur K (Silhouette maximal)
    best_k_idx = np.argmax(silhouettes)
    best_k = k_values[best_k_idx]
    
    # Re-fit avec le meilleur K
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans_best.fit_predict(_features)
    silhouette = silhouette_score(_features, labels)
    
    return labels, best_k, silhouette, inertias


@st.cache_data(show_spinner=False)
def perform_dbscan_clustering(_features: np.ndarray, eps: float = 0.5, min_samples: int = 100) -> Tuple[np.ndarray, int, int]:
    """
    DBSCAN clustering pour détecter les outliers (VERSION OPTIMISÉE).
    
    Args:
        _features: Matrice de features
        eps: Distance maximale entre deux échantillons
        min_samples: Nombre minimal de points pour former un cluster (100 = évite micro-clusters)
        
    Returns:
        labels: Labels de cluster (-1 = outlier)
        n_clusters: Nombre de clusters trouvés
        n_outliers: Nombre d'outliers détectés
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)  # n_jobs=-1 pour parallélisation
    labels = dbscan.fit_predict(_features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    
    return labels, n_clusters, n_outliers


@st.cache_data(show_spinner=False)
def calculate_k_distance(_features: np.ndarray, k: int = 100) -> Tuple[np.ndarray, float]:
    """
    Calcule les k-distances pour trouver le eps optimal (Elbow method pour DBSCAN).
    
    Args:
        _features: Matrice de features
        k: Valeur de k (devrait être = min_samples)
        
    Returns:
        k_distances: Distances triées (pour plot)
        suggested_eps: Eps suggéré (point du coude)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Calculer les k plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(_features)
    distances, indices = nbrs.kneighbors(_features)
    
    # Prendre la distance au k-ième voisin (dernière colonne)
    k_distances = np.sort(distances[:, -1])[::-1]  # Trier décroissant
    
    # Estimer eps avec méthode du coude (95e percentile comme approximation)
    suggested_eps = np.percentile(k_distances, 95)
    
    return k_distances, suggested_eps


@st.cache_data(show_spinner=False)
def perform_hierarchical_clustering(_features: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Clustering hiérarchique (Ward linkage).
    
    Args:
        _features: Matrice de features
        n_clusters: Nombre de clusters souhaité
        
    Returns:
        labels: Labels de cluster
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(_features)
    
    return labels


# ============================================================
# 3. RÉDUCTION DIMENSIONNELLE
# ============================================================

@st.cache_data(show_spinner=False)
def reduce_dimensions_tsne(_features: np.ndarray, perplexity: int = 30, n_components: int = 2) -> np.ndarray:
    """
    T-SNE pour visualisation 2D/3D.
    
    Args:
        _features: Matrice de features
        perplexity: Perplexité T-SNE (entre 5 et 50)
        n_components: 2 ou 3 pour visualisation
        
    Returns:
        coords: Coordonnées réduites (n_samples, n_components)
    """
    # PCA préalable pour accélérer T-SNE
    if _features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(_features)
    else:
        features_pca = _features
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(features_pca)
    
    return coords


@st.cache_data(show_spinner=False)
def reduce_dimensions_umap(_features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2) -> np.ndarray:
    """
    UMAP pour visualisation 2D/3D (plus rapide que T-SNE).
    
    Args:
        _features: Matrice de features
        n_neighbors: Taille du voisinage local
        min_dist: Distance minimale entre points
        n_components: 2 ou 3 pour visualisation
        
    Returns:
        coords: Coordonnées réduites (n_samples, n_components)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        metric='euclidean'
    )
    coords = reducer.fit_transform(_features)
    
    return coords


# ============================================================
# 4. SYSTÈME DE RECOMMANDATION
# ============================================================

@st.cache_data(show_spinner=False)
def calculate_similarity_matrix(_features: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de similarité cosine entre tous les jeux.
    
    Args:
        _features: Matrice de features
        
    Returns:
        similarity_matrix: Matrice de similarité (n_samples, n_samples)
    """
    similarity_matrix = cosine_similarity(_features)
    return similarity_matrix


def calculate_single_game_similarity(
    game_index: int,
    _features: np.ndarray
) -> np.ndarray:
    """
    Calcule la similarité d'un seul jeu vs tous les autres (économie mémoire).
    
    Args:
        game_index: Index du jeu
        _features: Matrice de features complète
        
    Returns:
        similarities: Vecteur de similarités (n_samples,)
    """
    # Récupérer le vecteur de features du jeu
    game_features = _features[game_index:game_index+1]  # Shape (1, n_features)
    
    # Calculer similarité avec tous les autres jeux
    similarities = cosine_similarity(game_features, _features)[0]  # Shape (n_samples,)
    
    return similarities


def get_recommendations(
    game_index: int,
    similarities: np.ndarray,
    df: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
    n: int = 10,
    same_cluster_only: bool = False
) -> pd.DataFrame:
    """
    Obtient les N jeux les plus similaires à un jeu donné.
    
    Args:
        game_index: Index du jeu dans le DataFrame
        similarities: Vecteur de similarité du jeu (shape: n_samples)
        df: DataFrame avec colonnes 'Name', 'AppID', etc.
        cluster_labels: Labels de cluster (optionnel)
        n: Nombre de recommandations
        same_cluster_only: Si True, recommande uniquement les jeux du même cluster
        
    Returns:
        DataFrame avec les recommandations (Name, AppID, Similarity, Cluster)
    """
    # Créer un DataFrame temporaire
    df_temp = df.copy()
    df_temp['Similarity'] = similarities
    
    if cluster_labels is not None:
        df_temp['Cluster'] = cluster_labels
    
    # Exclure le jeu lui-même
    df_temp = df_temp[df_temp.index != game_index]
    
    # Filtrer par cluster si demandé
    if same_cluster_only and cluster_labels is not None:
        game_cluster = cluster_labels[game_index]
        df_temp = df_temp[df_temp['Cluster'] == game_cluster]
    
    # Trier par similarité décroissante
    recommendations = df_temp.nlargest(n, 'Similarity')
    
    return recommendations


def get_cluster_statistics(cluster_id: int, df: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
    """
    Calcule les statistiques d'un cluster donné.
    
    Args:
        cluster_id: ID du cluster
        df: DataFrame complet
        cluster_labels: Labels de cluster
        
    Returns:
        dict avec statistiques (prix moyen, CCU moyen, etc.)
    """
    cluster_mask = cluster_labels == cluster_id
    df_cluster = df[cluster_mask]
    
    stats = {
        'size': len(df_cluster),
        'avg_price': df_cluster['Price'].mean() if 'Price' in df_cluster.columns else 0,
        'avg_ccu': df_cluster['Peak CCU'].mean() if 'Peak CCU' in df_cluster.columns else 0,
        'avg_metacritic': df_cluster['Metacritic score'].mean() if 'Metacritic score' in df_cluster.columns else 0,
        'avg_user_score': df_cluster['User score'].mean() if 'User score' in df_cluster.columns else 0,
        'pct_f2p': (df_cluster['Price'] == 0).mean() * 100 if 'Price' in df_cluster.columns else 0
    }
    
    # Genres dominants
    if 'Genres' in df_cluster.columns:
        genres_exploded = df_cluster.explode('Genres')['Genres'].dropna()
        top_genres = genres_exploded.value_counts().head(5).to_dict()
        stats['top_genres'] = top_genres
    
    # Tags dominants
    if 'Tags' in df_cluster.columns:
        tags_exploded = df_cluster.explode('Tags')['Tags'].dropna()
        top_tags = tags_exploded.value_counts().head(10).to_dict()
        stats['top_tags'] = top_tags
    
    return stats
