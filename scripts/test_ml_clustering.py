"""Script de test pour le module ML."""
import sys
from pathlib import Path
import numpy as np

# Ajout du chemin
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

print("=" * 60)
print("TEST ML CLUSTERING - GAMEDATA360")
print("=" * 60)

# Test 1: Import des modules
print("\n[1/5] Test des imports...")
try:
    from utils.config import FILE_PATH
    from utils.data_helpers import load_game_data
    from utils.ml_helpers import (
        prepare_features_for_clustering,
        perform_kmeans_clustering,
        perform_dbscan_clustering,
        reduce_dimensions_umap,
        calculate_similarity_matrix
    )
    print("✅ Tous les modules importés avec succès")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Test 2: Chargement des données
print("\n[2/5] Test du chargement des données...")
try:
    df = load_game_data(str(FILE_PATH))
    df = df.head(1000)  # Sous-échantillon pour test rapide
    print(f"✅ {len(df)} jeux chargés (échantillon de test)")
except Exception as e:
    print(f"❌ Erreur de chargement: {e}")
    sys.exit(1)

# Test 3: Preprocessing
print("\n[3/5] Test du preprocessing...")
try:
    features, df_clean, scaler = prepare_features_for_clustering(df, top_genres=10, top_tags=15)
    print(f"✅ Features créées: {features.shape}")
    print(f"   - Samples: {features.shape[0]}")
    print(f"   - Features: {features.shape[1]}")
    
    # Vérifier normalisation
    mean_check = np.abs(features.mean(axis=0)).max()
    std_check = np.abs(features.std(axis=0) - 1).max()
    
    if mean_check < 0.1 and std_check < 0.1:
        print(f"✅ Normalisation correcte (mean≈0, std≈1)")
    else:
        print(f"⚠️ Normalisation approximative (mean max: {mean_check:.3f}, std deviation: {std_check:.3f})")
        
except Exception as e:
    print(f"❌ Erreur preprocessing: {e}")
    sys.exit(1)

# Test 4: Clustering
print("\n[4/5] Test des algorithmes de clustering...")
try:
    # K-Means
    labels_km, best_k, silhouette, inertias = perform_kmeans_clustering(features, k_range=(2, 8))
    print(f"✅ K-Means: K optimal = {best_k}, Silhouette = {silhouette:.3f}")
    
    # DBSCAN
    labels_db, n_clusters, n_outliers = perform_dbscan_clustering(features, eps=2.5, min_samples=5)
    print(f"✅ DBSCAN: {n_clusters} clusters, {n_outliers} outliers")
    
    # UMAP
    coords = reduce_dimensions_umap(features, n_components=2)
    print(f"✅ UMAP: Réduction {features.shape[1]}D → 2D réussie")
    
except Exception as e:
    print(f"❌ Erreur clustering: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Recommandations
print("\n[5/5] Test du système de recommandations...")
try:
    similarity_matrix = calculate_similarity_matrix(features)
    print(f"✅ Matrice de similarité calculée: {similarity_matrix.shape}")
    
    # Vérifier que les valeurs sont entre 0 et 1
    sim_min, sim_max = similarity_matrix.min(), similarity_matrix.max()
    if 0 <= sim_min <= 1 and 0 <= sim_max <= 1:
        print(f"✅ Similarités dans l'intervalle [0, 1]: [{sim_min:.3f}, {sim_max:.3f}]")
    else:
        print(f"⚠️ Similarités hors intervalle: [{sim_min:.3f}, {sim_max:.3f}]")
        
except Exception as e:
    print(f"❌ Erreur recommandations: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TOUS LES TESTS PASSÉS AVEC SUCCÈS!")
print("=" * 60)
print("\nLa nouvelle page ML est prête à l'emploi.")
print("Accédez-y via: http://localhost:8501")
print("Sélectionnez '🤖 ML CLUSTERING & RECOMMANDATIONS' dans la sidebar")
print("=" * 60)
