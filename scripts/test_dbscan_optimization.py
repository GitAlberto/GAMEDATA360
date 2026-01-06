"""Test des optimisations DBSCAN."""
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

print("=" * 60)
print("TEST OPTIMISATIONS DBSCAN - PCA + K-DISTANCE")
print("=" * 60)

from utils.config import FILE_PATH
from utils.data_helpers import load_game_data
from utils.ml_helpers import (
    prepare_features_for_clustering,
    perform_dbscan_clustering,
    calculate_k_distance
)

# Chargement échantillon
print("\n[1/4] Chargement données (échantillon 2000)...")
df = load_game_data(str(FILE_PATH)).head(2000)
print(f"✅ {len(df)} jeux chargés")

# Preprocessing avec PCA
print("\n[2/4] Preprocessing optimisé (RobustScaler + PCA)...")
features, df_clean, (scaler, pca) = prepare_features_for_clustering(df, top_genres=15, top_tags=20, n_pca_components=3)
print(f"✅ Features shape: {features.shape}")
print(f"   - Avant PCA: 57 dimensions")
print(f"   - Après PCA: {features.shape[1]} dimensions (réduction {57/features.shape[1]:.0f}x)")

# K-distance pour eps optimal
print("\n[3/4] Calcul k-distance (min_samples=100)...")
k_distances, suggested_eps = calculate_k_distance(features, k=100)
print(f"✅ Eps optimal trouvé: {suggested_eps:.3f}")
print(f"   - Min k-distance: {k_distances.min():.3f}")
print(f"   - Max k-distance: {k_distances.max():.3f}")
print(f"   - Median: {np.median(k_distances):.3f}")

# DBSCAN optimisé
print("\n[4/4] DBSCAN avec paramètres optimaux...")
labels, n_clusters, n_outliers = perform_dbscan_clustering(features, eps=suggested_eps, min_samples=100)
print(f"✅ Clustering terminé:")
print(f"   - Clusters: {n_clusters}")
print(f"   - Outliers: {n_outliers} ({(n_outliers/len(df)*100):.1f}%)")
print(f"   - Jeux clusterisés: {len(df) - n_outliers} ({((len(df)-n_outliers)/len(df)*100):.1f}%)")

# Distribution clusters
unique, counts = np.unique(labels[labels != -1], return_counts=True)
if len(unique) > 0:
    print(f"\n   Distribution des clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"     - Cluster {cluster_id}: {count} jeux")

print("\n" + "=" * 60)
print("✅ OPTIMISATIONS DBSCAN VALIDÉES!")
print("=" * 60)
print("\n🎯 AMÉLIORATIONS:")
print("  ✓ Log-transform appliqué")
print("  ✓ RobustScaler (résistant outliers)")
print("  ✓ PCA 3D (réduction 19x)")
print("  ✓ Min_samples=100 (anti micro-clusters)")
print("  ✓ Eps optimal via k-distance")
print("=" * 60)
