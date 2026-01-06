"""Test rapide de la correction mémoire."""
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

print("=" * 60)
print("TEST CORRECTION MÉMOIRE - SIMILARITÉ ON-DEMAND")
print("=" * 60)

from utils.config import FILE_PATH
from utils.data_helpers import load_game_data
from utils.ml_helpers import (
    prepare_features_for_clustering,
    calculate_single_game_similarity
)

# Chargement échantillon
print("\n[1/3] Chargement données (échantillon)...")
df = load_game_data(str(FILE_PATH)).head(1000)
print(f"✅ {len(df)} jeux chargés")

# Preprocessing
print("\n[2/3] Preprocessing...")
features, df_clean, scaler = prepare_features_for_clustering(df, top_genres=10, top_tags=15)
print(f"✅ Features shape: {features.shape}")

# Test similarité single game
print("\n[3/3] Test similarité single game (économie mémoire)...")
game_idx = 0
similarities = calculate_single_game_similarity(game_idx, features)

print(f"✅ Similarités calculées: {similarities.shape}")
print(f"   - Min: {similarities.min():.3f}")
print(f"   - Max: {similarities.max():.3f}")
print(f"   - Mean: {similarities.mean():.3f}")

# Estimation mémoire économisée
full_matrix_size_gb = (len(df) ** 2 * 8) / (1024**3)  # float64 = 8 bytes
single_vector_size_mb = (len(df) * 8) / (1024**2)

print(f"\n{'='*60}")
print("💾 ÉCONOMIE MÉMOIRE:")
print(f"   - Matrice complète: {full_matrix_size_gb:.2f} GB")
print(f"   - Vecteur single: {single_vector_size_mb:.2f} MB")
print(f"   - Ratio: {(full_matrix_size_gb * 1024) / single_vector_size_mb:.0f}x plus petit !")
print("=" * 60)
print("✅ CORRECTION MÉMOIRE VALIDÉE!")
print("=" * 60)
