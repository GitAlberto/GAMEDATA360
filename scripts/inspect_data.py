"""Script d'inspection rapide des données pour le clustering."""
import pandas as pd
import sys
from pathlib import Path

# Ajout du chemin parent
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

from utils.config import FILE_PATH

# Chargement d'un échantillon
print("=" * 60)
print("INSPECTION DES DONNÉES - GAMEDATA360")
print("=" * 60)

df = pd.read_csv(FILE_PATH, nrows=3)

print(f"\nCOLONNES DISPONIBLES ({len(df.columns)}):")
print("-" * 60)
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    print(f"{i:2}. {col:35} [{dtype}]")

# Total de lignes
df_full = pd.read_csv(FILE_PATH)
print(f"\n{'='*60}")
print(f"TOTAL LIGNES: {len(df_full):,}")
print(f"TOTAL COLONNES: {len(df_full.columns)}")
print(f"\nClés pour le clustering:")
print("  ✓ Prix (Price)")
print("  ✓ CCU (Peak CCU)")
print("  ✓ Metacritic (Metacritic score)")
print("  ✓ User Score (User score)")
print("  ✓ Genres (liste)")
print("  ✓ Tags (liste)")
print("  ✓ Categories (liste)")
print("=" * 60)

# Affichage sample
print("\nÉCHANTILLON DE DONNÉES:")
print("-" * 60)
print(df[['Name', 'Price', 'Peak CCU', 'Metacritic score', 'User score']].to_string())
print("=" * 60)
