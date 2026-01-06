"""Diagnostic rapide de l'installation Streamlit."""
import sys

print("=" * 60)
print("DIAGNOSTIC STREAMLIT")
print("=" * 60)

# 1. Version Python
print(f"\n✓ Python version: {sys.version}")
print(f"✓ Python executable: {sys.executable}")

# 2. Vérifier Streamlit
print("\n--- Test import Streamlit ---")
try:
    import streamlit as st
    print(f"✅ Streamlit installé: version {st.__version__}")
    print(f"   Emplacement: {st.__file__}")
except ImportError as e:
    print(f"❌ Streamlit NON installé!")
    print(f"   Erreur: {e}")
    print("\n🔧 SOLUTION: Installez streamlit avec:")
    print("   python -m pip install streamlit")
    sys.exit(1)

# 3. Vérifier autres dépendances ML
print("\n--- Test dépendances ML ---")
packages = ['pandas', 'numpy', 'plotly', 'scikit-learn', 'umap-learn']
for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'N/A')
        print(f"✅ {pkg}: {version}")
    except ImportError:
        print(f"❌ {pkg}: NON INSTALLÉ")

# 4. Tester chargement config
print("\n--- Test modules projet ---")
try:
    sys.path.insert(0, 'app')
    from utils.config import FILE_PATH, COLORS
    print(f"✅ utils.config importé")
    print(f"   FILE_PATH: {FILE_PATH}")
except Exception as e:
    print(f"❌ Erreur import config: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)
