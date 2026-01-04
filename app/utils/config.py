# -*- coding: utf-8 -*-
"""
Configuration centralisée pour GameData360.
Contient les constantes, palettes de couleurs et chemins.
"""

from pathlib import Path

# ============================================================
# CHEMINS DES FICHIERS
# ============================================================
# Utilisation de Path pour la portabilité
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "nettoyes"
FILE_PATH = DATA_DIR / "jeux_analysis_final.csv"

# ============================================================
# PALETTE DE COULEURS GAMING (Thème Néon Sombre)
# ============================================================
COLORS = {
    # Couleurs principales
    "primary": "#00ff88",      # Vert néon (succès, positif)
    "secondary": "#ff00ff",    # Magenta néon (accent)
    "tertiary": "#00ffff",     # Cyan néon (info)
    "warning": "#ffaa00",      # Orange néon (attention)
    "danger": "#ff3366",       # Rouge rosé (négatif)
    
    # Variations pour graphiques
    "chart": [
        "#00ff88",  # Vert néon
        "#ff00ff",  # Magenta
        "#00ffff",  # Cyan
        "#ffaa00",  # Orange
        "#ff3366",  # Rouge rosé
        "#7c3aed",  # Violet
        "#3b82f6",  # Bleu
        "#10b981",  # Émeraude
        "#f59e0b",  # Ambre
        "#ec4899",  # Rose
    ],
    
    # Comparaisons spécifiques
    "solo": "#3b82f6",         # Bleu pour Solo
    "multi": "#f97316",        # Orange pour Multi
    "benchmark": "#ff3366",    # Rouge pour ligne de référence
    
    # Fond et grille
    "background": "#0a0a0a", # Noir
    "grid": "#1a1a1a", # Gris
    "text": "#e5e5e5",  # Gris clair
}

# ============================================================
# CONFIGURATION PLOTLY (Thème Gaming Sombre)
# ============================================================
PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "Rajdhani, sans-serif",
        "color": COLORS["text"],
        "size": 12
    },
    "hoverlabel": {
        "bgcolor": "#1a1a1a", # Gris
        "font_size": 13,
        "font_family": "Rajdhani"
    },
    "margin": {"l": 40, "r": 40, "t": 50, "b": 40},
}

# Configuration des axes (à appliquer séparément pour éviter les conflits)
PLOTLY_AXIS = {
    "gridcolor": COLORS["grid"],
    "zerolinecolor": COLORS["grid"],
}

# ============================================================
# COLONNES À CONVERTIR (listes Python)
# ============================================================
LIST_COLUMNS = ["Genres", "Categories", "Tags"]

# ============================================================
# COLONNES AVEC TYPES OPTIMISÉS (économie mémoire)
# ============================================================
DTYPE_OPTIMIZATIONS = {
    "AppID": "int32",
    "Price": "float32",
    "Peak CCU": "int32",
    "Recommendations": "int32",
    "Median playtime forever": "float32",
    "Average playtime forever": "float32",
    "Release Year": "int16",
    "Windows": "bool",
    "Mac": "bool",
    "Linux": "bool",
}

# ============================================================
# MOTS-CLÉS POUR CLASSIFICATION SOLO/MULTI
# ============================================================
MULTI_KEYWORDS = {
    'multi-player', 'mmo', 'co-op', 'online pvp', 
    'online co-op', 'cross-platform multiplayer',
    'pvp', 'massively multiplayer'
}

SOLO_KEYWORDS = {'single-player'}

# ============================================================
# GENRES À EXCLURE (Logiciels non-jeux)
# ============================================================
NON_GAME_GENRES = [
    "utilities", 
    "design & illustration", 
    "animation & modeling", 
    "software training", 
    "audio production", 
    "video production", 
    "web publishing", 
    "game development", 
    "photo editing", 
    "accounting", 
]

# ============================================================
# CONFIGURATION DES ONGLETS
# ============================================================
TAB_CONFIG = {
    "vue_ensemble": "📊 Vue d'ensemble",
    "recommandations": "🎮 Recommandations",
    "temps_jeu": "⏱️ Temps de Jeu",
    "solo_multi": "🎯 Solo vs Multi",
}
