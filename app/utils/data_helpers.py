# -*- coding: utf-8 -*-
"""
Fonctions utilitaires pour le chargement et la manipulation des données.
Optimisées pour la performance avec caching et vectorisation.
"""

import ast
from typing import List, Set, Optional
import pandas as pd
import streamlit as st

from .config import (
    FILE_PATH, 
    LIST_COLUMNS, 
    DTYPE_OPTIMIZATIONS,
    MULTI_KEYWORDS,
    SOLO_KEYWORDS
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_game_data(file_path: str = None) -> pd.DataFrame:
    """
    Charge les données des jeux avec optimisations mémoire et cache.
    
    Args:
        file_path: Chemin vers le fichier CSV. Utilise FILE_PATH par défaut.
        
    Returns:
        DataFrame avec les colonnes list parsées et types optimisés.
    """
    path = file_path or str(FILE_PATH)
    
    # Chargement avec types optimisés
    df = pd.read_csv(path, dtype={
        k: v for k, v in DTYPE_OPTIMIZATIONS.items() 
        if k not in LIST_COLUMNS
    })
    
    # Suppression des doublons
    df = df.drop_duplicates(subset=["AppID"])
    
    # Conversion des colonnes string → listes Python
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_safe_literal_eval)
    
    # Pré-calcul des versions lowercase pour filtrage rapide
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[f"{col}_lower"] = df[col].apply(
                lambda x: [i.lower().strip() for i in x] if isinstance(x, list) else []
            )
    
    return df


def _safe_literal_eval(x):
    """Parse sécurisé d'une string vers liste Python."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []


@st.cache_data(show_spinner=False)
def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """
    Extrait les valeurs uniques d'une colonne contenant des listes.
    Retourne une liste triée en lowercase.
    """
    lower_col = f"{column}_lower"
    if lower_col in df.columns:
        values = {
            item 
            for lst in df[lower_col] 
            if isinstance(lst, list) 
            for item in lst 
            if item
        }
    else:
        values = {
            item.lower().strip() 
            for lst in df[column] 
            if isinstance(lst, list) 
            for item in lst 
            if item
        }
    return sorted(values)


def apply_list_filter(
    df: pd.DataFrame, 
    column: str, 
    selected_values: List[str]
) -> pd.DataFrame:
    """
    Filtre vectorisé pour colonnes contenant des listes.
    Beaucoup plus performant que .apply() avec lambda.
    
    Args:
        df: DataFrame source
        column: Nom de la colonne (ex: "Genres")
        selected_values: Valeurs à filtrer
        
    Returns:
        DataFrame filtré
    """
    if not selected_values:
        return df
    
    lower_col = f"{column}_lower"
    selected_set = set(selected_values)
    
    if lower_col in df.columns:
        mask = df[lower_col].apply(
            lambda x: bool(set(x) & selected_set) if isinstance(x, list) else False
        )
    else:
        mask = df[column].apply(
            lambda x: bool(
                set(i.lower().strip() for i in x) & selected_set
            ) if isinstance(x, list) else False
        )
    
    return df[mask].copy()


def apply_all_filters(
    df: pd.DataFrame,
    genres: List[str] = None,
    categories: List[str] = None,
    tags: List[str] = None
) -> pd.DataFrame:
    """
    Applique tous les filtres en une seule passe.
    
    Returns:
        DataFrame filtré
    """
    result = df.copy()
    
    if genres:
        result = apply_list_filter(result, "Genres", genres)
    if categories:
        result = apply_list_filter(result, "Categories", categories)
    if tags:
        result = apply_list_filter(result, "Tags", tags)
    
    return result


@st.cache_data(show_spinner=False)
def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explose la colonne Genres une seule fois et met en cache.
    Évite de refaire cette opération coûteuse à chaque graphique.
    """
    return (
        df.explode("Genres")
        .dropna(subset=["Genres"])
        .copy()
    )


def categorize_game_mode(categories: list) -> str:
    """
    Classifie un jeu en Solo, Multijoueur ou Autre.
    
    Args:
        categories: Liste des catégories du jeu
        
    Returns:
        "Solo", "Multijoueur / Co-op", ou "Autre"
    """
    if not isinstance(categories, list):
        return "Inconnu"
    
    cats_lower = {c.lower() for c in categories}
    
    if cats_lower & MULTI_KEYWORDS:
        return "Multijoueur / Co-op"
    elif cats_lower & SOLO_KEYWORDS:
        return "Solo"
    return "Autre"


def calculate_genre_stats(
    df: pd.DataFrame, 
    metric_col: str, 
    agg_func: str = "sum",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Calcule les statistiques par genre de manière optimisée.
    
    Args:
        df: DataFrame (peut être déjà explosé ou non)
        metric_col: Colonne métrique (ex: "Peak CCU")
        agg_func: Fonction d'agrégation ("sum", "median", "mean")
        top_n: Nombre de résultats à retourner
        
    Returns:
        DataFrame avec Genre et valeur agrégée
    """
    # Explosion si nécessaire
    if df["Genres"].apply(lambda x: isinstance(x, list)).any():
        df_exploded = explode_genres(df)
    else:
        df_exploded = df
    
    # Agrégation
    result = (
        df_exploded.groupby("Genres")[metric_col]
        .agg(agg_func)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    
    result.columns = ["Genre", metric_col]
    return result


def format_number(num: float, decimals: int = 2) -> str:
    """Formate un nombre avec notation K/M/B."""
    if num >= 1e9:
        return f"${num/1e9:.{decimals}f}B"
    elif num >= 1e6:
        return f"${num/1e6:.{decimals}f}M"
    elif num >= 1e3:
        return f"${num/1e3:.{decimals}f}K"
    return f"${num:.{decimals}f}"
