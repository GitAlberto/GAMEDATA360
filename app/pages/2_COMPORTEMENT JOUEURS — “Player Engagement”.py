import streamlit as st
import pandas as pd
import altair as alt
import ast

# Paramétrage de la page
st.set_page_config(page_title="GameData360 — COMPORTEMENT JOUEURS", layout="wide")
st.title("🎮 COMPORTEMENT JOUEURS — Analyse des comportements des joueurs")

# Importation des données
df_analyse = pd.read_csv(
    r"C:\Users\bongu\Documents\GAMEDATA360\data\nettoyes\jeux_analysis_final.csv"
)

# Conversion automatique des colonnes en liste
for col in ["Genres", "Categories", "Tags"]:
    df_analyse[col] = df_analyse[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
# Titre des filtres
st.header("🔎 Filtres généraux")

# Extraction des valeurs uniques
unique_genres = sorted({g.strip().lower() for lst in df_analyse["Genres"] for g in lst if g})
unique_cats = sorted({c.strip().lower() for lst in df_analyse["Categories"] for c in lst if c})
unique_tags = sorted({t.strip().lower() for lst in df_analyse["Tags"] for t in lst if t})

col_f1, col_f2, col_f3 = st.columns(3) # Création de 3 colonnes pour les filtres

# Colonne 1 genres, Colonne 2 catégories, Colonne 3 tags
with col_f1:
    selected_genres = st.multiselect("🎭 Genres", unique_genres)
with col_f2:
    selected_categories = st.multiselect("📂 Catégories", unique_cats)
with col_f3:
    selected_tags = st.multiselect("🏷️ Tags", unique_tags)

# Bouton Reset pour réinitialiser les filtres
if st.button("🔄 Réinitialiser les filtres"):
    selected_genres = []
    selected_categories = []
    selected_tags = []
    st.experimental_rerun()

# ------------------------------------------------------------
# 🔥 APPLICATION DES FILTRES GLOBAUX
# ------------------------------------------------------------
df_filtered = df_analyse.copy()

# Filtre Genre
if selected_genres:
    df_filtered = df_filtered[
        df_filtered["Genres"].apply(
            lambda lst: any(g in [x.lower() for x in lst] for g in selected_genres)
        )
    ]

# Filtre Catégorie
if selected_categories:
    df_filtered = df_filtered[
        df_filtered["Categories"].apply(
            lambda lst: any(c in [x.lower() for x in lst] for c in selected_categories)
        )
    ]

# Filtre Tags
if selected_tags:
    df_filtered = df_filtered[
        df_filtered["Tags"].apply(
            lambda lst: any(t in [x.lower() for x in lst] for t in selected_tags)
        )
    ]

# Affichage du nombre de jeux après application des filtres
st.success(f"🎯 Jeux après filtres : {df_filtered.shape[0]} / {df_analyse.shape[0]}") 


# ------------------------------------------------------------
# 📊 KPI GLOBALS
# Nuage de points Peak CCU vs Users score
col1, col2 = st.columns(2)
with col1:
    st.metric("🎮 Nombre total de jeux", df_filtered.shape[0])

with col2:
    if "Estimated revenue" in df_filtered.columns:
        total_revenue = df_filtered["Estimated revenue"].sum() / 1e9
        st.metric("💰 Revenu total estimé (milliards USD)", f"${total_revenue:.2f}B")
    else:
        st.metric("💰 Revenu estimé", "Non disponible")

# Peak CCU vs User Score (scatter)
st.header("📊 Peak CCU vs User score")
scatter = (
    alt.Chart(df_filtered)
    .mark_circle(size=60, opacity=0.6)
    .encode(
        x=alt.X("Peak CCU", title="Peak CCU (échelle logarithmique)"),
        y=alt.Y("User score", title="User Score"),
        #color=alt.Color("Genres:N", title="Genre"),
        tooltip=["Name", "Peak CCU", "User score"],
    )
    .properties(width=700, height=400)
    .interactive()
)
st.altair_chart(scatter, use_container_width=True)