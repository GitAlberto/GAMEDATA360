import streamlit as st
import pandas as pd
import altair as alt
import ast

st.set_page_config(page_title="GameData360 — BETA vs PUBLISHED", layout="wide")
st.title("PUBLISHED vs BETA — Game Release Quality")

# 1. CHARGEMENT DOUBLE (PUBLISHED + BETA)
@st.cache_data
def load_combined_data(path_main, path_beta):
    # --- A. Chargement Published ---
    df_pub = pd.read_csv(path_main)
    # Parsing listes
    for col in ["Genres", "Categories", "Tags"]:
        if col in df_pub.columns:
            df_pub[col] = df_pub[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_pub['Status'] = 'Published'
    
    # --- B. Chargement Beta ---
    df_beta = pd.read_csv(path_beta)
    # Parsing chaînes (ex: "Action,Indie") -> Listes
    for col in ["Genres", "Categories", "Tags"]:
        if col in df_beta.columns:
            # On remplit les NaN par "" puis on split par virgule
            df_beta[col] = df_beta[col].fillna("").apply(lambda x: x.split(',') if x else [])
    df_beta['Status'] = 'Beta / Early Access'
    
    # --- C. Fusion ---
    # On garde les colonnes communes essentielles
    cols = ['Name', 'Price', 'User score', 'Peak CCU', 'Genres', 'Tags', 'Status', 'AppID']
    df_combined = pd.concat([df_pub[cols], df_beta[cols]], ignore_index=True)
    return df_combined, df_beta # On retourne aussi df_beta seul pour l'analyse spécifique

PATH_MAIN = r"data/nettoyes/jeux_analysis_final.csv"
PATH_BETA = r"data/nettoyes/jeux_beta.csv"

try:
    with st.spinner('Fusion des datasets Beta et Published...'):
        df_all, df_beta_only = load_combined_data(PATH_MAIN, PATH_BETA)
except Exception as e:
    st.error(f"Erreur chargement: {e}")
    st.stop()

# 2. ANALYSE COMPARATIVE
st.header("Comparaison Globale")

# Camembert Répartition
count_data = df_all['Status'].value_counts().reset_index()
count_data.columns = ['Status', 'Count']
c1, c2 = st.columns(2)
with c1:
    st.subheader("Répartition du Volume")
    st.altair_chart(alt.Chart(count_data).mark_arc(innerRadius=50).encode(
        theta='Count', color='Status', tooltip=['Status', 'Count']
    ), use_container_width=True)

# Barres KPI
with c2:
    st.subheader("KPI Moyens (Score & CCU)")
    kpi = df_all.groupby('Status')[['User score', 'Peak CCU', 'Price']].mean().reset_index().melt('Status')
    st.altair_chart(alt.Chart(kpi).mark_bar().encode(
        x='Status', y='value', color='Status', column='variable', tooltip=['value']
    ), use_container_width=True)

# 3. GENRES BETA
st.header("Genres les plus présents en Beta")
# On utilise df_beta_only qui vient du fichier Beta spécifique
df_beta_genres = df_beta_only.explode('Genres')
# On filtre les vides
df_beta_genres = df_beta_genres[df_beta_genres['Genres'] != ""]

if not df_beta_genres.empty:
    top_beta = df_beta_genres['Genres'].value_counts().head(10).reset_index()
    top_beta.columns = ['Genre', 'Nombre de jeux Beta']
    st.altair_chart(alt.Chart(top_beta).mark_bar(color='orange').encode(
        x='Nombre de jeux Beta', y=alt.Y('Genre', sort='-x')
    ), use_container_width=True)
else:
    st.warning("Pas assez d'info sur les genres dans le fichier Beta.")

# 4. TABLEAU
st.header("Tous les jeux Beta")
st.dataframe(df_beta_only[['Name', 'Genres', 'Price', 'User score', 'Peak CCU']])