import streamlit as st
import pandas as pd
import altair as alt
import ast

# 1. CONFIGURATION
st.set_page_config(page_title="GameData360 — TENDANCES", layout="wide")
st.title("TENDANCES — 10-Year Trends")

# 2. CHARGEMENT
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    for col in ["Genres", "Categories", "Tags"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

FILE_PATH = r"C:\Users\bongu\Documents\GAMEDATA360\data\nettoyes\jeux_analysis_final.csv"

try:
    with st.spinner('Chargement...'):
        df_analyse = load_data(FILE_PATH)
except FileNotFoundError:
    st.stop()

# 3. FILTRES GLOBAUX
st.header("Filtres généraux")
unique_genres = sorted({g.strip().lower() for lst in df_analyse["Genres"] if isinstance(lst, list) for g in lst if g})
unique_cats = sorted({c.strip().lower() for lst in df_analyse["Categories"] if isinstance(lst, list) for c in lst if c})
unique_tags = sorted({t.strip().lower() for lst in df_analyse["Tags"] if isinstance(lst, list) for t in lst if t})

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    selected_genres = st.multiselect("Genres", unique_genres)
with col_f2:
    selected_categories = st.multiselect("Catégories", unique_cats)
with col_f3:
    selected_tags = st.multiselect("Tags", unique_tags)

if st.button("Réinitialiser les filtres"):
    st.rerun()

df_filtered = df_analyse.copy()
if selected_genres:
    df_filtered = df_filtered[df_filtered["Genres"].apply(lambda lst: isinstance(lst, list) and any(g in [x.lower() for x in lst] for g in selected_genres))]
if selected_categories:
    df_filtered = df_filtered[df_filtered["Categories"].apply(lambda lst: isinstance(lst, list) and any(c in [x.lower() for x in lst] for c in selected_categories))]
if selected_tags:
    df_filtered = df_filtered[df_filtered["Tags"].apply(lambda lst: isinstance(lst, list) and any(t in [x.lower() for x in lst] for t in selected_tags))]

st.success(f"Jeux après filtres : {df_filtered.shape[0]} / {df_analyse.shape[0]}")
st.divider()

# ---------------------------------------------------------
# 6. CONTENU SPÉCIFIQUE : TENDANCES (CORRIGÉ)
# ---------------------------------------------------------

# On filtre pour avoir une timeline propre (ex: 2010 - 2024)
df_trends = df_filtered[(df_filtered['Release Year'] >= 2010) & (df_filtered['Release Year'] <= 2024)].copy()

# 1. Calcul des stats générales (Volume, CCU, Price, User Score) sur TOUT le dataset
# On ne met PAS Metacritic ici pour ne pas le calculer sur les 0
yearly_stats = df_trends.groupby('Release Year').agg({
    'AppID': 'count',
    'User score': 'mean',
    'Peak CCU': 'mean',
    'Price': 'mean' 
}).reset_index()

# 2. Calcul spécifique pour Metacritic (uniquement > 0)
# On isole les jeux qui ont un vrai score pour avoir une médiane représentative
df_meta_clean = df_trends[df_trends['Metacritic score'] > 0]
meta_stats = df_meta_clean.groupby('Release Year')['Metacritic score'].median().reset_index()

# 3. Fusion des deux tableaux
# On ajoute la colonne 'Metacritic score' propre dans yearly_stats
yearly_stats = pd.merge(yearly_stats, meta_stats, on='Release Year', how='left')

# --- A. Graphiques d'Évolution (Lignes) ---
st.header("Évolution Historique")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Volume de Jeux Publiés")
    chart_vol = alt.Chart(yearly_stats).mark_line(point=True, color="#3CC8BF").encode(
        x='Release Year:O', 
        y=alt.Y('AppID', title="Nombre de jeux"),
        tooltip=['Release Year', 'AppID']
    )
    st.altair_chart(chart_vol, use_container_width=True)

with c2:
    st.subheader("Metacritic : Moyenne vs Médiane")
    
    # 1. Filtrage spécifique : On ne garde que les jeux ayant un score Metacritic (> 0)
    # Sinon les 0 tirent la moyenne vers le bas violemment
    df_meta_clean = df_trends[df_trends['Metacritic score'] > 0].copy()

    # 2. Calculer Moyenne et Médiane par année en une seule fois
    meta_evol = df_meta_clean.groupby('Release Year')['Metacritic score'].agg(
        Moyenne='mean', 
        Médiane='median'
    ).reset_index()

    # 3. Transformation pour Altair (Format "Long")
    # On pivote le tableau pour avoir une colonne "Type" (Moyenne ou Médiane)
    meta_melted = meta_evol.melt(
        id_vars='Release Year', 
        value_vars=['Moyenne', 'Médiane'], 
        var_name='Indicateur', 
        value_name='Score'
    )

    # 4. Graphique Multi-lignes
    chart_meta = alt.Chart(meta_melted).mark_line(point=True).encode(
        x=alt.X('Release Year:O', title="Année"),
        
        # On ajuste l'échelle (souvent entre 60 et 80) pour bien voir l'écart
        y=alt.Y('Score', scale=alt.Scale(domain=[60, 85]), title="Score Metacritic"),
        
        # La couleur distingue les deux lignes automatiquement
        color=alt.Color('Indicateur', scale=alt.Scale(range=['#FF33A8', '#33FF57'])), # Rose (Moyenne) / Vert (Médiane)
        
        tooltip=['Release Year', 'Indicateur', alt.Tooltip('Score', format='.1f')]
    )

    st.altair_chart(chart_meta, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Peak CCU Moyen")
    chart_ccu = alt.Chart(yearly_stats).mark_line(point=True, color='#3357FF').encode(
        x='Release Year:O', 
        y='Peak CCU', 
        tooltip=['Release Year', 'Peak CCU']
    )
    st.altair_chart(chart_ccu, use_container_width=True)

# --- B. Évolution des Genres Dominants ---
with c4:
    st.subheader("Évolution des Genres Dominants (Top 5)")
    df_exp = df_trends.explode('Genres')
    if not df_exp.empty:
        top_genres = df_exp['Genres'].value_counts().head(5).index
        df_area = df_exp[df_exp['Genres'].isin(top_genres)].groupby(['Release Year', 'Genres']).size().reset_index(name='Count')
        
        chart_area = alt.Chart(df_area).mark_area().encode(
            x='Release Year:O',
            y=alt.Y('Count', stack='normalize', title='Part de Marché'),
            color='Genres',
            tooltip=['Release Year', 'Genres', 'Count']
        )
        st.altair_chart(chart_area, use_container_width=True)

# --- C. Évolution Modèle Éco ---
st.header("Évolution des Modèles Économiques")
df_trends['Model'] = df_trends['Price'].apply(lambda x: 'Free-to-Play' if x == 0 else 'Payant')
model_stats = df_trends.groupby(['Release Year', 'Model']).size().reset_index(name='Count')

chart_model = alt.Chart(model_stats).mark_bar().encode(
    x='Release Year:O',
    y='Count',
    color='Model',
    tooltip=['Release Year', 'Model', 'Count']
)
st.altair_chart(chart_model, use_container_width=True)

# --- D. Table Timeline ---
st.header("Timeline Complète")
st.dataframe(yearly_stats.style.format("{:.1f}", subset=['User score', 'Peak CCU', 'Price']), use_container_width=True)