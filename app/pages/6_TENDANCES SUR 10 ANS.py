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

FILE_PATH = r"data/nettoyes/jeux_analysis_final.csv"

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
# 6. CONTENU SPÉCIFIQUE : TENDANCES
# ---------------------------------------------------------

# On filtre pour avoir une timeline propre (ex: 2010 - 2024)
df_trends = df_filtered[(df_filtered['Release Year'] >= 2010) & (df_filtered['Release Year'] <= 2024)].copy()

# Aggrégation par année
yearly_stats = df_trends.groupby('Release Year').agg({
    'AppID': 'count',
    'User score': 'mean',
    'Peak CCU': 'mean',
    'Price': 'mean' # Ajouté pour le sentiment/prix
}).reset_index()

# --- A. Graphiques d'Évolution (Lignes) ---
st.header("Évolution Historique")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Volume de Jeux Publiés")
    chart_vol = alt.Chart(yearly_stats).mark_line(point=True, color='#FF5733').encode(
        x='Release Year:O', y='AppID', tooltip=['Release Year', 'AppID']
    )
    st.altair_chart(chart_vol, use_container_width=True)

with c2:
    st.subheader("Score Moyen & Sentiment")
    chart_score = alt.Chart(yearly_stats).mark_line(point=True, color='#33FF57').encode(
        x='Release Year:O', y=alt.Y('User score', scale=alt.Scale(domain=(50, 100))), tooltip=['Release Year', 'User score']
    )
    st.altair_chart(chart_score, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Peak CCU Moyen")
    chart_ccu = alt.Chart(yearly_stats).mark_line(point=True, color='#3357FF').encode(
        x='Release Year:O', y='Peak CCU', tooltip=['Release Year', 'Peak CCU']
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
st.header("volution des Modèles Économiques")
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