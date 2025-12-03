import streamlit as st
import pandas as pd
import ast
import base64

# 1. CONFIGURATION
st.set_page_config(page_title="GameData360 — DATA EXPLORER", layout="wide")
st.title("DATA EXPLORER — Raw Data & Filters")

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
# 9. FILTRES SUPPLÉMENTAIRES (Interface Libre)
# ---------------------------------------------------------

st.header("Filtres Avancés (Cumulatifs)")

c1, c2, c3 = st.columns(3)

with c1:
    min_score, max_score = st.slider("Score Utilisateur", 0, 100, (0, 100))
    os_filter = st.multiselect("Système d'exploitation", ['Windows', 'Mac', 'Linux'])

with c2:
    max_price = st.slider("Prix Maximum ($)", 0, 200, 200)
    # Filtre Beta
    show_beta = st.checkbox("Afficher uniquement Beta / Early Access")

with c3:
    if 'Release Year' in df_filtered.columns:
        min_year = int(df_filtered['Release Year'].min())
        max_year = int(df_filtered['Release Year'].max())
        sel_year = st.slider("Année de Sortie", min_year, max_year, (min_year, max_year))

# Application des filtres supplémentaires
df_explorer = df_filtered[
    (df_filtered['User score'] >= min_score) & 
    (df_filtered['User score'] <= max_score) & 
    (df_filtered['Price'] <= max_price) &
    (df_filtered['Release Year'] >= sel_year[0]) & 
    (df_filtered['Release Year'] <= sel_year[1])
]

if show_beta:
    df_explorer = df_explorer[df_explorer['Tags'].apply(lambda x: isinstance(x, list) and any('early access' in t.lower() for t in x))]

if os_filter:
    for os_col in os_filter:
        if os_col in df_explorer.columns:
            df_explorer = df_explorer[df_explorer[os_col] == True]

st.info(f"Résultat final : {len(df_explorer)} jeux.")

# Table Filtrable
st.dataframe(df_explorer, height=600, use_container_width=True)

# Bouton Download CSV
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="gamedata360_export.csv" style="padding:10px; background-color:#4CAF50; color:white; border-radius:5px; text-decoration:none;">📥 Télécharger en CSV</a>'

st.markdown(get_csv_download_link(df_explorer), unsafe_allow_html=True)