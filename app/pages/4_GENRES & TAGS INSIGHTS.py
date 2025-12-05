import streamlit as st
import pandas as pd
import altair as alt
import ast
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. CONFIGURATION
st.set_page_config(page_title="GameData360 — GAMEPLAY DNA", layout="wide")
st.title("GAMEPLAY DNA — Genres & Tags Insights")

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
    with st.spinner('Chargement des données...'):
        df_analyse = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {FILE_PATH}")
    st.stop()

# 3. FILTRES GLOBAUX (SUR PAGE PRINCIPALE)
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
# 4. CONTENU SPÉCIFIQUE : GAMEPLAY DNA
# ---------------------------------------------------------

PLAYTIME_COL = 'Median playtime forever'
CCU_COL = 'Peak CCU'

col1, col2 = st.columns(2)

# --- A. Top Genres & Tags par Engagement ---
with col1:
    df_exploded = df_filtered.explode("Genres").dropna(subset=["Genres"])
    genre_ccu = df_exploded.groupby("Genres")["Peak CCU"].sum().reset_index()
    top_genres_ccu = genre_ccu.nlargest(10, "Peak CCU")
    st.subheader("Top 10 des genres par Peak CCU")
        
    chart_ccu = (
        alt.Chart(top_genres_ccu)
        .mark_bar(color="teal")
        .encode(
            x=alt.X("Peak CCU:Q", title="Total Peak CCU"),
            y=alt.Y("Genres:N", sort="-x", title="Genre"),
            tooltip=["Genres", "Peak CCU"]
        )
        .properties(height=400)
    )
    st.altair_chart(chart_ccu, use_container_width=True)

with col2:
    st.subheader("Top Tags par Engagement (Playtime)")
    df_exp_tags = df_filtered.explode('Tags')
    if not df_exp_tags.empty:
        # Filtrer les tags rares (< 10 jeux)
        valid_tags = df_exp_tags['Tags'].value_counts()
        valid_tags = valid_tags[valid_tags > 10].index
        tag_play = df_exp_tags[df_exp_tags['Tags'].isin(valid_tags)].groupby('Tags')[PLAYTIME_COL].median().nlargest(10).reset_index()
        tag_play['Heures'] = tag_play[PLAYTIME_COL]
        
        chart_tags = (alt.Chart(tag_play).mark_bar(color='#1f77b4').encode(
            x=alt.X('Heures:Q', title="Heures Médianes"),
            y=alt.Y('Tags:N', sort='-x'),
            tooltip=['Tags', 'Heures']
            
        )
            .properties(height=400)
        )
        st.altair_chart(chart_tags, use_container_width=True)

st.divider()

# --- B. Graphe de Similarité (Co-occurrence) ---
st.header("Interactions Gameplay (Co-occurrence Tags)")
st.caption("Les tags qui apparaissent souvent ensemble définissent des mécaniques de jeu (ex: Survival + Open World).")

if len(df_filtered) > 0:
    # On prend les 30 tags les plus fréquents de la sélection actuelle
    all_tags = [t for sublist in df_filtered["Tags"].dropna() for t in sublist]
    top_tags = pd.Series(all_tags).value_counts().head(30).index.tolist()

    @st.cache_data
    def get_cooccurrence(df, allowed_tags):
        pair_counts = {}
        for tags in df["Tags"].dropna():
            # Garder seulement les tags du top 30
            relevant = sorted([t for t in tags if t in allowed_tags])
            for t1, t2 in combinations(relevant, 2):
                pair = (t1, t2)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    pairs = get_cooccurrence(df_filtered, top_tags)
    df_pairs = pd.DataFrame([(k[0], k[1], v) for k, v in pairs.items()], columns=['Tag1', 'Tag2', 'Count'])
    
    # Heatmap
    chart_heat = alt.Chart(df_pairs).mark_rect().encode(
        x=alt.X('Tag1:N', title=None),
        y=alt.Y('Tag2:N', title=None),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Tag1', 'Tag2', 'Count']
    ).properties(height=500)
    st.altair_chart(chart_heat, use_container_width=True)

st.divider()

