import streamlit as st
import pandas as pd
import altair as alt
import ast

# 1. CONFIGURATION
st.set_page_config(page_title="GameData360 — ÉCONOMIE", layout="wide")
st.title("ÉCONOMIE — Pricing & Monetization Insights")

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
# 5. CONTENU SPÉCIFIQUE : ÉCONOMIE
# ---------------------------------------------------------

PRICE_COL = 'Price'
CCU_COL = 'Peak CCU'

col1, col2 = st.columns(2)

# --- A. Prix Moyen par Année ---
with col1:
    st.subheader("Prix Moyen au fil des Années")
    if 'Release Year' in df_filtered.columns:
        df_price_year = df_filtered[df_filtered['Release Year'] >= 2010].groupby('Release Year')[PRICE_COL].mean().reset_index()
        chart_year = alt.Chart(df_price_year).mark_line(point=True).encode(
            x='Release Year:O', 
            y=alt.Y(PRICE_COL, title="Prix Moyen ($)"),
            tooltip=['Release Year', alt.Tooltip(PRICE_COL, format='$.2f')]
        )
        st.altair_chart(chart_year, use_container_width=True)
# --- B. Prix Moyen par Genre ---
with col2:
    st.subheader("Prix Moyen par Genre") # (Top 10 n'est plus pertinent si on filtre un seul genre)
    
    # 1. On "explose" la liste des genres
    df_exp = df_filtered.explode('Genres')

    # ---------------------------------------------------------
    # CORRECTION ICI :
    # Si des genres sont sélectionnés, on ne garde QUE ceux-là dans le graphique.
    # Sinon, on garde tout.
    # ---------------------------------------------------------
    if selected_genres:
        # On s'assure de comparer des minuscules car unique_genres est en minuscule
        df_exp = df_exp[df_exp['Genres'].astype(str).str.lower().isin(selected_genres)]

    if not df_exp.empty:
        df_genre_price = df_exp.groupby('Genres')[PRICE_COL].mean().nlargest(10).reset_index()
        
        chart_genre = alt.Chart(df_genre_price, height=350).mark_bar().encode(
            x=alt.X(PRICE_COL, title="Prix Moyen ($)"),
            y=alt.Y('Genres', sort='-x'),
            color=alt.Color(PRICE_COL, legend=None), 
            tooltip=['Genres', alt.Tooltip(PRICE_COL, format='$.2f')]
        )
        st.altair_chart(chart_genre, use_container_width=True)
    else:
        st.info("Aucune donnée disponible pour les genres sélectionnés.")
# --- C. Peak CCU vs Price ---
st.subheader("Peak CCU vs Price (Scatter)")
st.caption("Existe-t-il une corrélation entre le prix et le pic de joueurs ? (Échelle Log)")

# Filtrer les jeux gratuits pour le log scale, ou ajouter une petite constante
df_scatter = df_filtered[df_filtered[CCU_COL] > 0].copy()
# On ajoute une petite valeur pour afficher les jeux gratuits (0$) sur une échelle Log si besoin,
# mais Altair gère mal le log(0). On sépare souvent.
df_paid = df_scatter[df_scatter[PRICE_COL] > 0]

chart_scatter = alt.Chart(df_paid).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X(PRICE_COL, scale=alt.Scale(type='log'), title="Prix ($) - Log Scale"),
    y=alt.Y(CCU_COL, scale=alt.Scale(type='log'), title="Peak CCU - Log Scale"),
    color=alt.Color('Genres', legend=None),
    tooltip=['Name', PRICE_COL, CCU_COL]
).properties(height=400).interactive()
st.altair_chart(chart_scatter, use_container_width=True)

st.divider()

# --- D. Free-to-play vs Pay-to-play ---
st.header("Free-to-Play (F2P) vs Payant")

df_filtered['Model'] = df_filtered[PRICE_COL].apply(lambda x: 'Free-to-Play' if x == 0 else 'Payant')

col_f2p_1, col_f2p_2, col_f2p_3 = st.columns(3)

# Calcul des métriques
f2p_stats = df_filtered.groupby('Model').agg({
    'AppID': 'count',
    'User score': 'mean',
    'Median playtime forever': 'median'
}).reset_index()

with col_f2p_1:
    st.subheader("Quantité Publiée")
    base = alt.Chart(f2p_stats).encode(x='Model', color='Model')
    st.altair_chart(base.mark_bar().encode(y='AppID'), use_container_width=True)

with col_f2p_2:
    st.subheader("Satisfaction (Score)")
    st.altair_chart(base.mark_bar().encode(y=alt.Y('User score', scale=alt.Scale(domain=(0, 100)))), use_container_width=True)

with col_f2p_3:
    st.subheader("Engagement (Playtime)")
    # Conversion minutes -> heures
    f2p_stats['Heures'] = f2p_stats['Median playtime forever'] / 60
    st.altair_chart(base.mark_bar().encode(y=alt.Y('Heures', title="Heures Médianes")), use_container_width=True)

# --- E. Table Détails ---
st.header("Liste des Jeux (Prix & Engagement)")
st.dataframe(
    df_filtered[['Name', 'Price', 'Estimated revenue', 'Median playtime forever', 'User score']].sort_values(by='Estimated revenue', ascending=False).head(100),
    use_container_width=True
)