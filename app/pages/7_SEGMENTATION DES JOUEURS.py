import streamlit as st
import pandas as pd
import altair as alt
import ast
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. CONFIGURATION
st.set_page_config(page_title="GameData360 — PERSONAS", layout="wide")
st.title("PERSONAS JOUEURS — Segmentation Comportementale")

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
# 7. CONTENU SPÉCIFIQUE : PERSONAS
# ---------------------------------------------------------

st.markdown("""
Cette section utilise l'IA pour regrouper les jeux en **5 profils types (Personas)** basés sur :
* Le temps de jeu (Engagement)
* Le prix (Budget)
* Le score (Exigence Qualité)
* Le Peak CCU (Aspect Social/Popularité)
""")

if len(df_filtered) > 100:
    # 1. PRÉPARATION
    features = ['Median playtime forever', 'Price', 'User score', 'Peak CCU']
    df_ml = df_filtered.dropna(subset=features).copy()
    
    # Normalisation MinMax pour le "Radar" (0 à 1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_ml[features]), columns=features, index=df_ml.index)
    
    # 2. CLUSTERING (5 Clusters pour les 5 Personas demandés)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_ml['Cluster'] = kmeans.fit_predict(df_scaled)
    df_scaled['Cluster'] = df_ml['Cluster']
    
    # 3. VISUALISATION "RADAR" (APLATI)
    # On calcule la moyenne de chaque feature pour chaque cluster
    cluster_profiles = df_scaled.groupby('Cluster').mean().reset_index()
    # Melt pour format Altair
    cluster_melt = cluster_profiles.melt('Cluster', var_name='Caractéristique', value_name='Score Normalisé')
    
    st.header("Profils des Personas (Radar Chart Aplatit)")
    st.caption("Score proche de 1 = Très élevé (ex: Prix élevé, Playtime énorme).")
    
    # Heatmap pour représenter les Radars
    chart_radar = alt.Chart(cluster_melt).mark_rect().encode(
        x='Caractéristique:N',
        y='Cluster:N',
        color=alt.Color('Score Normalisé:Q', scale=alt.Scale(scheme='magma')),
        tooltip=['Cluster', 'Caractéristique', 'Score Normalisé']
    ).properties(height=300)
    st.altair_chart(chart_radar, use_container_width=True)
    
    # Nommage hypothétique (Automatique, peut ne pas être parfait)
    st.markdown("**Interprétation suggérée des Clusters (basé sur les couleurs) :**")
    st.markdown("""
    * **Budget Player** : Prix bas, Score variable.
    * **Competitive Hardcore** : Playtime & CCU très élevés.
    * **Casual Explorer** : Playtime bas, Prix moyen.
    * **Indie Lover** : Prix bas, Score élevé, CCU bas.
    """)

    # 4. CLUSTER PLOT 2D
    st.header("Cluster Plot 2D : Playtime vs Satisfaction")
    df_ml['Heures'] = df_ml['Median playtime forever'] / 60
    
    chart_2d = alt.Chart(df_ml).mark_circle(size=60).encode(
        x=alt.X('Heures', scale=alt.Scale(domain=(0, 300)), title="Playtime (Heures)"),
        y=alt.Y('User score', scale=alt.Scale(domain=(0, 100))),
        color='Cluster:N',
        tooltip=['Name', 'Cluster', 'Heures', 'User score']
    ).interactive().properties(height=500)
    st.altair_chart(chart_2d, use_container_width=True)
    
    # 5. TABLE
    st.header("Jeux classés par Persona")
    st.dataframe(df_ml[['Name', 'Cluster', 'Genres', 'Price', 'User score']], use_container_width=True)

else:
    st.warning("Pas assez de données pour générer 5 Personas fiables (Min 100 jeux).")