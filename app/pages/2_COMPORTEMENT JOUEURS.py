import streamlit as st
import pandas as pd
import altair as alt
import ast

# ------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ------------------------------------------------------------
st.set_page_config(page_title="GameData360 — COMPORTEMENT JOUEURS", layout="wide")
st.title("🎮 COMPORTEMENT JOUEURS — Analyse des comportements des joueurs")

# ------------------------------------------------------------
# 2. CHARGEMENT OPTIMISÉ (CACHE)
# ------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    # Chargement
    df = pd.read_csv(file_path)
    
    # Suppression des doublons (fait une seule fois ici)
    df = df.drop_duplicates()
    
    # Conversion des colonnes "string" en listes Python
    # C'est l'étape lourde qui est maintenant mise en cache
    cols_to_convert = ["Genres", "Categories", "Tags"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

# Chemin du fichier
FILE_PATH = r"C:\Users\bongu\Documents\GAMEDATA360\data\nettoyes\jeux_analysis_final.csv"

# Chargement avec spinner
try:
    with st.spinner('Chargement des données joueurs...'):
        df_analyse = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {FILE_PATH}")
    st.stop()

# ------------------------------------------------------------
# 3. FILTRES GLOBAUX
# ------------------------------------------------------------
st.header("Filtres généraux")

# Extraction des valeurs uniques (rapide car les données sont en cache)
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

# Bouton Reset
if st.button("Réinitialiser les filtres"):
    st.rerun()

# ------------------------------------------------------------
# 4. APPLICATION DES FILTRES
# ------------------------------------------------------------
df_filtered = df_analyse.copy()

# Filtre Genre
if selected_genres:
    df_filtered = df_filtered[
        df_filtered["Genres"].apply(
            lambda lst: isinstance(lst, list) and any(g in [x.lower() for x in lst] for g in selected_genres)
        )
    ]

# Filtre Catégorie
if selected_categories:
    df_filtered = df_filtered[
        df_filtered["Categories"].apply(
            lambda lst: isinstance(lst, list) and any(c in [x.lower() for x in lst] for c in selected_categories)
        )
    ]

# Filtre Tags
if selected_tags:
    df_filtered = df_filtered[
        df_filtered["Tags"].apply(
            lambda lst: isinstance(lst, list) and any(t in [x.lower() for x in lst] for t in selected_tags)
        )
    ]

st.success(f"Jeux après filtres : {df_filtered.shape[0]} / {df_analyse.shape[0]}")

# ------------------------------------------------------------
# 5. KPI GLOBAUX
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nombre total de jeux", df_filtered.shape[0],border=True)

with col2:
    if "Estimated revenue" in df_filtered.columns:
        total_revenue = df_filtered["Estimated revenue"].sum() / 1e9
        st.metric("Revenu total estimé (milliards USD)", f"${total_revenue:.2f}B",border=True)
    else:
        st.metric("Revenu estimé", "Non disponible")
with col3:
    st.metric("Prix moyen d'un jeu",f"${round(df_filtered["Price"].mean(),2)}",border=True)

with col4:
    st.metric("Prix médian d'un jeu", f"${round(df_filtered["Price"].median(),2)}",border=True)

# ------------------------------------------------------------
# 6. TOP 10 RECOMMANDATIONS (GRAPHIQUE + TABLEAU)
# ------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.header("Top 10 des jeux les plus recommandés")
    top_10_recommended = df_filtered.nlargest(10, "Recommendations")[["Name", "Recommendations"]]
    
    chart_rec = (
        alt.Chart(top_10_recommended)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("Recommendations:Q", title="Nombre de recommandations"),
            y=alt.Y("Name:N", sort="-x", title="Jeu"),
            tooltip=["Name", "Recommendations"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart_rec, use_container_width=True)

with colB:
    st.subheader("Détails des 10 jeux les plus recommandés")
    # On ajoute Estimated revenue pour plus de contexte
    cols_rec = ["AppID", "Name", "Recommendations"]
    if "Estimated revenue" in df_filtered.columns:
        cols_rec.append("Estimated revenue")
        
    top_10_table = df_filtered.nlargest(10, "Recommendations")[cols_rec]
    st.dataframe(top_10_table.reset_index(drop=True), height=400, use_container_width=True)


# ------------------------------------------------------------
# 7. TOP 10 GENRES PAR PEAK CCU (COMPLÉTÉ)
# ------------------------------------------------------------
# J'ai ajouté le code ici car il manquait dans votre snippet initial
colC, colD = st.columns(2)

# Préparation des données pour le graphique par Genre
# On "explode" la colonne Genres pour compter chaque genre individuellement
df_exploded = df_filtered.explode("Genres").dropna(subset=["Genres"])

# On groupe par genre et on somme le Peak CCU
if "Peak CCU" in df_exploded.columns:
    genre_ccu = df_exploded.groupby("Genres")["Peak CCU"].sum().reset_index()
    top_genres_ccu = genre_ccu.nlargest(10, "Peak CCU")

    with colC:
        st.header("Top 10 des genres par Peak CCU")
        
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

    with colD:
        st.header("Détails genres par Peak CCU")
        st.dataframe(top_genres_ccu, height=380, use_container_width=True)
else:
    st.warning("La colonne 'Peak CCU' est manquante pour l'analyse par genre.")

# Creation de 2 colonnes pour nombbre de jeux par genres et détailles du nombre jeux par gens
col1, col2 = st.columns(2)
with col1 :
    st.subheader("Top 10 des genres les plus populaires")
    # Explode seulement si la liste n'est pas vide
    genre_exploded = df_filtered.explode("Genres")
    # On retire les NaN ou listes vides qui auraient généré des NaN
    genre_exploded = genre_exploded.dropna(subset=["Genres"])
    
    if not genre_exploded.empty:
        genre_counts = genre_exploded["Genres"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Nombre de jeux"]
        top_genres = genre_counts.head(10)

        bar_chart_genres = (
            alt.Chart(top_genres)
            .mark_bar(color="red", opacity=0.7)
            .encode(
                x="Nombre de jeux:Q",
                y=alt.Y("Genre:N", sort="-x"),
                tooltip=["Genre", "Nombre de jeux"],
            )
            .properties(height=370)
        )
        st.altair_chart(bar_chart_genres, use_container_width=True)
    else:
        st.info("Pas assez de données de genres pour afficher le graphique.")
with col2:
    st.subheader("Détails du Top 10 des genres les plus populaires")
    # Explode seulement si la liste n'est pas vide
    genre_exploded = df_filtered.explode("Genres")
    # On retire les NaN ou listes vides qui auraient généré des NaN
    genre_exploded = genre_exploded.dropna(subset=["Genres"])
    
    if not genre_exploded.empty:
        genre_counts = genre_exploded["Genres"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Nombre de jeux"]
        top_genres = genre_counts.head(10)
        st.dataframe(top_genres, height=350)
    else:
        st.info("Pas assez de données de genres pour afficher le graphique.")

# ------------------------------------------------------------
# 🎯 ANALYSE DU TEMPS DE JEU (CORRIGÉE)
# ------------------------------------------------------------
st.divider()
st.header("Analyse du Temps de Jeu (Engagement)")

# 1. Préparation des données
# On explose pour séparer les genres multiples
df_playtime = df_filtered.explode("Genres").dropna(subset=["Genres"])

# 2. Calcul du Top 15 (Moyenne des Heures par Genre)
df_result = (
    df_playtime.groupby("Genres")["Average playtime forever"]
    .mean()                        # On utilise la MOYENNE pour coller à votre EDA
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

# Renommage pour l'affichage propre dans le graphique
df_result.columns = ["Genres", "Heures Moyennes"]

col_pt1, col_pt2 = st.columns(2) # Création de 2 colonnes

with col_pt1:
    st.subheader("Top 15 Genres par Temps de Jeu Moyen")
    
    chart_playtime = (
        alt.Chart(df_result)
        .mark_bar(color="#FF8C00")
        .encode(
            x=alt.X("Heures Moyennes:Q", title="Heures Moyennes"),
            y=alt.Y("Genres:N", sort="-x", title="Genre"),
            tooltip=["Genres", alt.Tooltip("Heures Moyennes", format=".1f")]
        )
        .properties(height=380)
    )
    st.altair_chart(chart_playtime, use_container_width=True)

with col_pt2:
    st.subheader("Données vérifiées")
    st.dataframe(df_result, use_container_width=True)

st.subheader("Performance Genre vs Marché")
st.caption("Barre Bleue = Médiane du Genre. Trait Rouge = Médiane Globale (Benchmark).")

# 1. NETTOYAGE CRUCIAL (Fix du problème "Médiane = 0")
# On ne garde que les jeux qui ont été joués (> 0). 
# Sinon les milliers de jeux non joués tirent la médiane vers 0.
col_ref = "Median playtime forever"
df_active = df_filtered[df_filtered[col_ref] > 0].copy()

# 2. Calcul du Benchmark Global (Le "Stick")
# C'est la ligne rouge : la médiane de TOUS les jeux filtrés confondus
global_median_val = df_active[col_ref].median()

# 3. Préparation par Genre
df_genre_prep = df_active.explode("Genres").dropna(subset=["Genres"])

# 4. Calcul de la Médiane par Genre (La Barre)
df_chart = (
    df_genre_prep.groupby("Genres")[col_ref]
    .median()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
df_chart.columns = ['Genres', 'Genre Median']

# On ajoute la valeur globale à chaque ligne pour pouvoir l'afficher en "Stick"
df_chart['Global Median'] = global_median_val

# 5. Graphique Combiné
# Base du graphique
base = alt.Chart(df_chart).encode(
    y=alt.Y('Genres', sort='-x', title="Genre")
)

# A. Les Barres (Performance du Genre)
bars = base.mark_bar(color="#4682B4", opacity=0.8).encode(
    x=alt.X('Genre Median', title="Temps de Jeu Médian (Heures)"),
    tooltip=['Genres', 'Genre Median']
)

# B. Le Stick (Référence Marché)
ticks = base.mark_tick(color="red", thickness=4, size=30).encode(
    x='Global Median',
    tooltip=[alt.Tooltip('Global Median', title="Médiane du Marché")]
)

st.altair_chart(bars + ticks, use_container_width=True)

# Petit indicateur textuel pour aider à la lecture
st.markdown(f"**Référence marché (Trait Rouge) :** {global_median_val:.1f} Heures")

# --- 4. ANALYSE SOLO VS MULTI (CORRIGÉE) ---
st.subheader("Engagement : Solo vs Multijoueur")

# Fonction pour classifier simplifié
def categorize_mode(cats):
    if not isinstance(cats, list): return "Inconnu"
    cats_lower = [c.lower() for c in cats]
    if any(x in cats_lower for x in ['multi-player', 'mmo', 'co-op', 'online pvp', 'online co-op']):
        return "Multijoueur / Co-op"
    elif 'single-player' in cats_lower:
        return "Solo"
    return "Autre"

# 1. Création de la catégorie
df_filtered["Mode"] = df_filtered["Categories"].apply(categorize_mode)

# 2. Filtrage des données pour le graphique
# On retire "Inconnu" ET les jeux avec 0 heure de jeu (sinon le Log Scale plante)
df_mode = df_filtered[
    (df_filtered["Mode"] != "Inconnu") & 
    (df_filtered["Median playtime forever"] > 0)
].copy()

# Boxplot pour voir la distribution
chart_mode = (
    alt.Chart(df_mode)
    .mark_boxplot(extent='min-max', size=50) # size=50 rend les boites plus visibles
    .encode(
        x=alt.X("Mode:N", title="Type de jeu"),
        # Titre corrigé en HEURES
        y=alt.Y("Median playtime forever:Q", scale=alt.Scale(type="log"), title="Playtime Médian (Heures, Log Scale)"),
        color=alt.Color("Mode:N", legend=None),
        tooltip=["Name", "Median playtime forever", "Mode"]
    )
    .properties(height=350)
)

st.altair_chart(chart_mode, use_container_width=True)