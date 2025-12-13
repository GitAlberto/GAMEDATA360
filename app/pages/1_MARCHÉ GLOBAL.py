# Importation des librairies
import streamlit as st
import pandas as pd
import altair as alt
import ast

# ------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ------------------------------------------------------------
st.set_page_config(page_title="GameData360 — Marché Global", layout="wide")
st.title("🎮 MARCHÉ GLOBAL — Analyse du Marché des Jeux Vidéo")

# ------------------------------------------------------------
# 2. CHARGEMENT OPTIMISÉ DES DONNÉES (CACHE)
# ------------------------------------------------------------
# Le décorateur @st.cache_data permet de ne lancer cette fonction qu'une seule fois
# et de garder le résultat en mémoire tant que l'application tourne.
@st.cache_data
def load_data(file_path):
    # Chargement du CSV
    df = pd.read_csv(file_path)
    
    # Conversion des chaînes de caractères "[...]" en vraies listes Python
    # C'est cette étape qui est lourde et qui nécessite le cache
    cols_to_convert = ["Genres", "Categories", "Tags"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

# Chemin vers votre fichier
FILE_PATH = r"data/nettoyes/jeux_analysis_final.csv"

# Chargement avec indicateur visuel (spinner)
try:
    with st.spinner('Chargement et traitement des données en cours...'):
        df_analyse = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Le fichier n'a pas été trouvé à l'emplacement : {FILE_PATH}")
    st.stop()

# ------------------------------------------------------------
# 3. FILTRES GLOBAUX
# ------------------------------------------------------------
st.header("Filtres généraux")

# Extraction des valeurs uniques (se fait rapidement une fois les données chargées)
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
    # On vide les clés de session si nécessaire ou on relance simplement
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
# 5. KPIs PRINCIPAUX
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
# 6. ANALYSE TEMPORELLE (TABLEAU & GRAPHIQUE)
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition des jeux par année")
    st.dataframe(
        df_filtered["Release Year"].value_counts().sort_index(ascending=False),
        height=280,
        use_container_width=True
    )

with col2:
    st.subheader("Évolution du nombre de jeux par année")
    years = df_filtered["Release Year"].value_counts().sort_index().reset_index()
    years.columns = ["Année", "Nombre de jeux"]
    # Nettoyage pour s'assurer que l'année est propre
    years = years[years["Année"] != 0] 
    years["Année"] = years["Année"].astype(int)

    chart = (
        alt.Chart(years)
        .mark_line(point=True, interpolate='monotone')
        .encode(
            x=alt.X("Année:O", title="Année"),
            y=alt.Y("Nombre de jeux:Q", title="Nombre de jeux"),
            tooltip=["Année", "Nombre de jeux"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

# ------------------------------------------------------------
# 7. EXTRAITS & DISTRIBUTION DÉTAILLÉE
# ------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Jeux compris")
    cols_to_show = ["AppID", "Name", "Genres", "Categories", "Tags", "Release Year"]
    # Vérification que les colonnes existent avant affichage
    cols_present = [c for c in cols_to_show if c in df_filtered.columns]
    st.dataframe(
        df_filtered[cols_present],
        height=300,
        use_container_width=True
    )

with colB:
    st.subheader("Distribution des jeux par année")
    bar_data = df_filtered["Release Year"].value_counts().sort_index().reset_index()
    bar_data.columns = ["Année", "Nombre de jeux"]
    
    bar_chart = (
        alt.Chart(bar_data)
        .mark_bar(color="purple", opacity=0.75)
        .encode(
            x="Année:O",
            y="Nombre de jeux:Q",
            tooltip=["Année", "Nombre de jeux"],
        )
        .properties(height=350)
    )
    st.altair_chart(bar_chart, use_container_width=True)

# ------------------------------------------------------------
# 8. RÉPARTITION OS & TOP GENRES
# ------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Répartition des jeux par OS")
    # On vérifie si les colonnes OS existent
    os_cols = ["Windows", "Mac", "Linux"]
    if all(col in df_filtered.columns for col in os_cols):
        os_counts = df_filtered[os_cols].sum().reset_index()
        os_counts.columns = ["OS", "Nombre de jeux"]

        pie_chart = (
            alt.Chart(os_counts)
            .mark_arc(innerRadius=50)
            .encode(
                theta="Nombre de jeux:Q",
                color="OS:N",
                tooltip=["OS", "Nombre de jeux"],
            )
            .properties(height=350)
        )
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.warning("Données OS manquantes")

with col4:
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
            .mark_bar(color="teal", opacity=0.7)
            .encode(
                x="Nombre de jeux:Q",
                y=alt.Y("Genre:N", sort="-x"),
                tooltip=["Genre", "Nombre de jeux"],
            )
            .properties(height=350)
        )
        st.altair_chart(bar_chart_genres, use_container_width=True)
    else:
        st.info("Pas assez de données de genres pour afficher le graphique.")

# ------------------------------------------------------------
# 9. FREE-TO-PLAY VS PAYANTS
# ------------------------------------------------------------
st.header("Free-to-Play vs Payants")

if "Price" in df_filtered.columns:
    df_filtered["Type de jeu"] = df_filtered["Price"].apply(lambda x: "Free-to-Play" if x == 0 else "Payant")

    col5, col6 = st.columns([1, 2])

    with col5:
        st.subheader("Volume total")
        ftp_data = df_filtered["Type de jeu"].value_counts().reset_index()
        ftp_data.columns = ["Type de jeu", "Nombre de jeux"]

        bar_chart_ftp = (
            alt.Chart(ftp_data)
            .mark_bar(color="orange", opacity=0.7)
            .encode(
                x="Type de jeu:N",
                y="Nombre de jeux:Q",
                tooltip=["Type de jeu", "Nombre de jeux"],
            )
            .properties(height=350)
        )
        st.altair_chart(bar_chart_ftp, use_container_width=True)

    with col6:
        st.subheader("Évolution temporelle")
        yearly_ftp = (
            df_filtered.groupby(["Release Year", "Type de jeu"])
            .size()
            .reset_index(name="Nombre de jeux")
        )

        line_chart_ftp = (
            alt.Chart(yearly_ftp)
            .mark_line(point=True, interpolate='monotone')
            .encode(
                x="Release Year:O",
                y="Nombre de jeux:Q",
                color="Type de jeu:N",
                tooltip=["Release Year", "Type de jeu", "Nombre de jeux"],
            )
            .properties(height=350)
        )
        st.altair_chart(line_chart_ftp, use_container_width=True)
else:
    st.warning("La colonne 'Price' est manquante, impossible d'afficher l'analyse F2P.")

# ------------------------------------------------------------
# 10. TOP 50 (PEAK CCU)
# ------------------------------------------------------------
st.header("Top 50 des jeux les plus populaires (Peak CCU)")
if "Peak CCU" in df_analyse.columns:
    # On utilise df_analyse (global) ou df_filtered selon si on veut le top du filtre ou le top global
    # Ici, prenons le top global comme référence, ou le top filtré :
    top_50_games = df_filtered.nlargest(50, "Peak CCU")[["AppID", "Name", "Peak CCU", "Genres", "Categories", "Tags"]]
    st.dataframe(top_50_games.reset_index(drop=True), height=500, use_container_width=True)
else:
    st.warning("La colonne 'Peak CCU' est manquante.")