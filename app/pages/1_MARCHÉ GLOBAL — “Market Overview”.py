# Importation des librairies
import streamlit as st
import pandas as pd
import altair as alt
import ast

# Parametrage de la page
st.set_page_config(page_title="GameData360 — Marché Global", layout="wide")
st.title("🎮 MARCHÉ GLOBAL — Analyse du Marché des Jeux Vidéo")

# Importation des données
df_analyse = pd.read_csv(
    r"C:\Users\bongu\Documents\GAMEDATA360\data\nettoyes\jeux_analysis_final.csv"
)

# Conversion automatique des colonnes en liste
for col in ["Genres", "Categories", "Tags"]:
    df_analyse[col] = df_analyse[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# Filtres globaux
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

# Application des filtres globaux
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

st.success(f"🎯 Jeux après filtres : {df_filtered.shape[0]} / {df_analyse.shape[0]}")

# 2 colonnes de métriques
col1, col2 = st.columns(2)

with col1:
    st.metric("🎮 Nombre total de jeux", df_filtered.shape[0])

with col2:
    if "Estimated revenue" in df_filtered.columns:
        total_revenue = df_filtered["Estimated revenue"].sum() / 1e9
        st.metric("💰 Revenu total estimé (milliards USD)", f"${total_revenue:.2f}B")
    else:
        st.metric("💰 Revenu estimé", "Non disponible")


# 2 colonnes pour la répartition par année
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Répartition des jeux par année (Tableau)")
    st.dataframe(
        df_filtered["Release Year"].value_counts().sort_index(ascending=False),
        height=280,
    )

with col2:
    st.subheader("📈 Évolution du nombre de jeux par année")

    years = (
        df_filtered["Release Year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    years.columns = ["Année", "Nombre de jeux"]
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
# 🎯 JEUX FILTRÉS + DISTRIBUTION PAR ANNÉE
# ------------------------------------------------------------

st.header("📂 Analyse filtrée (Genres & Catégories & Tags)")

colA, colB = st.columns(2)

with colA:
    st.subheader("🎮 Jeux filtrés (extraits)")
    st.dataframe(
        df_filtered[["AppID", "Name", "Genres", "Categories", "Tags", "Release Year"]],
        height=300,
    )

with colB:
    st.subheader("📊 Distribution des jeux filtrés par année")

    bar_data = (
        df_filtered["Release Year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
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
# 🖥️ RÉPARTITION PAR OS + TOP GENRES
# ------------------------------------------------------------

col3, col4 = st.columns(2)

with col3:
    st.subheader("🖥️ Répartition des jeux par OS")
    os_counts = df_filtered[["Windows", "Mac", "Linux"]].sum().reset_index()
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

with col4:
    st.subheader("🏆 Top 10 des genres les plus populaires")

    genre_exploded = df_filtered.explode("Genres")
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
        .properties(height=420)
    )
    st.altair_chart(bar_chart_genres, use_container_width=True)


# ------------------------------------------------------------
# 🆓 COMPARAISON FREE-TO-PLAY VS PAYANTS
# ------------------------------------------------------------

st.header("🆓 vs 💵 Free-to-Play vs Payants")

df_filtered["Type de jeu"] = df_filtered["Price"].apply(lambda x: "Free-to-Play" if x == 0 else "Payant")

col5, col6 = st.columns([1, 2])

with col5:
    st.subheader("📊 Nombre total de jeux")
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
    st.subheader("📈 Évolution Free-to-Play vs Payants")

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


# Top 50 des jeux les plus populaires par peak concurrent users
st.header("🏅 Top 50 des jeux les plus populaires (Peak Concurrent Users)")
top_50_games = df_analyse.nlargest(50, "Peak CCU")[["AppID", "Name", "Peak CCU", "Genres", "Categories", "Tags"]]
st.dataframe(top_50_games.reset_index(drop=True), height=500)
