-- GAMEDATA360 - Creation Tables (fidele au CSV)

DROP TABLE IF EXISTS jeu_categorie, jeu_tag, jeu_genre, jeu_editeur, jeu_developpeur CASCADE;
DROP TABLE IF EXISTS fait_jeux, dim_developpeur, dim_editeur, dim_genre, dim_tag, dim_categorie, dim_plateforme CASCADE;
DROP TABLE IF EXISTS staging_jeux CASCADE;

-- Staging (27 colonnes exactes du CSV)
CREATE TABLE staging_jeux (
    "AppID" INTEGER,
    "Name" TEXT,
    "Release date" TEXT,
    "Estimated owners" NUMERIC,
    "Price" NUMERIC,
    "Peak CCU" NUMERIC,
    "User score" NUMERIC,
    "Positive" INTEGER,
    "Negative" INTEGER,
    "Metacritic score" NUMERIC,
    "Average playtime forever" NUMERIC,
    "Median playtime forever" NUMERIC,
    "Achievements" INTEGER,
    "Recommendations" INTEGER,
    "Developers" TEXT,
    "Publishers" TEXT,
    "Genres" TEXT,
    "Tags" TEXT,
    "Categories" TEXT,
    "Windows" INTEGER,
    "Mac" INTEGER,
    "Linux" INTEGER,
    "Website" TEXT,
    "published" INTEGER,
    "Estimated sales" NUMERIC,
    "Estimated revenue" NUMERIC,
    "Release Year" INTEGER
);

-- Dimensions (TEXT au lieu de VARCHAR pour eviter les limites)
CREATE TABLE dim_developpeur (id SERIAL PRIMARY KEY, nom TEXT UNIQUE);
CREATE TABLE dim_editeur (id SERIAL PRIMARY KEY, nom TEXT UNIQUE);
CREATE TABLE dim_genre (id SERIAL PRIMARY KEY, nom TEXT UNIQUE);
CREATE TABLE dim_tag (id SERIAL PRIMARY KEY, nom TEXT UNIQUE);
CREATE TABLE dim_categorie (id SERIAL PRIMARY KEY, nom TEXT UNIQUE);
CREATE TABLE dim_plateforme (id SERIAL PRIMARY KEY, windows INTEGER, mac INTEGER, linux INTEGER, UNIQUE(windows, mac, linux));

-- Table de faits
CREATE TABLE fait_jeux (
    id SERIAL PRIMARY KEY,
    appid INTEGER UNIQUE,
    id_plateforme INTEGER REFERENCES dim_plateforme(id),
    nom TEXT,
    date_sortie TEXT,
    annee INTEGER,
    proprietaires NUMERIC,
    prix NUMERIC,
    peak_ccu NUMERIC,
    score_user NUMERIC,
    score_metacritic NUMERIC,
    avis_positifs INTEGER,
    avis_negatifs INTEGER,
    temps_moyen NUMERIC,
    temps_median NUMERIC,
    succes INTEGER,
    recommandations INTEGER,
    ventes NUMERIC,
    revenus NUMERIC,
    website TEXT,
    published INTEGER
);

-- Liaisons
CREATE TABLE jeu_developpeur (id_fait INTEGER REFERENCES fait_jeux(id), id_dim INTEGER REFERENCES dim_developpeur(id), PRIMARY KEY(id_fait, id_dim));
CREATE TABLE jeu_editeur (id_fait INTEGER REFERENCES fait_jeux(id), id_dim INTEGER REFERENCES dim_editeur(id), PRIMARY KEY(id_fait, id_dim));
CREATE TABLE jeu_genre (id_fait INTEGER REFERENCES fait_jeux(id), id_dim INTEGER REFERENCES dim_genre(id), PRIMARY KEY(id_fait, id_dim));
CREATE TABLE jeu_tag (id_fait INTEGER REFERENCES fait_jeux(id), id_dim INTEGER REFERENCES dim_tag(id), PRIMARY KEY(id_fait, id_dim));
CREATE TABLE jeu_categorie (id_fait INTEGER REFERENCES fait_jeux(id), id_dim INTEGER REFERENCES dim_categorie(id), PRIMARY KEY(id_fait, id_dim));
