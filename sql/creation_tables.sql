-- ============================================================================
-- BASE DE DONNÉES GAMEDATA360 - MODÈLE EN ÉTOILE (STAR SCHEMA)
-- ============================================================================
-- Description: Schéma dimensionnel pour l'analyse des jeux vidéo Steam
-- Auteur: Expert Base de Données
-- Date: 2026-01-01
-- ============================================================================
-- INSTRUCTIONS D'UTILISATION:
-- 1. Se connecter à PostgreSQL en tant que superutilisateur (postgres)
-- 2. Exécuter ce script complet pour créer la base de données et les tables
-- 3. Le script créera automatiquement un utilisateur 'gamedata_user'
-- ============================================================================
-- ============================================================================
-- SECTION 1: CRÉATION DE LA BASE DE DONNÉES
-- ============================================================================
-- Vérifier si la base existe et la supprimer si nécessaire (ATTENTION: PERTE DE DONNÉES!)
-- Décommenter les lignes suivantes uniquement pour une réinitialisation complète
-- DROP DATABASE IF EXISTS gamedata360;
-- DROP USER IF EXISTS gamedata_user;
-- Créer un utilisateur dédié pour la base de données
DO $$ BEGIN IF NOT EXISTS (
    SELECT
    FROM pg_catalog.pg_user
    WHERE usename = 'gamedata_user'
) THEN CREATE USER gamedata_user WITH PASSWORD 'GameData@2026#Secure';
END IF;
END $$;
-- Créer la base de données avec les paramètres optimaux
CREATE DATABASE gamedata360 WITH OWNER = gamedata_user ENCODING = 'UTF8' LC_COLLATE = 'French_France.1252' LC_CTYPE = 'French_France.1252' TABLESPACE = pg_default CONNECTION
LIMIT = -1 TEMPLATE = template0;
-- Ajouter un commentaire descriptif
COMMENT ON DATABASE gamedata360 IS 'Base de données analytique pour GameData360 - Analyse des jeux vidéo Steam';
-- ============================================================================
-- SECTION 2: CONFIGURATION DE LA BASE DE DONNÉES
-- ============================================================================
-- Se connecter à la base de données gamedata360
\ c gamedata360 -- Octroyer les privilèges à l'utilisateur
GRANT ALL PRIVILEGES ON DATABASE gamedata360 TO gamedata_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO gamedata_user;
-- Configuration des paramètres de la base de données
-- Augmenter la mémoire de travail pour les requêtes analytiques
ALTER DATABASE gamedata360
SET work_mem = '64MB';
-- Augmenter la mémoire de maintenance pour les index
ALTER DATABASE gamedata360
SET maintenance_work_mem = '256MB';
-- Optimiser pour les lectures analytiques
ALTER DATABASE gamedata360
SET random_page_cost = 1.1;
-- Augmenter le cache effectif
ALTER DATABASE gamedata360
SET effective_cache_size = '4GB';
-- Timezone
ALTER DATABASE gamedata360
SET timezone = 'Europe/Paris';
-- Encoding client
ALTER DATABASE gamedata360
SET client_encoding = 'UTF8';
-- ============================================================================
-- SECTION 3: EXTENSIONS POSTGRESQL UTILES
-- ============================================================================
-- Extension pour les fonctions de hachage (utile pour la performance)
CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- Extension pour les UUID (si besoin futur)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Extension pour les statistiques avancées
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
-- Extension pour le Full-Text Search (recherche sur les noms de jeux)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- ============================================================================
-- SECTION 4: SCHÉMA ET RÔLES
-- ============================================================================
-- Créer un schéma dédié pour les données brutes (staging)
CREATE SCHEMA IF NOT EXISTS staging AUTHORIZATION gamedata_user;
-- Créer un schéma pour les vues et les rapports
CREATE SCHEMA IF NOT EXISTS reporting AUTHORIZATION gamedata_user;
-- Octroyer les permissions
GRANT USAGE ON SCHEMA staging TO gamedata_user;
GRANT USAGE ON SCHEMA reporting TO gamedata_user;
GRANT CREATE ON SCHEMA staging TO gamedata_user;
GRANT CREATE ON SCHEMA reporting TO gamedata_user;
-- Permissions par défaut pour les futures tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL ON TABLES TO gamedata_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA staging
GRANT ALL ON TABLES TO gamedata_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA reporting
GRANT ALL ON TABLES TO gamedata_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL ON SEQUENCES TO gamedata_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA staging
GRANT ALL ON SEQUENCES TO gamedata_user;
-- ============================================================================
-- SECTION 5: PARAMÈTRES DE CONFIGURATION RECOMMANDÉS
-- ============================================================================
-- Les paramètres suivants sont des recommandations à ajouter dans postgresql.conf
-- pour optimiser les performances analytiques
/*
 PARAMÈTRES RECOMMANDÉS POUR postgresql.conf:
 -----------------------------------------------
 
 # Mémoire
 shared_buffers = 2GB                    # 25% de la RAM disponible
 effective_cache_size = 6GB              # 50-75% de la RAM disponible
 work_mem = 64MB                         # Pour les tris et hash joins
 maintenance_work_mem = 512MB            # Pour CREATE INDEX, VACUUM
 
 # Parallélisme (si PostgreSQL 10+)
 max_parallel_workers_per_gather = 4     # Nombre de workers parallèles
 max_parallel_workers = 8                # Maximum de workers parallèles
 max_worker_processes = 8                # Processus background
 
 # Optimisations disque
 random_page_cost = 1.1                  # Pour SSD (4.0 pour HDD)
 effective_io_concurrency = 200          # Pour SSD (2 pour HDD)
 
 # Checkpoint et WAL
 checkpoint_completion_target = 0.9
 wal_buffers = 16MB
 min_wal_size = 1GB
 max_wal_size = 4GB
 
 # Logs (pour le développement)
 log_statement = 'mod'                   # Log toutes les modifications
 log_duration = on                       # Log la durée des requêtes
 log_min_duration_statement = 1000       # Log les requêtes > 1s
 
 # Extensions statistiques
 shared_preload_libraries = 'pg_stat_statements'
 pg_stat_statements.track = all
 */
-- ============================================================================
-- SECTION 6: TABLES DE DONNÉES
-- ============================================================================
-- Suppression des tables existantes (dans l'ordre inverse des dépendances)
DROP TABLE IF EXISTS fait_jeux CASCADE;
DROP TABLE IF EXISTS dim_developpeur CASCADE;
DROP TABLE IF EXISTS dim_editeur CASCADE;
DROP TABLE IF EXISTS dim_genre CASCADE;
DROP TABLE IF EXISTS dim_tag CASCADE;
DROP TABLE IF EXISTS dim_categorie CASCADE;
DROP TABLE IF EXISTS dim_plateforme CASCADE;
DROP TABLE IF EXISTS dim_temps CASCADE;
-- ============================================================================
-- TABLES DE DIMENSIONS
-- ============================================================================
-- ----------------------------------------------------------------------------
-- Dimension Temps (DIM_TEMPS)
-- ----------------------------------------------------------------------------
-- Stocke les informations temporelles pour l'analyse par période
CREATE TABLE dim_temps (
    id_temps SERIAL PRIMARY KEY,
    date_complete DATE NOT NULL UNIQUE,
    annee INTEGER NOT NULL,
    mois INTEGER NOT NULL,
    trimestre INTEGER NOT NULL,
    jour INTEGER NOT NULL,
    jour_semaine INTEGER NOT NULL,
    nom_mois VARCHAR(20),
    nom_jour VARCHAR(20)
);
CREATE INDEX idx_dim_temps_annee ON dim_temps(annee);
CREATE INDEX idx_dim_temps_mois ON dim_temps(annee, mois);
-- ----------------------------------------------------------------------------
-- Dimension Développeur (DIM_DEVELOPPEUR)
-- ----------------------------------------------------------------------------
-- Catalogue des studios de développement
CREATE TABLE dim_developpeur (
    id_developpeur SERIAL PRIMARY KEY,
    nom_developpeur VARCHAR(255) NOT NULL UNIQUE,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_dim_developpeur_nom ON dim_developpeur(nom_developpeur);
-- ----------------------------------------------------------------------------
-- Dimension Éditeur (DIM_EDITEUR)
-- ----------------------------------------------------------------------------
-- Catalogue des éditeurs de jeux
CREATE TABLE dim_editeur (
    id_editeur SERIAL PRIMARY KEY,
    nom_editeur VARCHAR(255) NOT NULL UNIQUE,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_dim_editeur_nom ON dim_editeur(nom_editeur);
-- ----------------------------------------------------------------------------
-- Dimension Genre (DIM_GENRE)
-- ----------------------------------------------------------------------------
-- Classification des genres de jeux
CREATE TABLE dim_genre (
    id_genre SERIAL PRIMARY KEY,
    nom_genre VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);
CREATE INDEX idx_dim_genre_nom ON dim_genre(nom_genre);
-- ----------------------------------------------------------------------------
-- Dimension Tag (DIM_TAG)
-- ----------------------------------------------------------------------------
-- Tags/étiquettes associés aux jeux
CREATE TABLE dim_tag (
    id_tag SERIAL PRIMARY KEY,
    nom_tag VARCHAR(100) NOT NULL UNIQUE,
    popularite INTEGER DEFAULT 0
);
CREATE INDEX idx_dim_tag_nom ON dim_tag(nom_tag);
-- ----------------------------------------------------------------------------
-- Dimension Catégorie (DIM_CATEGORIE)
-- ----------------------------------------------------------------------------
-- Catégories Steam (Multijoueur, Solo, etc.)
CREATE TABLE dim_categorie (
    id_categorie SERIAL PRIMARY KEY,
    nom_categorie VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);
CREATE INDEX idx_dim_categorie_nom ON dim_categorie(nom_categorie);
-- ----------------------------------------------------------------------------
-- Dimension Plateforme (DIM_PLATEFORME)
-- ----------------------------------------------------------------------------
-- Combinaisons de plateformes supportées (Windows, Mac, Linux)
CREATE TABLE dim_plateforme (
    id_plateforme SERIAL PRIMARY KEY,
    windows BOOLEAN DEFAULT FALSE,
    mac BOOLEAN DEFAULT FALSE,
    linux BOOLEAN DEFAULT FALSE,
    description VARCHAR(100),
    UNIQUE(windows, mac, linux)
);
CREATE INDEX idx_dim_plateforme_combinaison ON dim_plateforme(windows, mac, linux);
-- ============================================================================
-- TABLE DE FAITS
-- ============================================================================
-- ----------------------------------------------------------------------------
-- Fait Jeux (FAIT_JEUX)
-- ----------------------------------------------------------------------------
-- Table centrale contenant les métriques et mesures des jeux
CREATE TABLE fait_jeux (
    id_fait SERIAL PRIMARY KEY,
    -- Clés étrangères vers les dimensions
    appid INTEGER NOT NULL UNIQUE,
    id_temps INTEGER REFERENCES dim_temps(id_temps),
    id_plateforme INTEGER REFERENCES dim_plateforme(id_plateforme),
    -- Informations de base
    nom_jeu VARCHAR(500) NOT NULL,
    date_sortie DATE,
    site_web VARCHAR(500),
    date_publication_steam DATE,
    -- Métriques de prix et ventes
    prix NUMERIC(10, 2),
    proprietaires_estimes VARCHAR(50),
    ventes_estimees NUMERIC(12, 2),
    revenus_estimes NUMERIC(15, 2),
    -- Métriques de joueurs
    pic_joueurs_simultanees INTEGER,
    temps_jeu_moyen_total INTEGER,
    -- en minutes
    temps_jeu_median_total INTEGER,
    -- en minutes
    -- Métriques de satisfaction
    score_utilisateurs INTEGER CHECK (
        score_utilisateurs >= 0
        AND score_utilisateurs <= 100
    ),
    avis_positifs INTEGER DEFAULT 0,
    avis_negatifs INTEGER DEFAULT 0,
    score_metacritic INTEGER CHECK (
        score_metacritic >= 0
        AND score_metacritic <= 100
    ),
    nombre_recommandations INTEGER DEFAULT 0,
    -- Métriques de contenu
    nombre_succes INTEGER DEFAULT 0,
    -- Métadonnées
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_mise_a_jour TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Index pour optimiser les requêtes
CREATE INDEX idx_fait_jeux_appid ON fait_jeux(appid);
CREATE INDEX idx_fait_jeux_date_sortie ON fait_jeux(date_sortie);
CREATE INDEX idx_fait_jeux_prix ON fait_jeux(prix);
CREATE INDEX idx_fait_jeux_ventes ON fait_jeux(ventes_estimees);
CREATE INDEX idx_fait_jeux_revenus ON fait_jeux(revenus_estimes);
CREATE INDEX idx_fait_jeux_score ON fait_jeux(score_utilisateurs);
-- ============================================================================
-- TABLES DE LIAISON (Many-to-Many)
-- ============================================================================
-- Ces tables permettent de gérer les relations N:N entre les jeux et les dimensions
-- ----------------------------------------------------------------------------
-- Liaison Jeu-Développeur
-- ----------------------------------------------------------------------------
CREATE TABLE jeu_developpeur (
    id_fait INTEGER REFERENCES fait_jeux(id_fait) ON DELETE CASCADE,
    id_developpeur INTEGER REFERENCES dim_developpeur(id_developpeur) ON DELETE CASCADE,
    PRIMARY KEY (id_fait, id_developpeur)
);
CREATE INDEX idx_jeu_developpeur_jeu ON jeu_developpeur(id_fait);
CREATE INDEX idx_jeu_developpeur_dev ON jeu_developpeur(id_developpeur);
-- ----------------------------------------------------------------------------
-- Liaison Jeu-Éditeur
-- ----------------------------------------------------------------------------
CREATE TABLE jeu_editeur (
    id_fait INTEGER REFERENCES fait_jeux(id_fait) ON DELETE CASCADE,
    id_editeur INTEGER REFERENCES dim_editeur(id_editeur) ON DELETE CASCADE,
    PRIMARY KEY (id_fait, id_editeur)
);
CREATE INDEX idx_jeu_editeur_jeu ON jeu_editeur(id_fait);
CREATE INDEX idx_jeu_editeur_ed ON jeu_editeur(id_editeur);
-- ----------------------------------------------------------------------------
-- Liaison Jeu-Genre
-- ----------------------------------------------------------------------------
CREATE TABLE jeu_genre (
    id_fait INTEGER REFERENCES fait_jeux(id_fait) ON DELETE CASCADE,
    id_genre INTEGER REFERENCES dim_genre(id_genre) ON DELETE CASCADE,
    PRIMARY KEY (id_fait, id_genre)
);
CREATE INDEX idx_jeu_genre_jeu ON jeu_genre(id_fait);
CREATE INDEX idx_jeu_genre_genre ON jeu_genre(id_genre);
-- ----------------------------------------------------------------------------
-- Liaison Jeu-Tag
-- ----------------------------------------------------------------------------
CREATE TABLE jeu_tag (
    id_fait INTEGER REFERENCES fait_jeux(id_fait) ON DELETE CASCADE,
    id_tag INTEGER REFERENCES dim_tag(id_tag) ON DELETE CASCADE,
    ordre_pertinence INTEGER DEFAULT 0,
    -- Pour garder l'ordre des tags
    PRIMARY KEY (id_fait, id_tag)
);
CREATE INDEX idx_jeu_tag_jeu ON jeu_tag(id_fait);
CREATE INDEX idx_jeu_tag_tag ON jeu_tag(id_tag);
-- ----------------------------------------------------------------------------
-- Liaison Jeu-Catégorie
-- ----------------------------------------------------------------------------
CREATE TABLE jeu_categorie (
    id_fait INTEGER REFERENCES fait_jeux(id_fait) ON DELETE CASCADE,
    id_categorie INTEGER REFERENCES dim_categorie(id_categorie) ON DELETE CASCADE,
    PRIMARY KEY (id_fait, id_categorie)
);
CREATE INDEX idx_jeu_categorie_jeu ON jeu_categorie(id_fait);
CREATE INDEX idx_jeu_categorie_cat ON jeu_categorie(id_categorie);
-- ============================================================================
-- VUES ANALYTIQUES UTILES
-- ============================================================================
-- ----------------------------------------------------------------------------
-- Vue: Vue complète des jeux avec toutes les dimensions
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_jeux_complet AS
SELECT f.id_fait,
    f.appid,
    f.nom_jeu,
    f.date_sortie,
    t.annee AS annee_sortie,
    t.mois AS mois_sortie,
    f.prix,
    f.proprietaires_estimes,
    f.ventes_estimees,
    f.revenus_estimes,
    f.pic_joueurs_simultanees,
    f.score_utilisateurs,
    f.score_metacritic,
    f.avis_positifs,
    f.avis_negatifs,
    CASE
        WHEN (f.avis_positifs + f.avis_negatifs) > 0 THEN ROUND(
            (
                f.avis_positifs::NUMERIC / (f.avis_positifs + f.avis_negatifs) * 100
            ),
            2
        )
        ELSE NULL
    END AS pourcentage_avis_positifs,
    f.nombre_recommandations,
    f.nombre_succes,
    f.temps_jeu_moyen_total,
    f.temps_jeu_median_total,
    p.windows,
    p.mac,
    p.linux,
    f.site_web
FROM fait_jeux f
    LEFT JOIN dim_temps t ON f.id_temps = t.id_temps
    LEFT JOIN dim_plateforme p ON f.id_plateforme = p.id_plateforme;
-- ----------------------------------------------------------------------------
-- Vue: Statistiques par développeur
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_stats_developpeur AS
SELECT dev.id_developpeur,
    dev.nom_developpeur,
    COUNT(DISTINCT jd.id_fait) AS nombre_jeux,
    AVG(f.score_utilisateurs) AS score_moyen,
    SUM(f.ventes_estimees) AS ventes_totales,
    SUM(f.revenus_estimes) AS revenus_totaux,
    AVG(f.prix) AS prix_moyen
FROM dim_developpeur dev
    LEFT JOIN jeu_developpeur jd ON dev.id_developpeur = jd.id_developpeur
    LEFT JOIN fait_jeux f ON jd.id_fait = f.id_fait
GROUP BY dev.id_developpeur,
    dev.nom_developpeur;
-- ----------------------------------------------------------------------------
-- Vue: Statistiques par genre
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_stats_genre AS
SELECT g.id_genre,
    g.nom_genre,
    COUNT(DISTINCT jg.id_fait) AS nombre_jeux,
    AVG(f.score_utilisateurs) AS score_moyen,
    SUM(f.ventes_estimees) AS ventes_totales,
    AVG(f.prix) AS prix_moyen,
    SUM(f.avis_positifs + f.avis_negatifs) AS total_avis
FROM dim_genre g
    LEFT JOIN jeu_genre jg ON g.id_genre = jg.id_genre
    LEFT JOIN fait_jeux f ON jg.id_fait = f.id_fait
GROUP BY g.id_genre,
    g.nom_genre;
-- ----------------------------------------------------------------------------
-- Vue: Évolution temporelle des ventes
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_evolution_ventes AS
SELECT t.annee,
    t.mois,
    COUNT(DISTINCT f.id_fait) AS nombre_jeux_sortis,
    SUM(f.ventes_estimees) AS ventes_totales,
    SUM(f.revenus_estimes) AS revenus_totaux,
    AVG(f.prix) AS prix_moyen,
    AVG(f.score_utilisateurs) AS score_moyen
FROM dim_temps t
    LEFT JOIN fait_jeux f ON t.id_temps = f.id_temps
GROUP BY t.annee,
    t.mois
ORDER BY t.annee,
    t.mois;
-- ============================================================================
-- COMMENTAIRES SUR LES TABLES
-- ============================================================================
COMMENT ON TABLE fait_jeux IS 'Table de faits centrale contenant les métriques des jeux vidéo';
COMMENT ON TABLE dim_temps IS 'Dimension temporelle pour l''analyse par période';
COMMENT ON TABLE dim_developpeur IS 'Dimension des studios de développement';
COMMENT ON TABLE dim_editeur IS 'Dimension des éditeurs de jeux';
COMMENT ON TABLE dim_genre IS 'Dimension des genres de jeux';
COMMENT ON TABLE dim_tag IS 'Dimension des tags/étiquettes';
COMMENT ON TABLE dim_categorie IS 'Dimension des catégories Steam';
COMMENT ON TABLE dim_plateforme IS 'Dimension des combinaisons de plateformes';
-- ============================================================================
-- FIN DU SCRIPT DE CRÉATION
-- ============================================================================
-- Message de confirmation
DO $$ BEGIN RAISE NOTICE '============================================================================';
RAISE NOTICE 'Base de données GAMEDATA360 créée avec succès !';
RAISE NOTICE 'Modèle: Star Schema (Étoile)';
RAISE NOTICE 'Tables de dimensions: 7';
RAISE NOTICE 'Table de faits: 1';
RAISE NOTICE 'Tables de liaison: 5';
RAISE NOTICE 'Vues analytiques: 4';
RAISE NOTICE '============================================================================';
END $$;