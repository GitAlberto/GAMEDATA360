-- GAMEDATA360 - ETL (conforme aux tables)

-- 1. Plateformes
INSERT INTO dim_plateforme (windows, mac, linux)
SELECT DISTINCT "Windows", "Mac", "Linux" FROM staging_jeux
ON CONFLICT DO NOTHING;

-- 2. Genres
INSERT INTO dim_genre (nom)
SELECT DISTINCT TRIM(REPLACE(REPLACE(unnest(string_to_array(REPLACE(REPLACE("Genres", '[', ''), ']', ''), ',')), '''', ''), '"', ''))
FROM staging_jeux WHERE "Genres" IS NOT NULL AND "Genres" != '[]'
ON CONFLICT DO NOTHING;

-- 3. Categories
INSERT INTO dim_categorie (nom)
SELECT DISTINCT TRIM(REPLACE(REPLACE(unnest(string_to_array(REPLACE(REPLACE("Categories", '[', ''), ']', ''), ',')), '''', ''), '"', ''))
FROM staging_jeux WHERE "Categories" IS NOT NULL AND "Categories" != '[]'
ON CONFLICT DO NOTHING;

-- 4. Tags
INSERT INTO dim_tag (nom)
SELECT DISTINCT TRIM(REPLACE(REPLACE(unnest(string_to_array(REPLACE(REPLACE("Tags", '[', ''), ']', ''), ',')), '''', ''), '"', ''))
FROM staging_jeux WHERE "Tags" IS NOT NULL AND "Tags" != '[]'
ON CONFLICT DO NOTHING;

-- 5. Developpeurs
INSERT INTO dim_developpeur (nom)
SELECT DISTINCT TRIM("Developers") FROM staging_jeux 
WHERE "Developers" IS NOT NULL AND "Developers" != ''
ON CONFLICT DO NOTHING;

-- 6. Editeurs
INSERT INTO dim_editeur (nom)
SELECT DISTINCT TRIM("Publishers") FROM staging_jeux 
WHERE "Publishers" IS NOT NULL AND "Publishers" != ''
ON CONFLICT DO NOTHING;

-- 7. Faits
INSERT INTO fait_jeux (appid, id_plateforme, nom, date_sortie, annee, proprietaires, prix, peak_ccu, 
                       score_user, score_metacritic, avis_positifs, avis_negatifs, 
                       temps_moyen, temps_median, succes, recommandations, ventes, revenus, website, published)
SELECT 
    s."AppID", p.id, s."Name", s."Release date", s."Release Year", s."Estimated owners", s."Price", s."Peak CCU",
    s."User score", s."Metacritic score", s."Positive", s."Negative",
    s."Average playtime forever", s."Median playtime forever", s."Achievements", s."Recommendations",
    s."Estimated sales", s."Estimated revenue", s."Website", s."published"
FROM staging_jeux s
LEFT JOIN dim_plateforme p ON p.windows = s."Windows" AND p.mac = s."Mac" AND p.linux = s."Linux"
ON CONFLICT (appid) DO NOTHING;

-- 8. Liaison jeu-genre
INSERT INTO jeu_genre (id_fait, id_dim)
SELECT f.id, g.id
FROM staging_jeux s
JOIN fait_jeux f ON f.appid = s."AppID"
CROSS JOIN LATERAL unnest(string_to_array(REPLACE(REPLACE(s."Genres", '[', ''), ']', ''), ',')) AS gname
JOIN dim_genre g ON g.nom = TRIM(REPLACE(REPLACE(gname, '''', ''), '"', ''))
WHERE s."Genres" IS NOT NULL AND s."Genres" != '[]'
ON CONFLICT DO NOTHING;

-- 9. Liaison jeu-categorie
INSERT INTO jeu_categorie (id_fait, id_dim)
SELECT f.id, c.id
FROM staging_jeux s
JOIN fait_jeux f ON f.appid = s."AppID"
CROSS JOIN LATERAL unnest(string_to_array(REPLACE(REPLACE(s."Categories", '[', ''), ']', ''), ',')) AS cname
JOIN dim_categorie c ON c.nom = TRIM(REPLACE(REPLACE(cname, '''', ''), '"', ''))
WHERE s."Categories" IS NOT NULL AND s."Categories" != '[]'
ON CONFLICT DO NOTHING;

-- 10. Liaison jeu-tag
INSERT INTO jeu_tag (id_fait, id_dim)
SELECT f.id, t.id
FROM staging_jeux s
JOIN fait_jeux f ON f.appid = s."AppID"
CROSS JOIN LATERAL unnest(string_to_array(REPLACE(REPLACE(s."Tags", '[', ''), ']', ''), ',')) AS tname
JOIN dim_tag t ON t.nom = TRIM(REPLACE(REPLACE(tname, '''', ''), '"', ''))
WHERE s."Tags" IS NOT NULL AND s."Tags" != '[]'
ON CONFLICT DO NOTHING;

-- 11. Liaison jeu-developpeur
INSERT INTO jeu_developpeur (id_fait, id_dim)
SELECT f.id, d.id
FROM staging_jeux s
JOIN fait_jeux f ON f.appid = s."AppID"
JOIN dim_developpeur d ON d.nom = TRIM(s."Developers")
WHERE s."Developers" IS NOT NULL AND s."Developers" != ''
ON CONFLICT DO NOTHING;

-- 12. Liaison jeu-editeur
INSERT INTO jeu_editeur (id_fait, id_dim)
SELECT f.id, e.id
FROM staging_jeux s
JOIN fait_jeux f ON f.appid = s."AppID"
JOIN dim_editeur e ON e.nom = TRIM(s."Publishers")
WHERE s."Publishers" IS NOT NULL AND s."Publishers" != ''
ON CONFLICT DO NOTHING;

-- Verification
SELECT 'fait_jeux' AS t, COUNT(*) FROM fait_jeux
UNION ALL SELECT 'dim_genre', COUNT(*) FROM dim_genre
UNION ALL SELECT 'dim_categorie', COUNT(*) FROM dim_categorie
UNION ALL SELECT 'dim_tag', COUNT(*) FROM dim_tag
UNION ALL SELECT 'dim_developpeur', COUNT(*) FROM dim_developpeur
UNION ALL SELECT 'dim_editeur', COUNT(*) FROM dim_editeur;
