# -*- coding: utf-8 -*-
"""ETL Simple - CSV vers PostgreSQL"""

import pandas as pd
from sqlalchemy import create_engine, text
import ast

# CONFIG
DB = "postgresql://postgres:Mot de passe@localhost:5432/GAMEDB"
CSV = "data/nettoyes/jeux_analysis_final.csv"

engine = create_engine(DB)
print("Connexion OK")

# 1. Charger CSV
print("Chargement CSV...")
df = pd.read_csv(CSV)
print(f"  {len(df)} jeux")

# 2. Import staging
print("Import staging_jeux...")
df.to_sql('staging_jeux', engine, if_exists='replace', index=False)

# 3. Plateformes
print("Plateformes...")
plat = df[['Windows', 'Mac', 'Linux']].drop_duplicates()
plat.columns = ['windows', 'mac', 'linux']
plat.to_sql('dim_plateforme', engine, if_exists='append', index=False)

# Fonction extraction liste
def get_list(val):
    if pd.isna(val) or val == '[]': return []
    try: return ast.literal_eval(val)
    except: return []

# 4. Dimensions
print("Dimensions...")
for col, table in [('Genres','dim_genre'), ('Categories','dim_categorie'), ('Tags','dim_tag'), ('Developers','dim_developpeur'), ('Publishers','dim_editeur')]:
    items = set()
    for v in df[col].dropna():
        for x in get_list(v): items.add(x.strip())
    pd.DataFrame({'nom': list(items)}).to_sql(table, engine, if_exists='append', index=False)
    print(f"  {table}: {len(items)}")

# 5. Fait
print("Table fait_jeux...")
with engine.connect() as c:
    plat_map = pd.read_sql("SELECT id_plateforme, windows, mac, linux FROM dim_plateforme", c)

fait = df[['AppID','Name','Release date','Release Year','Price','Estimated owners','Estimated sales','Estimated revenue',
           'Peak CCU','User score','Metacritic score','Positive','Negative','Recommendations','Achievements',
           'Average playtime forever','Median playtime forever','Windows','Mac','Linux']].copy()
fait.columns = ['appid','nom','date_sortie','annee_sortie','prix','proprietaires','ventes','revenus',
                'peak_ccu','score_user','score_metacritic','avis_positifs','avis_negatifs','recommandations','succes',
                'temps_moyen','temps_median','windows','mac','linux']
fait = fait.merge(plat_map, on=['windows','mac','linux'], how='left')
fait = fait.drop(columns=['windows','mac','linux'])
fait.to_sql('fait_jeux', engine, if_exists='append', index=False)
print(f"  {len(fait)} jeux")

# 6. Liaisons
print("Liaisons...")
with engine.connect() as c:
    fait_map = pd.read_sql("SELECT id_fait, appid FROM fait_jeux", c)

for col, table, dim_table, id_col in [
    ('Genres','jeu_genre','dim_genre','id_genre'),
    ('Categories','jeu_categorie','dim_categorie','id_categorie'),
    ('Tags','jeu_tag','dim_tag','id_tag'),
    ('Developers','jeu_developpeur','dim_developpeur','id_developpeur'),
    ('Publishers','jeu_editeur','dim_editeur','id_editeur')
]:
    with engine.connect() as c:
        dim_map = pd.read_sql(f"SELECT {id_col}, nom FROM {dim_table}", c)
    
    links = []
    for _, row in df.iterrows():
        for v in get_list(row[col]):
            links.append({'appid': row['AppID'], 'nom': v.strip()})
    
    if links:
        ldf = pd.DataFrame(links).merge(fait_map, on='appid').merge(dim_map, on='nom')
        ldf[['id_fait', id_col]].drop_duplicates().to_sql(table, engine, if_exists='append', index=False)
        print(f"  {table}: {len(ldf)}")

print("\nTermine!")
