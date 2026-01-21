# -*- coding: utf-8 -*-
"""GAMEDATA360 - Import CSV vers PostgreSQL"""

import pandas as pd
from sqlalchemy import create_engine, text

# CONFIG
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'GAMEDB',  
    'user': 'postgres',    
    'password': 'Mot de passe' 
}

CSV_PATH = 'data/nettoyes/jeux_analysis_final.csv'

# CONNEXION
connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(connection_string)

print("Connexion a PostgreSQL...")

# IMPORT DU CSV
print(f"Chargement du CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"{len(df):,} lignes chargees")

# Import dans staging_jeux
print("Import dans staging_jeux...")
df.to_sql('staging_jeux', engine, if_exists='replace', index=False)
print(f"Import termine !")

# VERIFICATION
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM staging_jeux"))
    count = result.scalar()
    print(f"\nVerification: {count:,} lignes dans staging_jeux")

print("\nImport reussi !")
