# -*- coding: utf-8 -*-
"""
GameData360 - Page d'Accueil
=============================
Dashboard d'analyse stratégique du marché du jeu vidéo sur Steam.
Présente le contexte, les objectifs, et guide l'utilisateur.

Auteur: GameData360 Team
Version: 3.0 (Professional Edition)
"""

import streamlit as st
import pandas as pd

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="GameData360 — Accueil",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le thème gaming
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    
    .stMetric {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(255,0,255,0.1) 100%);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0,255,136,0.05), rgba(0,255,255,0.05));
        border-left: 4px solid #00ff88;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. HEADER
# ============================================================
st.markdown("# 🎮 GameData360")
st.markdown("### Dashboard d'Analyse Stratégique du Marché du Jeu Vidéo")
st.markdown("---")

# ============================================================
# 3. CONTEXTE & OBJECTIFS
# ============================================================
st.markdown("## 📋 Contexte du Projet")

st.markdown("""
**GameData360** est un outil d'analyse stratégique basé sur les données de la plateforme **Steam**, 
le plus grand distributeur de jeux vidéo PC avec plus de **103,000 jeux** référencés.

Notre mission : **Transformer les données brutes en insights actionnables** pour comprendre 
les dynamiques du marché, identifier les opportunités, et guider les décisions stratégiques.
""")

col_obj1, col_obj2, col_obj3 = st.columns(3)

with col_obj1:
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Objectifs</h4>
    <ul>
        <li>Analyser les tendances du marché</li>
        <li>Identifier les segments porteurs</li>
        <li>Comprendre les comportements joueurs</li>
        <li>Évaluer la performance économique</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_obj2:
    st.markdown("""
    <div class="info-box">
    <h4>💪 Points Forts</h4>
    <ul>
        <li>Base de données exhaustive (+103k jeux)</li>
        <li>Analyses multi-dimensionnelles</li>
        <li>Visualisations interactives (Plotly)</li>
        <li>Insights automatiques & actionnables</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_obj3:
    st.markdown("""
    <div class="info-box">
    <h4>⚠️ Défis & Limitations</h4>
    <ul>
        <li>Données limitées à Steam (PC)</li>
        <li>Revenus estimés (non officiels)</li>
        <li>Metacritic incomplet (~30% jeux)</li>
        <li>Biais vers jeux récents/populaires</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 4. STRUCTURE DU DASHBOARD
# ============================================================
st.markdown("## 🗺️ Navigation & Analyses Disponibles")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown("""
    ### 📊 Analyses Générales
    
    1. **🌍 Marché Global**  
       Vue d'ensemble du marché : volume, revenus, prix, plateformes, genres dominants
    
    2. **👥 Comportement Joueurs**  
       Engagement, playtime, popularité (Peak CCU), patterns de consommation
    
    3. **⭐ Ratings & Sentiment**  
       Qualité critique (Metacritic), sentiment communauté, polarisation, ROI qualité
    
    4. **🎮 Genres & Tags**  
       Combinaisons gagnantes, co-occurrences, tags émergents, analyse mix genres
    """)

with nav_col2:
    st.markdown("""
    ### 🔍 Analyses Avancées
    
    5. **💰 Économie**  
       Analyse économique : Pareto, pricing power, market share, value for money
    
    6. **📈 Tendances 10 Ans**  
       Évolution temporelle : croissance marché, boom F2P, saturation, COVID impact
    
    7. **👥 Segmentation Joueurs**  
       Segments comportementaux : Casual/Hardcore, Budget, Social, Quality Seekers
    
    8. **⚔️ Published vs Beta**  
       Comparaison jeux publiés vs Early Access : pricing, qualité, engagement
    
    9. **🔍 Exploration de Données**  
       Outil de recherche avancée, filtrage granulaire, comparateur, data browser
    """)

st.markdown("---")

# ============================================================
# 5. DICTIONNAIRE DES DONNÉES
# ============================================================
st.markdown("## 📊 Dictionnaire des Données")
st.markdown("Colonnes principales utilisées dans les analyses")

# Créer le dictionnaire
data_dict = pd.DataFrame({
    'Colonne': [
        'AppID',
        'Name',
        'Release Year',
        'Price',
        'Estimated revenue',
        'Positive / Negative',
        'Metacritic score',
        'Peak CCU',
        'Median playtime forever',
        'Genres',
        'Categories',
        'Tags',
        'Windows / Mac / Linux',
        'Developers / Publishers'
    ],
    'Type': [
        'Identifiant',
        'Texte',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Numérique',
        'Liste',
        'Liste',
        'Liste',
        'Booléen',
        'Texte'
    ],
    'Description': [
        'Identifiant unique Steam du jeu',
        'Nom du jeu',
        'Année de sortie du jeu (1997-2024)',
        'Prix en USD ($0 = Free-to-Play)',
        'Revenus estimés cumulés (USD) - Non officiel',
        'Nombre de reviews positives/négatives Steam',
        'Score critique Metacritic (0-100, ~30% des jeux)',
        'Pic de joueurs simultanés (Concurrent Users)',
        'Temps de jeu médian des joueurs (minutes)',
        'Genres du jeu (Action, RPG, Strategy...)',
        'Catégories (Single-player, Multi-player, Co-op...)',
        'Tags communautaires descriptifs',
        'Support des plateformes PC',
        'Studio développeur / Éditeur'
    ],
    'Utilisation': [
        'Identification unique',
        'Recherche, filtrage',
        'Analyse temporelle, tendances',
        'Segmentation économique, F2P vs Payant',
        'Analyse économique, market share',
        'Sentiment communauté, popularité',
        'Qualité critique, benchmarking',
        'Popularité temps réel, engagement',
        'Engagement joueur, rétention',
        'Segmentation marché, combinaisons',
        'Préférences sociales (Solo/Multi)',
        'Tendances émergentes, co-occurrences',
        'Analyse support multi-plateforme',
        'Analyse par studio/éditeur'
    ]
})

# Afficher le tableau avec style
st.dataframe(
    data_dict,
    hide_index=True,
    use_container_width=True,
    height=500
)

st.markdown("---")

# ============================================================
# 6. INSIGHTS CLÉS & ENJEUX
# ============================================================
st.markdown("## 💡 Insights Clés du Marché")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.markdown("""
    ### 📈 Croissance
    
    - **+103,000 jeux** sur Steam
    - Croissance exponentielle depuis 2015
    - Boom du **Free-to-Play** (~30% du marché)
    - Saturation du marché visible post-2020
    """)

with insight_col2:
    st.markdown("""
    ### 💰 Économie
    
    - **Long Tail** prononcée (20% jeux = 80% revenus)
    - Prix médian : **~$10-15**
    - Segment **Indie** dominant en volume
    - **AAA** dominant en revenus par titre
    """)

with insight_col3:
    st.markdown("""
    ### 🎮 Comportement
    
    - **Casual** majoritaire (playtime < 5h)
    - **Multiplayer** en forte croissance
    - Metacritic moyen : **~70-75**
    - Polarisation forte sur certains titres
    """)

st.markdown("---")

# ============================================================
# 7. GUIDE D'UTILISATION
# ============================================================
st.markdown("## 🚀 Guide de Démarrage Rapide")

st.markdown("""
### Pour commencer votre analyse :

1. **📊 Commencez par "Marché Global"** pour comprendre le paysage général
2. **🎯 Utilisez les filtres** (sidebar) pour affiner vos analyses par genre, prix, année...
3. **📈 Explorez les tendances temporelles** pour comprendre l'évolution du marché
4. **💰 Analysez l'économie** pour identifier les segments rentables
5. **🔍 Utilisez "Exploration de Données"** pour des recherches spécifiques et comparaisons

### 💡 Conseils :
- Les graphiques sont **interactifs** : survolez, zoomez, cliquez pour plus de détails
- Les **insights automatiques** en haut de chaque page vous guident
- Utilisez le **bouton Reset** pour revenir à la vue complète
- Les **KPIs** affichent des deltas quand des filtres sont actifs
""")

st.markdown("---")

# ============================================================
# 8. FOOTER
# ============================================================
st.markdown("## 📞 Contact & Support")

st.markdown("""
**GameData360** — Dashboard d'Analyse Stratégique du Marché du Jeu Vidéo  
Data source: Steam Platform | Analyse: 103,367 jeux

Pour toute question ou suggestion d'amélioration, contactez l'équipe GameData360.

---

🎮 **Bonne exploration !**
""")