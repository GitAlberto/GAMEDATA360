# 🎮 GameData360 — Dashboard d'Analyse Stratégique du Marché du Jeu Vidéo

Dashboard interactif d'analyse du marché Steam, basé sur un dataset de **103 367 jeux**. 
Transforme les données brutes en insights actionnables via 9 modules d'analyse dédiés.

## Contexte

Projet réalisé en équipe (4 personnes) dans le cadre de [préciser : formation/Bachelor ECE].
Objectif : construire un outil d'aide à la décision stratégique pour comprendre les dynamiques 
du marché du jeu vidéo PC (tendances, économie, comportement joueurs, segmentation).

## Stack technique

- **Python** — pandas, scikit-learn (clustering ML)
- **Streamlit** — interface dashboard multi-pages
- **Plotly** — visualisations interactives
- **PostgreSQL** — stockage et requêtage des données

## Modules d'analyse

1. Marché Global — volume, revenus, prix, plateformes, genres dominants
2. Comportement Joueurs — engagement, playtime, popularité (Peak CCU)
3. Ratings & Sentiment — score Metacritic, sentiment communauté
4. Genres & Tags — combinaisons gagnantes, tendances émergentes
5. Économie — analyse Pareto, pricing power, market share
6. Tendances sur 10 ans — croissance, boom Free-to-Play, impact COVID
7. Segmentation des joueurs — Casual/Hardcore, Budget, Quality Seekers
8. Published vs Beta — comparaison Early Access vs jeux publiés
9. Exploration de données — recherche avancée, comparateur

## Insights clés

- Effet Long Tail marqué : ~20% des jeux génèrent ~80% des revenus
- Croissance du Free-to-Play (~30% du marché)
- Segment Casual majoritaire (playtime médian < 5h)
- Score Metacritic moyen : ~70-75

## Ma contribution

Chef de projet et contributeur principal : conception de l'architecture du dashboard, 
implémentation du pipeline PostgreSQL, développement de la majorité des modules d'analyse 
(dont le module de clustering ML pour la segmentation des joueurs), coordination de l'équipe 
de 4 personnes.

## Limites

- Données limitées à la plateforme Steam (PC uniquement)
- Revenus estimés, non officiels
- Couverture Metacritic partielle (~30% des jeux)

---
Équipe : Katia Boussad, Amine Kone, Ulrich Eneli Eneli, Alberto Bonguele
